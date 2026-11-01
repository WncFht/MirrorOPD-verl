# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Fixed-token actor settings probe; excludes teacher preparation, Ray/TQ and rollout.

Use original cached teacher states and actual production objective/optimizer paths.
Repeated inputs scale per-rank work only; they are not fresh scientific samples.
LR is zero, but AdamW, clipping, backward and model/optimizer transfers execute.
"""

import argparse
import hashlib
import inspect
import json
import time
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from tensordict import TensorDict

from examples.mirroropd.benchmarks.profile.benchmark_termination_training import parameters_hash
from examples.mirroropd.benchmarks.profile.pipeline import loss_fn, worker
from verl.trainer.distillation.profiling import full_kl_range
from verl.utils import tensordict_utils as tu
from verl.utils.distributed import initialize_global_process_group
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions
from verl.workers.config.distillation import DistillationConfig, DistillationLossConfig, FullVocabConfig
from verl.workers.engine_workers import ActorRolloutRefWorker
from verl.workers.utils.padding import prediction_positions


def run(args):
    torch.set_num_threads(1)
    _, rank, world = initialize_global_process_group(timeout_second=600)
    if args.output.exists() or args.output.with_suffix(".progress.jsonl").exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cached = torch.load(args.cache, map_location="cpu", weights_only=True)
    global_rows = cached["rows"] * args.repeat_input
    global_states = cached["teacher_hidden"] * args.repeat_input
    if len(global_rows) % world:
        raise ValueError("Effective input rows must divide across ranks")
    lengths = [len(r["prompt_ids"]) + len(r["response_ids"]) for r in global_rows]
    if max(lengths) > args.budget:
        raise ValueError("Token budget must hold every complete input")
    partitions = get_seqlen_balanced_partitions(lengths, world, equal_size=True)
    indices = partitions[rank]
    rows = [global_rows[i] for i in indices]

    def nested(values):
        return torch.nested.nested_tensor([torch.as_tensor(v) for v in values], layout=torch.jagged)

    data = TensorDict(
        {
            "input_ids": nested([r["prompt_ids"] + r["response_ids"] for r in rows]),
            "position_ids": nested([list(range(len(r["prompt_ids"]) + len(r["response_ids"]))) for r in rows]),
            "prompts": nested([r["prompt_ids"] for r in rows]),
            "responses": nested([r["response_ids"] for r in rows]),
            "response_mask": nested([torch.ones(len(r["response_ids"]), dtype=torch.long) for r in rows]),
            "loss_mask": nested([torch.ones(len(r["response_ids"]), dtype=torch.long) for r in rows]),
            "teacher_hidden": nested([global_states[i] for i in indices]),
        },
        batch_size=[len(rows)],
    )
    positions, _ = prediction_positions(data["input_ids"], data["loss_mask"])
    data["teacher_positions"] = nested(positions)
    actor = worker(
        str(args.student),
        world,
        False,
        args.budget,
        args.checkpoint,
        no_sync=args.no_sync,
        forward_precision="bf16",
    )
    initial_hash = parameters_hash(actor)
    loss = DistillationLossConfig(
        loss_mode="mirror",
        mirror=dict(objective="mdpo", kappa=1, eta=0.1, diagnostics=args.diagnostics),
        full_vocab=FullVocabConfig(
            backend="precompute_fused", token_chunk_size=args.chunk, termination_token_ids=[151643, 151645]
        ),
        use_policy_gradient=False,
        use_task_rewards=False,
        policy_loss_mode="vanilla",
        loss_max_clamp=None,
        log_prob_min_clamp=None,
    )
    config = DistillationConfig(distillation_loss=loss)
    actor.engine.configure_full_vocab(loss.full_vocab, cached["teacher"], spec=loss.loss_settings)
    actor.set_loss_fn(partial(loss_fn, config=config))
    records = []

    def execute(index, *, warmup=False, trace=False):
        batch = data.clone()
        reuse = 2 if warmup or trace else args.reuse
        tu.assign_non_tensor(
            batch,
            temperature=1.0,
            compute_loss=False,
            full_vocab_teacher=False,
            full_vocab_train=True,
            mirror_anchor_version=(index, 0),
            global_batch_size=len(global_rows),
            mini_batch_size=len(global_rows),
            epochs=reuse,
            seed=42,
            dataloader_kwargs={"shuffle": False},
            collect_metrics=True,
            collect_update_records=True,
            metrics_attempt_start=0,
        )
        dist.barrier()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start_ns = time.monotonic_ns()
        with full_kl_range("actor_phase_including_offload"):
            output = inspect.unwrap(ActorRolloutRefWorker.update_actor)(
                SimpleNamespace(actor=actor, mirror_spec=loss.loss_settings), batch
            )
        torch.cuda.synchronize()
        end_ns = time.monotonic_ns()
        elapsed = torch.tensor(
            [
                (end_ns - start_ns) / 1e9,
                torch.cuda.max_memory_allocated() / 2**30,
                torch.cuda.max_memory_reserved() / 2**30,
            ],
            device="cuda",
            dtype=torch.float64,
        )
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        updates = tu.get(output, "update_records")
        assert tu.get(output, "optimizer_steps") == len(updates) == reuse
        values = [r["metrics"]["optim/actor/loss"] for r in updates]
        norms = [r["metrics"]["optim/actor/grad_norm_preclip"] for r in updates]
        assert bool(torch.isfinite(torch.tensor(values + norms)).all())
        record = dict(
            index=index,
            warmup=warmup,
            trace=trace,
            reuse=reuse,
            actor_seconds=elapsed[0].item(),
            allocated_peak_gib=elapsed[1].item(),
            reserved_peak_gib=elapsed[2].item(),
            loss=values,
            grad_norm=norms,
            interval_monotonic_ns=[start_ns, end_ns],
            updates=updates,
        )
        records.append(record)
        if rank == 0:
            with args.output.with_suffix(".progress.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            print(json.dumps({k: v for k, v in record.items() if k != "updates"}), flush=True)

    execute(0, warmup=True)
    for index in range(1, args.samples + 1):
        execute(index)
    if args.trace:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        ) as profiler:
            execute(args.samples + 1, trace=True)
        profiler.export_chrome_trace(str(args.output.with_name(args.output.stem + f"-rank{rank}.json.gz")))
    final_hash = parameters_hash(actor)
    assert initial_hash == final_hash
    hashes = [None] * world
    dist.all_gather_object(hashes, dict(rank=rank, before=initial_hash, after=final_hash))
    if rank == 0:
        args.output.write_text(
            json.dumps(
                dict(
                    scope=__doc__,
                    arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    world_size=world,
                    gpu=torch.cuda.get_device_name(),
                    torch=torch.__version__,
                    unique_cached_rows=len(cached["rows"]),
                    effective_rows=len(global_rows),
                    input_tokens=sum(lengths),
                    loss_tokens=sum(len(r["response_ids"]) for r in global_rows),
                    per_rank_input_tokens=[sum(lengths[i] for i in partition) for partition in partitions],
                    cache_sha256=hashlib.sha256(args.cache.read_bytes()).hexdigest(),
                    parameters_unchanged_on_all_ranks=True,
                    parameter_hashes=hashes,
                    execution=dict(
                        lr=0,
                        precision="bf16",
                        kappa=1,
                        eta=0.1,
                        backend="precompute_fused",
                        param_optimizer_offload=True,
                    ),
                    records=records,
                ),
                indent=2,
            )
            + "\n"
        )
    dist.barrier()
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeat-input", type=int, default=2)
    parser.add_argument("--reuse", type=int, default=4)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--budget", type=int, default=32768)
    parser.add_argument("--chunk", type=int, default=512)
    parser.add_argument("--checkpoint", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--diagnostics", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-sync", action="store_true")
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()
    if min(args.repeat_input, args.reuse, args.samples, args.budget, args.chunk) <= 0:
        parser.error("Input repeat, reuse, samples and token budgets must be positive")
    run(args)


if __name__ == "__main__":
    main()
