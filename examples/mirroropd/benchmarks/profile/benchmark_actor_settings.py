# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Fixed-token actor settings probe; excludes teacher preparation, Ray/TQ and rollout.

Use original cached teacher states and actual production objective/optimizer paths.
Repeated inputs scale per-rank work only; they are not fresh scientific samples.
LR is zero, but AdamW, clipping, backward and model/optimizer transfers execute.
"""

import argparse
import copy
import hashlib
import inspect
import json
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from tensordict import TensorDict

from examples.mirroropd.benchmarks.profile.benchmark_termination_training import parameters_hash
from examples.mirroropd.benchmarks.profile.pipeline import worker
from verl.trainer.distillation.losses import distillation_loss
from verl.trainer.distillation.profiling import full_kl_range
from verl.utils import tensordict_utils as tu
from verl.utils.checkpoint.checkpoint_manager import BaseCheckpointManager
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
            backend=args.backend,
            token_chunk_size=args.chunk,
            termination_token_ids=[151643, 151645],
            future={"mode": args.future_mode, "gamma": args.gamma},
        ),
        use_policy_gradient=False,
        use_task_rewards=False,
        policy_loss_mode="vanilla",
        loss_max_clamp=None,
        log_prob_min_clamp=None,
    )
    config = DistillationConfig(distillation_loss=loss)

    def configure(case):
        actor.engine.configure_full_vocab(
            case.distillation_loss.full_vocab,
            cached["teacher"],
            spec=case.distillation_loss.loss_settings,
            loss_agg_mode=args.loss_agg_mode,
        )

        def loss_fn(model_output, data, dp_group):
            return distillation_loss(SimpleNamespace(loss_agg_mode=args.loss_agg_mode), case, model_output, data)

        actor.set_loss_fn(loss_fn)

    configure(config)
    records = []
    current_case = "single"
    current_block = 0
    forward_counts = {"train": 0, "capture": 0}
    original_forward = actor.engine.forward_step

    def counted_forward(micro_batch, loss_function, forward_only):
        if tu.get_non_tensor_data(micro_batch, "full_vocab_train", default=False):
            forward_counts["train"] += 1
        if tu.get_non_tensor_data(micro_batch, "full_vocab_capture", default=None):
            forward_counts["capture"] += 1
        return original_forward(micro_batch, loss_function, forward_only)

    actor.engine.forward_step = counted_forward

    def execute(index, *, warmup=False, trace=False):
        batch = data.clone()
        before_counts = dict(forward_counts)
        reuse = args.reuse if args.paired_cases else (2 if warmup or trace else args.reuse)
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
        local_counts = {k: forward_counts[k] - before_counts[k] for k in forward_counts}
        rank_counts = [None] * world
        dist.all_gather_object(rank_counts, local_counts)
        updates = tu.get(output, "update_records")
        assert tu.get(output, "optimizer_steps") == len(updates) == reuse
        values = [r["metrics"]["optim/actor/loss"] for r in updates]
        norms = [r["metrics"]["optim/actor/grad_norm_preclip"] for r in updates]
        assert bool(torch.isfinite(torch.tensor(values + norms)).all())
        record = dict(
            index=index,
            case=current_case,
            block=current_block,
            settings={k: getattr(args, k) for k in ("backend", "future_mode", "gamma", "chunk", "loss_agg_mode")},
            forward_counts=local_counts,
            per_rank_forward_counts=rank_counts,
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

    if args.paired_cases:
        cases = json.loads(args.paired_cases.read_text())
        if len(cases) < 2 or any(
            set(c) - {"name", "backend", "future_mode", "gamma", "chunk", "loss_agg_mode"} for c in cases
        ):
            raise ValueError("Cases require a name and only backend/future/gamma/chunk/aggregation overrides")
        if len({c["name"] for c in cases}) != len(cases):
            raise ValueError("Paired case names must be unique")
        initial_optimizer = copy.deepcopy(actor.engine.optimizer.state_dict())
        initial_scheduler = copy.deepcopy(actor.engine.lr_scheduler.state_dict())
        initial_rng = BaseCheckpointManager.get_rng_state()
        defaults = {k: getattr(args, k) for k in ("backend", "future_mode", "gamma", "chunk", "loss_agg_mode")}
        # Every alternative gets its own A-B-B-A comparison with the first case.
        order = [c for alternative in cases[1:] for c in (cases[0], alternative, alternative, cases[0])]
        for current_block, case in enumerate(order):
            current_case = case["name"]
            for key, default in defaults.items():
                setattr(args, key, case.get(key, default))
            loss = replace(
                loss,
                full_vocab=FullVocabConfig(
                    backend=args.backend,
                    token_chunk_size=args.chunk,
                    termination_token_ids=[151643, 151645],
                    future={"mode": args.future_mode, "gamma": args.gamma},
                ),
            )
            config = DistillationConfig(distillation_loss=loss)
            configure(config)
            actor.engine.optimizer.load_state_dict(copy.deepcopy(initial_optimizer))
            actor.engine.lr_scheduler.load_state_dict(copy.deepcopy(initial_scheduler))
            BaseCheckpointManager.load_rng_state(initial_rng)
            for index in range(args.warmups + args.samples):
                execute(index, warmup=index < args.warmups)
        for key, value in defaults.items():
            setattr(args, key, value)
    else:
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
                    paired_cases=json.loads(args.paired_cases.read_text()) if args.paired_cases else None,
                    parameters_unchanged_on_all_ranks=True,
                    parameter_hashes=hashes,
                    execution=dict(
                        lr=0,
                        precision="bf16",
                        kappa=1,
                        eta=0.1,
                        backend=args.backend,
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
    parser.add_argument("--backend", choices=["precompute", "precompute_fused"], default="precompute_fused")
    parser.add_argument("--loss-agg-mode", choices=["token-mean", "seq-mean-token-mean"], default="token-mean")
    parser.add_argument("--future-mode", default="none")
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument(
        "--paired-cases", type=Path, help="JSON cases; run each alternative against baseline in ABBA order"
    )
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--checkpoint", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--diagnostics", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-sync", action="store_true")
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()
    if min(args.repeat_input, args.reuse, args.samples, args.budget, args.chunk, args.warmups) <= 0:
        parser.error("Input repeat, reuse, samples and token budgets must be positive")
    if args.paired_cases and args.trace:
        parser.error("Trace the single-case probe separately from paired timing")
    run(args)


if __name__ == "__main__":
    main()
