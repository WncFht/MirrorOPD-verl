# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Paired real-model actor benchmark with frozen original tokens and EOS semantics.

Teacher hidden states are prepared once. Timed calls include native actor/anchor
FSDP2 execution, reuse, backward, AdamW (lr=0), and model/optimizer offload. They
exclude teacher preparation, TQ and rollout. Parameter hashes verify fixed weights.
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

from examples.mirroropd.benchmarks.profile.pipeline import loss_fn, worker
from verl.trainer.distillation.profiling import full_kl_range
from verl.utils import tensordict_utils as tu
from verl.utils.distributed import initialize_global_process_group
from verl.workers.config.distillation import DistillationConfig, DistillationLossConfig, FullVocabConfig
from verl.workers.engine_workers import ActorRolloutRefWorker
from verl.workers.utils.padding import prediction_positions


def prepare(args):
    from examples.mirroropd.research.backends.hf_models import hidden, load

    if args.output.exists():
        raise FileExistsError(args.output)
    bank = json.loads(args.bank.read_text())
    if args.batch > len(bank):
        raise ValueError("Requested more distinct samples than the bank contains")
    indices = torch.linspace(0, len(bank) - 1, args.batch).long().tolist()
    rows = [
        {
            "id": bank[i]["id"],
            "prompt_ids": bank[i]["prompt_ids"],
            "response_ids": bank[i]["response_ids"][: args.max_response],
        }
        for i in indices
    ]
    model = load(None, path=str(args.teacher))
    states = []
    with torch.inference_mode():
        for row in rows:
            states.append(hidden(model, [row], pad_id=0).cpu())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"rows": rows, "teacher_hidden": states, "teacher": str(args.teacher.resolve())}, args.output)
    args.output.with_suffix(".json").write_text(
        json.dumps(
            {
                "teacher": str(args.teacher.resolve()),
                "bank": str(args.bank.resolve()),
                "bank_sha256": hashlib.sha256(args.bank.read_bytes()).hexdigest(),
                "responses": [len(r["response_ids"]) for r in rows],
                "precision": "native BF16 teacher, original FP32 buffers; no new generation",
            },
            indent=2,
        )
        + "\n"
    )


def parameters_hash(actor):
    digest = hashlib.sha256()
    for name, value in actor.engine.module.named_parameters():
        local = value.detach().to_local().cpu().contiguous()
        digest.update(name.encode())
        digest.update(local.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def run(args):
    torch.set_num_threads(1)
    _, rank, world = initialize_global_process_group(timeout_second=600)
    cached = torch.load(args.cache, map_location="cpu", weights_only=True)
    rows, states = cached["rows"][rank::world], cached["teacher_hidden"][rank::world]
    if len(cached["rows"]) % world:
        raise ValueError("Batch must divide across ranks")
    if args.output.exists() or args.output.with_suffix(".progress.jsonl").exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)

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
            "teacher_hidden": nested(states),
        },
        batch_size=[len(rows)],
    )
    positions, _ = prediction_positions(data["input_ids"], data["loss_mask"])
    data["teacher_positions"] = nested(positions)
    actor = worker(str(args.student), world, False, args.budget, True, no_sync=False)
    initial_hash = parameters_hash(actor)
    records = []
    baseline_loss = None
    baseline_norm = None

    def execute(backend, step, warmup=False, trace=False):
        nonlocal baseline_loss, baseline_norm
        loss = DistillationLossConfig(
            loss_mode="mirror",
            mirror=dict(objective="mdpo", kappa=args.kappa, eta=args.eta, diagnostics=True),
            full_vocab=FullVocabConfig(
                backend=backend, token_chunk_size=args.chunk, termination_token_ids=[151643, 151645]
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
        batch = data.clone()
        reuse = 2 if trace else args.reuse
        tu.assign_non_tensor(
            batch,
            temperature=1.0,
            compute_loss=False,
            full_vocab_teacher=False,
            full_vocab_train=True,
            mirror_anchor_version=(step, 0),
            global_batch_size=len(cached["rows"]),
            mini_batch_size=len(cached["rows"]),
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
        started = time.perf_counter()
        with full_kl_range("actor_phase_including_offload"):
            output = inspect.unwrap(ActorRolloutRefWorker.update_actor)(
                SimpleNamespace(actor=actor, mirror_spec=loss.loss_settings), batch
            )
        torch.cuda.synchronize()
        elapsed = torch.tensor(
            [time.perf_counter() - started, torch.cuda.max_memory_allocated() / 2**30], device="cuda"
        )
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        updates = tu.get(output, "update_records")
        assert tu.get(output, "optimizer_steps") == reuse == len(updates)
        values = [r["metrics"]["optim/actor/loss"] for r in updates]
        norms = [r["metrics"]["optim/actor/grad_norm_preclip"] for r in updates]
        assert all(torch.isfinite(torch.tensor(values))) and all(torch.isfinite(torch.tensor(norms)))
        if backend == "precompute" and baseline_loss is None:
            baseline_loss, baseline_norm = values[0], norms[0]
        record = dict(
            backend=backend,
            warmup=warmup,
            trace=trace,
            reuse=reuse,
            actor_seconds=elapsed[0].item(),
            allocated_peak_gib=elapsed[1].item(),
            loss=values,
            grad_norm=norms,
        )
        records.append(record)
        if rank == 0:
            with args.output.with_suffix(".progress.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
        torch.testing.assert_close(torch.tensor(values), torch.full((reuse,), baseline_loss), atol=2e-5, rtol=2e-4)
        torch.testing.assert_close(torch.tensor(norms), torch.full((reuse,), baseline_norm), atol=2e-5, rtol=1e-3)
        return record

    step = 0
    for backend in ("precompute", "precompute_fused"):
        execute(backend, step, warmup=True)
        step += 1
    for repeat in range(args.repeats):
        order = ("precompute", "precompute_fused") if repeat % 2 == 0 else ("precompute_fused", "precompute")
        for backend in order:
            execute(backend, step)
            step += 1
    if args.trace:
        for backend in ("precompute", "precompute_fused"):
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
            ) as prof:
                execute(backend, step, trace=True)
            prof.export_chrome_trace(str(args.output.parent / f"{backend}-rank{rank}.json.gz"))
            step += 1
    final_hash = parameters_hash(actor)
    assert initial_hash == final_hash
    hashes = [None] * world
    dist.all_gather_object(hashes, {"rank": rank, "before": initial_hash, "after": final_hash})
    if rank == 0:
        args.output.write_text(
            json.dumps(
                {
                    "scope": __doc__,
                    "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    "world_size": world,
                    "gpu": torch.cuda.get_device_name(),
                    "torch": torch.__version__,
                    "input_tokens": sum(len(r["prompt_ids"]) + len(r["response_ids"]) for r in cached["rows"]),
                    "loss_tokens": sum(len(r["response_ids"]) for r in cached["rows"]),
                    "cache_sha256": hashlib.sha256(args.cache.read_bytes()).hexdigest(),
                    "parameters_unchanged_on_all_ranks": True,
                    "parameter_hashes": hashes,
                    "execution": {
                        "lr": 0.0,
                        "gradient_checkpointing": True,
                        "gradient_no_sync": False,
                        "diagnostics": True,
                        "shuffle": False,
                    },
                    "records": records,
                },
                indent=2,
            )
            + "\n"
        )
    dist.barrier()
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--bank", type=Path, required=True)
    prep.add_argument("--teacher", type=Path, required=True)
    prep.add_argument("--batch", type=int, default=8)
    prep.add_argument("--max-response", type=int, default=512)
    prep.add_argument("--output", type=Path, required=True)
    bench = sub.add_parser("run")
    bench.add_argument("--student", type=Path, required=True)
    bench.add_argument("--cache", type=Path, required=True)
    bench.add_argument("--output", type=Path, required=True)
    bench.add_argument("--reuse", type=int, default=16)
    bench.add_argument("--repeats", type=int, default=2)
    bench.add_argument("--chunk", type=int, default=512)
    bench.add_argument("--budget", type=int, default=8192)
    bench.add_argument("--kappa", type=float, default=0.5)
    bench.add_argument("--eta", type=float, default=0.1)
    bench.add_argument("--trace", action="store_true")
    args = parser.parse_args()
    positive = (
        (args.batch, args.max_response)
        if args.action == "prepare"
        else (args.reuse, args.repeats, args.chunk, args.budget)
    )
    if any(value <= 0 for value in positive):
        parser.error("Batch, lengths, reuse, repeats and chunk/budget values must be positive")
    prepare(args) if args.action == "prepare" else run(args)


if __name__ == "__main__":
    main()
