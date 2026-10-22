# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Fixed-prefix real-model FSDP2 probe. Direct worker calls; excludes Ray/TQ/vLLM.

Re-tokenized saved math generations provide realistic unequal lengths, not exact
original rollout token IDs. lr=0 keeps all variants on identical model weights;
AdamW still executes and allocates its ordinary state. Production loss/normalizer,
head hooks, checkpointing, micro-batching and model offload are exercised.
"""

import argparse
import json
import time
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from tensordict import TensorDict
from transformers import AutoTokenizer

from verl.trainer.distillation.losses import distillation_loss
from verl.trainer.distillation.profiling import full_kl_range
from verl.utils import tensordict_utils as tu
from verl.utils.distributed import initialize_global_process_group
from verl.workers.config import (
    DistillationConfig,
    DistillationLossConfig,
    FSDPEngineConfig,
    FSDPOptimizerConfig,
    HFModelConfig,
    TrainingWorkerConfig,
)
from verl.workers.config.distillation import FullVocabConfig
from verl.workers.engine_workers import TrainingWorker


def loss_fn(model_output, data, dp_group, *, config):
    return distillation_loss(SimpleNamespace(loss_agg_mode="token-mean"), config, model_output, data)


def worker(path, world, teacher, budget, checkpoint):
    cfg = TrainingWorkerConfig(
        model_type="language_model",
        model_config=HFModelConfig(
            path=path,
            enable_gradient_checkpointing=checkpoint and not teacher,
            use_fused_kernels=False,
            enable_full_vocab=True,
            use_remove_padding=True,
        ),
        engine_config=FSDPEngineConfig(
            strategy="fsdp2",
            forward_only=teacher,
            model_dtype="bfloat16" if teacher else "float32",
            fsdp_size=world,
            mixed_precision={"param_dtype": "bfloat16", "reduce_dtype": "float32"},
            forward_only_cpu_offload=False,
            param_offload=True,
            optimizer_offload=not teacher,
            use_torch_compile=False,
            infer_max_token_len_per_gpu=budget,
            max_token_len_per_gpu=budget,
            use_no_sync_for_gradient_accumulation=True,
        ),
        optimizer_config=None if teacher else FSDPOptimizerConfig(lr=0.0, total_training_steps=20),
    )
    w = TrainingWorker(cfg)
    w.reset()
    return w


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--student", required=True)
    p.add_argument("--teacher", required=True)
    p.add_argument("--samples", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--backend",
        default="precompute",
        choices=["precompute", "precompute_fused", "fused_candidate", "streaming", "liger"],
    )
    p.add_argument("--chunk", type=int, default=512)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--max-length", type=int, default=8192)
    p.add_argument("--budget", type=int, default=16384)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--warmup-steps", type=int, default=1)
    p.add_argument("--no-checkpoint", action="store_true")
    p.add_argument("--trace", action="store_true")
    p.add_argument("--memory", action="store_true")
    args = p.parse_args()
    if not 0 <= args.warmup_steps < args.steps - int(args.trace):
        p.error("warmup must leave at least one unprofiled measured step")
    if args.backend == "fused_candidate":
        from fused_candidate import FusedPrecomputedKL

        import verl.trainer.distillation.full_vocab as fv

        fv.PrecomputedKL = FusedPrecomputedKL  # process-local experimental substitution only
    torch.set_num_threads(1)
    _, rank, world = initialize_global_process_group(timeout_second=600)
    assert args.batch % world == 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.student)
    samples = []
    with args.samples.open() as f:
        for line in f:
            r = json.loads(line)
            prompt = tokenizer.encode(r["input"], add_special_tokens=False)[:1024]
            response = tokenizer.encode(r["output"], add_special_tokens=False)[: args.max_length - len(prompt)]
            if prompt and response:
                samples.append((prompt, response))
            if len(samples) == args.batch:
                break
    assert len(samples) == args.batch
    local = samples[rank::world]

    def nested(rows):
        return torch.nested.nested_tensor([torch.tensor(x, dtype=torch.long) for x in rows], layout=torch.jagged)

    data = TensorDict(
        {
            "input_ids": nested([a + b for a, b in local]),
            "position_ids": nested([list(range(len(a) + len(b))) for a, b in local]),
            "prompts": nested([a for a, b in local]),
            "responses": nested([b for a, b in local]),
            "loss_mask": nested([[1] * len(b) for a, b in local]),
            "response_mask": nested([[1] * len(b) for a, b in local]),
        },
        batch_size=[len(local)],
    )
    actor = worker(args.student, world, False, args.budget, not args.no_checkpoint)
    teacher = worker(args.teacher, world, True, args.budget, False)
    config = FullVocabConfig(
        backend="precompute" if args.backend == "fused_candidate" else args.backend, token_chunk_size=args.chunk
    )
    actor.engine.configure_full_vocab(config, args.teacher)
    dcfg = DistillationConfig(
        distillation_loss=DistillationLossConfig(
            loss_mode="full_reverse_kl",
            use_policy_gradient=False,
            use_task_rewards=False,
            loss_max_clamp=None,
            log_prob_min_clamp=None,
            full_vocab=config,
        )
    )
    actor.set_loss_fn(partial(loss_fn, config=dcfg))
    records = []
    for step in range(args.steps):
        batch = data.clone()
        tu.assign_non_tensor(batch, temperature=1.0, compute_loss=False, full_vocab_teacher=True)
        dist.barrier()
        torch.cuda.synchronize()
        # Probe-specific counters: this standalone process does not publish production lifetime metrics.
        torch.cuda.reset_peak_memory_stats()
        capture = args.trace and step == args.steps - 1
        if args.memory and capture:
            torch.cuda.memory._record_memory_history(max_entries=100000)
        prof = (
            torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
            )
            if capture
            else nullcontext()
        )
        with prof:
            start = time.perf_counter()
            with full_kl_range("teacher_phase_including_offload"):
                labels = teacher.infer_batch(batch)
            torch.cuda.synchronize()
            teacher_s = time.perf_counter() - start
            batch.update(labels.select("teacher_hidden", "teacher_positions"))
            tu.assign_non_tensor(batch, full_vocab_teacher=False, full_vocab_train=True)
            begin = time.perf_counter()
            with full_kl_range("actor_phase_including_offload"):
                output = actor.train_batch(batch)
                actor.engine.full_vocab_teacher_head.release()
            torch.cuda.synchronize()
            actor_s = time.perf_counter() - begin
        elapsed = torch.tensor(
            [
                teacher_s,
                actor_s,
                time.perf_counter() - start,
                torch.cuda.max_memory_allocated() / 2**30,
                torch.cuda.max_memory_reserved() / 2**30,
            ],
            device="cuda",
        )
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        metrics = tu.get(output, "metrics")
        row = dict(
            step=step,
            warmup=step < args.warmup_steps,
            profiled=capture,
            teacher_s=elapsed[0].item(),
            actor_s=elapsed[1].item(),
            wall_s=elapsed[2].item(),
            peak_allocated_gib=elapsed[3].item(),
            peak_reserved_gib=elapsed[4].item(),
            loss=metrics["loss"],
            grad_norm=metrics.get("grad_norm"),
        )
        records.append(row)
        if capture:
            prof.export_chrome_trace(str(args.output.parent / f"pipeline-rank{rank}.json.gz"))
        if args.memory and capture:
            torch.cuda.memory._dump_snapshot(str(args.output.parent / f"memory-rank{rank}.pickle"))
            torch.cuda.memory._record_memory_history(enabled=None)
        if rank == 0:
            result = {
                "scope": __doc__,
                "arguments": vars(args) | {"samples": str(args.samples), "output": str(args.output)},
                "world_size": world,
                "input_tokens": sum(len(a) + len(b) for a, b in samples),
                "loss_tokens": sum(len(b) for a, b in samples),
                "records": records,
                "gpu": torch.cuda.get_device_name(),
                "torch": torch.__version__,
            }
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(row), flush=True)
        del batch, labels, output
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
