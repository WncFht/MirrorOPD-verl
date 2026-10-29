#!/usr/bin/env python3
# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Four-rank real-teacher mirror diagnostic on four fixed math continuations.

Requires four GPUs and saved fixed-prefix records. Packed SDPA comparisons are
valid here only because each rank/micro-batch contains exactly one sequence.
Use on an allocated GPU pool; the memory cap is not a scheduler reservation.

Uses the production FSDP2 engine, global token normalization, anchor scope and
AdamW. This creates a local low-KL state, not a reconstruction of the original
on-policy E45/E47 history. Pre/post evaluation uses FP64 row arithmetic.
"""

import argparse
import gc
import importlib
import json
import os
import sys
import time
from functools import partial
from pathlib import Path

import torch
import torch.distributed as dist
from tensordict import TensorDict

import verl.trainer.distillation.full_vocab as kernels
from examples.mirroropd.research.backends.fsdp import evaluate, loss_fn, make_worker, nested, scalar_metrics
from verl.trainer.distillation.anchor import anchor_scope
from verl.utils import tensordict_utils as tu
from verl.utils.distributed import initialize_global_process_group
from verl.workers.config import (
    DistillationConfig,
    DistillationLossConfig,
)
from verl.workers.config.distillation import FullVocabConfig


def main(extra_kernels=None, *, worker_factory=make_worker):
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--objective", choices=["mdpo", "bpo"], default="mdpo")
    parser.add_argument("--kappa", type=float, default=0.0)
    parser.add_argument("--eta", type=float, default=0.3)
    parser.add_argument("--warm-outers", type=int, default=32)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--start-branch", type=int, default=0)
    parser.add_argument("--student", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--prefixes", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--memory-fraction", type=float, default=0.30)
    parser.add_argument("--branch-config", type=Path)
    parser.add_argument("--warm-precision", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--warm-attention", choices=["flash_attention_2", "sdpa"], default="flash_attention_2")
    parser.add_argument("--warm-kernel", choices=["baseline", "fixed"], default="baseline")
    parser.add_argument("--adapt", action="store_true", help="Warm from resume state and save a new checkpoint")
    parser.add_argument("--stop-at-low-kl", action="store_true", help="Stop warming below 1e-3 even without rebound")
    args = parser.parse_args()
    if not 0 < args.eta < 1 or not 0 <= args.kappa <= 1:
        parser.error("This anchored replay requires 0 < eta < 1 and 0 <= kappa <= 1")
    if args.warm_precision == "float32" and args.warm_attention != "sdpa":
        parser.error("FP32 requires SDPA")
    if args.adapt and not args.resume_checkpoint:
        parser.error("--adapt requires --resume-checkpoint")
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    _, rank, world = initialize_global_process_group(timeout_second=600)
    assert world == 4
    sys.path.insert(0, str(args.baseline_dir.resolve()))
    baseline = importlib.import_module("mirror_perf_baseline.full_vocab").full_vocab_objective_result
    fixed = kernels.full_vocab_objective_result
    implementations = {"baseline": baseline, "fixed": fixed, **(extra_kernels or {})}
    records = json.loads(args.prefixes.read_text())["records"]
    r = records[rank]
    prompt = r["positions"][0] + 1
    tokens = torch.tensor(r["ids"][: prompt + 128])
    response = tokens[prompt:]
    data = TensorDict(
        {
            "input_ids": nested([tokens]),
            "position_ids": nested([torch.arange(len(tokens))]),
            "prompts": nested([tokens[:prompt]]),
            "responses": nested([response]),
            "response_mask": nested([torch.ones(len(response), dtype=torch.long)]),
            "loss_mask": nested([torch.ones(len(response), dtype=torch.long)]),
        },
        batch_size=[1],
    )
    teacher_path = args.teacher
    student_path = args.student
    teacher = worker_factory(teacher_path, world, True)
    tu.assign_non_tensor(data, temperature=1.0, compute_loss=False, full_vocab_teacher=True, global_token_num=None)
    labels = teacher.infer_batch(data)
    data.update(labels.select("teacher_hidden", "teacher_positions"))
    del teacher
    gc.collect()
    torch.cuda.empty_cache()
    config = DistillationConfig(
        distillation_loss=DistillationLossConfig(
            loss_mode="mirror",
            mirror={
                "objective": args.objective,
                "estimator": "full_vocab",
                "kappa": args.kappa,
                "eta": args.eta,
                "diagnostics": True,
            },
            full_vocab=FullVocabConfig(backend="precompute_fused", token_chunk_size=128),
            use_task_rewards=False,
            use_policy_gradient=False,
            policy_loss_mode="vanilla",
            loss_max_clamp=None,
            log_prob_min_clamp=None,
        )
    )
    spec = config.distillation_loss.loss_settings

    def configure(actor):
        actor.engine.configure_full_vocab(config.distillation_loss.full_vocab, str(teacher_path), spec=spec)
        actor.set_loss_fn(partial(loss_fn, config=config))

    def batch(version):
        b = data.clone()
        tu.assign_non_tensor(
            b,
            temperature=1.0,
            compute_loss=True,
            full_vocab_teacher=False,
            full_vocab_train=True,
            mirror_anchor_version=(version, 0),
            global_token_num=None,
            global_batch_size=4,
            mini_batch_size=4,
            epochs=16,
            seed=42,
            dataloader_kwargs={"shuffle": False},
            collect_metrics=True,
            collect_update_records=True,
            metrics_attempt_start=0,
        )
        return b

    actor = worker_factory(student_path, world, precision=args.warm_precision, attention=args.warm_attention)
    configure(actor)
    warm = []
    branches = []
    checkpoint = args.out / "warm-checkpoint" if args.adapt else args.resume_checkpoint or args.out / "warm-checkpoint"
    if args.adapt:
        actor.load_checkpoint(str(args.resume_checkpoint))
    clock = time.monotonic()

    def save_result():
        if rank == 0:
            (args.out / "result.json").write_text(
                json.dumps(
                    {
                        "scope": __doc__,
                        "world_size": world,
                        "batch_size": 4,
                        "response_cap": 128,
                        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                        "warmup_kernel": args.warm_kernel,
                        "objective": args.objective,
                        "kappa": args.kappa,
                        "eta": args.eta,
                        "warmup": warm,
                        "branches": branches,
                        "checkpoint": str(checkpoint),
                        "elapsed_seconds": time.monotonic() - clock,
                    },
                    indent=2,
                )
                + "\n"
            )

    kernels.full_vocab_objective_result = baseline if args.warm_kernel == "baseline" else fixed
    for outer in range(1, 1 if args.resume_checkpoint and not args.adapt else args.warm_outers + 1):
        b = batch(outer)
        with anchor_scope(actor, b, spec):
            out = actor.train_mini_batch(b)
        actor.engine.full_vocab_teacher_head.release()
        updates = scalar_metrics(tu.get(out, "update_records"))
        assert len(updates) == 16
        entry = {"outer": outer, "updates": updates, "optimizer_steps": tu.get(out, "optimizer_steps")}
        warm.append(entry)
        save_result()
        first, last = updates[0], updates[-1]
        if rank == 0:
            print(
                "WARM",
                outer,
                "teacher",
                first["distill/teacher_kl"],
                last["distill/teacher_kl"],
                "target",
                first["distill/target_kl"],
                last["distill/target_kl"],
                flush=True,
            )
        if (
            outer >= 4
            and first["distill/teacher_kl"] < 1e-3
            and (args.stop_at_low_kl or last["distill/target_kl"] > first["distill/target_kl"] * 1.05)
        ):
            break
    if not args.resume_checkpoint or args.adapt:
        actor.save_checkpoint(str(checkpoint), global_step=len(warm))
    precision = args.warm_precision
    attention = args.warm_attention
    settings = [
        ("baseline", 1e-6, "keep", "bfloat16", "flash_attention_2"),
        ("fixed", 1e-6, "keep", "bfloat16", "flash_attention_2"),
        ("fixed", 3e-7, "keep", "bfloat16", "flash_attention_2"),
        ("fixed", 1e-7, "keep", "bfloat16", "flash_attention_2"),
        ("fixed", 1e-6, "reset_m", "bfloat16", "flash_attention_2"),
        ("fixed", 1e-6, "reset_all", "bfloat16", "flash_attention_2"),
        ("fixed", 1e-6, "keep", "bfloat16", "sdpa"),
        ("fixed", 3e-7, "keep", "bfloat16", "sdpa"),
        ("fixed", 1e-6, "keep", "float32", "sdpa"),
        ("fixed", 3e-7, "keep", "float32", "sdpa"),
        ("fixed", 1e-7, "keep", "float32", "sdpa"),
    ]
    if args.branch_config:
        settings = json.loads(args.branch_config.read_text())
    for index, (kernel, lr, state, dtype, attn) in enumerate(settings):
        if index < args.start_branch:
            continue
        if (dtype, attn) != (precision, attention):
            del actor
            gc.collect()
            torch.cuda.empty_cache()
            actor = worker_factory(student_path, world, precision=dtype, attention=attn)
            configure(actor)
            precision, attention = dtype, attn
        actor.load_checkpoint(str(checkpoint))
        if state == "reset_all":
            actor.engine.optimizer.state.clear()
        elif state == "reset_m":
            for st in actor.engine.optimizer.state.values():
                # Retaining v/step also changes effective step magnitude, not just direction.
                if "exp_avg" in st:
                    st["exp_avg"].zero_()
        elif state == "round_weights":
            with torch.no_grad():
                for parameter in actor.engine.module.parameters():
                    parameter.copy_(parameter.to(torch.bfloat16).to(parameter.dtype))
        elif state != "keep":
            raise ValueError(f"Unknown optimizer/weight state: {state}")
        for group in actor.engine.optimizer.param_groups:
            group["lr"] = lr
        actor.engine.lr_scheduler.base_lrs = [lr] * len(actor.engine.optimizer.param_groups)
        kernels.full_vocab_objective_result = implementations[kernel]
        steps = [
            (v["step"].to_local() if hasattr(v["step"], "to_local") else v["step"]).item()
            for v in actor.engine.optimizer.state.values()
            if "step" in v
        ]
        optimizer_start = {
            "state_entries": len(steps),
            "min_step": min(steps) if steps else None,
            "max_step": max(steps) if steps else None,
        }
        step_stats = []
        original_step = actor.engine.optimizer_step
        counter = [0]

        def measured_step(counter=counter, original_step=original_step, actor=actor, step_stats=step_stats):
            counter[0] += 1
            if counter[0] not in (1, 16):
                return original_step()
            parameters = [p for p in actor.engine.module.parameters() if p.grad is not None]
            local = lambda p: p.to_local() if hasattr(p, "to_local") else p
            before = [local(p).detach().clone() for p in parameters]
            norm = original_step()
            dot = torch.zeros((), device="cuda", dtype=torch.float64)
            distance = dot.clone()
            for p, old in zip(parameters, before, strict=True):
                delta = local(p).detach() - old
                gradient = local(p.grad).detach()
                dot += (delta.double() * gradient.double()).sum()
                distance += delta.double().square().sum()
            values = torch.stack([dot, distance])
            dist.all_reduce(values)
            step_stats.append(
                {
                    "update": counter[0],
                    "grad_dot_delta": values[0].item(),
                    "parameter_delta_norm": values[1].sqrt().item(),
                    "grad_norm": norm,
                }
            )
            return norm

        actor.engine.optimizer_step = measured_step
        b = batch(1000 + index)
        try:
            with anchor_scope(actor, b, spec):
                before = evaluate(actor, b)
                out = actor.train_mini_batch(b)
                after = evaluate(actor, b)
        finally:
            actor.engine.optimizer_step = original_step
            actor.engine.full_vocab_teacher_head.release()
        updates = scalar_metrics(tu.get(out, "update_records"))
        assert len(updates) == 16
        branch = {
            "kernel": kernel,
            "lr": lr,
            "optimizer_state": state,
            "precision": dtype,
            "attention": attn,
            "optimizer_start": optimizer_start,
            "optimizer_steps": tu.get(out, "optimizer_steps"),
            "before_fp64_rows": before,
            "after_fp64_rows": after,
            "updates": updates,
            "step_statistics": step_stats,
        }
        branches.append(branch)
        save_result()
        if rank == 0:
            print(
                "BRANCH",
                kernel,
                lr,
                state,
                dtype,
                attn,
                "target",
                before["distill/target_kl"],
                after["distill/target_kl"],
                step_stats,
                flush=True,
            )
    kernels.full_vocab_objective_result = fixed
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
