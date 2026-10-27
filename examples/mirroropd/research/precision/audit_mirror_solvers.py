#!/usr/bin/env python3
# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Actual multi-step MDPO solver controls with independent post-update scores.

All branches restore the same checkpoint. Teacher and anchor stay frozen for the
entire inner solve. This is a short fixed-prefix diagnostic, not a training recipe.
"""

import argparse
import copy
import gc
import json
import math
import os
import time
from functools import partial
from pathlib import Path

import torch
import torch.distributed as dist
from tensordict import TensorDict

from verl.trainer.distillation import full_vocab as kernels
from verl.trainer.distillation.anchor import anchor_scope
from verl.utils import tensordict_utils as tu
from verl.utils.distributed import initialize_global_process_group
from verl.workers.config import DistillationConfig, DistillationLossConfig
from verl.workers.config.distillation import FullVocabConfig
from verl.workers.update_records import UpdateResult, finalize_update

from . import engine as replay
from .solver_optimizers import FullMatrixMuonAdamW, constant_scheduler


def local(value):
    return value.to_local() if hasattr(value, "to_local") else value


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu", copy=True)
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return copy.deepcopy(value)


def equal_tree(a, b):
    if isinstance(a, torch.Tensor):
        return (
            isinstance(b, torch.Tensor)
            and a.dtype == b.dtype
            and a.shape == b.shape
            and torch.equal(local(a).cpu(), local(b).cpu())
        )
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal_tree(a[k], b[k]) for k in a)
    if isinstance(a, list | tuple):
        return len(a) == len(b) and all(equal_tree(x, y) for x, y in zip(a, b, strict=True))
    return a == b


def validate_arm(arm):
    allowed = {
        "solver",
        "lr",
        "steps",
        "precision",
        "rounded",
        "trainable",
        "eval_interval",
        "save_best",
        "aux_lr",
        "momentum",
        "weight_decay",
        "save_checkpoint",
    }
    if set(arm) - allowed:
        raise ValueError(f"Unsupported branch keys: {set(arm) - allowed}")
    if arm["solver"] not in ("adam_saved", "adam_fresh", "adam_beta0", "sgd", "muon"):
        raise ValueError(f"Unknown solver: {arm['solver']}")
    if not math.isfinite(arm["lr"]) or arm["lr"] < 0:
        raise ValueError("lr must be finite and nonnegative")
    if arm.get("precision", "bfloat16") == "float16":
        raise ValueError("This solver audit does not snapshot AMP scaler state; use BF16/FP32 controls")
    for name in ("steps", "eval_interval"):
        value = arm.get(name, 16 if name == "steps" else 1)
        if not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if arm["solver"] != "muon" and ("aux_lr" in arm or "momentum" in arm):
        raise ValueError("aux_lr and momentum apply only to the Muon adapter")
    if arm["solver"] in ("sgd", "muon") and arm.get("weight_decay", 0) != 0:
        raise ValueError("SGD/Muon controls use zero weight decay")
    if arm.get("save_checkpoint") and arm["solver"] != "adam_saved":
        raise ValueError("Reusable worker checkpoints require the native saved Adam optimizer")


def canonical_score(metrics, spec):
    """Expose comparable objective fields without inventing an OPD anchor KL."""
    if spec.teacher_only_kl:
        value = metrics["distill/full_kl"]
        return metrics | {
            "distill/teacher_kl": value,
            "distill/target_kl": value,
            "distill/objective": value,
            "distill/target_log_z": 0.0,
        }
    return metrics | {"distill/objective": metrics["distill/mdpo/objective"]}


def rope_buffer_summary(worker):
    rope = getattr(getattr(worker.engine.module, "model", None), "rotary_emb", None)
    if rope is None or not hasattr(rope, "original_inv_freq"):
        return None
    current = rope.inv_freq.detach().float().cpu()
    original = rope.original_inv_freq.detach().float().cpu()
    return {
        "dtype": str(rope.inv_freq.dtype),
        "original_dtype": str(rope.original_inv_freq.dtype),
        "max_abs_change": (current - original).abs().max().item(),
        "values": current.tolist(),
        "original_values": original.tolist(),
    }


def set_teacher_rope(worker, mode):
    """Diagnostic control of nonpersistent Qwen2 RoPE buffers, never weights."""
    if mode == "native":
        return
    module = worker.engine.module
    if module.config.model_type != "qwen2":
        raise ValueError("The teacher buffer control is audited only for Qwen2")
    rope = module.model.rotary_emb
    if mode == "canonical":
        fresh = type(rope)(rope.config, device="cpu")
        device = rope.inv_freq.device
        rope.inv_freq = fresh.inv_freq.to(device)
        rope.original_inv_freq = fresh.original_inv_freq.to(device)
    elif mode == "rounded_bf16":
        rope.inv_freq = rope.inv_freq.bfloat16()
        rope.original_inv_freq = rope.original_inv_freq.bfloat16()
    else:
        raise ValueError(mode)


def main(*, worker_factory=replay.make_worker, precision_factory=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--prefixes", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--warm-outers", type=int, default=0, help="Create a new state with ordinary full KL from the initial student"
    )
    parser.add_argument("--branches", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--memory-fraction", type=float, default=0.25)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--response-start", type=int, default=0)
    parser.add_argument("--response-count", type=int, default=128)
    parser.add_argument("--eta", type=float, default=0.3, help="eta=1 uses the native ordinary full-KL endpoint")
    parser.add_argument("--teacher-rope", choices=("native", "canonical", "rounded_bf16"), default="native")
    parser.add_argument("--monitor-canonical-teacher", action="store_true")
    args = parser.parse_args()
    if not math.isfinite(args.eta) or not 0 < args.eta <= 1:
        parser.error("Require 0 < eta <= 1")
    if bool(args.checkpoint) == bool(args.warm_outers):
        parser.error("Choose exactly one of checkpoint and warm-outers")
    arms = json.loads(args.branches.read_text())
    for arm in arms:
        validate_arm(arm)
    if args.response_start < 0 or args.response_count < 1 or args.warm_outers < 0:
        parser.error("Require a nonnegative start/warm budget and a positive response count")
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    torch.backends.cuda.matmul.allow_tf32 = False
    _, rank, world = initialize_global_process_group(timeout_second=600)
    assert world == 4
    rec = json.loads(args.prefixes.read_text())["records"][rank]
    prompt = rec["positions"][0] + 1
    end = args.response_start + args.response_count
    tokens = torch.tensor(rec["ids"][: prompt + end])
    response = tokens[prompt:]
    if len(response) != end:
        raise ValueError("Requested prefix exceeds saved response length")
    mask = torch.zeros_like(response)
    mask[args.response_start : end] = 1
    data = TensorDict(
        {
            "input_ids": replay.nested([tokens]),
            "position_ids": replay.nested([torch.arange(len(tokens))]),
            "prompts": replay.nested([tokens[:prompt]]),
            "responses": replay.nested([response]),
            "response_mask": replay.nested([torch.ones_like(mask)]),
            "loss_mask": replay.nested([mask]),
        },
        batch_size=[1],
    )
    teacher = worker_factory(args.teacher, world, True)
    set_teacher_rope(teacher, args.teacher_rope)
    tu.assign_non_tensor(data, temperature=1.0, compute_loss=False, full_vocab_teacher=True, global_token_num=None)
    payload = teacher.infer_batch(data)
    data.update(payload.select("teacher_hidden", "teacher_positions"))
    teacher_buffers = rope_buffer_summary(teacher)
    canonical_payload = None
    teacher_equals_canonical = None
    if args.monitor_canonical_teacher:
        if args.teacher_rope == "canonical":
            canonical_payload = payload.select("teacher_hidden", "teacher_positions")
        else:
            set_teacher_rope(teacher, "canonical")
            canonical_payload = teacher.infer_batch(data).select("teacher_hidden", "teacher_positions")
        equal = torch.tensor(
            int(torch.equal(payload["teacher_hidden"].values(), canonical_payload["teacher_hidden"].values())),
            device="cuda",
        )
        dist.all_reduce(equal, op=dist.ReduceOp.MIN)
        teacher_equals_canonical = bool(equal.item())
    del teacher, payload
    gc.collect()
    torch.cuda.empty_cache()
    cfg = DistillationConfig(
        distillation_loss=DistillationLossConfig(
            loss_mode="mirror",
            mirror={"objective": "mdpo", "estimator": "full_vocab", "kappa": 0.0, "eta": args.eta, "diagnostics": True},
            full_vocab=FullVocabConfig(backend="precompute_fused", token_chunk_size=128),
            use_task_rewards=False,
            use_policy_gradient=False,
            policy_loss_mode="vanilla",
            loss_max_clamp=None,
            log_prob_min_clamp=None,
        )
    )
    spec = cfg.distillation_loss.loss_settings
    checkpoint = args.checkpoint
    if args.warm_outers:
        warmer = worker_factory(args.student, world)
        warm_cfg = DistillationConfig(
            distillation_loss=DistillationLossConfig(
                loss_mode="mirror",
                mirror={"objective": "mdpo", "estimator": "full_vocab", "kappa": 0.0, "eta": 1.0},
                full_vocab=FullVocabConfig(backend="precompute", token_chunk_size=128),
                use_task_rewards=False,
                use_policy_gradient=False,
                policy_loss_mode="vanilla",
                loss_max_clamp=None,
                log_prob_min_clamp=None,
            )
        )
        warmer.engine.configure_full_vocab(
            warm_cfg.distillation_loss.full_vocab, str(args.teacher), spec=warm_cfg.distillation_loss.loss_settings
        )
        warmer.set_loss_fn(partial(replay.loss_fn, config=warm_cfg))
        warm_records = []
        for outer in range(1, args.warm_outers + 1):
            wb = data.clone()
            tu.assign_non_tensor(
                wb,
                temperature=1.0,
                compute_loss=True,
                full_vocab_teacher=False,
                full_vocab_train=True,
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
            out = warmer.train_mini_batch(wb)
            warmer.engine.full_vocab_teacher_head.release()
            updates = replay.scalar_metrics(tu.get(out, "update_records"))
            assert tu.get(out, "optimizer_steps") == 16
            warm_records.append({"outer": outer, "updates": updates})
            if rank == 0:
                (args.out / "warmup.json").write_text(
                    json.dumps({"objective": "ordinary full reverse KL, eta=1", "records": warm_records}, indent=2)
                    + "\n"
                )
                print("WARM", outer, updates[0]["distill/full_kl"], updates[-1]["distill/full_kl"], flush=True)
            if outer >= 4 and updates[0]["distill/full_kl"] < 1e-3:
                break
        checkpoint = args.out / "warm-checkpoint"
        warmer.save_checkpoint(str(checkpoint), global_step=len(warm_records))
        del warmer
        gc.collect()
        torch.cuda.empty_cache()
    results = []
    actor = None
    precision = None
    original_optimizer = original_scheduler = None

    def save():
        if rank == 0:
            path = args.out / "result.json"
            temporary = path.with_suffix(".tmp")
            temporary.write_text(
                json.dumps(
                    {
                        "scope": __doc__,
                        "checkpoint": str(checkpoint),
                        "teacher_rope": teacher_buffers,
                        "teacher_equals_canonical": teacher_equals_canonical,
                        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                        "branches": results,
                    },
                    indent=2,
                )
                + "\n"
            )
            temporary.replace(path)

    for index, arm in enumerate(arms):
        if index < args.start:
            continue
        dtype = arm.get("precision", "bfloat16")
        if precision != dtype:
            if actor is not None:
                actor = original_optimizer = original_scheduler = None
                gc.collect()
                torch.cuda.empty_cache()
            factory = worker_factory
            if dtype not in ("bfloat16", "float16", "float32"):
                if precision_factory is None:
                    from .isolate_mirror_precision import make_worker as factory
                else:
                    factory = precision_factory
            actor = factory(
                args.student,
                world,
                precision=dtype,
                attention="sdpa" if dtype in ("float32", "bf16_weights_fp32_compute") else "flash_attention_2",
            )
            actor.engine.configure_full_vocab(cfg.distillation_loss.full_vocab, str(args.teacher), spec=spec)
            actor.set_loss_fn(partial(replay.loss_fn, config=cfg))
            original_optimizer = actor.engine.optimizer
            original_scheduler = actor.engine.lr_scheduler
            precision = dtype
        actor.engine.optimizer = original_optimizer
        actor.engine.lr_scheduler = original_scheduler
        actor.engine.checkpoint_manager.optimizer = original_optimizer
        actor.engine.checkpoint_manager.lr_scheduler = original_scheduler
        actor.load_checkpoint(str(checkpoint))
        if arm.get("rounded", False):
            with torch.no_grad():
                for p in actor.engine.module.parameters():
                    p.copy_(p.to(torch.bfloat16).to(p.dtype))
        solver = arm["solver"]
        lr = arm["lr"]
        if solver.startswith("adam"):
            if solver == "adam_fresh":
                actor.engine.optimizer.state.clear()
            if solver not in ["adam_saved", "adam_fresh", "adam_beta0"]:
                raise ValueError(solver)
            for group in actor.engine.optimizer.param_groups:
                group["lr"] = lr
                group["weight_decay"] = arm.get("weight_decay", 0.0)
                group["betas"] = (0.0 if solver == "adam_beta0" else 0.9, 0.999)
        elif solver == "sgd":
            actor.engine.optimizer = torch.optim.SGD(
                actor.engine.module.parameters(), lr=lr, momentum=0, weight_decay=0
            )
        elif solver == "muon":
            actor.engine.optimizer = FullMatrixMuonAdamW(
                actor.engine.module.named_parameters(),
                lr=lr,
                aux_lr=arm.get("aux_lr", lr),
                momentum=arm.get("momentum", 0.95),
                weight_decay=0,
            )
        else:
            raise ValueError(solver)
        actor.engine.lr_scheduler = constant_scheduler(actor.engine.optimizer)
        expected_lrs = [lr, arm.get("aux_lr", lr)] if solver == "muon" else [lr]
        assert [g["lr"] for g in actor.engine.optimizer.param_groups] == expected_lrs
        # Runtime swapping is diagnostic only. Checkpoint reload always restores
        # the original Adam reference above before constructing another solver.
        batch = data.clone()
        steps = arm.get("steps", 16)
        interval = arm.get("eval_interval", 1)
        tu.assign_non_tensor(
            batch,
            temperature=1.0,
            compute_loss=True,
            full_vocab_teacher=False,
            full_vocab_train=True,
            mirror_anchor_version=(4000 + index, 0),
            global_token_num=None,
            global_batch_size=4,
            mini_batch_size=4,
            epochs=steps,
            seed=42,
            dataloader_kwargs={"shuffle": False},
            collect_metrics=True,
            collect_update_records=True,
            metrics_attempt_start=0,
        )
        row = {
            "index": index,
            **arm,
            "post_updates": [],
            "steps": steps,
            "eta": args.eta,
            "objective": "full_reverse_kl" if spec.teacher_only_kl else "mdpo",
            "partition": getattr(actor.engine.optimizer, "partition", None),
            "actor_rope": rope_buffer_summary(actor),
        }
        results.append(row)
        counter = [0]
        best_store = [None]
        started = time.monotonic()

        def score(actor=actor, batch=batch, counter=counter):
            probe = batch.clone()
            tu.assign_non_tensor(
                probe,
                compute_loss=True,
                disable_auto_offload=next(actor.engine.module.parameters()).device.type == "cuda",
                global_token_num=None,
            )
            old = kernels.full_vocab_objective_result
            training = actor.engine.module.training
            previous_mode = actor.engine.mode
            kernels.full_vocab_objective_result = replay.reference

            def evaluate_once(probe):
                out = actor.infer_batch(probe)
                _, record = finalize_update(
                    UpdateResult(None, tu.get(out, "metrics")),
                    role="actor",
                    collect_records=True,
                    collect_metrics=True,
                    update_index=counter[0],
                    minibatches_per_epoch=1,
                )
                metrics = {
                    k: v for k, v in record["metrics"].items() if k.startswith("distill/") and "_position/" not in k
                }
                return canonical_score(metrics, spec)

            try:
                metrics = evaluate_once(probe)
                if canonical_payload is not None:
                    if args.teacher_rope == "canonical":
                        canonical_kl = metrics["distill/teacher_kl"]
                    else:
                        canonical_probe = probe.clone()
                        canonical_probe.update(canonical_payload)
                        canonical_kl = evaluate_once(canonical_probe)["distill/teacher_kl"]
                    metrics["distill/canonical_teacher_kl"] = canonical_kl
            finally:
                kernels.full_vocab_objective_result = old
                actor.engine.module.train(training)
                actor.engine.mode = previous_mode
            return metrics

        trainable = arm.get("trainable", "all")
        if trainable != "all" and actor.engine.module.config.tie_word_embeddings:
            raise ValueError("Head/backbone ablations require an untied output head")
        if trainable not in ("all", "head", "backbone"):
            raise ValueError(trainable)

        def allowed(name, trainable=trainable):
            return trainable == "all" or ((name == "lm_head.weight") == (trainable == "head"))

        frozen_weights = {
            name: local(p).detach().cpu().clone()
            for name, p in actor.engine.module.named_parameters()
            if not allowed(name)
        }
        original_step = actor.engine.optimizer_step

        def measured_step(
            counter=counter,
            actor=actor,
            steps=steps,
            interval=interval,
            original_step=original_step,
            score=score,
            row=row,
            index=index,
            solver=solver,
            lr=lr,
            dtype=dtype,
            allowed=allowed,
            best_store=best_store,
            arm=arm,
        ):
            counter[0] += 1
            for name, p in actor.engine.module.named_parameters():
                if not allowed(name):
                    p.grad = None
            selected = counter[0] == 1 or counter[0] % interval == 0 or counter[0] == steps
            parameters = [p for p in actor.engine.module.parameters() if p.grad is not None]
            before = [local(p).detach().clone() for p in parameters] if selected else []
            grad_norm = original_step()
            if selected:
                stats = torch.zeros(2, device="cuda", dtype=torch.float64)
                for p, old in zip(parameters, before, strict=True):
                    delta = (local(p).detach() - old).double()
                    gradient = local(p.grad).detach().double()
                    stats[0] += delta.square().sum()
                    stats[1] += (delta * gradient).sum()
                dist.all_reduce(stats)
                del before
                metrics = score()
                if metrics["distill/target_kl"] < row["best"]["metrics"]["distill/target_kl"]:
                    row["best"] = {"update": counter[0], "metrics": metrics}
                    if arm.get("save_best", False):
                        best_store[0] = {
                            "parameters": {
                                n: local(p).detach().cpu().clone() for n, p in actor.engine.module.named_parameters()
                            },
                            "optimizer": cpu_tree(actor.engine.optimizer.state_dict()),
                            "scheduler": cpu_tree(actor.engine.lr_scheduler.state_dict()),
                        }
                row["post_updates"].append(
                    {
                        "update": counter[0],
                        "metrics": metrics,
                        "grad_norm": grad_norm,
                        "delta_norm": stats[0].sqrt().item(),
                        "gradient_dot_delta": stats[1].item(),
                    }
                )
                save()
                if rank == 0:
                    print("STEP", index, solver, lr, dtype, counter[0], metrics["distill/target_kl"], flush=True)
            return grad_norm

        actor.engine.optimizer_step = measured_step
        try:
            with anchor_scope(actor, batch, spec):
                row["before"] = score()
                row["best"] = {"update": 0, "metrics": row["before"]}
                save()
                out = actor.train_mini_batch(batch)
                row["after"] = score()
                if best_store[0] is not None:
                    best = best_store[0]
                    with torch.no_grad():
                        for name, p in actor.engine.module.named_parameters():
                            local(p).copy_(best["parameters"][name])
                    actor.engine.optimizer.load_state_dict(cpu_tree(best["optimizer"]))
                    actor.engine.lr_scheduler.load_state_dict(cpu_tree(best["scheduler"]))
                    row["best_optimizer_restored_exact"] = equal_tree(
                        best["optimizer"], actor.engine.optimizer.state_dict()
                    )
                    assert row["best_optimizer_restored_exact"]
                    row["best_rechecked"] = score()
                    assert (
                        abs(row["best_rechecked"]["distill/target_kl"] - row["best"]["metrics"]["distill/target_kl"])
                        < 1e-10
                    )
                    torch.save(best, args.out / f"best-arm{index}-rank{rank}.pt")
        finally:
            actor.engine.optimizer_step = original_step
            actor.engine.full_vocab_teacher_head.release()
        row["frozen_weights_unchanged"] = all(
            torch.equal(local(p).detach().cpu(), frozen_weights[name])
            for name, p in actor.engine.module.named_parameters()
            if name in frozen_weights
        )
        assert row["frozen_weights_unchanged"]
        del frozen_weights
        row["optimizer_steps"] = tu.get(out, "optimizer_steps")
        row["updates"] = replay.scalar_metrics(tu.get(out, "update_records"))
        row["elapsed_seconds"] = time.monotonic() - started
        assert row["optimizer_steps"] == steps
        assert all(r["optim/actor/lr"] == lr for r in row["updates"]), "Actual LR differs from requested LR"
        assert abs(row["before"]["distill/target_log_z"] - row["after"]["distill/target_log_z"]) < 1e-12
        if arm.get("save_checkpoint"):
            destination = args.out / f"checkpoint-arm{index}"
            actor.engine.checkpoint_manager.optimizer = actor.engine.optimizer
            actor.engine.checkpoint_manager.lr_scheduler = actor.engine.lr_scheduler
            actor.save_checkpoint(str(destination), global_step=steps)
            row["saved_checkpoint"] = str(destination)
        save()
        if rank == 0:
            print(
                "ARM",
                index,
                solver,
                lr,
                dtype,
                row["before"]["distill/target_kl"],
                row["after"]["distill/target_kl"],
                flush=True,
            )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
