#!/usr/bin/env python3
# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Same-gradient optimizer counterfactuals and paired forward-precision probes.

Four ranks, one fixed response per rank. Capture one real native FSDP2 gradient
and Adam step; replay the exact optimizer state on local shards before any
intervention. Zero-gradient controls use actual zero tensors, not grad=None.
All precision evaluations keep native q/T/m frozen; mode-relative metrics also
remove the static change caused by switching forward arithmetic.
"""

import argparse
import copy
import gc
import json
import math
import os
from functools import partial
from pathlib import Path

import replay_fixed_prefix as replay
import torch
import torch.distributed as dist
from probe_mirror_forward import assign, local, snapshot
from tensordict import TensorDict

from verl.trainer.distillation import full_vocab as kernels
from verl.trainer.distillation.anchor import anchor_scope
from verl.utils import tensordict_utils as tu
from verl.utils.distributed import initialize_global_process_group
from verl.workers.config import DistillationConfig, DistillationLossConfig
from verl.workers.config.distillation import FullVocabConfig


def cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return local(value).detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_copy(v) for k, v in value.items()}
    return copy.deepcopy(value)


def dot(a, b):
    result = torch.zeros((), device="cuda", dtype=torch.float64)
    for name, value in a.items():
        x = value.to("cuda", dtype=torch.float64)
        y = b[name].to("cuda", dtype=torch.float64)
        result += (x * y).sum()
    dist.all_reduce(result)
    return result.item()


def norm(a):
    return dot(a, a) ** 0.5


def subtract(a, b):
    return {k: v - b[k] for k, v in a.items()}


def replay_adam_update(theta, gradient, states, group_settings, kind, device="cuda"):
    if kind not in ("saved", "zero", "decay_only", "reset_m", "fresh"):
        raise ValueError(f"Unknown counterfactual: {kind}")
    parameters = {n: torch.nn.Parameter(v.to(device).clone()) for n, v in theta.items()}
    groups = []
    for original in group_settings:
        groups.append(
            {
                **{k: copy.deepcopy(v) for k, v in original.items() if k != "names"},
                "params": [parameters[n] for n in original["names"]],
            }
        )
    optimizer = torch.optim.AdamW(groups)
    for name, p in parameters.items():
        if kind != "fresh":
            st = {}
            for key, value in states[name].items():
                st[key] = (
                    value.clone()
                    if key == "step"
                    else value.to(device).clone()
                    if isinstance(value, torch.Tensor)
                    else copy.deepcopy(value)
                )
            if kind in ("reset_m", "decay_only"):
                st["exp_avg"].zero_()
            optimizer.state[p] = st
        p.grad = torch.zeros_like(p) if kind in ("zero", "decay_only") else gradient[name].to(device).clone()
    optimizer.step()
    delta = {n: (p.detach().cpu() - theta[n]) for n, p in parameters.items()}
    del optimizer, parameters
    gc.collect()
    if device != "cpu":
        torch.cuda.empty_cache()
    return delta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--prefixes", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--response-start", type=int, default=0)
    parser.add_argument("--response-count", type=int, default=128)
    parser.add_argument(
        "--response-length", type=int, default=128, help="Keep the entire model input fixed across masks"
    )
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--scales", type=float, nargs="+", default=[1.0])
    parser.add_argument("--save-state", action="store_true")
    parser.add_argument("--capture-dir", type=Path, help="Reuse a verified captured gradient and updates")
    parser.add_argument(
        "--candidates", nargs="+", help="Restrict candidate names, e.g. adam zero reset_m_matched sgd_matched"
    )
    args = parser.parse_args()
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("lr must be finite and positive")
    if args.capture_dir and args.capture_dir.resolve() == args.out.resolve():
        parser.error("out must differ from capture-dir")
    if args.response_start < 0 or args.response_count < 1:
        parser.error("Require response-start >= 0 and response-count > 0")
    if args.response_start + args.response_count > args.response_length:
        parser.error("Loss interval exceeds response-length")
    if any(x < 0 or not math.isfinite(x) for x in args.scales):
        parser.error("scales must be finite and nonnegative")
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_per_process_memory_fraction(0.40)
    torch.backends.cuda.matmul.allow_tf32 = False
    _, rank, world = initialize_global_process_group(timeout_second=600)
    assert world == 4
    r = json.loads(args.prefixes.read_text())["records"][rank]
    prompt = r["positions"][0] + 1
    end = args.response_start + args.response_count
    tokens = torch.tensor(r["ids"][: prompt + args.response_length])
    response = tokens[prompt:]
    if len(response) != args.response_length:
        raise ValueError("Requested response interval exceeds the saved prefix")
    mask = torch.zeros(len(response), dtype=torch.long)
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
    teacher = replay.make_worker(args.teacher, world, True)
    tu.assign_non_tensor(data, temperature=1.0, compute_loss=False, full_vocab_teacher=True, global_token_num=None)
    payload = teacher.infer_batch(data)
    data.update(payload.select("teacher_hidden", "teacher_positions"))
    del teacher, payload
    gc.collect()
    torch.cuda.empty_cache()
    cfg = DistillationConfig(
        distillation_loss=DistillationLossConfig(
            loss_mode="mirror",
            mirror={"objective": "mdpo", "estimator": "full_vocab", "kappa": 0.0, "eta": 0.3, "diagnostics": True},
            full_vocab=FullVocabConfig(backend="precompute_fused", token_chunk_size=128),
            use_task_rewards=False,
            use_policy_gradient=False,
            policy_loss_mode="vanilla",
            loss_max_clamp=None,
            log_prob_min_clamp=None,
        )
    )
    spec = cfg.distillation_loss.loss_settings

    def make_actor(dtype):
        worker = replay.make_worker(
            args.student, world, precision=dtype, attention="flash_attention_2" if dtype == "bfloat16" else "sdpa"
        )
        worker.engine.configure_full_vocab(cfg.distillation_loss.full_vocab, str(args.teacher), spec=spec)
        worker.set_loss_fn(partial(replay.loss_fn, config=cfg))
        return worker

    frozen, mode_initial = {}, {}
    current_mode, initializing = [None], [False]

    def evaluate_rows(hs, ws, ht, wt, *, spec, hq, wq, **kwargs):
        lp = kernels.project(hs, ws).double().log_softmax(-1)
        p = lp.exp()
        if not frozen:
            lq = kernels.project(hq, wq).double().log_softmax(-1)
            lt = kernels.project(ht, wt).double().log_softmax(-1)[:, : len(ws)]
            b = lq + spec.eta * (lt - lq)
            frozen.update(q=lq, t=lt, b=b)
        mode = current_mode[0]
        if mode is not None and initializing[0]:
            mode_initial[mode] = lp.detach().clone()
        q = mode_initial.get(mode, frozen["q"])
        own_b = q + spec.eta * (frozen["t"] - q)
        raw = (p * (lp - frozen["b"])).sum(-1)
        return kernels.FullVocabObjectiveResult(
            raw.sum(),
            raw,
            {
                "target_kl": raw + frozen["b"].logsumexp(-1),
                "target_log_z": frozen["b"].logsumexp(-1),
                "teacher_kl": (p * (lp - frozen["t"])).sum(-1),
                "anchor_kl": (p * (lp - frozen["q"])).sum(-1),
                "mode_anchor_kl": (p * (lp - q)).sum(-1),
                "mode_target_kl": (p * (lp - own_b)).sum(-1) + own_b.logsumexp(-1),
            },
        )

    replay.reference = evaluate_rows
    actor = make_actor("bfloat16")
    actor.load_checkpoint(str(args.checkpoint))
    for group in actor.engine.optimizer.param_groups:
        group["lr"] = args.lr
    actor.engine.lr_scheduler.base_lrs = [args.lr] * len(actor.engine.optimizer.param_groups)
    theta = snapshot(actor)
    capture = {}
    loaded = None
    source_result = None
    if args.capture_dir:
        source_result = json.loads((args.capture_dir / "result.json").read_text())
        for key in ("response_start", "response_count", "lr"):
            assert source_result["arguments"][key] == getattr(args, key), key
        for key in ("checkpoint", "prefixes", "student", "teacher"):
            assert Path(source_result["arguments"][key]).resolve() == getattr(args, key).resolve(), key
        source_length = source_result["arguments"].get(
            "response_length",
            source_result["arguments"]["response_start"] + source_result["arguments"]["response_count"],
        )
        assert source_length == args.response_length
        loaded = torch.load(args.capture_dir / f"capture-rank{rank}.pt", map_location="cpu", weights_only=True)
        assert all(torch.equal(theta[n], loaded["theta"][n]) for n in theta)
        capture.update(groups=loaded["groups"], state=loaded["optimizer"], gradient=loaded["gradient"])
    native_step = actor.engine.optimizer_step

    def capture_step():
        named = dict(actor.engine.module.named_parameters())
        lookup = {id(p): name for name, p in named.items()}
        capture["groups"] = [
            {
                **{k: copy.deepcopy(v) for k, v in group.items() if k != "params"},
                "names": [lookup[id(p)] for p in group["params"]],
            }
            for group in actor.engine.optimizer.param_groups
        ]
        capture["state"] = {lookup[id(p)]: cpu_copy(st) for p, st in actor.engine.optimizer.state.items()}
        grad_norm = native_step()
        capture["gradient"] = {
            name: local(p.grad).detach().cpu().clone() for name, p in named.items() if p.grad is not None
        }
        capture["grad_norm"] = grad_norm
        capture["native_after"] = snapshot(actor)
        return grad_norm

    actor.engine.optimizer_step = capture_step
    batch = data.clone()
    tu.assign_non_tensor(
        batch,
        temperature=1.0,
        compute_loss=True,
        full_vocab_teacher=False,
        full_vocab_train=True,
        mirror_anchor_version=(3000, 0),
        global_token_num=None,
        global_batch_size=4,
        mini_batch_size=4,
        epochs=1,
        seed=42,
        dataloader_kwargs={"shuffle": False},
        collect_metrics=True,
        collect_update_records=True,
        metrics_attempt_start=0,
    )
    with anchor_scope(actor, batch, spec):
        native_before = replay.evaluate(actor, batch)
        frozen_batch, frozen_head = batch.clone(), actor.engine.mirror_anchor_head
        if loaded is None:
            update = actor.train_mini_batch(batch)
            assert tu.get(update, "optimizer_steps") == 1
        else:
            assign(actor, {n: theta[n] + loaded["deltas"]["adam"][n] for n in theta})
        native_after = replay.evaluate(actor, batch)
    actor.engine.optimizer_step = native_step
    actor.engine.full_vocab_teacher_head.release()
    gradient = capture["gradient"]
    assert set(gradient) == set(theta), "Every parameter needs an explicit gradient for controlled zero updates"
    native_delta = loaded["deltas"]["adam"] if loaded is not None else subtract(capture.pop("native_after"), theta)
    target_norm = norm(native_delta)
    gnorm = norm(gradient)
    states = [st["step"].item() for st in capture["state"].values()]
    result = {
        "scope": __doc__,
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "global_valid_tokens": int(args.response_count * world),
        "gradient_norm": gnorm,
        "native_before": native_before,
        "native_after": native_after,
        "native_update_norm": target_norm,
        "optimizer_start": {"entries": len(states), "step_min": min(states), "step_max": max(states)},
        "optimizer_groups": capture["groups"],
        "directions": {},
        "evaluations": [],
    }

    def save():
        if rank == 0:
            (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")

    save()
    if rank == 0:
        print(
            "CAPTURE",
            result["native_update_norm"],
            native_before["distill/target_kl"],
            native_after["distill/target_kl"],
            flush=True,
        )

    def adam_update(kind):
        return replay_adam_update(theta, gradient, capture["state"], capture["groups"], kind)

    if loaded is not None:
        raw = loaded["deltas"]
        for key in (
            "native_replica_relative_error",
            "native_replica_exact",
            "directions",
            "current_increment_to_adam_norm_ratio",
        ):
            result[key] = source_result[key]
        result["reused_verified_capture"] = str(args.capture_dir)
        loaded.clear()
    else:
        raw = {"adam": native_delta}
        replica = adam_update("saved")
        error = subtract(replica, native_delta)
        result["native_replica_relative_error"] = norm(error) / target_norm
        result["native_replica_exact"] = all(torch.equal(replica[k], native_delta[k]) for k in theta)
        exact = torch.tensor(int(result["native_replica_exact"]), device="cuda")
        dist.all_reduce(exact, op=dist.ReduceOp.MIN)
        result["native_replica_exact"] = bool(exact.item())
        assert result["native_replica_relative_error"] < 1e-6, result["native_replica_relative_error"]
        del error, replica
        for kind in ("zero", "decay_only", "reset_m", "fresh"):
            raw[kind] = adam_update(kind)
        raw["sgd"] = {n: -g for n, g in gradient.items()}
        raw["current_increment"] = subtract(raw["adam"], raw["zero"])
        raw_norms = {name: norm(d) for name, d in raw.items()}
        for name, d in raw.items():
            nd = raw_norms[name]
            gdot = dot(gradient, d)
            result["directions"][name] = {
                "norm": nd,
                "cosine_with_adam": dot(d, raw["adam"]) / (nd * target_norm) if nd else None,
                "cosine_with_zero": dot(d, raw["zero"]) / (nd * raw_norms["zero"])
                if nd and raw_norms["zero"]
                else None,
                "gradient_dot": gdot,
                "cosine_with_negative_gradient": -gdot / (gnorm * nd) if nd else None,
            }
        result["current_increment_to_adam_norm_ratio"] = raw_norms["current_increment"] / target_norm
    save()
    if rank == 0:
        print("DIRECTIONS", json.dumps(result["directions"]), flush=True)
    # Free unused native optimizer buffers before evaluating counterfactuals.
    if args.save_state:
        torch.save(
            {
                "theta": theta,
                "gradient": gradient,
                "optimizer": capture["state"],
                "groups": capture["groups"],
                "deltas": raw,
            },
            args.out / f"capture-rank{rank}.pt",
        )
    capture.clear()
    gc.collect()
    candidates = [("initial", None, False)]
    for name in ("adam", "zero", "decay_only", "reset_m", "fresh", "current_increment"):
        candidates.append((name, name, False))
    for name in ("reset_m", "fresh", "current_increment", "sgd"):
        candidates.append((name + "_matched", name, True))
    if args.candidates:
        allowed = set(args.candidates) | {"initial"}
        unknown = allowed - {c[0] for c in candidates}
        if unknown:
            raise ValueError(f"Unknown candidates: {unknown}")
        candidates = [c for c in candidates if c[0] in allowed]
    candidate_states = {}
    for label, name, matched in candidates:
        if name is None:
            candidate_states[label] = theta
            continue
        scale = target_norm / result["directions"][name]["norm"] if matched else 1.0
        for multiplier in args.scales:
            key = label if multiplier == 1 else f"{label}@{multiplier}"
            values = {n: theta[n] + (scale * multiplier) * v for n, v in raw[name].items()}
            actual = subtract(values, theta)
            actual_norm = norm(actual)
            result.setdefault("actual_updates", {})[key] = {
                "requested_multiplier": multiplier,
                "direction_scale": scale,
                "norm": actual_norm,
                "relative_to_adam_norm": actual_norm / target_norm,
                "gradient_dot": dot(gradient, actual),
            }
            candidate_states[key] = values
    save()
    for dtype in ("bfloat16", "float32"):
        if dtype == "float32":
            del actor
            gc.collect()
            torch.cuda.empty_cache()
            actor = make_actor(dtype)
        current_mode[0] = dtype
        for label, values in candidate_states.items():
            assign(actor, values, rounded=(dtype == "float32"))
            actor.engine.mirror_anchor_head = frozen_head
            initializing[0] = label == "initial"
            metrics = replay.evaluate(actor, frozen_batch)
            actor.engine.full_vocab_teacher_head.release()
            row = {"precision": dtype, "rounded_weights": dtype == "float32", "candidate": label, "metrics": metrics}
            result["evaluations"].append(row)
            save()
            if rank == 0:
                print("EVAL", dtype, label, metrics, flush=True)
    actor.engine.mirror_anchor_head = None
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
