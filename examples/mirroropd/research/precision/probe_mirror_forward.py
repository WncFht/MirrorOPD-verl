#!/usr/bin/env python3
# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Hold the exact Adam displacement fixed and isolate forward precision.

Four ranks, four fixed prefixes. Native teacher/anchor log probabilities stay
fixed across all evaluations. mode_anchor_kl additionally measures each execution
mode relative to its own initial output, removing static dtype-switch offsets.
Movement mode captures actual master shards (optional --save-snapshots); layer
mode reuses those shards and promotes Qwen2 prefixes/suffixes to FP32 arithmetic
while retaining the same BF16-valued weights. Single sequence per rank only.
"""

import argparse
import gc
import json
import os
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

from . import engine as replay
from .engine import assign, local, snapshot
from .factor_mirror_precision import MODES as FACTOR_PRECISIONS
from .factor_mirror_precision import reset_interventions


def patch_layers(actor):
    selected = set()
    for name, module in actor.engine.module.named_modules():
        if isinstance(module, torch.nn.Linear) and name != "lm_head":
            from types import MethodType

            def linear(module, inputs):
                if torch.is_autocast_enabled("cuda"):
                    return torch.nn.functional.linear(inputs, module.weight, module.bias)
                return torch.nn.functional.linear(
                    inputs.float(), module.weight.float(), None if module.bias is None else module.bias.float()
                )

            module.forward = MethodType(linear, module)
        if hasattr(module, "self_attn") and hasattr(module, "mlp"):
            original = module.forward
            layer_index = int(name.split(".")[-1])

            def forward(hidden_states, *args, original=original, layer_index=layer_index, **kwargs):
                if layer_index in selected:
                    with torch.autocast("cuda", enabled=False):
                        return original(hidden_states.float(), *args, **kwargs)
                return original(hidden_states.to(torch.bfloat16), *args, **kwargs)

            module.forward = forward
    return selected


def forward_error_decomposition(lp0, lp1, ref0, ref1):
    """Exact variance decomposition of centered log-ratio motion, plus true KL.

    The variance is a local quadratic KL approximation; its components are not
    an additive causal decomposition of the actual training objective.
    """
    w = lp0.exp()

    def center(v):
        return v - (w * v).sum(-1, keepdim=True)

    motion = center(lp1 - lp0)
    signal = center(ref1 - ref0)
    change = motion - signal
    e0, e1 = center(lp0 - ref0), center(lp1 - ref1)
    per_row = [
        0.5 * (w * motion.square()).sum(-1),
        0.5 * (w * signal.square()).sum(-1),
        0.5 * (w * change.square()).sum(-1),
        (w * signal * change).sum(-1),
        (w * e0.square()).sum(-1),
        (w * e1.square()).sum(-1),
        (w * e0 * e1).sum(-1),
        (lp1.exp() * (lp1 - lp0)).sum(-1),
        (ref1.exp() * (ref1 - ref0)).sum(-1),
    ]
    totals = torch.stack([x.sum() for x in per_row] + [lp0.new_tensor(len(lp0))])
    if dist.is_initialized():
        dist.all_reduce(totals)
    values = (totals[:-1] / totals[-1]).tolist()
    denom = (values[4] * values[5]) ** 0.5
    return dict(
        zip(
            [
                "quadratic_motion",
                "quadratic_reference_motion",
                "quadratic_error_change",
                "cross_term",
                "initial_error_energy",
                "final_error_energy",
                "error_inner_product",
                "actual_policy_kl",
                "reference_policy_kl",
            ],
            values,
            strict=True,
        )
    ) | {
        "error_correlation": values[6] / denom if denom else None,
        "decomposition_residual": values[0] - values[1] - values[2] - values[3],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--student", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--prefixes", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--snapshots", type=Path, help="Read previously saved rank shards instead of training")
    parser.add_argument("--save-snapshots", action="store_true")
    parser.add_argument("--mode", choices=["movement", "layers"], default="movement")
    parser.add_argument("--lrs", type=float, nargs="+", default=[1e-6, 1e-7])
    parser.add_argument("--layer-counts", type=int, nargs="+", default=[1, 4, 8, 14, 28])
    parser.add_argument("--memory-fraction", type=float, default=0.30)
    parser.add_argument("--math-attention", action="store_true", help="Force the same SDPA math backend across dtypes")
    parser.add_argument(
        "--forward-precisions",
        nargs="+",
        default=["bfloat16", "float32"],
        choices=[
            "bfloat16",
            "float32",
            "bf16_weights_fp32_compute",
            "bf16_gemm_fp32_outputs",
            "bf16_mlp_fp32_outputs",
            "bf16_attention_fp32_outputs",
            "fp32_residual",
            "residual_only",
            "fp32_norm",
            "fp32_final_norm",
            *FACTOR_PRECISIONS,
        ],
    )
    args = parser.parse_args()
    if args.forward_precisions[0] != "bfloat16":
        parser.error("The first forward precision must be bfloat16 as the baseline")
    if args.mode == "layers" and not args.snapshots:
        parser.error("layers requires --snapshots")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.math_attention:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    torch.set_num_threads(1)
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    torch.backends.cuda.matmul.allow_tf32 = False
    _, rank, world = initialize_global_process_group(timeout_second=600)
    ckpt = args.checkpoint
    rec = json.loads(args.prefixes.read_text())["records"][rank]
    prompt = rec["positions"][0] + 1
    tokens = torch.tensor(rec["ids"][: prompt + 128])
    response = tokens[prompt:]
    data = TensorDict(
        {
            "input_ids": replay.nested([tokens]),
            "position_ids": replay.nested([torch.arange(len(tokens))]),
            "prompts": replay.nested([tokens[:prompt]]),
            "responses": replay.nested([response]),
            "response_mask": replay.nested([torch.ones(len(response), dtype=torch.long)]),
            "loss_mask": replay.nested([torch.ones(len(response), dtype=torch.long)]),
        },
        batch_size=[1],
    )
    teacher_path = args.teacher
    student_path = args.student
    teacher = replay.make_worker(teacher_path, world, True)
    tu.assign_non_tensor(data, temperature=1.0, compute_loss=False, full_vocab_teacher=True, global_token_num=None)
    labels = teacher.infer_batch(data)
    data.update(labels.select("teacher_hidden", "teacher_positions"))
    del teacher
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

    def actor_for(dtype, attn):
        import verl.workers.engine.fsdp.transformer_impl as impl

        reset_interventions()
        old_policy = impl.MixedPrecisionPolicy
        if args.mode == "layers":
            impl.MixedPrecisionPolicy = lambda **kw: old_policy(**(kw | {"cast_forward_inputs": False}))
        try:
            factory = replay.make_worker
            if dtype in FACTOR_PRECISIONS:
                from .factor_mirror_precision import make_worker as factory
            elif dtype not in ("bfloat16", "float16", "float32"):
                from .isolate_mirror_precision import make_worker as factory
            a = factory(student_path, world, precision=dtype, attention=attn)
        finally:
            impl.MixedPrecisionPolicy = old_policy
        a.engine.configure_full_vocab(cfg.distillation_loss.full_vocab, str(teacher_path), spec=spec)
        a.set_loss_fn(partial(replay.loss_fn, config=cfg))
        return a

    frozen = {}
    mode_anchor = {}
    activations = {}
    initial_activations = {}
    current_mode = [None]
    current_snapshot = [None]
    saved_logps = {}
    error_records = []

    def fixed_reference(hs, ws, ht, wt, *, hq, wq, spec, **kwargs):
        z = kernels.project(hs, ws).double()
        lp = z.log_softmax(-1)
        p = lp.exp()
        if not frozen:
            lq = kernels.project(hq, wq).double().log_softmax(-1)
            lt = kernels.project(ht, wt).double().log_softmax(-1)
            frozen.update(lq=lq, lt=lt, b=lq + spec.eta * (lt - lq))
        if current_mode[0] is not None and current_snapshot[0] == "initial":
            mode_anchor[current_mode[0]] = lp.detach().clone()
        mode_kl = (
            (p * (lp - mode_anchor[current_mode[0]])).sum(-1) if current_mode[0] is not None else lp.new_zeros(len(lp))
        )
        if args.mode == "movement" and current_mode[0] is not None:
            saved_logps[(current_mode[0], current_snapshot[0])] = lp.detach().cpu()
        b = frozen["b"]
        rows = (p * (lp - b)).sum(-1)
        lz = b.logsumexp(-1)
        return kernels.FullVocabObjectiveResult(
            rows.sum(),
            rows,
            {
                "mode_anchor_kl": mode_kl,
                "target_kl": rows + lz,
                "target_log_z": lz,
                "anchor_kl": (p * (lp - frozen["lq"])).sum(-1),
                "teacher_kl": (p * (lp - frozen["lt"])).sum(-1),
            },
        )

    actor = actor_for("bfloat16", "sdpa" if args.mode == "layers" else "flash_attention_2")
    actor.load_checkpoint(str(ckpt))
    snapshots = {"initial": snapshot(actor)}
    stats = []
    evaluations = []

    def output():
        if rank == 0:
            (args.out / "result.json").write_text(
                json.dumps(
                    {
                        "scope": __doc__,
                        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                        "fixed_target": True,
                        "step_statistics": stats,
                        "evaluations": evaluations,
                        "forward_error_decomposition": error_records,
                    },
                    indent=2,
                )
                + "\n"
            )

    if args.snapshots:
        snapshots = {}
        files = sorted(
            args.snapshots.glob(f"snapshot-*-rank{rank}.pt"), key=lambda p: ("initial" not in p.name, p.name)
        )
        for path in files:
            tag = path.name.removeprefix("snapshot-").removesuffix(f"-rank{rank}.pt")
            name = "initial" if tag == "initial" else "/".join(tag.rsplit("-", 1))
            snapshots[name] = torch.load(path, map_location="cpu", weights_only=True)
        assign(actor, snapshots["initial"])
        b = data.clone()
        tu.assign_non_tensor(
            b,
            temperature=1.0,
            compute_loss=True,
            full_vocab_teacher=False,
            full_vocab_train=True,
            mirror_anchor_version=(2000, 0),
            global_token_num=None,
        )
        with anchor_scope(actor, b, spec):
            replay.evaluate(actor, b, reference_fn=fixed_reference)
            frozen_batch = b.clone()
            frozen_head = actor.engine.mirror_anchor_head
    for lr in [] if args.snapshots else args.lrs:
        actor.load_checkpoint(str(ckpt))
        for group in actor.engine.optimizer.param_groups:
            group["lr"] = lr
        actor.engine.lr_scheduler.base_lrs = [lr] * len(actor.engine.optimizer.param_groups)
        b = data.clone()
        tu.assign_non_tensor(
            b,
            temperature=1.0,
            compute_loss=True,
            full_vocab_teacher=False,
            full_vocab_train=True,
            mirror_anchor_version=(2000, 0),
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
        orig = actor.engine.optimizer_step
        counter = [0]

        def step(counter=counter, orig=orig, lr=lr, actor=actor):
            counter[0] += 1
            if counter[0] not in (1, 16):
                return orig()
            named = list(actor.engine.module.named_parameters())
            old = [local(p).detach().clone() for _, p in named]
            norm = orig()
            values = torch.zeros(7, device="cuda", dtype=torch.float64)
            per_layer = {}
            for (name, p), v0 in zip(named, old, strict=True):
                v1 = local(p).detach()
                d = (v1 - v0).double()
                effective = (v1.to(torch.bfloat16).float() - v0.to(torch.bfloat16).float()).double()
                g = local(p.grad).double()
                row = torch.stack(
                    [
                        d.square().sum(),
                        effective.square().sum(),
                        (g * d).sum(),
                        (g * effective).sum(),
                        (d * effective).sum(),
                        (effective != 0).sum().double(),
                        torch.tensor(d.numel(), device="cuda", dtype=torch.float64),
                    ]
                )
                values += row
                key = ".".join(name.split(".")[:3]) if "layers." in name else name
                per_layer[key] = per_layer.get(key, torch.zeros_like(row)) + row
            dist.all_reduce(values)
            layers = {}
            for key, val in per_layer.items():
                dist.all_reduce(val)
                layers[key] = val.tolist()
            stats.append(
                {
                    "lr": lr,
                    "update": counter[0],
                    "grad_norm": norm,
                    "columns": [
                        "master_delta_sq",
                        "bf16_effective_delta_sq",
                        "g_dot_master",
                        "g_dot_effective",
                        "master_dot_effective",
                        "changed_effective_count",
                        "parameters",
                    ],
                    "values": values.tolist(),
                    "layers": layers,
                }
            )
            snapshots[f"{lr}/{counter[0]}"] = snapshot(actor)
            output()
            return norm

        actor.engine.optimizer_step = step
        with anchor_scope(actor, b, spec):
            if not frozen:
                replay.evaluate(actor, b, reference_fn=fixed_reference)
                frozen_batch = b.clone()
                frozen_head = actor.engine.mirror_anchor_head
            actor.train_mini_batch(b)
        actor.engine.optimizer_step = orig
        actor.engine.full_vocab_teacher_head.release()
    for name in snapshots if args.save_snapshots else []:
        torch.save(snapshots[name], args.out / f"snapshot-{name.replace('/', '-')}-rank{rank}.pt")
    # All evaluations use exactly the same cached native BF16 q/T log probabilities.
    if args.mode == "layers":
        if actor.engine.module.config.model_type != "qwen2":
            raise ValueError("layer probes require Qwen2")
        selected = patch_layers(actor)
        depth = actor.engine.module.config.num_hidden_layers
        if any(n < 1 or n > depth for n in args.layer_counts):
            raise ValueError("layer-counts outside model depth")
        modes = [
            ("native", []),
            *[(f"last{n}", list(range(depth - n, depth))) for n in args.layer_counts],
            *[(f"first{n}", list(range(n))) for n in args.layer_counts if n < depth],
        ]
    else:
        modes = [(dtype, []) for dtype in args.forward_precisions]
    actor_precision = "bfloat16"
    for dtype, indices in modes:
        rounded = args.mode == "movement" and dtype == "float32"
        attn = (
            "sdpa"
            if args.mode == "layers" or dtype in ("float32", "bf16_weights_fp32_compute")
            else "flash_attention_2"
        )
        if args.mode == "layers":
            selected.clear()
            selected.update(indices)
        elif dtype != actor_precision:
            del actor
            gc.collect()
            torch.cuda.empty_cache()
            actor = actor_for(dtype, attn)
            actor_precision = dtype
        current_mode[0] = dtype
        handles = []
        for name, module in actor.engine.module.named_modules():
            if hasattr(module, "self_attn") and hasattr(module, "mlp"):

                def hook(module, inputs, output, name=name):
                    activations[name] = output.detach().float().cpu()

                handles.append(module.register_forward_hook(hook))
        for name, values in snapshots.items():
            current_snapshot[0] = name
            activations.clear()
            assign(actor, values, rounded=rounded)
            actor.engine.mirror_anchor_head = frozen_head
            metrics = replay.evaluate(actor, frozen_batch, reference_fn=fixed_reference)
            layer_changes = {}
            if name == "initial":
                initial_activations[dtype] = {k: v.clone() for k, v in activations.items()}
            for key, value in activations.items():
                ref = initial_activations[dtype][key]
                cross = initial_activations["native" if args.mode == "layers" else "bfloat16"][key]
                sums = torch.tensor(
                    [
                        (value - ref).double().square().sum().item(),
                        ref.double().square().sum().item(),
                        (value - cross).double().square().sum().item(),
                    ],
                    device="cuda",
                    dtype=torch.float64,
                )
                dist.all_reduce(sums)
                layer_changes[key] = {
                    "relative_to_own_initial": (sums[0] / sums[1]).sqrt().item(),
                    "relative_to_bf16_initial": (sums[2] / sums[1]).sqrt().item(),
                }
            actor.engine.full_vocab_teacher_head.release()
            row = {
                "precision": dtype,
                "attention": attn,
                "rounded_weights": rounded,
                "snapshot": name,
                "metrics": metrics,
                "layer_changes": layer_changes,
            }
            evaluations.append(row)
            if args.mode == "movement" and dtype == "float32":
                for mode in args.forward_precisions:
                    if mode == "float32" or (mode, name) not in saved_logps:
                        continue
                    tensors = [
                        saved_logps[key].to("cuda")
                        for key in [(mode, "initial"), (mode, name), ("float32", "initial"), ("float32", name)]
                    ]
                    error_records.append({"precision": mode, "snapshot": name, **forward_error_decomposition(*tensors)})
                    del tensors
            output()
            if rank == 0:
                print("EVAL", dtype, name, metrics, flush=True)
        for handle in handles:
            handle.remove()
    actor.engine.mirror_anchor_head = None
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
