# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Compare Liger PR #1555's compensated gap with our production difference.

Research probe only: does not install a dependency or change production dispatch.
The compensated algorithm follows head 740a415a of kashif/Liger-Kernel.
"""

import argparse
import json
from pathlib import Path

import torch

from verl.trainer.distillation.numerics import relative_difference


def compensated_gap(x, y, pivot):
    a, b = x.gather(-1, pivot), y.gather(-1, pivot)
    delta, shift = x - y, a - b
    error = torch.where(x.abs() >= y.abs(), (x - delta) - y, x - (delta + y))
    shift_error = torch.where(a.abs() >= b.abs(), (a - shift) - b, a - (shift + b))
    return (delta - shift) + (error - shift_error)


def cuda_gap(x, y, pivot):
    import triton
    import triton.language as tl

    @triton.jit
    def kernel(X, Y, P, OUT, V: tl.constexpr, B: tl.constexpr):
        row, tile = tl.program_id(0), tl.program_id(1)
        c = tile * B + tl.arange(0, B)
        p = tl.load(P + row)
        a, b = tl.load(X + row * V + p), tl.load(Y + row * V + p)
        x, y = tl.load(X + row * V + c, c < V, 0), tl.load(Y + row * V + c, c < V, 0)
        d, s = x - y, a - b
        de = tl.where(tl.abs(x) >= tl.abs(y), (x - d) - y, x - (d + y))
        se = tl.where(tl.abs(a) >= tl.abs(b), (a - s) - b, a - (s + b))
        tl.store(OUT + row * V + c, (d - s) + (de - se), c < V)

    out = torch.empty_like(x)
    kernel[(len(x), triton.cdiv(x.shape[1], 1024))](x, y, pivot, out, x.shape[1], 1024, enable_fp_fusion=False)
    return out


def compare(x, y, name):
    pivot = x.argmax(-1, keepdim=True)
    ref = ((x.double() - y.double()) - (x.double() - y.double()).gather(-1, pivot)).float()
    actual = relative_difference(x, y, pivot)
    candidate = cuda_gap(x, y, pivot) if x.is_cuda else compensated_gap(x, y, pivot)
    return {
        "case": name,
        "rows": len(x),
        "vocab": x.shape[-1],
        "production_max_abs_error": (actual - ref).abs().max().item(),
        "candidate_max_abs_error": (candidate - ref).abs().max().item(),
        "different_elements": int((actual != candidate).sum()),
        "candidate_zero_but_reference_nonzero": int(((candidate == 0) & (ref != 0)).sum()),
        "production_zero_but_reference_nonzero": int(((actual == 0) & (ref != 0)).sum()),
    }


def external_losses():
    from flash_attn.ops.triton.cross_entropy import cross_entropy_loss

    from verl.trainer.distillation.full_vocab import full_reverse_kl_result
    from verl.utils.stable_logprobs import selected_logprobs

    result = []
    labels = torch.tensor([0], device="cuda")
    for name in ("flash_attn_ce", "project_ce"):
        z = torch.tensor([[9.0, -9.0, -9.0]], device="cuda", requires_grad=True)
        value = (
            cross_entropy_loss(z, labels)[0].sum() if name == "flash_attn_ce" else -selected_logprobs(z, labels).sum()
        )
        g = torch.autograd.grad(value, z)[0]
        result.append({"backend": name, "value": value.item(), "gradient": g.tolist()})
    for offset in (0.0, 32.0):
        for backend in ("precompute", "liger"):
            z = (torch.tensor([[0.0, -24.0, -26.0]], device="cuda") + offset).requires_grad_()
            teacher = torch.tensor([[0.0, -23.97, -26.02]], device="cuda")
            head = torch.eye(3, device="cuda")
            value = full_reverse_kl_result(z, head, teacher, head, backend=backend).loss_sum
            g = torch.autograd.grad(value, z)[0]
            result.append({"backend": backend, "offset": offset, "value": value.item(), "gradient": g.tolist()})
    return result


def liger_patch_scope():
    """Run in this disposable probe process: the upstream patch changes globals."""
    from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from transformers.models.qwen2 import modeling_qwen2

    config = Qwen2Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
    )
    actor = Qwen2ForCausalLM(config)
    original_rope = modeling_qwen2.apply_rotary_pos_emb
    before = type(actor.model.norm).__name__
    _apply_liger_kernel_to_instance(actor, fused_linear_cross_entropy=False, swiglu=True)
    teacher = Qwen2ForCausalLM(config)
    return {
        "before_norm": before,
        "actor_forward_root": actor.forward.__func__.__qualname__,
        "module_rope_changed": original_rope is not modeling_qwen2.apply_rotary_pos_emb,
        "new_teacher_norm": type(teacher.model.norm).__name__,
        "new_teacher_mlp": type(teacher.model.layers[0].mlp).__name__,
        "note": "CPU construction only; no model forward, no explicit teacher Liger patch",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--real-logits", type=Path)
    parser.add_argument(
        "--liger-scope", action="store_true", help="also probe the installed patch's process-global scope"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(27)
    if args.device == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.05)
    rows = []
    for scale in (0.001, 1.0, 128.0):
        base = torch.randn(256, 129, device=args.device) * scale
        for offset in (0.0, 32.0, 1024.0):
            for delta in (0.0, 1e-5, 0.01):
                x = (base + delta * torch.randn_like(base) + offset).contiguous()
                rows.append(compare(x, base, f"random-scale{scale}-offset{offset}-delta{delta}"))
    # Independent exact-input witnesses: the small residual is already an FP32
    # input, but subtracting the pivot separately can swallow it even in FP64.
    x = torch.tensor([[1.0, 1e-20, 0.0]], device=args.device)
    y = torch.tensor([[1.0, 0.0, 0.0]], device=args.device)
    rows.append(compare(x, y, "tiny_input_below_fp64_pivot_ulp"))
    # A compensated pair difference does not justify doing the eta combination
    # in FP32: float32(0.3) - float64(0.3) is nonzero by construction.
    anchor = torch.zeros(1, 2, device=args.device)
    teacher = torch.tensor([[0.0, 1.0]], device=args.device)
    z = torch.tensor([[0.0, 0.3]], device=args.device)
    pivot = torch.zeros(1, 1, dtype=torch.long, device=args.device)
    geometric = {
        "production": relative_difference(z, teacher, pivot, anchor=anchor, eta=0.3).tolist(),
        "naive_fp32_extension": (
            compensated_gap(z, anchor, pivot) - 0.3 * compensated_gap(teacher, anchor, pivot)
        ).tolist(),
        "reference": (z.double() - 0.3 * teacher.double()).tolist(),
    }
    if args.real_logits:
        saved = torch.load(args.real_logits, map_location="cpu", weights_only=True)
        for start in range(0, len(saved["student"]), 8):
            x = saved["student"][start : start + 8].to(args.device).float().contiguous()
            y = saved["teacher"][start : start + 8, : x.shape[1]].to(args.device).float().contiguous()
            rows.append(compare(x, y, f"saved_logits_{start}"))
    result = {
        "device": args.device,
        "torch": torch.__version__,
        "reference": (
            "FP64 difference-first, then FP32; analytical tiny witnesses, not arbitrary-precision exhaustive proof"
        ),
        "rows": rows,
        "geometric": geometric,
    }
    if args.device == "cuda":
        result["external_losses"] = external_losses()
    if args.liger_scope:
        result["liger_patch_scope"] = liger_patch_scope()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "cases": len(rows),
                "different_elements": sum(r["different_elements"] for r in rows),
                "tiny": rows[27],
                "geometric": geometric,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
