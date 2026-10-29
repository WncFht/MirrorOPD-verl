#!/usr/bin/env python3
# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Qwen2-only diagnostic precision factors; never production configuration.

Residual-stream and midstream modes explicitly control block/head boundaries.
The rope_table_math_fp32 mode intentionally retains the FSDP recast failure as
a negative control; rope_table_preserved_fp32 protects the actual block input. Attention VJP oracles preserve the
native FA2 forward bitwise and replace only its backward on one causal sequence.
"""

import json
import os
from pathlib import Path
from types import MethodType

import torch

from examples.mirroropd.research.backends import fsdp as replay
from verl.trainer.distillation import full_vocab as kernels

MODES = (
    "gemm_no_reduced",
    "residual_stream_fp32",
    "norm_inner_fp32",
    "norm_vjp32",
    "rope_math_fp32",
    "rope_table_math_fp32",
    "rope_table_preserved_fp32",
    "midstream_head_bf16",
    "midstream_head_fp32",
    "mlp_pointwise_fp32",
    "mlp_internal_fp32",
    "attention_fp32_cast_back",
    "attention_vjp32",
    "attention_vjp64",
    "attention_deterministic",
    "capture_native",
)
native_worker = replay.make_worker
_restore = []
_captured = set()


def reset_interventions():
    while _restore:
        _restore.pop()()
    _captured.clear()


def audit_boundaries(worker, mode):
    """Record actual inference/training boundary dtypes once, without tensors."""
    directory = os.environ.get("MIRROR_PRECISION_BOUNDARY_DIR")
    if directory is None and os.environ.get("MIRROR_ATTENTION_CAPTURE_DIR"):
        directory = str(Path(os.environ["MIRROR_ATTENTION_CAPTURE_DIR"]).parent / "boundaries")
    if directory is None:
        return
    records, dirty = {}, [False]

    def describe(value):
        if isinstance(value, torch.Tensor):
            return {"dtype": str(value.dtype), "shape": list(value.shape)}
        if isinstance(value, tuple | list):
            return [describe(item) for item in value]
        return type(value).__name__

    def hook(module, args, kwargs, output, name):
        phase = "grad" if torch.is_grad_enabled() else "no_grad"
        key = f"{phase}/{name}"
        if key not in records:
            records[key] = {"input": describe(args or kwargs.get("hidden_states")), "output": describe(output)}
            if "position_embeddings" in kwargs:
                records[key]["position_embeddings"] = describe(kwargs["position_embeddings"])
            dirty[0] = True

    for name, module in worker.engine.module.named_modules():
        kind = module.__class__.__name__
        if (
            kind.endswith(("Qwen2RMSNorm", "Qwen2DecoderLayer", "Qwen2Attention", "Qwen2MLP", "Qwen2RotaryEmbedding"))
            or name == "lm_head"
        ):
            module.register_forward_hook(lambda m, a, kw, o, name=name: hook(m, a, kw, o, name), with_kwargs=True)

    def save(module, args, output):
        if dirty[0]:
            directory_path = Path(directory)
            directory_path.mkdir(parents=True, exist_ok=True)
            path = directory_path / f"rank{os.environ.get('RANK', '0')}-{mode}.json"
            temporary = path.with_suffix(".tmp")
            reduced = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
            temporary.write_text(
                json.dumps(
                    {
                        "mode": mode,
                        "records": records,
                        "bf16_reduced_precision_reduction": reduced,
                    },
                    indent=2,
                )
                + "\n"
            )
            temporary.replace(path)
            dirty[0] = False

    worker.engine.module.register_forward_hook(save)


def dense_attention_vjp(q, k, v, go, scale, dtype=torch.float64):
    """Analytic causal GQA VJP, with independent softmax row centering."""
    assert q.ndim == 4 and q.shape[2] == k.shape[2] == v.shape[2]
    groups = q.shape[1] // k.shape[1]
    assert groups * k.shape[1] == q.shape[1]
    with torch.autocast(q.device.type, enabled=False):
        qh = q.to(dtype)
        kh = k.to(dtype).repeat_interleave(groups, dim=1)
        vh = v.to(dtype).repeat_interleave(groups, dim=1)
        gh = go.transpose(1, 2).to(dtype)
        scores = (qh @ kh.transpose(-1, -2)) * scale
        mask = torch.ones(scores.shape[-2:], dtype=torch.bool, device=q.device).triu(1)
        scores.masked_fill_(mask, -torch.inf)
        p = scores.softmax(-1)
        dp = gh @ vh.transpose(-1, -2)
        ds = p * (dp - (p * dp).sum(-1, keepdim=True))
        dq = (ds @ kh) * scale
        dk = (ds.transpose(-1, -2) @ qh) * scale
        dv = p.transpose(-1, -2) @ gh
        shape = (k.shape[0], k.shape[1], groups, k.shape[2], k.shape[3])
        return dq, dk.reshape(shape).sum(2), dv.reshape(shape).sum(2)


class NativeForwardReferenceBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, module, native, kwargs, dtype):
        out, weights = native(module, q, k, v, None, **kwargs)
        assert weights is None
        assert out.shape == (q.shape[0], q.shape[2], q.shape[1], q.shape[3])
        ctx.save_for_backward(q, k, v)
        ctx.scale = kwargs["scaling"]
        ctx.dtype = dtype
        return out

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, go):
        q, k, v = ctx.saved_tensors
        dq, dk, dv = dense_attention_vjp(q, k, v, go, ctx.scale, ctx.dtype)
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), None, None, None, None


class NativeNormReferenceBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, epsilon, native):
        ctx.save_for_backward(x, weight)
        ctx.epsilon = epsilon
        return native(x)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, upstream):
        x, weight = ctx.saved_tensors
        dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        with torch.autocast(x.device.type, enabled=False):
            h, go = x.to(dtype), upstream.to(dtype)
            inv = torch.rsqrt(h.square().mean(-1, keepdim=True) + ctx.epsilon)
            z = h * inv
            v = go * weight.to(dtype)
            dx = inv * (v - z * (v * z).mean(-1, keepdim=True))
            dw = go * z
            if dw.ndim > 1:
                dw = dw.sum(tuple(range(dw.ndim - 1)))
        return dx.to(x.dtype), dw.to(weight.dtype), None, None


def require_single_causal(module, q, k, v, mask, kwargs):
    assert mask is None and q.shape[0] == 1
    assert q.shape[2] == k.shape[2] == v.shape[2]
    assert module.is_causal and kwargs.get("dropout", 0) == 0
    assert kwargs.get("sliding_window") is None and kwargs.get("softcap") is None
    positions = kwargs.get("position_ids")
    if positions is not None:
        assert torch.equal(positions.flatten(), torch.arange(q.shape[2], device=q.device))


def install_attention(mode):
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    native = ALL_ATTENTION_FUNCTIONS["flash_attention_2"]
    _restore.append(lambda: ALL_ATTENTION_FUNCTIONS.register("flash_attention_2", native))

    def interface(module, q, k, v, attention_mask, **kwargs):
        require_single_causal(module, q, k, v, attention_mask, kwargs)
        if mode == "attention_deterministic":
            return native(module, q, k, v, attention_mask, **(kwargs | {"deterministic": True}))
        if mode.startswith("attention_vjp"):
            dtype = torch.float64 if mode.endswith("64") else torch.float32
            return NativeForwardReferenceBackward.apply(q, k, v, module, native, kwargs, dtype), None
        if mode == "attention_fp32_cast_back":
            from torch.nn.attention import SDPBackend, sdpa_kernel

            groups = q.shape[1] // k.shape[1]
            with torch.autocast(q.device.type, enabled=False), sdpa_kernel(SDPBackend.MATH):
                out = torch.nn.functional.scaled_dot_product_attention(
                    q.float(),
                    k.float().repeat_interleave(groups, 1),
                    v.float().repeat_interleave(groups, 1),
                    is_causal=True,
                    scale=kwargs["scaling"],
                )
            return out.transpose(1, 2).to(q.dtype), None
        out, weights = native(module, q, k, v, attention_mask, **kwargs)
        key = (int(os.environ.get("RANK", 0)), module.layer_idx)
        if torch.is_grad_enabled() and out.requires_grad and key not in _captured:
            _captured.add(key)
            directory = Path(os.environ["MIRROR_ATTENTION_CAPTURE_DIR"])
            directory.mkdir(parents=True, exist_ok=True)
            record = {name: x.detach().cpu().clone() for name, x in (("q", q), ("k", k), ("v", v), ("out", out))}
            record.update(rank=key[0], layer=key[1], scale=kwargs["scaling"])
            keys = (
                "position_ids",
                "cu_seq_lens_q",
                "cu_seq_lens_k",
                "max_length_q",
                "max_length_k",
                "scaling",
                "dropout",
                "sliding_window",
                "is_causal",
                "deterministic",
            )
            record["native_kwargs"] = {
                key: value.detach().cpu().clone() if isinstance(value, torch.Tensor) else value
                for key, value in kwargs.items()
                if key in keys
            }
            record["fa_deterministic_env"] = os.environ.get("FLASH_ATTENTION_DETERMINISTIC", "0")

            def receive(name, grad):
                record[name] = grad.detach().cpu().clone()
                if all(name in record for name in ("go", "dq", "dk", "dv")):
                    torch.save(record, directory / f"rank{key[0]}-layer{key[1]}.pt")

            for name, x in (("go", out), ("dq", q), ("dk", k), ("dv", v)):
                x.register_hook(lambda grad, name=name: receive(name, grad))
        return out, weights

    ALL_ATTENTION_FUNCTIONS.register("flash_attention_2", interface)


def make_worker(path, world, teacher=False, precision="bfloat16", attention="flash_attention_2"):
    from transformers.models.qwen2 import modeling_qwen2 as qwen2

    import verl.workers.engine.fsdp.transformer_impl as implementation

    reset_interventions()
    if precision == "gemm_no_reduced":
        previous_reduction = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        _restore.append(
            lambda: setattr(torch.backends.cuda.matmul, "allow_bf16_reduced_precision_reduction", previous_reduction)
        )
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    if precision not in MODES:
        worker = native_worker(path, world, teacher=teacher, precision=precision, attention=attention)
        if not teacher:
            audit_boundaries(worker, precision)
        return worker
    old_policy = implementation.MixedPrecisionPolicy
    if precision in ("residual_stream_fp32", "rope_table_preserved_fp32", "midstream_head_bf16", "midstream_head_fp32"):
        implementation.MixedPrecisionPolicy = lambda **kw: old_policy(**(kw | {"cast_forward_inputs": False}))
    try:
        worker = native_worker(path, world, teacher=teacher, precision="bfloat16", attention="flash_attention_2")
    finally:
        implementation.MixedPrecisionPolicy = old_policy
    assert worker.engine.module.config.model_type == "qwen2"
    model = worker.engine.module
    if precision in ("residual_stream_fp32", "midstream_head_bf16", "midstream_head_fp32"):
        model.model.embed_tokens.register_forward_hook(lambda m, args, out: out.float())
        if precision == "residual_stream_fp32":
            model.model.rotary_emb.register_forward_hook(lambda m, args, out: tuple(v.bfloat16() for v in out))
        for name, module in model.named_modules():
            cast_norm = precision == "residual_stream_fp32" or (
                precision == "midstream_head_bf16" and name == "model.norm"
            )
            if cast_norm and module.__class__.__name__.endswith("Qwen2RMSNorm"):
                original = module.forward
                module.forward = lambda x, original=original: original(x.bfloat16())
    elif precision == "norm_vjp32":
        for module in model.modules():
            if module.__class__.__name__.endswith("Qwen2RMSNorm"):
                original = module.forward

                def norm_reference(inputs, module=module, original=original):
                    return NativeNormReferenceBackward.apply(inputs, module.weight, module.variance_epsilon, original)

                module.forward = norm_reference
    elif precision == "norm_inner_fp32":

        def norm(module, inputs):
            x = inputs.float()
            return module.weight.float() * x * torch.rsqrt(x.square().mean(-1, keepdim=True) + module.variance_epsilon)

        for name, module in model.named_modules():
            if ".layers." in name and module.__class__.__name__.endswith("Qwen2RMSNorm"):
                module.forward = MethodType(norm, module)
    elif precision.startswith("rope_"):
        original = qwen2.apply_rotary_pos_emb
        _restore.append(lambda: setattr(qwen2, "apply_rotary_pos_emb", original))

        def rotate(q, k, cos, sin, *args, **kwargs):
            with torch.autocast(q.device.type, enabled=False):
                qr, kr = original(q.float(), k.float(), cos.float(), sin.float(), *args, **kwargs)
            return qr.to(q.dtype), kr.to(k.dtype)

        qwen2.apply_rotary_pos_emb = rotate
        if precision in ("rope_table_math_fp32", "rope_table_preserved_fp32"):
            original_rope = model.model.rotary_emb.forward
            model.model.rotary_emb.forward = lambda x, positions: original_rope(x.float(), positions)
    elif precision.startswith("mlp_"):

        def mlp(module, inputs):
            if precision == "mlp_pointwise_fp32":
                gate, up = module.gate_proj(inputs).float(), module.up_proj(inputs).float()
                return module.down_proj(torch.nn.functional.silu(gate) * up)
            with torch.autocast(inputs.device.type, enabled=False):
                shape = inputs.shape
                x = inputs.reshape(-1, shape[-1]).bfloat16()
                gate = kernels.project(x, module.gate_proj.weight)
                up = kernels.project(x, module.up_proj.weight)
                h = (torch.nn.functional.silu(gate) * up).bfloat16()
                out = kernels.project(h, module.down_proj.weight).bfloat16()
                return out.reshape(shape)

        for module in model.modules():
            if module.__class__.__name__ == "Qwen2MLP":
                assert all(getattr(module, name).bias is None for name in ("gate_proj", "up_proj", "down_proj"))
                module.forward = MethodType(mlp, module)
    elif precision.startswith("attention_") or precision == "capture_native":
        install_attention(precision)
    audit_boundaries(worker, precision)
    return worker


if __name__ == "__main__":
    from examples.mirroropd.research.studies.precision import audit_mirror_solvers as audit

    audit.main(worker_factory=make_worker, precision_factory=make_worker)
