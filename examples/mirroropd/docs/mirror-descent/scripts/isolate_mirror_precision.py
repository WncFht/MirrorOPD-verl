#!/usr/bin/env python3
# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic precision interventions for replay_fixed_prefix.py.

Same CLI and branch JSON, with additional kernel names row64/head64 and precision
names bf16_weights_fp32_compute, fp32_residual, residual_only, fp32_norm and
fp32_final_norm. These Qwen2-only interventions are NOT production configuration.
Each precision branch captures its own q. For identical-displacement evidence,
use probe_mirror_forward.py instead. No Ray/vLLM/TQ is started here.
"""

from types import MethodType

import replay_fixed_prefix as replay
import torch

from verl.trainer.distillation import full_vocab as kernels
from verl.trainer.distillation.full_vocab_triton import fused_log_weights


def rows64(hs, ws, ht, wt, hq, wq, spec):
    """Native projections and exact original FP32 b; only row arithmetic changes."""
    z = kernels.project(hs, ws)
    lt = kernels.project(ht, wt).log_softmax(-1)[:, : len(ws)]
    lq = kernels.project(hq, wq).log_softmax(-1)
    b = fused_log_weights(lt, lq, kappa=spec.kappa, eta=spec.eta)
    lp = z.double().log_softmax(-1)
    p = lp.exp()
    gap = lp - b.double()
    rows = (p * gap).sum(-1)
    dz = p * (gap - rows[:, None])
    log_z = b.double().logsumexp(-1)
    diag = torch.stack([rows + log_z, log_z, (p * (lp - lq.double())).sum(-1), (p * (lp - lt.double())).sum(-1)], -1)
    return rows, dz, diag


def rows64_cast(*args):
    rows, dz, diag = rows64(*args)
    return rows, dz.to(args[0].dtype), diag


def row64_result(hs, ws, ht, wt, *, spec, hq, wq, token_chunk, **kwargs):
    total, rows, diag = kernels.PrecomputedKL.apply(hs, ws, ht, wt, token_chunk, rows64_cast, hq, wq, spec)
    return kernels.FullVocabObjectiveResult(
        total, rows, dict(zip(kernels._MIRROR_DIAGNOSTICS, diag.unbind(-1), strict=True))
    )


class Head64(torch.autograd.Function):
    """Keep native forward projections; compute row and head VJPs in FP64."""

    @staticmethod
    def forward(ctx, hs, ws, ht, wt, hq, wq, spec):
        rows, dz, diag = rows64(hs, ws, ht, wt, hq, wq, spec)
        dh = dz @ ws.double()
        dw = dz.T @ hs.double()
        ctx.save_for_backward(dh, dw)
        ctx.hdtype, ctx.wdtype = hs.dtype, ws.dtype
        ctx.mark_non_differentiable(rows, diag)
        return rows.sum(), rows, diag

    @staticmethod
    def backward(ctx, upstream, *unused):
        dh, dw = ctx.saved_tensors
        return (
            (dh * upstream.double()).to(ctx.hdtype),
            (dw * upstream.double()).to(ctx.wdtype),
            None,
            None,
            None,
            None,
            None,
        )


def head64_result(hs, ws, ht, wt, *, spec, hq, wq, **kwargs):
    total, rows, diag = Head64.apply(hs, ws, ht, wt, hq, wq, spec)
    return kernels.FullVocabObjectiveResult(
        total, rows, dict(zip(kernels._MIRROR_DIAGNOSTICS, diag.unbind(-1), strict=True))
    )


def make_worker(path, world, teacher=False, precision="bfloat16", attention="flash_attention_2"):
    import verl.workers.engine.fsdp.transformer_impl as implementation

    promoted_residual = precision in ("bf16_weights_fp32_compute", "fp32_residual", "residual_only")
    norm_only = precision in ("fp32_norm", "fp32_final_norm")
    special = promoted_residual or norm_only
    original_policy = implementation.MixedPrecisionPolicy
    if special:
        # Preserve FP32 activations at FSDP block/head boundaries while keeping
        # parameter all-gathers and optimizer master storage unchanged.
        implementation.MixedPrecisionPolicy = lambda **kw: original_policy(**(kw | {"cast_forward_inputs": False}))
    try:
        worker = native_worker(
            path, world, teacher=teacher, precision="bfloat16" if special else precision, attention=attention
        )
    finally:
        implementation.MixedPrecisionPolicy = original_policy
    if not special:
        return worker
    if worker.engine.module.config.model_type != "qwen2":
        raise ValueError("Precision interventions were audited only on Qwen2")
    if promoted_residual:
        worker.engine.module.model.embed_tokens.register_forward_hook(lambda module, args, out: out.float())
    if precision == "bf16_weights_fp32_compute":
        worker.engine._autocast_dtype = torch.float32

        def linear_fp32(module, inputs):
            return torch.nn.functional.linear(
                inputs.float(), module.weight.float(), None if module.bias is None else module.bias.float()
            )

        for name, module in worker.engine.module.named_modules():
            if isinstance(module, torch.nn.Linear) and name != "lm_head":
                module.forward = MethodType(linear_fp32, module)
    elif precision == "residual_only":
        for module in worker.engine.module.modules():
            if module.__class__.__name__.endswith("Qwen2RMSNorm"):
                original = module.forward

                def norm_bf16(inputs, original=original):
                    return original(inputs.to(torch.bfloat16))

                module.forward = norm_bf16
        worker.engine.module.model.rotary_emb.register_forward_hook(
            lambda module, args, out: tuple(v.to(torch.bfloat16) for v in out)
        )
    elif norm_only:

        def norm_fp32(module, inputs):
            h = inputs.float()
            return module.weight.float() * h * torch.rsqrt(h.square().mean(-1, keepdim=True) + module.variance_epsilon)

        for name, module in worker.engine.module.named_modules():
            if module.__class__.__name__.endswith("Qwen2RMSNorm") and (
                precision == "fp32_norm" or name == "model.norm"
            ):
                module.forward = MethodType(norm_fp32, module)
    return worker


native_worker = replay.make_worker

if __name__ == "__main__":
    replay.make_worker = make_worker
    replay.main(extra_kernels={"row64": row64_result, "head64": head64_result})
