# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Selected log probabilities with no softmax-minus-one cancellation."""

import torch
import torch.nn.functional as F


class _SelectedLogProb(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, labels):
        x = logits if logits.dtype == torch.float64 else logits.float()
        index = labels.clamp_min(0).unsqueeze(-1)
        target = x.gather(-1, index)
        others = x.scatter(-1, index, -torch.inf)
        maximum = others.amax(-1, keepdim=True)
        maximum = torch.where(torch.isfinite(maximum), maximum, 0.0)
        total = (others - maximum).exp().sum(-1, keepdim=True)
        log_total = total.clamp_min(torch.finfo(x.dtype).tiny).log()
        margin = (maximum - target) + log_total
        bad_target = torch.isposinf(target) | ((target == -torch.inf) & (total == 0))
        margin = torch.where(total == 0, -torch.inf, margin)
        margin = torch.where(bad_target, torch.nan, margin)
        log_mass = -F.softplus(-margin)
        loss = F.softplus(margin).squeeze(-1)
        ctx.save_for_backward(logits, labels, maximum, log_total, log_mass)
        return torch.where(labels == -100, 0.0, -loss)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, upstream):
        logits, labels, maximum, log_total, log_mass = ctx.saved_tensors
        index = labels.clamp_min(0).unsqueeze(-1)
        x = logits.to(maximum.dtype)
        other = ((x - maximum) - log_total + log_mass).exp().scatter(-1, index, 0.0)
        grad = -other * upstream.to(maximum.dtype).unsqueeze(-1)
        grad = grad.scatter(-1, index, -grad.sum(-1, keepdim=True))
        grad = torch.where((labels != -100).unsqueeze(-1), grad, 0.0)
        return grad.to(logits.dtype), None


def selected_logprobs(logits, labels, *, fused=True, inplace_backward=False):
    shape = logits.shape[:-1]
    if tuple(labels.shape) != tuple(shape) or logits.shape[-1] < 1:
        raise ValueError("Labels must match logits leading dimensions and vocabulary must be nonempty")
    if labels.dtype != torch.long:
        raise ValueError("Labels must be int64")
    if labels.numel() == 0:
        return logits.sum(-1)
    valid = ((labels >= 0) & (labels < logits.shape[-1])) | (labels == -100)
    torch._assert_async(valid.all(), "Label index outside vocabulary")
    x, y = logits.reshape(-1, logits.shape[-1]), labels.reshape(-1).contiguous()
    if fused and x.is_cuda and x.dtype in (torch.float32, torch.float16, torch.bfloat16):
        from verl.utils.stable_logprobs_triton import selected_logprobs_cuda

        result = selected_logprobs_cuda(x.contiguous(), y, inplace_backward=inplace_backward)
    else:
        result = _SelectedLogProb.apply(x, y)
    return result.reshape(shape)
