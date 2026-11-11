# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Explicit FP32 AdamW state offload for single-GPU large-model controls."""

import torch


class WeightAdam:
    """Keep AdamW moments and master weights on CPU, copying one update back.

    This changes execution placement, not the weight-space objective. Gradients
    and Adam states remain FP32. Small controls keep the original GPU AdamW.
    """

    def __init__(self, parameters, lr):
        self.entries = [(p, torch.nn.Parameter(p.detach().cpu().clone())) for _, p in parameters]
        self.optimizer = torch.optim.AdamW([value for _, value in self.entries], lr=lr, weight_decay=0)
        self.offload = True

    @torch.no_grad()
    def step(self):
        for parameter, value in self.entries:
            if parameter.grad is None:
                raise ValueError("Selected parameter has no loss gradient")
            value.grad = parameter.grad.detach().cpu()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        for parameter, value in self.entries:
            parameter.copy_(value.to(parameter.device))
