# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic Muon+AdamW with exact full-matrix orthogonalization under FSDP2.

Uses the installed PyTorch Muon Newton-Schulz implementation unchanged. This is
an experiment adapter, not a production optimizer registration. Muon applies to
hidden 2D matrices; embeddings, output heads, biases and norms use AdamW.
"""

import torch
from torch.distributed.tensor import DTensor, Replicate


class FullMatrixMuonAdamW(torch.optim.Optimizer):
    def __init__(self, named_parameters, lr, aux_lr=None, momentum=0.95, weight_decay=0.0):
        from torch.optim._muon import DEFAULT_A, DEFAULT_B, DEFAULT_C, _adjust_lr, _zeropower_via_newtonschulz

        self.ns_coefficients = (DEFAULT_A, DEFAULT_B, DEFAULT_C)
        self.orthogonalize = _zeropower_via_newtonschulz
        self.adjust_lr = _adjust_lr
        named = list(named_parameters)
        matrices = [p for n, p in named if p.ndim == 2 and ".layers." in n]
        auxiliary = [p for n, p in named if not (p.ndim == 2 and ".layers." in n)]
        groups = [
            {"params": matrices, "kind": "muon", "lr": lr, "momentum": momentum},
            {"params": auxiliary, "kind": "adamw", "lr": lr if aux_lr is None else aux_lr},
        ]
        super().__init__(groups, {"weight_decay": weight_decay})
        self.auxiliary = torch.optim.AdamW(
            [self.param_groups[1]], betas=(0.9, 0.999), eps=1e-8, weight_decay=weight_decay, foreach=False
        )
        self.auxiliary.state = self.state
        self.partition = {
            "muon_tensors": len(matrices),
            "aux_tensors": len(auxiliary),
            "muon_parameters": sum(p.numel() for p in matrices),
            "aux_parameters": sum(p.numel() for p in auxiliary),
        }

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError("This experiment adapter does not use optimizer closures")
        group = self.param_groups[0]
        for p in group["params"]:
            if p.grad is None:
                continue
            state = self.state[p]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(p.grad)
            buffer = state["momentum_buffer"]
            buffer.lerp_(p.grad, 1 - group["momentum"])
            direction = p.grad.lerp(buffer, group["momentum"])
            full = direction.full_tensor() if isinstance(direction, DTensor) else direction
            orthogonal = self.orthogonalize(full, self.ns_coefficients, 5, 1e-7)
            if isinstance(p, DTensor):
                orthogonal = DTensor.from_local(
                    orthogonal, p.device_mesh, [Replicate()] * p.device_mesh.ndim, run_check=False
                ).redistribute(placements=p.placements)
            adjusted_lr = self.adjust_lr(group["lr"], "match_rms_adamw", p.shape)
            p.mul_(1 - group["lr"] * group["weight_decay"])
            p.add_(orthogonal, alpha=-adjusted_lr)
        self.auxiliary.step()

    def load_state_dict(self, state_dict):
        result = super().load_state_dict(state_dict)
        self.auxiliary.param_groups = [self.param_groups[1]]
        self.auxiliary.state = self.state
        return result


def constant_scheduler(optimizer):
    """Respect a diagnostic LR override even after restoring checkpoint groups."""
    for group in optimizer.param_groups:
        group["initial_lr"] = group["lr"]
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
