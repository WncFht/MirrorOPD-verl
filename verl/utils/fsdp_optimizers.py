# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Muon+AdamW with full-matrix Newton-Schulz under FSDP2.

Uses the installed PyTorch Muon Newton-Schulz implementation unchanged. This is
a correctness-first FSDP2 implementation. Muon applies to
hidden 2D matrices; embeddings, output heads, biases and norms use AdamW.
"""

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Replicate, Shard


def muon_parameter_groups(model):
    """Select hidden matrices structurally; tied embedding/head weights stay in AdamW.

    Custom architectures can instead supply explicit ``use_muon`` parameter groups
    directly to :class:`MuonAdamW`.
    """
    excluded = {
        id(p) for module in model.modules() if isinstance(module, torch.nn.Embedding) for p in module.parameters()
    }
    for accessor in ("get_input_embeddings", "get_output_embeddings"):
        getter = getattr(model, accessor, None)
        if getter is not None:
            module = getter()
            if module is not None:
                excluded.update(id(p) for p in module.parameters())
    matrices, auxiliary = [], []
    for p in model.parameters():
        if p.requires_grad:
            (matrices if p.ndim == 2 and id(p) not in excluded else auxiliary).append(p)
    return [
        {"params": params, "use_muon": use_muon}
        for params, use_muon in ((matrices, True), (auxiliary, False))
        if params
    ]


class MuonAdamW(torch.optim.Optimizer):
    """Native-PyTorch Muon for explicit matrix groups, AdamW for auxiliary groups.

    Every group must specify ``use_muon``. All ranks must use identical group order
    and gradient presence, as required by the FSDP2 collective update path.
    """

    def __init__(
        self,
        params,
        lr,
        aux_lr=None,
        momentum=0.95,
        weight_decay=0.0,
        betas=(0.9, 0.999),
        eps=1e-8,
        nesterov=True,
        ns_steps=5,
        ns_eps=1e-7,
        adjust_lr_fn="match_rms_adamw",
        distributed_mode="replicated",
    ):
        if not hasattr(torch.optim, "Muon"):
            raise RuntimeError("MuonAdamW requires PyTorch with native torch.optim.Muon")
        from torch.optim._muon import DEFAULT_A, DEFAULT_B, DEFAULT_C, _adjust_lr, _zeropower_via_newtonschulz

        self.ns_coefficients = (DEFAULT_A, DEFAULT_B, DEFAULT_C)
        self.orthogonalize = _zeropower_via_newtonschulz
        self.adjust_lr = _adjust_lr
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
            nesterov=nesterov,
            ns_steps=ns_steps,
            ns_eps=ns_eps,
            adjust_lr_fn=adjust_lr_fn,
            distributed_mode=distributed_mode,
        )
        groups = list(params)
        for group in groups:
            if not isinstance(group, dict) or "use_muon" not in group:
                raise ValueError("MuonAdamW requires parameter groups with explicit use_muon")
            if not group["use_muon"] and aux_lr is not None:
                group.setdefault("lr", aux_lr)
        super().__init__(groups, defaults)
        self._bind_auxiliary()

    def add_param_group(self, param_group):
        group = dict(self.defaults, **param_group)
        if not isinstance(group.get("use_muon"), bool):
            raise ValueError("Each MuonAdamW parameter group must specify boolean use_muon")
        group["params"] = list(group["params"])
        if group["use_muon"]:
            # Validate against the same native implementation as the numerical oracle.
            torch.optim.Muon(
                [torch.nn.Parameter(torch.zeros(1, 1))],
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                momentum=group["momentum"],
                nesterov=group["nesterov"],
                ns_steps=group["ns_steps"],
                eps=group["ns_eps"],
                adjust_lr_fn=group["adjust_lr_fn"],
            )
            if group["distributed_mode"] not in ("replicated", "owner"):
                raise ValueError("Muon distributed_mode must be replicated or owner")
            for p in group["params"]:
                if p.ndim != 2:
                    raise ValueError("Muon parameters must be 2D matrices")
                if (
                    group["distributed_mode"] == "owner"
                    and isinstance(p, DTensor)
                    and (p.device_mesh.ndim != 1 or p.placements != (Shard(0),))
                ):
                    raise ValueError("Owner Muon requires a one-dimensional FSDP2 mesh with row shards")
        else:
            torch.optim.AdamW([dict(group)], foreach=False)
        super().add_param_group(group)
        if hasattr(self, "auxiliary"):
            self._bind_auxiliary()

    def _bind_auxiliary(self):
        groups = [g for g in self.param_groups if not g["use_muon"]]
        self.auxiliary = torch.optim.AdamW(groups, foreach=False) if groups else None
        if self.auxiliary is not None:
            self.auxiliary.state = self.state

    @property
    def partition(self):
        result = {}
        for label, use_muon in (("muon", True), ("aux", False)):
            params = [p for g in self.param_groups if g["use_muon"] == use_muon for p in g["params"]]
            result[label + "_tensors"] = len(params)
            result[label + "_parameters"] = sum(p.numel() for p in params)
        return result

    def _owner_orthogonalize(self, direction, group, index):
        """Gather to one rotating owner, run NS once, scatter BF16 update shards.

        Momentum stays sharded. Other ranks allocate only padded local shards;
        full matrices and Newton-Schulz intermediates exist only on the owner.
        Padding supports uneven and empty FSDP row shards.
        """
        mesh = direction.device_mesh
        process_group = mesh.get_group()
        world = mesh.size()
        owner = dist.get_global_rank(process_group, index % world)
        is_owner = dist.get_rank() == owner
        local = direction.to_local()
        rows = (direction.shape[0] + world - 1) // world
        # Native NS starts by casting to BF16: cast before communication to halve
        # gather traffic without changing the full-matrix numerical calculation.
        padded = torch.zeros((rows, direction.shape[1]), dtype=torch.bfloat16, device=local.device)
        padded[: local.shape[0]].copy_(local)
        full_buffer = padded.new_empty((rows * world, direction.shape[1])) if is_owner else None
        gathered = list(full_buffer.chunk(world, dim=0)) if is_owner else None
        dist.gather(padded, gather_list=gathered, dst=owner, group=process_group)
        chunks = None
        if is_owner:
            full = full_buffer[: direction.shape[0]]
            orthogonal = self.orthogonalize(full, self.ns_coefficients, group["ns_steps"], group["ns_eps"])
            if direction.shape[0] == rows * world:
                output = orthogonal.contiguous()
            else:
                output = orthogonal.new_zeros((rows * world, direction.shape[1]))
                output[: direction.shape[0]].copy_(orthogonal)
            chunks = list(output.chunk(world, dim=0))
        # Native Newton-Schulz produces BF16 updates on CPU and CUDA.
        update = torch.empty(padded.shape, dtype=torch.bfloat16, device=local.device)
        dist.scatter(update, scatter_list=chunks, src=owner, group=process_group)
        return DTensor.from_local(
            update[: local.shape[0]].contiguous(),
            mesh,
            direction.placements,
            shape=direction.shape,
            stride=direction.stride(),
            run_check=False,
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        index = -1
        for group in self.param_groups:
            if not group["use_muon"]:
                continue
            for p in group["params"]:
                index += 1
                if p.grad is None:
                    continue
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p.grad)
                buffer = state["momentum_buffer"]
                buffer.lerp_(p.grad, 1 - group["momentum"])
                direction = p.grad.lerp(buffer, group["momentum"]) if group["nesterov"] else buffer
                if isinstance(direction, DTensor) and group["distributed_mode"] == "owner":
                    orthogonal = self._owner_orthogonalize(direction, group, index)
                else:
                    full = direction.full_tensor() if isinstance(direction, DTensor) else direction
                    orthogonal = self.orthogonalize(full, self.ns_coefficients, group["ns_steps"], group["ns_eps"])
                    if isinstance(p, DTensor):
                        orthogonal = DTensor.from_local(
                            orthogonal, p.device_mesh, [Replicate()] * p.device_mesh.ndim, run_check=False
                        ).redistribute(placements=p.placements)
                adjusted_lr = self.adjust_lr(group["lr"], group["adjust_lr_fn"], p.shape)
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(orthogonal, alpha=-adjusted_lr)
        if self.auxiliary is not None:
            self.auxiliary.step()
        return loss

    def load_state_dict(self, state_dict):
        result = super().load_state_dict(state_dict)
        self._bind_auxiliary()
        return result
