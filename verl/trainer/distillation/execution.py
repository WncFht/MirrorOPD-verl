# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Deterministic backend capabilities; no tensor state, fallback or autotuning."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ExecutionPlan:
    backend: str
    strategy: str
    fused: bool
    future: bool


BACKENDS = {
    "dense": "dense",
    "precompute": "precompute",
    "precompute_fused": "precompute",
    "recompute": "recompute",
    "cached": "cached",
    "streaming": "streaming",
    "liger": "external",
}
OBJECTIVE_BACKENDS = {
    "reverse_kl": frozenset(BACKENDS),
    "forward_kl": frozenset(("dense", "recompute", "precompute", "precompute_fused")),
    "mdpo": frozenset(("dense", "precompute", "precompute_fused")),
    "bpo": frozenset(("dense", "precompute", "precompute_fused")),
}


def resolve_execution(
    backend,
    *,
    objective="reverse_kl",
    future="none",
    distribution="native",
    termination=False,
    aggregate="token-mean",
    reference=False,
):
    """Validate a declaration at config time and again against the actual objective."""
    if backend not in BACKENDS:
        raise ValueError(f"Unknown full-vocab backend: {backend}")
    if objective not in OBJECTIVE_BACKENDS or backend not in OBJECTIVE_BACKENDS[objective]:
        raise ValueError(f"Unsupported {objective} backend: {backend}")
    if future != "none":
        if objective not in ("reverse_kl", "mdpo"):
            raise ValueError("Future credit supports full-vocabulary reverse KL and MDPO only")
        if backend not in ("dense", "precompute", "precompute_fused"):
            raise ValueError("Future credit requires dense/precompute/precompute_fused")
        if distribution != "native":
            raise ValueError("Future credit does not support common-conditioned distributions")
    if distribution != "native" and objective != "reverse_kl":
        raise ValueError("Conditional distributions require teacher-only reverse KL")
    if termination:
        if objective not in ("reverse_kl", "mdpo"):
            raise ValueError(f"{objective} does not support termination equivalence")
        if backend not in ("precompute", "precompute_fused") and not (reference and backend == "dense"):
            raise ValueError("termination equivalence requires precompute/precompute_fused")
    if backend == "liger" and aggregate == "seq-mean-token-mean":
        raise ValueError("liger does not support per-row aggregation; use precompute")
    return ExecutionPlan(backend, BACKENDS[backend], backend == "precompute_fused", future != "none")


def validate_fused_backend(backend, hidden):
    """A fused request must execute CUDA at a supported dtype, including empty ranks."""
    import torch

    if backend == "precompute_fused" and (
        not hidden.is_cuda or hidden.dtype not in (torch.float16, torch.bfloat16, torch.float32)
    ):
        raise ValueError("precompute_fused requires CUDA FP16/BF16/FP32 inputs; select precompute for a reference run")
