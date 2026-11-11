# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Exact, chunked gradient moments; finite-bank and cross-record quantities differ."""

import math

import numpy as np
import torch


def _finite(value):
    return float(value) if math.isfinite(float(value)) else float("nan")


def chunked_gram(vectors, chunk=262144, *, device="cpu", progress=None):
    """One bounded-memory Gram implementation for tensors, arrays or row maps.

    Convert blocks to FP64 on the compute device, rather than materializing
    full FP64 vectors on the host. Accelerator results return to CPU once.
    """
    if chunk < 1:
        raise ValueError("chunk must be positive")
    if isinstance(vectors, torch.Tensor | np.ndarray):
        if vectors.ndim != 2:
            raise ValueError("Expected a matrix of parameter vectors")
        n, d = vectors.shape
        block_at = lambda start: torch.as_tensor(vectors[:, start : start + chunk])
    else:
        n, d = len(vectors), len(vectors[0]) if len(vectors) else 0
        if any(len(row) != d for row in vectors):
            raise ValueError("Parameter vectors differ in length")
        block_at = lambda start: torch.stack([torch.as_tensor(row[start : start + chunk]) for row in vectors])
    gram = torch.zeros((n, n), dtype=torch.float64, device=device)
    for start in range(0, d, chunk):
        # Transfer FP32 first. A combined CPU->CUDA dtype conversion can cast
        # the entire block on the CPU and double PCIe traffic.
        block = block_at(start).to(device=device).to(dtype=torch.float64)
        gram.add_(block @ block.T)
        if progress is not None:
            progress(start, d)
    return gram.cpu()


def tensor_norm_sq(vector, chunk=4194304):
    value = torch.as_tensor(vector).reshape(-1)
    total = torch.zeros((), device=value.device, dtype=torch.float64)
    for block in value.split(chunk):
        total.add_(torch.linalg.vector_norm(block, dtype=torch.float64).square())
    return total


def tensor_inner(left, right, *, chunk=1048576):
    """FP64 dot with bounded temporaries, on the inputs' existing device.

    Parameter reductions must not move full vectors or synchronize per tensor.
    Callers accumulate these scalar tensors and convert to Python once.
    """
    if left.shape != right.shape or left.device != right.device:
        raise ValueError("Inner product requires matching shape and device")
    if chunk < 1:
        raise ValueError("chunk must be positive")
    a, b = left.reshape(-1), right.reshape(-1)
    total = torch.zeros((), device=a.device, dtype=torch.float64)
    for start in range(0, a.numel(), chunk):
        av = a[start : start + chunk].to(torch.float64)
        bv = b[start : start + chunk].to(torch.float64)
        total.add_(torch.dot(av, bv))
    return total


def squared_norm(vector, chunk=4194304):
    return float(tensor_norm_sq(vector, chunk))


def moment_summary(*, pressure, second, mean_norm, samples):
    """P uses fixed-cap token averaging; A/C use trajectory-averaged gradients.

    ``cross_record_inner`` is the U-statistic of distinct-record dot products.
    Its numerator is unbiased under iid sampling. The normalized ratio is not
    itself unbiased; negative estimates must be retained, rather than clipped.
    """
    p, second, mean_norm = float(pressure), float(second), float(mean_norm)
    divide = lambda x, y: x / y if y > 0 else float("nan")
    cross = (samples * mean_norm - second) / (samples - 1) if samples > 1 else float("nan")
    return {
        "P": p,
        "A": divide(second, p),
        "C": divide(mean_norm, second),
        "rho": divide(second - mean_norm, mean_norm),
        "Xi": divide(mean_norm, p),
        "mean_gradient_norm_sq": mean_norm,
        "mean_sample_gradient_norm_sq": second,
        "trace_covariance": second - mean_norm,
        "samples": samples,
        "orthogonal_baseline": 1 / samples,
        "cross_record_inner": cross,
        "cross_record_normalized": divide(cross, second),
        "cross_record_Xi": divide(cross, p),
    }


def pac_summary(*, pressure, gradient_vectors, gradient_norm_sq=None):
    g = torch.as_tensor(gradient_vectors)
    if g.ndim != 2 or g.shape[0] == 0:
        raise ValueError("gradient_vectors must have shape [samples, parameters]")
    gram = chunked_gram(g)
    second = (
        float(gram.diag().mean())
        if gradient_norm_sq is None
        else float(torch.as_tensor(gradient_norm_sq).double().mean())
    )
    n = g.shape[0] if gradient_norm_sq is None else len(gradient_norm_sq)
    return moment_summary(
        pressure=torch.as_tensor(pressure).double().mean(), second=second, mean_norm=gram.mean(), samples=n
    )


def batch_summary(batch_gradients):
    g = torch.as_tensor(batch_gradients)
    if g.ndim != 2 or g.shape[0] < 2:
        raise ValueError("batch_gradients must have shape [batches, parameters], with at least two batches")
    gram = chunked_gram(g)
    second, mean_sq = float(gram.diag().mean()), float(gram.mean())
    shared = mean_sq / second if second > 0 else float("nan")
    off = (gram.sum() - gram.diag().sum()) / (len(g) * (len(g) - 1))
    return {
        "C_batch": shared,
        "rho_batch": (second - mean_sq) / mean_sq if mean_sq > 0 else float("nan"),
        "E_shared": shared,
        "E_residual": 1 - shared,
        "cross_batch_inner": float(off),
        "cross_batch_normalized": float(off) / second if second > 0 else float("nan"),
        "mean_batch_gradient_norm_sq": mean_sq,
        "mean_batch_norm_sq": second,
        "batches": len(g),
    }


def summarize_gram(gram):
    gram = torch.as_tensor(gram, dtype=torch.float64)
    norms = gram.diag().clamp_min(0).sqrt()
    cosine = gram / (norms[:, None] * norms[None, :]).clamp_min(1e-30)
    eigenvalues = torch.linalg.eigvalsh(gram).clamp_min(0)
    total = eigenvalues.sum()
    effective_rank = (total.square() / eigenvalues.square().sum()).item() if total > 0 else float("nan")
    offdiag = cosine[~torch.eye(len(gram), dtype=torch.bool)]
    return {
        "gram": gram.tolist(),
        "cosine": cosine.tolist(),
        "eigenvalues": eigenvalues.tolist(),
        "effective_rank": _finite(effective_rank),
        "mean_offdiagonal_cosine": _finite(offdiag.mean()) if len(offdiag) else float("nan"),
    }


def gram_summary(batch_gradients):
    out = summarize_gram(chunked_gram(batch_gradients))
    out["parameter_count"] = int(batch_gradients.shape[1])
    return out
