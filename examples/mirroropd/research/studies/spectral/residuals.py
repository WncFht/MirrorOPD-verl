# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Blockwise Fisher-weighted output residuals, calibration SVD and heldout scores."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from examples.mirroropd.research.core.artifacts import file_hash
from examples.mirroropd.research.core.io import dump
from examples.mirroropd.research.studies.teacher_difference.runtime import write_tensors

from .decomposition import energy, ratio


def fisher_residual(log_base, log_other):
    """Row-centered log ratio in the base categorical Fisher metric."""
    probability = log_base.softmax(-1)
    difference = log_other - log_base
    centered = difference - (probability * difference).sum(-1, keepdim=True)
    return probability.sqrt() * centered


def residual_distribution(log_base, residual):
    """Invert full Fisher whitening up to a row constant on strictly positive support."""
    probability = log_base.softmax(-1)
    if not (probability > 0).all():
        raise ValueError("Inverse whitening requires strictly positive base probabilities")
    return (log_base + residual / probability.sqrt()).log_softmax(-1)


class ResidualBank:
    """Aligned, full-vocabulary FP32 log-probability memmaps; CPU blocks only.

    Equal token-ID semantics are required. Extra teacher rows retain their native
    partition function in RKL; Fisher bands use the student support. Different
    hidden widths are allowed. Incompatible token IDs are rejected.
    """

    def __init__(self, base, teacher, opd, *, metric="fisher"):
        if metric not in ("fisher", "logit", "probability"):
            raise ValueError("Unknown residual metric")
        self.metric = metric
        paths = [Path(p) for p in (base, teacher, opd)]
        self.metadata = [json.loads((p / "metadata.json").read_text()) for p in paths]
        for meta in self.metadata:
            if not meta.get("complete"):
                raise ValueError("Incomplete log-probability cache; extract to a new directory")
        for meta in self.metadata[1:]:
            for key in ("positions", "bank"):
                if meta[key] != self.metadata[0][key]:
                    raise ValueError(f"Residual cache {key} mismatch; identical states are required")
        self.positions = self.metadata[0]["positions"]
        self.shape = tuple(self.metadata[0]["shape"])
        if (
            self.metadata[2]["shape"] != list(self.shape)
            or self.metadata[2]["vocabulary"] != self.metadata[0]["vocabulary"]
        ):
            raise ValueError("Base and OPD vocabulary/shape must match")
        if self.metadata[1]["shape"][0] != self.shape[0] or self.metadata[1]["shape"][1] < self.shape[1]:
            raise ValueError("Teacher must cover student native support")
        if self.metadata[1]["vocabulary"] != self.metadata[0]["vocabulary"]:
            files = [p / "vocabulary.json" for p in paths[:2]]
            if not all(p.exists() for p in files):
                raise ValueError("Teacher vocabulary mismatch without auditable token maps")
            student, teacher_tokens = [json.loads(p.read_text()) for p in files]
            if any(teacher_tokens.get(token) != index for token, index in student.items()):
                raise ValueError("Teacher vocabulary mismatch: unaligned token IDs")
        self.arrays = [np.load(p / "logprobs.npy", mmap_mode="r") for p in paths]
        if any(
            list(x.shape) != meta["shape"] or x.dtype != np.float32
            for x, meta in zip(self.arrays, self.metadata, strict=True)
        ):
            raise ValueError("Cache shape/dtype does not match metadata")
        for path, meta in zip(paths, self.metadata, strict=True):
            expected = meta.get("sha256", {}).get("logprobs.npy")
            if expected is not None and file_hash(path / "logprobs.npy") != expected:
                raise ValueError(f"Log-probability cache content hash mismatch: {path}")
        counts = {}
        for row in self.positions:
            counts[row["row_id"]] = counts.get(row["row_id"], 0) + 1
        # Equal sampled mass per question, independent of response length.
        self.weights = torch.tensor([1 / counts[r["row_id"]] for r in self.positions], dtype=torch.float64)

    def indices(self, split):
        selected = [i for i, row in enumerate(self.positions) if row["split"] == split]
        if not selected:
            raise ValueError(f"No positions with split={split!r}")
        return selected

    def blocks(self, indices, block=32, *, teacher_only=False):
        if block < 1:
            raise ValueError("block must be positive")
        for start in range(0, len(indices), block):
            ix = indices[start : start + block]
            logs = [
                torch.from_numpy(np.array(x[ix], copy=True)).double().log_softmax(-1)
                for x in (self.arrays[:2] if teacher_only else self.arrays)
            ]
            b, t = logs[:2]
            o = None if teacher_only else logs[2]
            t = t[:, : self.shape[1]]  # Keep native teacher normalization, including extra rows.
            if any(not torch.isfinite(x).all() for x in logs):
                raise ValueError("Nonfinite log probabilities in cache")
            weights = self.weights[ix].sqrt()[:, None]

            def residual(other, b=b):
                if self.metric == "fisher":
                    return fisher_residual(b, other)
                if self.metric == "probability":
                    return other.exp() - b.exp()
                difference = other - b
                return difference - difference.mean(-1, keepdim=True)

            yield ix, residual(t) * weights, None if teacher_only else residual(o) * weights, (b, t, o)


def fit_right_basis(bank, indices, *, rank=16, oversample=8, iterations=1, seed=0, block=32):
    """Randomized SVD using O((positions+vocab)*(rank+oversample)) memory.

    No power is applied to heldout rows. Error is measured against the exact
    calibration Frobenius norm; returned values are truncated, not a rank claim.
    """
    n, v = len(indices), bank.shape[1]
    if not 0 < rank <= min(n, v) or oversample < 0 or iterations < 0:
        raise ValueError("Invalid randomized SVD rank/oversample/iterations")
    width = min(rank + oversample, n, v)
    generator = torch.Generator().manual_seed(seed)
    omega = torch.randn(v, width, generator=generator, dtype=torch.float64)

    def forward(matrix):
        return torch.cat([rt @ matrix for _, rt, _, _ in bank.blocks(indices, block, teacher_only=True)])

    def transpose(matrix):
        result = torch.zeros(v, matrix.shape[1], dtype=torch.float64)
        offset = 0
        for _, rt, _, _ in bank.blocks(indices, block, teacher_only=True):
            result += rt.T @ matrix[offset : offset + len(rt)]
            offset += len(rt)
        return result

    q = torch.linalg.qr(forward(omega), mode="reduced").Q
    for _ in range(iterations):
        z = torch.linalg.qr(transpose(q), mode="reduced").Q
        q = torch.linalg.qr(forward(z), mode="reduced").Q
    compressed = transpose(q).T
    _, singular, vh = torch.linalg.svd(compressed, full_matrices=False)
    basis = vh[:rank].T.contiguous()
    total, captured = 0.0, 0.0
    for _, rt, _, _ in bank.blocks(indices, block, teacher_only=True):
        total += float(energy(rt))
        captured += float(energy(rt @ basis))
    return basis, dict(
        singular_values=singular[:rank].tolist(),
        calibration_energy=total,
        relative_frobenius_error=(max(0.0, 1 - captured / total) ** 0.5 if total > 0 else None),
        rank=rank,
        oversample=oversample,
        iterations=iterations,
        seed=seed,
    )


def absorption(requested_energy, learned_energy, inner):
    return dict(
        teacher_energy=requested_energy,
        learned_energy=learned_energy,
        absorption=ratio(inner, requested_energy),
        missing=ratio(max(0.0, requested_energy + learned_energy - 2 * inner), requested_energy),
    )


def score_bands(bank, indices, basis, *, edges=(1, 4, 16), block=32):
    """Question-level band scores including unresolved complement and native RKL."""
    if basis.ndim != 2 or basis.shape[0] != bank.shape[1]:
        raise ValueError("Basis vocabulary dimension mismatch")
    if not torch.allclose(basis.T @ basis, torch.eye(basis.shape[1], dtype=basis.dtype), atol=1e-7, rtol=1e-7):
        raise ValueError("Basis must have orthonormal columns")
    bounds = sorted(set([0, basis.shape[1], *[k for k in edges if 0 < k < basis.shape[1]]]))
    accum = {}
    for ix, rt, ro, logs in bank.blocks(indices, block):
        ct, co = rt @ basis, ro @ basis
        b, t, o = logs
        native_base = (b.exp() * (b - t)).sum(-1)
        native_opd = (o.exp() * (o - t)).sum(-1)
        for j, position in enumerate(ix):
            row = bank.positions[position]
            result = accum.setdefault(
                row["row_id"],
                dict(row_id=row["row_id"], split=row["split"], native_base_rkl=0.0, native_opd_rkl=0.0, bands={}),
            )
            weight = float(bank.weights[position])
            result["native_base_rkl"] += float(native_base[j]) * weight
            result["native_opd_rkl"] += float(native_opd[j]) * weight
            totals = (float(energy(rt[j])), float(energy(ro[j])), float((rt[j] * ro[j]).sum()))
            projected = (float(energy(ct[j])), float(energy(co[j])), float((ct[j] * co[j]).sum()))
            values = {
                "all": totals,
                "complement": (
                    max(0.0, totals[0] - projected[0]),
                    max(0.0, totals[1] - projected[1]),
                    totals[2] - projected[2],
                ),
            }
            for lo, hi in zip(bounds[:-1], bounds[1:], strict=True):
                values[f"{lo}:{hi}"] = (
                    float(energy(ct[j, lo:hi])),
                    float(energy(co[j, lo:hi])),
                    float((ct[j, lo:hi] * co[j, lo:hi]).sum()),
                )
            for name, triple in values.items():
                total = result["bands"].setdefault(name, [0.0, 0.0, 0.0])
                for k, value in enumerate(triple):
                    total[k] += value
    for row in accum.values():
        row["bands"] = {name: absorption(*values) for name, values in row["bands"].items()}
    return list(accum.values())


def analyze(
    base,
    teacher,
    opd,
    output,
    *,
    calibration="train",
    evaluation="heldout",
    rank=16,
    oversample=8,
    iterations=1,
    seed=0,
    block=32,
    metric="fisher",
):
    if calibration == evaluation:
        raise ValueError("Calibration and evaluation splits must differ")
    bank = ResidualBank(base, teacher, opd, metric=metric)
    cal, test = bank.indices(calibration), bank.indices(evaluation)
    if {bank.positions[i]["row_id"] for i in cal} & {bank.positions[i]["row_id"] for i in test}:
        raise ValueError("A question cannot appear in calibration and evaluation")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    basis, diagnostics = fit_right_basis(
        bank, cal, rank=rank, oversample=oversample, iterations=iterations, seed=seed, block=block
    )
    identity = hashlib.sha256(
        json.dumps(dict(inputs=bank.metadata, metric=metric), sort_keys=True).encode()
    ).hexdigest()
    generator = torch.Generator().manual_seed(seed + 1)
    random_basis = torch.linalg.qr(torch.randn(basis.shape, dtype=basis.dtype, generator=generator), mode="reduced").Q
    write_tensors(
        output / "basis.pt",
        dict(basis=basis, random_basis=random_basis, calibration_indices=cal, cache_identity=identity),
    )
    dump(
        output / "scores.json",
        dict(
            cache_identity=identity,
            metric=metric,
            inputs=bank.metadata,
            diagnostics=diagnostics,
            random_evaluation=score_bands(bank, test, random_basis, block=block),
            calibration=score_bands(bank, cal, basis, block=block),
            evaluation=score_bands(bank, test, basis, block=block),
        ),
    )
