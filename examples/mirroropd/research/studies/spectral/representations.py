# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Same-coordinate representation residuals and exact centered-head bands."""

import json
from pathlib import Path

import numpy as np
import torch

from examples.mirroropd.research.core.artifacts import file_hash
from examples.mirroropd.research.core.io import dump
from examples.mirroropd.research.studies.teacher_difference.runtime import write_tensors

from .decomposition import energy, head_terms, ratio


def analyze(base, teacher, opd, output, *, layer="final", calibration="train", evaluation="heldout", rank=16):
    """Fit hidden residual directions on calibration only; shared ancestry required."""
    if calibration == evaluation:
        raise ValueError("Calibration and evaluation must differ")
    paths = [Path(p) for p in (base, teacher, opd)]
    metadata = [json.loads((p / "metadata.json").read_text()) for p in paths]
    for meta in metadata:
        if not meta.get("complete"):
            raise ValueError("Incomplete extraction cache")
        if meta["positions"] != metadata[0]["positions"] or meta["bank"] != metadata[0]["bank"]:
            raise ValueError("Representation caches must use identical positions and banks")
    filename = "hidden-final.npy" if layer == "final" else f"hidden-layer-{int(layer)}.npy"
    arrays = [np.load(p / filename, mmap_mode="r") for p in paths]
    if any(a.shape != arrays[0].shape for a in arrays):
        raise ValueError("Representation coordinates require equal hidden dimensions")
    positions = metadata[0]["positions"]
    counts = {}
    for p in positions:
        counts[p["row_id"]] = counts.get(p["row_id"], 0) + 1
    weight = torch.tensor([counts[p["row_id"]] ** -0.5 for p in positions], dtype=torch.float64)[:, None]
    b, t, o = [torch.from_numpy(np.array(a)).double() for a in arrays]
    dt, do = (t - b) * weight, (o - b) * weight
    cal = [i for i, p in enumerate(positions) if p["split"] == calibration]
    test = [i for i, p in enumerate(positions) if p["split"] == evaluation]
    if not cal or not test or not 0 < rank <= min(len(cal), dt.shape[1]):
        raise ValueError("Nonempty calibration/evaluation and valid rank are required")
    if {positions[i]["row_id"] for i in cal} & {positions[i]["row_id"] for i in test}:
        raise ValueError("Calibration/evaluation question overlap")
    _, s, vh = torch.linalg.svd(dt[cal], full_matrices=False)
    basis = vh[:rank].T.contiguous()
    rows = []
    for split, indices in ((calibration, cal), (evaluation, test)):
        requested, learned = dt[indices] @ basis, do[indices] @ basis
        rows.append(
            dict(
                split=split,
                energy=float(energy(dt[indices])),
                captured=ratio(energy(requested), energy(dt[indices])),
                absorption=ratio((requested * learned).sum(), energy(requested)),
                missing=ratio(energy(requested - learned), energy(requested)),
            )
        )
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=False)
    write_tensors(destination / "basis.pt", dict(basis=basis, singular_values=s, layer=layer))
    dump(destination / "scores.json", dict(inputs=metadata, layer=layer, shared_coordinates_asserted=True, rows=rows))


def head_band_summary(h0, ht, ho, w0, wt, *, rank=16, frame=None):
    """Exact centered-head SVD; caller chooses feasible matrix device/dtype.

    Strong and weak refer to the ordinary centered head, not Fisher eigenvalues.
    Returns the exact representation/head identity error and residual energy by
    hidden band. No cross-model hidden alignment is inferred.
    """
    centered = w0 - w0.mean(0, keepdim=True)
    if frame is None:
        _, s, vh = torch.linalg.svd(centered, full_matrices=False)
    else:
        s, vh = frame
    if not 0 < rank <= len(s):
        raise ValueError("Invalid head band rank")
    squared_error = 0.0
    for start in range(0, len(h0), 32):
        base, teacher = h0[start : start + 32], ht[start : start + 32]
        terms = head_terms(base, teacher, w0, wt)
        squared_error += float(energy(teacher @ wt.T - base @ w0.T - sum(terms.values())))
    identity_error = squared_error**0.5
    result = dict(identity_error=float(identity_error), singular_values=s.tolist(), bands={})
    for name, basis in (("strong", vh[:rank].T), ("weak", vh[rank:].T)):
        requested, learned = (ht - h0) @ basis, (ho - h0) @ basis
        result["bands"][name] = dict(
            teacher_energy=float(energy(requested)),
            absorption=ratio((requested * learned).sum(), energy(requested)),
            missing=ratio(energy(requested - learned), energy(requested)),
            output_energy=sum(
                float(energy((requested[start : start + 32] @ basis.T) @ centered.T))
                for start in range(0, len(requested), 32)
            ),
        )
    return result


def analyze_head(base, teacher, opd, output, *, base_weights, teacher_weights, rank=16, device="cpu"):
    """Analyze native post-norm caches in the base centered LM-head frame."""
    from .weights import TensorCheckpoint

    paths = [Path(p) for p in (base, teacher, opd)]
    metadata = [json.loads((p / "metadata.json").read_text()) for p in paths]
    if any(
        not m.get("complete") or m["positions"] != metadata[0]["positions"] or m["bank"] != metadata[0]["bank"]
        for m in metadata
    ):
        raise ValueError("Complete, aligned hidden caches are required")
    checkpoints = [TensorCheckpoint(p) for p in (base_weights, teacher_weights)]
    for checkpoint, meta in zip(checkpoints, metadata[:2], strict=True):
        hashes = meta.get("model_sha256")
        matches = (
            all(file_hash(checkpoint.path / name) == digest for name, digest in hashes.items())
            if hashes
            else checkpoint.identity == meta["model"]
        )
        if not matches:
            raise ValueError("Head checkpoint does not match extracted hidden-state content")
    weights = []
    for checkpoint, meta in zip(checkpoints, metadata[:2], strict=True):
        key = "lm_head.weight"
        if key not in checkpoint.files:
            config = json.loads((checkpoint.path / "config.json").read_text())
            if not config.get("tie_word_embeddings"):
                raise ValueError("Missing output head without an audited tied embedding")
            key = "model.embed_tokens.weight"
        weight = checkpoint.get(key)
        if meta.get("precision") == "native" and meta.get("device_type") == "cuda":
            weight = weight.bfloat16()
        weights.append(weight.float().to(device=device, dtype=torch.float64))
    if weights[0].shape != weights[1].shape:
        raise ValueError("Head analysis requires identical vocabulary and hidden coordinates")
    hidden = [np.load(p / "hidden-final.npy", mmap_mode="r") for p in paths]
    if any(h.shape != hidden[0].shape for h in hidden):
        raise ValueError("Head analysis requires equal hidden shapes and shared ancestry")
    centered = weights[0] - weights[0].mean(0, keepdim=True)
    _, singular, vh = torch.linalg.svd(centered, full_matrices=False)
    frame = singular, vh
    rows = []
    for split in sorted({p["split"] for p in metadata[0]["positions"]}):
        indices = [i for i, p in enumerate(metadata[0]["positions"]) if p["split"] == split]
        states = [torch.from_numpy(np.array(h[indices])).to(device=device, dtype=torch.float64) for h in hidden]
        rows.append(dict(split=split, **head_band_summary(*states, *weights, rank=rank, frame=frame)))
    dump(output, dict(inputs=metadata, rank=rank, shared_coordinates_asserted=True, rows=rows))
