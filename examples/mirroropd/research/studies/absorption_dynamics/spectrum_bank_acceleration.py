# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Guarded exact spectrum restoration and bounded, device-resident reference reuse."""

from collections import OrderedDict

import torch

from verl.trainer.evaluation.artifacts import identity


@torch.inference_mode()
def gram_restore(original, initial_singular, *, fallback):
    """Use the smaller FP64 Gram matrix, falling back when conditioning is unsafe.

    For tall W, W + W V diag(s0/s-1) V.T equals U diag(s0) V.T.
    For wide W, apply the analogous correction on the left. No rank truncation.
    """
    w = original.double()
    left = w.shape[0] < w.shape[1]
    gram = w @ w.T if left else w.T @ w
    eigenvalues, vectors = torch.linalg.eigh(gram)
    largest, smallest = float(eigenvalues[-1]), float(eigenvalues[0])
    # Squaring the condition number loses accuracy for near-null directions.
    if not 0 < smallest <= largest or largest / smallest > 1e8:
        edited, receipt = fallback(original, initial_singular)
        return edited, dict(receipt, method="svd-fallback", gram_condition=largest / smallest if smallest > 0 else None)
    residual = float((gram @ vectors - vectors * eigenvalues).norm() / gram.norm())
    if not residual < 1e-10:
        edited, receipt = fallback(original, initial_singular)
        return edited, dict(receipt, method="svd-fallback", gram_residual=residual)
    singular = eigenvalues.sqrt().flip(0)
    if singular.shape != initial_singular.shape:
        raise ValueError("Base/target singular value dimensions differ")
    vectors = vectors.flip(1)
    factors = initial_singular / singular - 1
    if left:
        correction = (vectors * factors) @ (vectors.T @ w)
    else:
        correction = (w @ vectors * factors) @ vectors.T
    edited = (w + correction).float()
    return edited, dict(
        method="gram-fp64",
        gram_condition=largest / smallest,
        gram_residual=residual,
        correction_norm=float(correction.norm()),
        original_norm=float(w.norm()),
        edited_norm=float(edited.double().norm()),
    )


class ReferenceCache:
    """Keep the head and a bounded LRU of validated hidden states on one device."""

    def __init__(self, directory, device, *, max_bytes):
        self.directory, self.device, self.max_bytes = directory, device, max_bytes
        self.head = torch.load(directory / "head.pt", weights_only=True, map_location=device)
        self.entries = OrderedDict()
        self.bytes = self.hits = self.misses = 0

    def get(self, row):
        key = identity(row)
        if key in self.entries:
            self.hits += 1
            self.entries.move_to_end(key)
            return self.entries[key]
        self.misses += 1
        saved = torch.load(self.directory / f"{identity(row['id'])}.pt", weights_only=True, map_location="cpu")
        if saved["row_identity"] != key:
            raise ValueError("Teacher cache row changed")
        hidden = saved["hidden"].to(self.device)
        size = hidden.numel() * hidden.element_size()
        while self.entries and self.bytes + size > self.max_bytes:
            _, removed = self.entries.popitem(last=False)
            self.bytes -= removed.numel() * removed.element_size()
        if size <= self.max_bytes:
            self.entries[key] = hidden
            self.bytes += size
        return hidden

    def stats(self):
        return dict(hits=self.hits, misses=self.misses, resident_bytes=self.bytes, limit_bytes=self.max_bytes)
