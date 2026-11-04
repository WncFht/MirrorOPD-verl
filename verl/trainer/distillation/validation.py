# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Real tensor contracts checked before any vocabulary projection."""

from verl.trainer.distillation.execution import resolve_execution, validate_fused_backend


def validate_mirror_inputs(hs, ws, ht, wt, *, spec, hq, wq, backend, token_chunk, vocab_chunk, termination_token_ids):
    """Check native MDPO/BPO inputs, including actual teacher and anchor rows."""
    if termination_token_ids:
        ids = tuple(termination_token_ids)
        if len(wt) != len(ws) or len(ids) < 2 or len(set(ids)) != len(ids):
            raise ValueError("Mirror termination equivalence requires equal vocabularies and distinct stop IDs")
        if any(type(i) is not int or i < 0 or i >= len(ws) for i in ids):
            raise ValueError("Mirror termination token IDs must be within the shared vocabulary")
    if spec.payload_kind != "hidden" or spec.objective not in ("mdpo", "bpo"):
        raise ValueError("Full-vocab mirror objective requires MDPO or BPO with hidden payload")
    resolve_execution(backend, objective=spec.objective, termination=bool(termination_token_ids), reference=True)
    if hq is None or wq is None:
        raise ValueError("Mirror full-vocab objective requires frozen anchor hidden/head")
    if any(x.ndim != 2 for x in (hs, ws, ht, wt, hq, wq)):
        raise ValueError("full-vocab expects 2-D hidden states and heads")
    if (
        len(hs) != len(ht)
        or len(hs) != len(hq)
        or len(wq) != len(ws)
        or len(wt) < len(ws)
        or hs.shape[1] != ws.shape[1]
        or ht.shape[1] != wt.shape[1]
        or hq.shape[1] != wq.shape[1]
    ):
        raise ValueError("Mirror hidden/head shapes or teacher/anchor support are incompatible")
    if any(type(c) is not int or c <= 0 for c in (token_chunk, vocab_chunk)) or not len(ws):
        raise ValueError("Positive chunk sizes and nonempty vocabulary required")
    validate_fused_backend(backend, hs)
