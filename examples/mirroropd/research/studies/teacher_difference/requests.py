# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Trajectory requests through the production bounded-memory KL/VJP program."""

from functools import partial

import torch

from verl.trainer.distillation.rows import precompute_kl_rows
from verl.trainer.distillation.vjp import precompute_rows


def _prepare(hs, ws, ht, wt, h0, w0, *, frozen):
    live_loss, live = precompute_kl_rows(hs, ws, ht, wt)
    if frozen:
        initial_loss, initial = precompute_kl_rows(h0, w0, ht, wt)
        # All reductions concern the unweighted logit requests. The production
        # executor applies 1/cap once to the selected VJP, including tied heads.
        info = torch.stack(
            (live.square().sum(-1), initial.square().sum(-1), (live * initial).sum(-1), initial_loss), -1
        )
        return initial_loss, initial, live_loss, info, None
    info = live.square().sum(-1, keepdim=True)
    return live_loss, live, live_loss, info, None


def trajectory_projection(hs, ws, ht, wt, *, initial=None, cap=1024, token_chunk=128):
    """Return the production scalar VJP and detached per-position statistics.

    With ``initial=(h0,w0)``, the derivative is J_current.T B_initial.
    Its displayed scalar is the initial KL, NOT a current KL objective.
    Output 1 always contains the current KL; output 3 contains request moments.
    No token-by-full-vocabulary tensor survives the current token block.
    """
    if cap <= 0 or token_chunk <= 0 or len(hs) > cap:
        raise ValueError("positive chunks/cap and at most cap positions are required")
    return precompute_rows(
        hs,
        ws,
        ht,
        wt,
        token_chunk,
        partial(_prepare, frozen=initial is not None),
        hq=None if initial is None else initial[0],
        wq=None if initial is None else initial[1],
        row_weights=hs.new_full((len(hs),), 1 / cap),
        diagnostic_count=1 if initial is None else 4,
    )
