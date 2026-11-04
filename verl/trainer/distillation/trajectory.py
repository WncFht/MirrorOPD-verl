# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Sampled action score in native coordinates; execution belongs to the common VJP loop."""

import torch

from verl.trainer.distillation.numerics import log_softmax


def action_score(logits, sampled, actions):
    """Sampled action log-probability and score VJP, without a dense one-hot."""
    lp = log_softmax(logits)
    value = lp.gather(1, sampled[:, None]).squeeze(1)
    gradient = -lp.exp()
    if actions is None:
        gradient.scatter_add_(1, sampled[:, None], torch.ones_like(value[:, None]))
    else:
        # Differentiate log P(stop class) directly in native coordinates. Only
        # the tiny stop group uses FP64, not a second full-vocabulary workspace.
        is_stop = torch.isin(sampled, actions.stop)
        stop_lp = lp.index_select(1, actions.stop).double()
        value = torch.where(is_stop, stop_lp.logsumexp(-1).to(lp.dtype), value)
        gradient.scatter_add_(1, sampled[:, None], (~is_stop[:, None]).to(gradient.dtype))
        conditional = stop_lp.softmax(-1).to(gradient.dtype) * is_stop[:, None]
        gradient.index_add_(1, actions.stop, conditional)
    # Preserve a tiny 1-p at a near-certain sampled action, as the local VJP does.
    pivot = logits.argmax(-1, keepdim=True)
    other = gradient.scatter(-1, pivot, 0.0)
    gradient = other.scatter(-1, pivot, -other.sum(-1, keepdim=True))
    return value, gradient
