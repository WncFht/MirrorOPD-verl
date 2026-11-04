# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Research control for target rounding; production consumes separate native logits."""

import torch

from verl.trainer.distillation.full_vocab_triton import difference, fused_bpo_rows, fused_rows, weighted_rows


def encoded_target_rows(z, b, log_q, log_teacher, dtype, *, objective, diagnostics):
    """Measure already-rounded target weights; lost increments cannot be recovered."""
    if objective == "bpo":
        value, gradient = fused_bpo_rows(z, b, log_q, dtype)
    else:
        value, gradient = fused_rows(z, b, dtype)
        value = value - b.logsumexp(-1)
    info = z.new_empty((len(z), 0))
    if diagnostics:
        target, _, _ = weighted_rows(z, difference(z, b), dtype, gradient=False)
        anchor, _, _ = weighted_rows(z, difference(z, log_q), dtype, gradient=False)
        teacher, _, _ = weighted_rows(z, difference(z, log_teacher), dtype, gradient=False)
        info = torch.stack([target, b.logsumexp(-1), anchor, teacher - log_teacher.logsumexp(-1)], -1)
    return value, gradient, info
