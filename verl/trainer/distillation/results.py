# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Tensor result contracts with no dependency on execution or engine state."""

from typing import NamedTuple

import torch


class FullKLResult(NamedTuple):
    """Differentiable objective and detached raw vocabulary-KL diagnostics.

    With termination equivalence, loss_sum optimizes the coarsened action space;
    token_kl deliberately retains the original vocabulary metric.
    """

    loss_sum: torch.Tensor
    token_kl: torch.Tensor | None
    objective_rows: torch.Tensor | None = None


class FullVocabObjectiveResult(NamedTuple):
    """Unscaled objective sum and detached rows; final loss applies 1/eta once.

    local_objective_sum is always the unweighted, undiscounted local numerator,
    including scalar-only backends and empty ranks. It never contains score credit.
    For teacher-only KL, token_objective retains raw vocabulary KL even when
    loss_sum uses termination equivalence, so position metrics stay comparable.
    """

    loss_sum: torch.Tensor
    token_objective: torch.Tensor | None
    token_diagnostics: dict[str, torch.Tensor]
    objective_rows: torch.Tensor | None
    local_objective_sum: torch.Tensor
    future_coefficients: torch.Tensor | None = None
    future_score_sum: torch.Tensor | None = None
