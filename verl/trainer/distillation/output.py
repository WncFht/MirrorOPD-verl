# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""One full-vocabulary training result, independent of engine transport keys."""

from typing import NamedTuple

import torch

from verl.utils.metric.observations import Observation
from verl.utils.metric.position import position_observations, position_statistics


class FullVocabOutput(NamedTuple):
    """Expose the numerator to FSDP pytree hooks; keep observations detached."""

    objective_sum: torch.Tensor
    divisor: float
    observations: dict[str, Observation]

    @classmethod
    def from_result(cls, result, spec, count, *, termination=False, bins=None, measure_name=None):
        """Translate objective semantics once, preserving historical metric meanings."""
        endpoint = spec.teacher_only_kl or spec.teacher_only_forward_kl
        divisor = 1.0 if endpoint else spec.eta
        primary = (
            "full_forward_kl"
            if spec.teacher_only_forward_kl
            else "full_kl"
            if spec.teacher_only_kl
            else f"{spec.objective}/objective"
        )
        conditioned = measure_name is not None and measure_name.startswith("conditional")
        if spec.teacher_only_kl and conditioned:
            primary = "conditional_rkl"
        value = result.loss_sum.detach() / divisor
        if termination and spec.teacher_only_kl:
            if result.token_objective is None:
                raise ValueError("Termination diagnostics require native-vocabulary KL rows")
            value = result.token_objective.detach().sum()
        statistics = {primary: value, **{k: v.detach().sum() for k, v in result.token_diagnostics.items()}}
        if measure_name is not None:
            statistics[measure_name + "/objective"] = result.loss_sum.detach() / divisor
        if termination and spec.teacher_only_kl:
            if not conditioned:
                statistics["termination_equiv_kl"] = result.loss_sum.detach()
            statistics["conditional_eos_identity_kl" if conditioned else "eos_identity_kl"] = (
                value - result.loss_sum.detach()
            )
        observations = {f"distillation/{k}": Observation.ratio(v, count) for k, v in statistics.items()}
        if bins is not None and result.token_objective is not None:
            sums, counts = position_statistics(result.token_objective, bins)
            observations.update(
                position_observations(sums / divisor, counts, prefix=f"distillation/{primary}_position")
            )
            if spec.teacher_only_forward_kl:
                reverse_sums, _ = position_statistics(result.token_diagnostics["full_kl"], bins)
                observations.update(position_observations(reverse_sums, counts, prefix="distillation/full_kl_position"))
        return cls(result.loss_sum, divisor, observations)
