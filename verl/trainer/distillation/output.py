# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""One full-vocabulary training result, independent of engine transport keys."""

from typing import NamedTuple

import torch

from verl.trainer.distillation.results import FullVocabObjectiveResult
from verl.utils.metric.observations import Observation
from verl.utils.metric.position import position_observations, position_statistics


class FullVocabOutput(NamedTuple):
    """Expose the numerator to FSDP pytree hooks; keep observations detached."""

    objective_sum: torch.Tensor
    divisor: float
    observations: dict[str, Observation]

    @classmethod
    def from_result(
        cls,
        result: FullVocabObjectiveResult,
        spec,
        count,
        *,
        termination=False,
        bins=None,
        measure_name=None,
        num_sequences=None,
    ):
        """Translate objective semantics once, preserving historical metric meanings."""
        if not isinstance(result, FullVocabObjectiveResult):
            raise TypeError("Full-vocabulary output requires FullVocabObjectiveResult")
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
        local = result.local_objective_sum.detach()
        value = local / divisor
        if termination and spec.teacher_only_kl:
            if result.token_objective is None:
                raise ValueError("Termination diagnostics require native-vocabulary KL rows")
            value = result.token_objective.detach().sum()
        statistics = {primary: value, **{k: v.detach().sum() for k, v in result.token_diagnostics.items()}}
        if measure_name is not None:
            statistics[measure_name + "/objective"] = local / divisor
        if termination and spec.teacher_only_kl:
            if not conditioned:
                statistics["termination_equiv_kl"] = local
            statistics["conditional_eos_identity_kl" if conditioned else "eos_identity_kl"] = value - local
        observations = {f"distillation/{k}": Observation.ratio(v, count) for k, v in statistics.items()}
        teacher_rkl = result.token_diagnostics.get("teacher_rkl")
        if teacher_rkl is None:
            if spec.teacher_only_kl:
                teacher_rkl = result.token_objective
            elif spec.teacher_only_forward_kl:
                teacher_rkl = result.token_diagnostics.get("full_kl")
        if teacher_rkl is not None:
            observations["distillation/teacher_rkl"] = Observation.mean(teacher_rkl)
            if num_sequences is not None:
                observations["distillation/teacher_rkl_per_sequence_sum"] = Observation.ratio(
                    teacher_rkl.sum(), num_sequences
                )
            if bins is not None:
                sums, counts = position_statistics(teacher_rkl, bins)
                observations.update(position_observations(sums, counts, prefix="distillation/teacher_rkl_position"))
        coefficients = result.future_coefficients
        if coefficients is not None:
            score = result.future_score_sum.detach()
            observations.update(
                {
                    "distillation/future/coefficient": Observation.mean(coefficients),
                    "distillation/future/coefficient_std": Observation.std(coefficients),
                    "distillation/future/coefficient_p95": Observation.quantile(coefficients, 0.95),
                    "distillation/future/coefficient_max": Observation.maximum(coefficients),
                    "distillation/future/score_surrogate": Observation.ratio(score / divisor, count),
                    "distillation/training_surrogate": Observation.ratio(result.loss_sum.detach() / divisor, count),
                    "distillation/local_surrogate": Observation.ratio(
                        (result.loss_sum.detach() - score) / divisor, count
                    ),
                }
            )
            quantities = {"coefficient": coefficients}
            if result.future_mass is not None:
                quantities["effective_mass"] = result.future_mass
                quantities["suffix_mean_cost"] = result.future_mean_cost
                observations["distillation/future/suffix_valid_tokens"] = Observation.total(result.future_mass > 0)
            for name, values in quantities.items():
                valid = (
                    result.future_mass > 0 if name == "suffix_mean_cost" else torch.ones_like(values, dtype=torch.bool)
                )
                observations[f"distillation/future/{name}"] = Observation.mean(values[valid])
                if bins is not None:
                    sums, counts = position_statistics(values[valid], bins[valid])
                    observations.update(
                        position_observations(sums, counts, prefix=f"distillation/future/{name}_position")
                    )
        observations["distillation/valid_tokens"] = Observation.total(count)
        if num_sequences is not None:
            observations["distillation/valid_sequences"] = Observation.total(num_sequences)
        if bins is not None and result.token_objective is not None:
            sums, counts = position_statistics(result.token_objective, bins)
            observations.update(
                position_observations(sums / divisor, counts, prefix=f"distillation/{primary}_position")
            )
            if spec.teacher_only_forward_kl:
                reverse_sums, _ = position_statistics(result.token_diagnostics["full_kl"], bins)
                observations.update(position_observations(reverse_sums, counts, prefix="distillation/full_kl_position"))
        return cls(result.loss_sum, divisor, observations)
