# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Validation config, native engine stopping semantics, and offline replay parity."""

from types import SimpleNamespace

import numpy as np
import pytest
from omegaconf import OmegaConf

from examples.mirroropd.research.studies.termination.replay_repetition import first_repetitions
from verl.workers.config.rollout import RolloutConfig, SamplingConfig, validation_repetition_detection


@pytest.mark.parametrize(
    "value",
    [
        {},
        True,
        {"min_pattern_size": 1, "max_pattern_size": 32, "min_count": 1},
        {"min_pattern_size": 33, "max_pattern_size": 32, "min_count": 8},
        {"min_pattern_size": 1, "max_pattern_size": 32, "min_count": True},
        {"min_pattern_size": 1, "max_pattern_size": 32, "min_count": 8, "extra": 0},
    ],
)
def test_invalid_config_rejected(value):
    with pytest.raises(ValueError, match="repetition_detection"):
        SamplingConfig(repetition_detection=value)


def test_backend_and_disable_contract():
    params = {"min_pattern_size": 16, "max_pattern_size": 256, "min_count": 8}
    rollout = OmegaConf.create({"name": "vllm", "val_kwargs": {"repetition_detection": params}})
    assert validation_repetition_detection(rollout) == params
    with pytest.raises(ValueError, match="requires rollout.name=vllm"):
        RolloutConfig(name="sglang", val_kwargs=SamplingConfig(repetition_detection=params))
    rollout.val_kwargs.repetition_detection = None
    rollout.name = "sglang"
    assert validation_repetition_detection(rollout) is None


def test_vectorized_replay_matches_native_detector_at_every_prefix():
    from vllm.sampling_params import RepetitionDetectionParams
    from vllm.v1.core.sched.utils import check_sequence_repetition

    rng = np.random.default_rng(17)
    profiles = {"a": (1, 8, 3), "b": (4, 16, 4), "c": (16, 32, 8)}
    cases = [rng.integers(0, 12, 160).tolist() for _ in range(15)]
    cases += [[999] + list(range(p)) * c + [998] for p in (1, 3, 8, 16, 24, 33) for c in (3, 4, 8)]
    cases += [[], [1], [1, 2] * 3, [1, 2] * 3 + [9]]
    for ids in cases:
        replay = first_repetitions(ids, profiles)
        for name, (low, high, count) in profiles.items():
            params = RepetitionDetectionParams(min_pattern_size=low, max_pattern_size=high, min_count=count)
            expected = next((n for n in range(1, len(ids)) if check_sequence_repetition(ids[:n], params)), len(ids))
            assert replay[name][0] == expected


def test_native_repetition_stop_and_eos_cap_precedence():
    from vllm.sampling_params import RepetitionDetectionParams, SamplingParams
    from vllm.v1.core.sched.utils import check_stop
    from vllm.v1.request import RequestStatus

    def request(ids, max_tokens=100, stop_ids=None):
        return SimpleNamespace(
            pooling_params=None,
            sampling_params=SamplingParams(
                stop_token_ids=stop_ids,
                repetition_detection=RepetitionDetectionParams(min_pattern_size=2, max_pattern_size=4, min_count=4),
            ),
            num_output_tokens=len(ids),
            num_tokens=len(ids) + 3,
            output_token_ids=ids,
            max_tokens=max_tokens,
            status=None,
            stop_reason=None,
        )

    req = request([1, 2] * 4)
    assert check_stop(req, 200) and req.status == RequestStatus.FINISHED_REPETITION
    assert req.stop_reason == "repetition_detected"
    req = request([1, 2] * 4, stop_ids=[2])
    assert check_stop(req, 200) and req.status == RequestStatus.FINISHED_STOPPED
    req = request([1, 2] * 4, max_tokens=8)
    assert check_stop(req, 200) and req.status == RequestStatus.FINISHED_LENGTH_CAPPED
    # A changed token breaks exact adjacency, despite a high global n-gram repeat score.
    assert not check_stop(request([1, 2, 3, 1, 2, 4, 1, 2, 5, 1, 2, 6]), 200)
