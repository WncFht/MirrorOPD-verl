# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Collection one uses actor zero; frozen components never claim actor freshness."""

import pytest

from verl.workers.rollout.vllm_fusion.metadata import output_metadata


@pytest.mark.parametrize("version", [(0, 0), (1, 2), (9, 15)])
def test_current_reports_loaded_actor_coordinate(version):
    metadata = output_metadata("request", "phase", version, 0.5, "teacher")
    assert metadata["global_steps"] == version[0]
    assert metadata["actor_version"] == version
    assert metadata["student_version"] == version


@pytest.mark.parametrize("version,alpha", [(("checkpoint", "initial"), 0.5), ((2, 4), 1.0)])
def test_initial_and_teacher_endpoint_do_not_claim_current(version, alpha):
    metadata = output_metadata("request", "phase", version, alpha, "teacher")
    assert metadata["global_steps"] is None
    assert metadata["actor_version"] is None
    assert metadata["student_version"] == (version if alpha < 1 else None)
