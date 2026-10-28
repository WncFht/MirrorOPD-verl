# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
import json

import pytest
from omegaconf import OmegaConf

from verl.experimental.rollout_policy.plan import CoefficientRule
from verl.experimental.rollout_policy.validation import validate_resume_clock


def test_unknown_checkpoint_optimizer_clock_rejected_before_model_setup(tmp_path):
    checkpoint = tmp_path / "global_step_2"
    checkpoint.mkdir()
    config = OmegaConf.create({"trainer": {"resume_mode": "resume_path", "resume_from_path": str(checkpoint)}})
    rule = CoefficientRule(kind="linear", clock="optimizer_step")
    with pytest.raises(ValueError, match="known applied"):
        validate_resume_clock(config, rule)
    (checkpoint / "trainer_state.json").write_text(json.dumps({"actor_optimizer_step": None}))
    with pytest.raises(ValueError, match="known applied"):
        validate_resume_clock(config, rule)
    (checkpoint / "trainer_state.json").write_text(json.dumps({"actor_optimizer_step": 4}))
    validate_resume_clock(config, rule)
