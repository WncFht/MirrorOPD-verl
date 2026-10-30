# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Readouts must preserve RNG and the exact generated answer across grading retries."""

import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from examples.mirroropd.research.studies.prefix_fusion.callback import preserve_driver_rng
from examples.mirroropd.research.studies.prefix_fusion.tasks import grade_pending, validate_raw
from verl.trainer.ppo.readout_callback import build_readout_callback


def test_readout_is_disabled_by_default():
    assert build_readout_callback(OmegaConf.create({"trainer": {}})) is None


def test_failed_readout_preserves_driver_rng():
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    expected = random.random(), np.random.rand(), torch.rand(1)
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    with pytest.raises(RuntimeError), preserve_driver_rng():
        random.random()
        np.random.rand()
        torch.rand(1)
        raise RuntimeError("scorer failed")
    assert random.random() == expected[0]
    assert np.random.rand() == expected[1]
    assert torch.equal(torch.rand(1), expected[2])


def sample():
    prompt = dict(prompt_id="q", seed_index=0, data_source="math", input="question", gts="2")
    row = dict(
        prompt_id="q",
        sample_index=0,
        data_source="math",
        input="question",
        gts="2",
        output="2",
        response_ids=[2],
        response_length=1,
        seed=42,
        finish_reason="stop",
    )
    return [prompt], row, SimpleNamespace(samples=1, seed=42, response_cap=64)


def test_raw_identity_rejects_duplicates_and_seed_changes():
    prompts, row, profile = sample()
    assert validate_raw([row], prompts, profile)[0] == {("q", 0)}
    with pytest.raises(ValueError, match="Duplicate"):
        validate_raw([row, row], prompts, profile)
    with pytest.raises(ValueError, match="identity"):
        validate_raw([row | {"seed": 43}], prompts, profile)


def test_offline_grade_recovery_checks_raw_text(tmp_path):
    prompts, row, profile = sample()
    (tmp_path / "raw-responses.jsonl").write_text(json.dumps(row) + "\n")
    graded = row | {"status": "ok", "score": 1}
    (tmp_path / "records.jsonl").write_text(json.dumps(graded) + "\n")
    assert grade_pending(tmp_path, prompts, profile) == [graded]
    (tmp_path / "records.jsonl").write_text(json.dumps(graded | {"output": "replacement"}) + "\n")
    with pytest.raises(ValueError, match="changed the saved response"):
        grade_pending(tmp_path, prompts, profile)
