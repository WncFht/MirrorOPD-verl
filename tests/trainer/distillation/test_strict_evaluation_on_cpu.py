# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Strict construction and content identity fail before creating evaluation artifacts."""

import shutil

import pytest
from omegaconf import OmegaConf

from verl.trainer.evaluation.config import SourceHistoryConfig
from verl.trainer.evaluation.identity import model_identity
from verl.trainer.ppo.readout_callback import build_readout_callback


@pytest.mark.parametrize(
    "field",
    [
        "readout_callback_class",
        "source_history_readout",
        "rkl_readout",
        "hf_reload_readout",
        "rkl_probe",
        "scientific_evaluation",
    ],
)
def test_retired_entry_points_are_rejected_even_when_empty(field):
    with pytest.raises(ValueError, match="Retired"):
        build_readout_callback(OmegaConf.create({"trainer": {field: None}}))


def test_explicit_target_is_required_and_disabled_is_explicit():
    assert build_readout_callback(OmegaConf.create({"trainer": {"evaluation": None}})) is None
    with pytest.raises(ValueError, match="_target_"):
        build_readout_callback(OmegaConf.create({"trainer": {"evaluation": {}}}))


@pytest.mark.parametrize(
    "extra",
    [
        {"audit_state": True},
        {"every": 1},
        {"typo": 1},
        {"initial_bank": ""},
        {"output": True},
        {"source_count": True},
        {"score": {"audit_state": "false"}},
    ],
)
def test_source_configuration_has_one_strict_representation(extra):
    with pytest.raises(ValueError):
        SourceHistoryConfig.parse({"output": "unused", "source_file": "source", **extra})


def test_model_identity_binds_weights_and_nested_templates_but_not_location(tmp_path):
    first = tmp_path / "first"
    first.mkdir()
    (first / "config.json").write_text("{}")
    (first / "model.safetensors").write_bytes(b"original")
    (first / "chat_templates").mkdir()
    (first / "chat_templates/default.jinja").write_text("one")
    original = model_identity(first)
    second = tmp_path / "second"
    shutil.copytree(first, second)
    assert model_identity(second) == original
    (second / "model.safetensors").write_bytes(b"modified")
    assert model_identity(second) != original
    (first / "chat_templates/default.jinja").write_text("two")
    assert model_identity(first) != original
    with pytest.raises(ValueError, match="Incomplete"):
        model_identity(tmp_path)
