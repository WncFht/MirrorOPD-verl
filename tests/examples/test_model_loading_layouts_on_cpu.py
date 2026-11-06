# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Public launchers select installed model layouts before tokenizer initialization."""

import subprocess
import sys

import pytest
from test_mirroropd_math_launcher_on_cpu import SCRIPT, _config

from examples.mirroropd.assets.paths import registered_model


@pytest.mark.parametrize("layout", ["flat", "revision", "both"])
@pytest.mark.parametrize("selector", ["E54", "E54-valid-student", "E54-valid-teacher", "custom"])
def test_public_training_and_validation_select_existing_layout_before_loading(tmp_path, layout, selector):
    root = tmp_path / "models with spaces"
    names = (
        ("DeepSeek-R1-Distill-Qwen-1.5B", "JustRL-DeepSeek-1.5B")
        if selector == "custom"
        else ("Qwen3-1.7B-Base", "Qwen3-30B-A3B-Instruct-2507")
    )
    for name in names:
        if layout in {"flat", "both"}:
            (root / name).mkdir(parents=True)
        if layout in {"revision", "both"}:
            registered_model(name, root=root).mkdir(parents=True)
    cfg = _config(tmp_path, command=["bash", str(SCRIPT), selector])
    selected = [root / name if layout == "flat" else registered_model(name, root=root) for name in names]
    generator = selected[1] if selector.endswith("valid-teacher") else selected[0]
    assert cfg.actor_rollout_ref.model.path == str(generator)
    if "-valid-" not in selector:
        assert cfg.distillation.teacher_models.teacher_model.model_path == str(selected[1])
        assert cfg.rollout_model_resolver == "examples.mirroropd.assets.paths.rollout_model_path"
    else:
        assert not cfg.distillation.enabled
    assert not (tmp_path / "run").exists()


def test_explicit_model_overrides_keep_priority_over_installed_layouts(tmp_path):
    root = tmp_path / "models with spaces"
    for name in ("Qwen3-1.7B-Base", "Qwen3-30B-A3B-Instruct-2507"):
        (root / name).mkdir(parents=True)
    cfg = _config(
        tmp_path,
        command=["bash", str(SCRIPT), "E54", "actor_rollout_ref.model.path=/hydra/student"],
        STUDENT_MODEL="/explicit/student",
        TEACHER_MODEL="/explicit/teacher",
    )
    assert cfg.actor_rollout_ref.model.path == "/hydra/student"
    assert cfg.distillation.teacher_models.teacher_model.model_path == "/explicit/teacher"


def test_path_cli_separates_local_loading_from_canonical_storage(tmp_path):
    name = "Qwen3-1.7B-Base"
    flat = tmp_path / name
    flat.mkdir()
    command = [sys.executable, str(SCRIPT.parent / "assets/paths.py"), "--alias", name, "--model-root", str(tmp_path)]
    local = subprocess.run(command, capture_output=True, text=True, check=True)
    canonical = subprocess.run([*command, "--canonical"], capture_output=True, text=True, check=True)
    assert local.stdout.strip() == str(flat)
    assert "has not been verified" in local.stderr
    assert canonical.stdout.strip() == str(registered_model(name, root=tmp_path))
    assert not canonical.stderr
