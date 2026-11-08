# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Check that a recovery audit detects missing state and separates finite scans."""

import json
import random

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from examples.mirroropd.research.studies.fixed_transfer import teacher_construction_checkpoint_audit as audit


@pytest.fixture
def checkpoint(tmp_path):
    tensors = {"weight": torch.zeros(2, 3), "bias": torch.zeros(2)}
    save_file(tensors, tmp_path / "model.safetensors")
    state = dict(
        step=16,
        optimizer=dict(
            param_groups=[dict(params=[0, 1])],
            state={
                i: dict(step=torch.tensor(16.0), exp_avg=torch.zeros_like(v), exp_avg_sq=torch.zeros_like(v))
                for i, v in enumerate(tensors.values())
            },
        ),
        torch_rng=torch.get_rng_state(),
        cuda_rng=torch.zeros(16, dtype=torch.uint8),
        python_rng=random.getstate(),
        numpy_rng=np.random.get_state(),
        sampler_rng=np.random.default_rng(7).bit_generator.state,
    )
    torch.save(state, tmp_path / "training-state.pt")
    (tmp_path / "checkpoint-complete.json").write_text(json.dumps(dict(step=16, parameter_dtype="float32")))
    return tmp_path, state


def test_complete_audit_checks_restorable_rng_and_keeps_global_state(checkpoint):
    path, _ = checkpoint
    before = torch.get_rng_state().clone()
    result = audit.audit_checkpoint(path, expected_step=16, require_sampler=True, scan=True)
    assert torch.equal(before, torch.get_rng_state())
    assert result["model_parameters"] == 8
    assert result["model_tensors"] == result["optimizer_parameter_states"] == 2
    assert result["sampler_rng_restore_checked"]
    assert result["mode"] == "full-finite-and-hash-scan"
    assert result["files"]["training-state.pt"]["sha256"] == audit.file_hash(path / "training-state.pt")


def test_incomplete_sampler_or_adam_step_is_rejected(checkpoint):
    path, state = checkpoint
    state["sampler_rng"] = None
    torch.save(state, path / "training-state.pt")
    with pytest.raises(ValueError, match="sampling RNG"):
        audit.audit_checkpoint(path, expected_step=16, require_sampler=True)
    state["optimizer"]["state"][0]["step"] = torch.tensor(15.0)
    torch.save(state, path / "training-state.pt")
    with pytest.raises(ValueError, match="update count mismatch"):
        audit.audit_checkpoint(path, expected_step=16)


def test_model_truncation_and_missing_optimizer_coverage_are_rejected(checkpoint):
    path, state = checkpoint
    file = path / "model.safetensors"
    raw = file.read_bytes()
    file.write_bytes(raw[:-1])
    with pytest.raises(ValueError, match="Truncated"):
        audit.audit_checkpoint(path, expected_step=16)
    file.write_bytes(raw)
    state["optimizer"]["state"].pop(1)
    torch.save(state, path / "training-state.pt")
    with pytest.raises(ValueError, match="cover the saved states"):
        audit.audit_checkpoint(path, expected_step=16)


def test_metadata_mode_does_not_claim_finite_tensor_scan(checkpoint):
    path, _ = checkpoint
    save_file({"weight": torch.full((2, 3), float("nan")), "bias": torch.zeros(2)}, path / "model.safetensors")
    result = audit.audit_checkpoint(path, expected_step=16)
    assert result["mode"] == "metadata-and-rng-only"
    assert "sha256" not in result["files"]["model.safetensors"]
    with pytest.raises(ValueError, match="Nonfinite checkpoint tensor: model"):
        audit.audit_checkpoint(path, expected_step=16, scan=True)
