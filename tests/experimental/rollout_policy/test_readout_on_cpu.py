# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Evaluation readouts cannot enter resume retention or perturb RNG on failure."""

from types import SimpleNamespace

import pytest
import torch

from verl.experimental.rollout_policy.readout import export_scientific_readout


@pytest.mark.parametrize("fail", [False, True])
def test_readout_restores_manager_rng_and_retention(tmp_path, fail):
    contents = ["model", "optimizer", "extra"]
    paths = ["resume20"]
    restored = []
    manager = SimpleNamespace(
        world_size=1,
        per_tensor_param_fn=lambda: None,
        checkpoint_config={},
        checkpoint_save_contents=contents,
        hf_export_dtype=torch.bfloat16,
        previous_global_step=20,
        previous_saved_paths=paths,
        get_rng_state=lambda: "original_rng",
        load_rng_state=restored.append,
    )

    def save_checkpoint(**kwargs):
        assert manager.checkpoint_save_contents == ["hf_model"]
        assert manager.hf_export_dtype == torch.float32
        assert kwargs["max_ckpt_to_keep"] is None
        manager.previous_global_step = 25
        manager.previous_saved_paths.append("readout25")
        if fail:
            raise RuntimeError("injected export failure")

    engine = SimpleNamespace(checkpoint_manager=manager, save_checkpoint=save_checkpoint)
    if fail:
        with pytest.raises(RuntimeError, match="injected"):
            export_scientific_readout(engine, str(tmp_path / "readout25"), 25)
    else:
        export_scientific_readout(engine, str(tmp_path / "readout25"), 25)
    assert manager.checkpoint_save_contents is contents
    assert manager.previous_saved_paths is paths and paths == ["resume20"]
    assert manager.previous_global_step == 20 and manager.hf_export_dtype == torch.bfloat16
    assert restored == ["original_rng"]


def test_readout_refuses_existing_directory(tmp_path):
    with pytest.raises(FileExistsError):
        export_scientific_readout(SimpleNamespace(), str(tmp_path), 0)


def test_readout_evidence_detects_optimizer_or_rng_mutations():
    from verl.experimental.rollout_policy.readout import state_digest

    state = {"step": torch.tensor(1.0), "exp_avg": torch.tensor([1.0, 2.0]), "rng": torch.get_rng_state()}
    before = state_digest(state)
    assert state_digest({key: value.clone() for key, value in state.items()}) == before
    state["exp_avg"][0] += 1
    assert state_digest(state) != before
