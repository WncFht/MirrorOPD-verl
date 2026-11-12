# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Intermediate exports must retain prior weights and publish only verified models."""

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from safetensors.torch import save_file

from verl.trainer.evaluation import checkpoint
from verl.trainer.evaluation.artifacts import ArtifactStore
from verl.trainer.evaluation.config import EvaluationConfig
from verl.trainer.evaluation.reporting import evaluation_metrics


def trainer(tmp_path):
    calls = []

    def save(path, remote, outer, max_ckpt_to_keep):
        from pathlib import Path

        root = Path(path)
        (root / "huggingface").mkdir(parents=True)
        save_file({"weight": torch.ones(2)}, root / "huggingface/model.safetensors")
        torch.save({}, root / "model_world_size_1_rank_0.pt")
        calls.append((outer, max_ckpt_to_keep))

    return SimpleNamespace(
        global_steps=1,
        actor_optimizer_step=16,
        config=OmegaConf.create({"actor_rollout_ref": {"actor": {"checkpoint": {"preserve_hf_master_weights": True}}}}),
        actor_rollout_wg=SimpleNamespace(save_checkpoint=save, load_checkpoint=lambda *args, **kwargs: None),
        calls=calls,
    )


def test_retains_two_verified_fp32_snapshots(tmp_path, monkeypatch):
    bank = ArtifactStore.create(tmp_path / "bank", kind="test", metadata={})
    bank.finish()
    reference = {"id": "r", "valid_tokens": 2, "bank": "public_initial", "artifact": str(bank.root)}
    monkeypatch.setattr(
        checkpoint, "score_native_rows", lambda *args, **kwargs: ([{"id": "r", "tokens": 2, "rkl_sum": 1}], {})
    )
    model = trainer(tmp_path)
    for outer in (1, 2):
        model.global_steps = outer
        checkpoint.final_model_export(
            model,
            [reference],
            hidden_size=2,
            budget=None,
            destination=tmp_path / f"outer-{outer}",
            max_ckpt_to_keep=None,
        )
    assert model.calls == [(1, None), (2, None)]
    for outer in (1, 2):
        root = tmp_path / f"outer-{outer}"
        assert (root / "huggingface/model.safetensors").exists()
        assert not list(root.glob("model_world*"))
        assert ArtifactStore(root / "export-audit").manifest(complete=True)["metadata"]["native_reload_rkl"] == "passed"


def test_failed_reload_does_not_publish_snapshot(tmp_path, monkeypatch):
    reference = {"id": "r", "valid_tokens": 2}
    outputs = iter([([{"id": "r", "tokens": 2, "rkl_sum": 1}], {}), ([{"id": "r", "tokens": 2, "rkl_sum": 2}], {})])
    monkeypatch.setattr(checkpoint, "score_native_rows", lambda *args, **kwargs: next(outputs))
    with pytest.raises(RuntimeError, match="changed fixed-input"):
        checkpoint.final_model_export(
            trainer(tmp_path),
            [reference],
            hidden_size=2,
            budget=None,
            destination=tmp_path / "snapshot",
            max_ckpt_to_keep=None,
        )
    assert not (tmp_path / "snapshot").exists()


def test_three_domain_summary_uses_question_weights():
    rows = [
        dict(id=str(i), bank="public_initial", domain=d, tokens=2, rkl_sum=v)
        for i, (d, v) in enumerate([("math", 2), ("math", 2), ("code", 6), ("if", 10)])
    ]
    metrics = evaluation_metrics(rows, baselines={}, public_equal_domains=False)
    assert metrics["eval/rkl/public_initial/all/question_mean"] == 2.5
    assert metrics["eval/rkl/public_initial/all/fixed_h1024"] == 20 / 4096


def test_rejects_unpaired_banks_and_unverified_snapshots():
    base = {"output": "out", "bootstrap": "boot", "datasets": [], "protocols": {}}
    for extra in ({"fixed_banks": {"public_initial": "a"}}, {"export_outers": [1], "verify_export": False}):
        with pytest.raises(ValueError):
            EvaluationConfig.parse(base | extra)
