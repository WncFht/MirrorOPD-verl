# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""A resumed stochastic Adam trajectory must exactly reproduce uninterrupted updates."""

import copy
import random

import numpy as np
import pytest
import torch

from examples.mirroropd.research.core import training_state


def test_stage_resume_preserves_adam_and_all_cpu_rngs(tmp_path):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Dropout(0.25), torch.nn.Linear(4, 1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

    def step(m, opt):
        x = torch.randn(8, 4) + random.random() + np.random.rand()
        opt.zero_grad(set_to_none=True)
        m(x).square().mean().backward()
        opt.step()

    for _ in range(4):
        step(model, optimizer)
    weights = copy.deepcopy(model.state_dict())
    training_state.save(
        tmp_path,
        model,
        optimizer,
        step=4,
        signature={"teacher": "a", "pool": 20},
        weights="fixed-checkpoint",
        weight_identity="weight-hash",
    )
    for _ in range(4):
        step(model, optimizer)
    resumed = copy.deepcopy(model)
    resumed.load_state_dict(weights)
    opt = torch.optim.AdamW(resumed.parameters(), lr=0.01)
    assert (
        training_state.restore(
            tmp_path, resumed, opt, signature={"teacher": "a", "pool": 20}, weight_identity="weight-hash"
        )
        == 4
    )
    for _ in range(4):
        step(resumed, opt)
    for a, b in zip(model.parameters(), resumed.parameters(), strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    with pytest.raises(ValueError, match="identity mismatch"):
        training_state.restore(
            tmp_path, resumed, opt, signature={"teacher": "b", "pool": 20}, weight_identity="weight-hash"
        )

    later = tmp_path / "later"
    training_state.save(
        later,
        resumed,
        opt,
        step=8,
        signature={"teacher": "a", "pool": 20},
        weights="later-weights",
        weight_identity="later-hash",
    )
    training_state.retire(tmp_path, replacement=later)
    assert not (tmp_path / "optimizer.pt").exists()
    assert (later / "optimizer.pt").exists()
    with pytest.raises(ValueError, match="superseded"):
        training_state.restore(
            tmp_path, resumed, opt, signature={"teacher": "a", "pool": 20}, weight_identity="weight-hash"
        )


def test_fixed_pool_schedule_and_risk_only_readouts():
    from dataclasses import replace

    from examples.mirroropd.research.core.paths import MIRROROPD
    from examples.mirroropd.research.studies.fixed_transfer.spec import Arm, Study

    base = Study.load(MIRROROPD / "research/studies/fixed_transfer/plans/qwen3-lineage.json")
    study = replace(base, continuation={"pool_batches": 20, "save_resume": True, "accuracy_gate": False})
    assert [study.training_group(s) for s in [1, 16, 17, 320, 321, 640, 641, 1280]] == [
        "A",
        "A",
        "B",
        "T",
        "A",
        "T",
        "A",
        "T",
    ]
    arm = Arm("risk-only", "e3", updates=640, evaluate_at=())
    assert study.risk_readouts(arm) == (64, 320, 640)
    with pytest.raises(ValueError, match="continuation"):
        base.training_group(321)


def test_cross_size_numerical_reference_uses_matching_layout(monkeypatch):
    from pathlib import Path
    from types import SimpleNamespace

    from transformers import AutoConfig

    from examples.mirroropd.research.studies.fixed_transfer.training import validation_teacher

    common = dict(
        model_type="qwen3",
        vocab_size=8,
        hidden_size=4,
        intermediate_size=8,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=4,
        tie_word_embeddings=True,
    )
    configs = {"student": common, "reference": common, "small": dict(common, hidden_size=2)}
    monkeypatch.setattr(
        AutoConfig, "from_pretrained", lambda path, **kwargs: SimpleNamespace(**configs[Path(path).name])
    )
    models = {name: SimpleNamespace(directory=Path(name), method="sft") for name in configs}
    study = SimpleNamespace(
        student="student", models=models, arms=[SimpleNamespace(teacher="small")], continuation=None
    )
    assert validation_teacher(study) == "student"
    study.continuation = {"validation_teacher": "reference"}
    assert validation_teacher(study) == "reference"
    study.continuation = {"validation_teacher": "small"}
    with pytest.raises(ValueError, match="parameter layout"):
        validation_teacher(study)
