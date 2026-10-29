# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Canonical Full-KL and method-overridden run names must survive local analysis."""

import json

import pytest

from examples.mirroropd.research.core import inventory


@pytest.mark.parametrize(
    "mode,objective,estimator,eta,expected",
    [
        ("full_reverse_kl", None, None, None, True),
        ("mirror", "mdpo", "full_vocab", 1, True),
        ("mirror", "mdpo", "full_vocab", 0.3, False),
        ("mirror", "bpo", "full_vocab", 1, False),
        ("mirror", "mdpo", "sampled", 1, False),
        ("k1", None, None, None, False),
    ],
)
def test_full_kl_identity(mode, objective, estimator, eta, expected):
    cfg = {
        "distillation": {
            "distillation_loss": {
                "loss_mode": mode,
                "mirror": {"objective": objective, "estimator": estimator, "eta": eta, "kappa": 0.5},
            }
        }
    }
    assert inventory.is_full_kl(cfg) is expected
    assert (inventory.method_label(cfg) == "Full-KL") is expected


@pytest.mark.parametrize(
    "name,preset,role,variant",
    [
        ("math-opd-E13-20261025-123", "E13", None, None),
        ("math-opd-E13-valid-teacher-20261025-123", "E13", "valid-teacher", None),
        ("math-opd-E13-bpo-full_vocab-k0.5-eta0.3-20261025-123", "E13", None, "bpo-full_vocab-k0.5-eta0.3"),
        ("math-opd-custom-mdpo-sampled-k0.5-eta1-20261025-123", "custom", None, "mdpo-sampled-k0.5-eta1"),
        ("math-opd-E1-full_kl-20261025-123", "E1", None, "full_kl"),
    ],
)
def test_run_names_keep_variants_separate(tmp_path, name, preset, role, variant):
    (tmp_path / "run.json").write_text(
        json.dumps(
            {"run": {"display_name": name, "run_id": "test"}, "status": "finished", "exit_code": 0, "history_rows": 0}
        )
    )
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "distillation": {
                    "distillation_loss": {
                        "loss_mode": "mirror",
                        "mirror": {"objective": "bpo", "estimator": "full_vocab", "eta": 0.3, "kappa": 0.5},
                    }
                }
            }
        )
    )
    (tmp_path / "history.jsonl").write_text("")
    run = inventory.inspect(tmp_path, {})
    assert run["name"] == name
    assert "BPO/full_vocab" in run["method"]
    if variant:
        assert variant in run["name"]
        assert inventory.family(name) == "method-overrides"


@pytest.mark.parametrize(
    "name",
    [
        "math-opd-E44-20261025-123",
        "math-opd-E45-mdpo-full_vocab-k0-eta0.3-20261025-123",
        "math-opd-E46-bpo-full_vocab-k0-eta0.3-20261025-123",
        "math-opd-E47-mdpo-full_vocab-k0.5-eta0.3-20261025-123",
    ],
)
def test_registered_mirror_comparison_family(name):
    assert inventory.family(name) == "mirror-comparison"
