# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Fixed-window risk preserves native teacher support and does not pad EOS losses."""

from types import SimpleNamespace

import pytest
import torch

from examples.mirroropd.research.backends import hf_rkl


def test_score_windows_and_native_extra_teacher_mass(monkeypatch):
    hidden = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    monkeypatch.setattr(hf_rkl, "hidden", lambda model, rows, pad_id: hidden)
    model = SimpleNamespace(get_output_embeddings=lambda: SimpleNamespace(weight=torch.eye(2)))
    teacher_head = torch.zeros(3, 2)
    row = dict(id=1, group="M01", domain="math", split="heldout", source="S0", response_ids=[0, 1, 0])
    result = hf_rkl.score(
        model,
        teacher_head,
        {1: torch.zeros(3, 2)},
        [row],
        80,
        SimpleNamespace(horizon=5),
        SimpleNamespace(microbatch=1, token_chunk=2),
        0,
        bands=[("all", 0, 5), ("early", 0, 1), ("late", 1, 5), ("after_eos", 3, 5)],
    )[0]
    logp = hidden.double().log_softmax(-1)
    oracle = (logp.exp() * (logp + torch.tensor(3.0).double().log())).sum().item()
    stats = result["stats"]
    assert stats["all"]["rkl"] == pytest.approx(oracle, abs=2e-6)
    assert stats["early"]["rkl"] + stats["late"]["rkl"] == pytest.approx(stats["all"]["rkl"], abs=2e-6)
    assert stats["all"]["tokens"] == 3 and stats["after_eos"]["tokens"] == 0
    assert stats["after_eos"]["rkl"] == 0
    assert result["step"] == 80 and result["source"] == "S0"


@pytest.mark.parametrize("bands", [[("bad", 0, 6)], [("bad", 2, 2)], [("same", 0, 1), ("same", 1, 5)]])
def test_invalid_score_windows_fail_before_model_execution(bands):
    with pytest.raises(ValueError, match="windows"):
        hf_rkl.score(None, None, {}, [], 0, SimpleNamespace(horizon=5), None, 0, bands=bands)
