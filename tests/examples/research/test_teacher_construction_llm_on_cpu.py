# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Audit the signed, sequence-weighted CE VJP used by SFT and RLOO."""

import copy

import pytest
import torch

from examples.mirroropd.research.backends import hf_models
from examples.mirroropd.research.studies.fixed_transfer.teacher_construction_llm import ce_backward


@pytest.mark.parametrize("tied", [False, True])
def test_signed_ce_matches_complete_autograd(monkeypatch, tied):
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(23, 7)
            self.transform = torch.nn.Linear(7, 7, bias=False)
            self.head = torch.nn.Linear(7, 23, bias=False)
            if tied:
                self.head.weight = self.embed.weight

        def get_output_embeddings(self):
            return self.head

    def hidden(model, batch, *, pad_id):
        assert len(batch) == 1
        value = batch[0]
        ids = value["prompt_ids"] + value["response_ids"][:-1]
        states = model.transform(model.embed(torch.tensor(ids)).cumsum(0)).tanh()
        return states[len(value["prompt_ids"]) - 1 :]

    torch.manual_seed(41)
    model = Tiny()
    reference = copy.deepcopy(model)
    batch = [dict(prompt_ids=[2, 3], response_ids=[5, 7, 11]), dict(prompt_ids=[4], response_ids=([3, 5, 9] * 24))]
    coefficients = [0.6, -1.2]
    monkeypatch.setattr(hf_models, "hidden", hidden)
    actual = ce_backward(model, batch, coefficients, denominator=31)
    expected = 0
    for value, coefficient in zip(batch, coefficients, strict=True):
        logits = reference.head(hidden(reference, [value], pad_id=0))
        expected += (
            coefficient
            * torch.nn.functional.cross_entropy(logits, torch.tensor(value["response_ids"]), reduction="sum")
            / 31
        )
    expected.backward()
    assert actual == pytest.approx(float(expected.detach()), rel=1e-6, abs=1e-6)
    for parameter, other in zip(model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(parameter.grad, other.grad, rtol=2e-5, atol=2e-6)
