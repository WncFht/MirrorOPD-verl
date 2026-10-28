# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from verl.experimental.rollout_policy.sampling import (
    SamplingConfig,
    fusion_log_probs,
    sample_tokens,
    stop_event_probabilities,
    validate_tokenizers,
)


@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("temperature", [0.7, 1.0, 2.0])
def test_fp64_oracle(alpha, temperature):
    s = torch.tensor([[0.8, 0.2]], dtype=torch.float64).log()
    t = torch.tensor([[0.4, 0.6]], dtype=torch.float64).log()
    oracle = ((1 - alpha) * s + alpha * t).div(temperature).log_softmax(-1)
    actual = fusion_log_probs(s, t, alpha, SamplingConfig(temperature=temperature))
    torch.testing.assert_close(actual.double(), oracle, atol=2e-7, rtol=2e-7)
    if alpha == 0.5 and temperature == 1:
        assert actual.exp()[0, 0].item() == pytest.approx(0.6202041)


def test_eos_disagreement():
    s = torch.tensor([[0.9, 0.001, 0.099]]).log()
    t = torch.tensor([[0.001, 0.9, 0.099]]).log()
    fused = fusion_log_probs(s, t, 0.5, SamplingConfig()).exp()
    assert fused[0, :2].sum().item() == pytest.approx(0.37735849)
    diagnostics = stop_event_probabilities({"student": s, "teacher": t}, 0.5, (0, 1), (0, 1))
    assert diagnostics["fusion/total_stop_mass"].item() == pytest.approx(0.37735849)
    assert diagnostics["native_event/stop_mass"].item() == pytest.approx(0.9 / 0.999)
    assert diagnostics["union_event/stop_mass"].item() == pytest.approx(0.901)


@pytest.mark.parametrize("alpha,name", [(0.0, "student"), (1.0, "teacher")])
def test_event_diagnostics_do_not_require_inactive_component(alpha, name):
    values = torch.tensor([[0.9, 0.001, 0.099]]).log()
    result = stop_event_probabilities({name: values}, alpha, (0, 1), (0, 1))
    assert result["native_event/stop_mass"].item() == pytest.approx(0.9 if alpha == 0 else 0.001)
    assert result["union_event/stop_mass"].item() == pytest.approx(0.901)


def test_truncation_and_selected_logprob():
    s = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
    config = SamplingConfig(temperature=2.0, top_k=3, top_p=0.7)
    actual = fusion_log_probs(s, s, 0.5, config)
    oracle = torch.tensor([[2.0, 1.5, -torch.inf, -torch.inf]], dtype=torch.float64).log_softmax(-1)
    torch.testing.assert_close(actual.double(), oracle, atol=2e-7, rtol=2e-7)
    tokens, lp = sample_tokens(actual, [torch.Generator().manual_seed(1)])
    assert lp.item() == actual[0, tokens.item()].item()
    assert actual.exp().sum().item() == pytest.approx(1.0)


def test_endpoints_do_not_multiply_negative_infinity():
    bad = torch.full((1, 2), -torch.inf)
    good = torch.tensor([[0.0, -torch.inf]])
    assert fusion_log_probs(good, bad, 0, SamplingConfig())[0, 0] == 0
    assert fusion_log_probs(bad, good, 1, SamplingConfig())[0, 0] == 0
    with pytest.raises(ValueError, match="support"):
        fusion_log_probs(good, bad, 0.5, SamplingConfig())


def test_equal_head_size_does_not_skip_mapping_check():
    class Tokenizer:
        def __init__(self, mapping):
            self.mapping = mapping

        def get_vocab(self):
            return self.mapping

    with pytest.raises(ValueError, match="mappings"):
        validate_tokenizers(Tokenizer({"a": 0, "b": 1}), Tokenizer({"a": 1, "b": 0}), 2, 2)
