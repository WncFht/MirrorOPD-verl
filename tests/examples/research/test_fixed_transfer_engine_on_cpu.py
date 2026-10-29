# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Full-gradient equivalence includes shared embedding/head parameters and extra teacher support."""

import pytest
import torch

from examples.mirroropd.research.backends import hf_rkl as engine
from examples.mirroropd.research.core.metrics import fixed_metrics, native_metrics
from examples.mirroropd.research.studies.fixed_transfer.spec import Execution, Training


@pytest.mark.parametrize("tied", [False, True])
def test_vjp_matches_whole_graph_for_tied_and_untied_models(monkeypatch, tied):
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(23, 7)
            self.transform = torch.nn.Linear(7, 7, bias=False)
            self.lm_head = torch.nn.Linear(7, 23, bias=False)
            if tied:
                self.lm_head.weight = self.embedding.weight

        def get_output_embeddings(self):
            return self.lm_head

    def states(model, rows, *, pad_id):
        result = []
        for row in rows:
            ids = row["prompt_ids"] + row["response_ids"][:-1]
            embeddings = model.embedding(torch.tensor(ids))
            h = model.transform(embeddings.cumsum(0)).tanh()
            result.append(h[len(row["prompt_ids"]) - 1 :])
        return torch.cat(result)

    torch.manual_seed(42)
    model = Tiny()
    rows = [dict(id=0, prompt_ids=[1, 2], response_ids=[3, 4, 5]), dict(id=1, prompt_ids=[2], response_ids=[6, 7])]
    q = {r["id"]: torch.randn(len(r["response_ids"]), 31).log_softmax(-1) for r in rows}
    profile = Training(batch_size=2, horizon=3)
    p = model.lm_head(states(model, rows, pad_id=0)).log_softmax(-1)
    target = torch.cat([q[r["id"]] for r in rows])[:, :23]
    reference = (p.exp() * (p - target)).sum() / profile.denominator
    reference.backward()
    gradients = {name: parameter.grad.clone() for name, parameter in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    monkeypatch.setattr(engine, "hidden", states)
    actual = engine.backward(model, q, rows, profile, Execution(microbatch=2, token_chunk=2), 0)
    assert abs(actual - float(reference.detach())) < 1e-6
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter.grad, gradients[name], atol=1e-6, rtol=1e-5)


def test_native_metrics_match_historical_full_vocabulary_metrics():
    torch.manual_seed(7)
    p, q = torch.randn(11, 23), torch.randn(11, 31)
    old, new = fixed_metrics(p, q), native_metrics(p, q)
    for key in ["rkl", "js", "tv", "hp", "hq"]:
        torch.testing.assert_close(new[key], old[key], rtol=0, atol=0)
    torch.testing.assert_close(new["extra_teacher_mass"], q.softmax(-1)[:, 23:].sum(-1))
    with pytest.raises(ValueError):
        engine.head_vjp(torch.randn(2, 7), torch.randn(23, 7), torch.randn(2, 22), 6)


def test_hf_weight_export_preserves_tied_fp32_model(tmp_path):
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM, Qwen3Config, Qwen3ForCausalLM

    config = Qwen3Config(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        tie_word_embeddings=True,
    )
    model = Qwen3ForCausalLM(config).float().eval()
    tokens = torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        before = model(tokens).logits
    model.save_pretrained(tmp_path, safe_serialization=True)
    with safe_open(tmp_path / "model.safetensors", framework="pt") as state:
        assert "lm_head.weight" not in state.keys()
        assert state.get_tensor("model.embed_tokens.weight").dtype == torch.float32
    loaded = AutoModelForCausalLM.from_pretrained(tmp_path, dtype=torch.float32).eval()
    assert loaded.get_decoder() is loaded.model
    assert loaded.get_output_embeddings().weight is loaded.get_input_embeddings().weight
    with torch.no_grad():
        torch.testing.assert_close(loaded(tokens).logits, before, rtol=0, atol=0)
