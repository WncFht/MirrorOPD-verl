# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Native registration contract and heterogeneous component wiring."""

import pytest
import torch
from vllm.model_executor.models.interfaces_base import is_text_generation_model

from verl.workers.rollout.vllm_fusion.model import MirrorOPDFusionForCausalLM


class Branch(torch.nn.Module):
    def __init__(self, width, offset, vocab_size=5):
        super().__init__()
        self.table = torch.nn.Embedding(5, width)
        self.head = torch.nn.Linear(width, vocab_size, bias=False)
        with torch.no_grad():
            self.table.weight.fill_(offset)
            self.head.weight.copy_(torch.arange(vocab_size * width).view(vocab_size, width) / 10)

    def embed_input_ids(self, input_ids):
        return self.table(input_ids)

    def forward(self, input_ids, positions, inputs_embeds=None):
        return self.embed_input_ids(input_ids) if inputs_embeds is None else inputs_embeds

    def compute_logits(self, hidden):
        return self.head(hidden)


def composite():
    model = MirrorOPDFusionForCausalLM.__new__(MirrorOPDFusionForCausalLM)
    torch.nn.Module.__init__(model)
    model.student = Branch(2, 0.25)
    model.teacher = Branch(3, 0.5)
    model.student_width = 2
    model.register_buffer("legal_tokens", torch.tensor([True, True, True, True, False]))
    return model


def test_native_model_protocol_includes_embedding_entrypoint():
    assert is_text_generation_model(MirrorOPDFusionForCausalLM)


@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
def test_independent_embeddings_heads_and_fusion(alpha):
    model = composite()
    model.set_alpha(alpha)
    ids, positions = torch.tensor([1, 2]), torch.tensor([0, 1])
    hidden = model(ids, positions)
    embedded_hidden = model(None, positions, inputs_embeds=model.embed_input_ids(ids))
    torch.testing.assert_close(hidden, embedded_hidden, atol=0, rtol=0)
    assert hidden.shape == (2, 5)
    student = model.student.compute_logits(model.student.embed_input_ids(ids)).double()
    teacher = model.teacher.compute_logits(model.teacher.embed_input_ids(ids)).double()
    expected = (1 - alpha) * student + alpha * teacher
    expected[:, -1] = -torch.inf
    torch.testing.assert_close(model.compute_logits(hidden).double(), expected, atol=2e-6, rtol=0)


@pytest.mark.parametrize("alpha", [-1.0, 1.1, float("nan"), float("inf")])
def test_invalid_alpha_rejected(alpha):
    with pytest.raises(ValueError, match="alpha"):
        composite().set_alpha(alpha)


@pytest.mark.parametrize("teacher_vocab", [4, 7])
@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
def test_unequal_heads_condition_sampling_without_mutating_components(teacher_vocab, alpha):
    model = composite()
    model.teacher = Branch(3, 0.5, teacher_vocab)
    model.capture_components = True
    model.set_alpha(alpha)
    ids = torch.tensor([1, 2])
    student = model.student.compute_logits(model.student.embed_input_ids(ids)).double()
    teacher = model.teacher.compute_logits(model.teacher.embed_input_ids(ids)).double()
    before = model.teacher.head.weight.detach().clone()
    # Normalize each COMPLETE native head first, then condition the geometric
    # mixture on the common legal IDs. Extra teacher rows must remain present.
    expected = (1 - alpha) * student.log_softmax(-1)[:, :4] + alpha * teacher.log_softmax(-1)[:, :4]
    expected = expected.log_softmax(-1)
    actual = model.compute_logits(model(ids, torch.tensor([0, 1])))
    assert actual.shape == (2, 5)
    assert torch.isneginf(actual[:, 4]).all()
    torch.testing.assert_close(actual.double().log_softmax(-1)[:, :4], expected, atol=2e-6, rtol=0)
    torch.testing.assert_close(model.component_logits["teacher"].double(), teacher, atol=0, rtol=0)
    torch.testing.assert_close(model.component_logits["student"].double(), student, atol=0, rtol=0)
    torch.testing.assert_close(model.teacher.head.weight, before, atol=0, rtol=0)
    assert model.teacher.head.out_features == teacher_vocab


def test_native_family_contract_keeps_qwen2_and_qwen3_pairs_explicit():
    from types import SimpleNamespace

    from verl.workers.rollout.vllm_fusion.compat import validate_component_families

    for student, teacher in [("qwen2", "qwen2"), ("qwen3", "qwen3"), ("qwen3", "qwen3_moe")]:
        validate_component_families(SimpleNamespace(model_type=student), SimpleNamespace(model_type=teacher))
    for student, teacher in [("qwen2", "qwen3"), ("qwen3_moe", "qwen3_moe"), ("llama", "llama")]:
        with pytest.raises(ValueError, match="requires"):
            validate_component_families(SimpleNamespace(model_type=student), SimpleNamespace(model_type=teacher))


def test_child_configuration_replaces_both_hf_views(monkeypatch):
    from types import SimpleNamespace

    from transformers import Qwen3Config, Qwen3MoeConfig
    from vllm import ModelRegistry

    from verl.workers.rollout.vllm_fusion.compat import component_config

    class Config:
        def __init__(self, hf):
            self.hf_config = self.hf_text_config = hf
            self.model = "student"
            self.hf_overrides = {"architectures": ["OuterFusion"]}

        @property
        def architectures(self):
            return self.hf_config.architectures

        def get_model_arch_config(self):
            assert self.hf_config is self.hf_text_config
            return (self.hf_config.hidden_size, self.hf_config.num_hidden_layers)

    student = Qwen3Config(hidden_size=16, num_hidden_layers=2)
    teacher = Qwen3MoeConfig(hidden_size=32, num_hidden_layers=3, num_experts=4)
    teacher.architectures = ["Qwen3MoeForCausalLM"]
    parent = SimpleNamespace(model_config=Config(student), compilation_config=object())
    monkeypatch.setattr(ModelRegistry, "inspect_model_cls", lambda arch, config: ("teacher-info", arch[0]))
    child = component_config(parent, teacher, "teacher")
    assert child.model_config.hf_text_config.num_experts == 4
    assert child.model_config.model_arch_config == (32, 3)
    assert child.model_config._model_info == "teacher-info"
    assert parent.model_config.hf_text_config is student
    assert child.compilation_config is parent.compilation_config
