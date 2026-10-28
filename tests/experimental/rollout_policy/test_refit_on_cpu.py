# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
import asyncio
from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from verl.workers.rollout.hf_fusion.adapter import ServerAdapter


def receiver():
    target = object.__new__(ServerAdapter)
    target.settings = {"current": True}
    config = Qwen3Config(
        vocab_size=17,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        tie_word_embeddings=True,
    )
    target.student = Qwen3ForCausalLM(config).eval().requires_grad_(False)
    target.teacher = Qwen3ForCausalLM(config).eval().requires_grad_(False)
    target.generating = False
    target.valid = True
    target.loaded_version = (0, 0)
    return target


def test_refit_commits_only_after_complete_stream_and_preserves_teacher():
    target = receiver()
    teacher_before = {k: v.clone() for k, v in target.teacher.state_dict().items()}
    state = [(name, torch.full_like(value, 0.25)) for name, value in target.student.named_parameters()]
    assert asyncio.run(target.update_weights(iter(state), version=(1, 16))) is None
    assert not target.valid and target.loaded_version is None
    target.commit_version((1, 16))
    assert target.valid and target.loaded_version == (1, 16)
    assert all(torch.equal(value, torch.full_like(value, 0.25)) for value in target.student.parameters())
    assert target.student.lm_head.weight.data_ptr() == target.student.model.embed_tokens.weight.data_ptr()
    for name, value in target.teacher.state_dict().items():
        torch.testing.assert_close(value, teacher_before[name], rtol=0, atol=0)


@pytest.mark.parametrize("fault", ["shape", "name", "missing", "duplicate"])
def test_failed_refit_exhausts_collective_stream_and_stays_invalid(fault):
    target = receiver()
    state = [(name, value.clone()) for name, value in target.student.named_parameters()]
    if fault == "shape":
        state[0] = state[0][0], torch.zeros(1)
    if fault == "name":
        state[0] = "wrong_name", state[0][1]
    if fault == "missing":
        state = state[1:]
    if fault == "duplicate":
        state.insert(1, state[0])
    visited = []

    def stream():
        for name, value in state:
            visited.append(name)
            yield name, value

    error = asyncio.run(target.update_weights(stream(), version=(1, 1)))
    assert error
    assert len(visited) == len(state)
    assert not target.valid and target.loaded_version is None


def test_tied_parameter_conflict_cannot_be_committed():
    target = receiver()
    state = [(name, value.clone()) for name, value in target.student.state_dict().items()]
    state = [(name, value + 1 if name == "lm_head.weight" else value) for name, value in state]
    assert asyncio.run(target.update_weights(iter(state), version=(1, 1))) is not None
    with pytest.raises(RuntimeError, match="version mismatch"):
        target.commit_version((1, 1))


def test_residency_backup_tracks_refits_and_restores_tied_storage():
    target = receiver()
    target.student_cpu_parameters = target._cpu_parameters(target.student)
    state = [(name, torch.full_like(value, 0.375)) for name, value in target.student.named_parameters()]
    assert asyncio.run(target.update_weights(iter(state), version=(2, 4))) is None
    target.commit_version((2, 4))
    assert all(torch.equal(value, torch.full_like(value, 0.375)) for value in target.student_cpu_parameters.values())
    # Simulate Module.to replacing each parameter's storage, without needing a GPU.
    for parameter in target.student.parameters():
        parameter.data = torch.full_like(parameter, 9.0)
    target._park_model(target.student, target.student_cpu_parameters)
    assert all(torch.equal(value, torch.full_like(value, 0.375)) for value in target.student.parameters())
    assert target.student.lm_head.weight.data_ptr() == target.student.model.embed_tokens.weight.data_ptr()


@pytest.mark.parametrize("alpha,path", [(0.0, "student"), (1.0, "teacher")])
def test_endpoint_does_not_load_inactive_generation_model(monkeypatch, alpha, path):
    target = receiver()
    target.settings = {"current": False, "teacher_path": "teacher"}
    target.student = target.teacher = None
    target.model_config = SimpleNamespace(path="student")
    target.device = torch.device("cpu")
    target.legal_ids = (0, 1)
    loaded = []

    def load(path):
        loaded.append(path)
        return torch.nn.Linear(2, 2)

    target._load = load
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    asyncio.run(target.prepare(alpha, (0, 0)))
    assert loaded == [path]
    assert (target.student is None) == (alpha == 1)
    assert (target.teacher is None) == (alpha == 0)
