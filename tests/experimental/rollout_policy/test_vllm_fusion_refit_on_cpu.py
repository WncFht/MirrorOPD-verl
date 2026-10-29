# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Receiver failure must drain buckets and leave the model unpublished."""

from types import SimpleNamespace

import pytest
import torch

from verl.workers.rollout.vllm_fusion.worker_extension import FusionWorkerExtension


class Student(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(3, 2))
        self.bias = torch.nn.Parameter(torch.zeros(3))
        self.config = SimpleNamespace(tie_word_embeddings=False)

    def load_weights(self, weights):
        loaded = set()
        with torch.no_grad():
            for name, value in weights:
                getattr(self, name).copy_(value)
                loaded.add(name)
        return loaded


@pytest.mark.parametrize("fault", [None, "missing", "shape", "name", "duplicate", "loader"])
def test_receiver_drains_and_commits_only_complete_versions(monkeypatch, fault):
    from verl.workers.rollout.vllm_rollout import bucketed_weight_transfer as transport

    target = FusionWorkerExtension()
    student = Student()
    teacher = torch.nn.Linear(2, 2)
    teacher_before = {name: tensor.clone() for name, tensor in teacher.state_dict().items()}
    target.model_runner = SimpleNamespace(model=SimpleNamespace(student=student, teacher=teacher))
    target.device = torch.device("cpu")
    target.local_rank = 0
    target.fusion_current = True
    target.fusion_shapes = {name: tuple(value.shape) for name, value in student.named_parameters()}
    rows = [(name, torch.ones_like(value)) for name, value in student.named_parameters()]
    if fault == "missing":
        rows = rows[1:]
    elif fault == "shape":
        rows[0] = rows[0][0], torch.zeros(1)
    elif fault == "name":
        rows[0] = "unknown", rows[0][1]
    elif fault == "duplicate":
        rows.insert(1, rows[0])
    elif fault == "loader":

        def fail(weights):
            raise RuntimeError("injected native loader failure")

        student.load_weights = fail
    consumed = []

    class Receiver:
        def __init__(self, **kwargs):
            pass

        def receive_weights(self, callback):
            for index, row in enumerate(rows):
                consumed.append(row[0])
                callback([row], index == len(rows) - 1)

    monkeypatch.setattr(transport, "BucketedWeightReceiver", Receiver)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    error = target.fusion_receive_weights((2, 4))
    assert len(consumed) == len(rows)
    assert not target.fusion_valid and target.fusion_version is None
    if fault is None:
        assert error is None
        target.fusion_commit_version((2, 4))
        assert target.fusion_valid and target.fusion_version == (2, 4)
    else:
        assert error
        with pytest.raises(RuntimeError, match="incomplete"):
            target.fusion_commit_version((2, 4))
    for name, tensor in teacher.state_dict().items():
        torch.testing.assert_close(tensor, teacher_before[name], rtol=0, atol=0)


def test_frozen_student_rejects_refit():
    target = FusionWorkerExtension()
    target.fusion_current = False
    with pytest.raises(ValueError, match="Frozen"):
        target.fusion_receive_weights((1, 1))


@pytest.mark.parametrize("conflict", [False, True])
def test_tied_alias_validation_uses_each_native_vocab_shard(monkeypatch, conflict):
    from verl.workers.rollout.vllm_fusion import worker_extension
    from verl.workers.rollout.vllm_rollout import bucketed_weight_transfer as transport

    source = torch.ones(4, 2)
    alias = source.clone()
    if conflict:
        alias[3] = 2

    class ShardedStudent(torch.nn.Module):
        def __init__(self, rank):
            super().__init__()
            self.rank = rank
            self.model = torch.nn.Module()
            self.model.embed_tokens = torch.nn.Embedding(2, 2)
            self.lm_head = self.model.embed_tokens
            self.config = SimpleNamespace(tie_word_embeddings=True)

        def load_weights(self, weights):
            with torch.no_grad():
                for _, value in weights:
                    self.model.embed_tokens.weight.copy_(value[self.rank * 2 : (self.rank + 1) * 2])
            return {"model.embed_tokens.weight"}

    class Receiver:
        def __init__(self, **kwargs):
            pass

        def receive_weights(self, callback):
            callback([("model.embed_tokens.weight", source)], False)
            callback([("lm_head.weight", alias)], True)

    monkeypatch.setattr(transport, "BucketedWeightReceiver", Receiver)
    monkeypatch.setattr(worker_extension, "get_torch_device", lambda: SimpleNamespace(synchronize=lambda: None))
    errors = []
    for rank in range(2):
        target = FusionWorkerExtension()
        target.model_runner = SimpleNamespace(model=SimpleNamespace(student=ShardedStudent(rank)))
        target.fusion_current = True
        target.fusion_shapes = {"model.embed_tokens.weight": (4, 2)}
        target.device, target.local_rank = torch.device("cpu"), rank
        errors.append(target.fusion_receive_weights((2, 2)))
        assert not target.fusion_valid and target.fusion_version is None
    assert errors[0] is None
    assert (errors[1] is not None) == conflict
