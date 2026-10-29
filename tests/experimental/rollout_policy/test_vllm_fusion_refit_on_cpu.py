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
