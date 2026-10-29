# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Exercise local DP/TP receiver identity and collective head ordering."""

import pytest
import torch

from verl.experimental.rollout_policy.topology import FusionTopology
from verl.workers.rollout.vllm_fusion.model import MirrorOPDFusionForCausalLM


@pytest.mark.parametrize("world,tp,receivers", [(1, 1, [(0, 0)]), (2, 1, [(0, 0), (1, 0)]), (2, 2, [(0, 0), (0, 1)])])
def test_rank_mapping_covers_each_actor_once(world, tp, receivers):
    topology = FusionTopology(world, tp)
    assert [topology.receiver(rank) for rank in range(world)] == receivers
    workers = list(range(world))
    assert [
        rank for replica in range(topology.replica_count) for rank in workers[topology.worker_slice(replica)]
    ] == workers


def test_incomplete_tp_pool_rejected():
    with pytest.raises(ValueError, match="divide"):
        FusionTopology(1, 2)


def test_non_output_rank_enters_both_head_collectives():
    calls = []

    class Head(torch.nn.Module):
        def __init__(self, name):
            super().__init__()
            self.name = name

        def compute_logits(self, hidden):
            calls.append(self.name)
            return None

    model = MirrorOPDFusionForCausalLM.__new__(MirrorOPDFusionForCausalLM)
    torch.nn.Module.__init__(model)
    model.student, model.teacher = Head("student"), Head("teacher")
    model.student_width = 2
    model.alpha = 0.0
    assert model.compute_logits(torch.zeros(1, 5)) is None
    assert calls == ["student", "teacher"]
