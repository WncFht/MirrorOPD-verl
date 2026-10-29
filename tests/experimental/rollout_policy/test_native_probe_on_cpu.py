# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Keep diagnostic prefixes aligned when requests enter and leave a native batch."""

from types import SimpleNamespace

import torch

from examples.mirroropd.benchmarks.fusion.compare_native_components import measure
from examples.mirroropd.benchmarks.fusion.probe_worker import PrefixProbe


def test_forced_replay_tracks_requests_across_batch_changes(tmp_path):
    model = SimpleNamespace(compute_logits=lambda hidden: hidden.clone())
    batch = SimpleNamespace(req_ids=["a"])
    probe = PrefixProbe()
    probe.model_runner = SimpleNamespace(get_model=lambda: model, input_batch=batch)
    probe.configure_probe([4, 3], [0, 1], requests=2)
    a0 = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.float32)
    assert model.compute_logits(a0).argmax(-1).tolist() == [4]
    batch.req_ids = ["b", "a"]
    ba = torch.tensor([[6, 7, 8, 9, 10], [11, 12, 13, 14, 15]], dtype=torch.float32)
    assert model.compute_logits(ba).argmax(-1).tolist() == [4, 3]
    batch.req_ids = ["b"]
    b1 = torch.tensor([[16, 17, 18, 19, 20]], dtype=torch.float32)
    assert model.compute_logits(b1).argmax(-1).tolist() == [3]
    path = tmp_path / "probe.pt"
    assert probe.save_probe(path)["batch_shapes"] == [1, 2, 1]
    saved = torch.load(path, weights_only=True)
    torch.testing.assert_close(saved[0]["raw"], torch.cat([a0, ba[:1]]))
    torch.testing.assert_close(saved[1]["raw"], torch.cat([ba[1:], b1]))


def test_distribution_metric_ignores_logit_shift_and_masked_tail():
    reference = torch.tensor([1.0, 2.0, 3.0, 100.0])
    candidate = torch.tensor([8.0, 9.0, 10.0, -100.0])
    metrics = measure(reference, candidate, torch.tensor([0, 1, 2]), 1)
    assert all(value < 1e-12 for value in metrics.values())
    candidate[1] += 2
    assert measure(reference, candidate, torch.tensor([0, 1, 2]), 1)["total_variation"] > 0.1
