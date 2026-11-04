# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Worker outputs may mix host counters and CUDA sufficient statistics."""

import pytest
import torch

from verl.utils.metric.observations import Observation, observations_to_cpu

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_mixed_host_device_observations_preserve_statistics_and_shared_samples():
    values = torch.tensor([1.0, 2.0, 4.0], device="cuda")
    metrics = {
        "host_count": Observation.total(3),
        "device_mean": Observation.mean(values),
        "host_ratio": Observation.ratio(5, 2),
        "device_empty": Observation.maximum(values[:0]),
        "median": Observation.quantile(values, 0.5),
        "maximum": Observation.quantile(values, 1.0),
        "ordinary": 7,
    }
    expected = {key: value.aggregate() for key, value in metrics.items() if isinstance(value, Observation)}
    observations_to_cpu(metrics)
    for key, value in metrics.items():
        if isinstance(value, Observation):
            assert value.state.device.type == "cpu"
            assert all(sample.device.type == "cpu" for sample in value.samples)
            assert value.aggregate() == pytest.approx(expected[key], nan_ok=True)
    assert metrics["median"].samples[0].data_ptr() == metrics["maximum"].samples[0].data_ptr()
    assert metrics["ordinary"] == 7
