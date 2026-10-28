# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Actor-version age applies only to requests with a current learner component."""

import numpy as np


def rollout_version_metrics(tags, global_step):
    current = [tag for tag in tags if tag.get("rollout_has_current_component", True)]
    metrics = {}
    if any("rollout_has_current_component" in tag for tag in tags):
        metrics["rollout/policy/current_component_version_fraction"] = len(current) / len(tags) if tags else 0.0
    if not current:
        return metrics
    if any(tag.get("min_global_steps") is None or tag.get("max_global_steps") is None for tag in current):
        raise ValueError("Current rollout component is missing an actor version")
    minimum = np.asarray([tag["min_global_steps"] for tag in current], dtype=int)
    maximum = np.asarray([tag["max_global_steps"] for tag in current], dtype=int)
    quantities = {
        "trajectory_spans": maximum - minimum + 1,
        "trajectory_staleness": (global_step - 1) - maximum,
        "trajectory_staleness_worst": (global_step - 1) - minimum,
    }
    for name, values in quantities.items():
        metrics.update(
            {
                f"training/off_policy/{name}/mean": values.mean(),
                f"training/off_policy/{name}/max": values.max(),
                f"training/off_policy/{name}/min": values.min(),
            }
        )
    return metrics
