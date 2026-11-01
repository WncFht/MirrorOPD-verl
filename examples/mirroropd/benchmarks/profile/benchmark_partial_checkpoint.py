# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Probe alternating native HF layer checkpoint flags; no global model-class patch.

The parent actor probe keeps its fixed-token, lr0, BF16 and parameter-hash checks.
Only even-indexed decoder layers retain their initialized checkpoint wrapper.
"""

import json
import os
import sys
from pathlib import Path

from examples.mirroropd.benchmarks.profile import benchmark_actor_settings as probe

_original_worker = probe.worker


def partial_checkpoint_worker(*args, **kwargs):
    actor = _original_worker(*args, **kwargs)
    layers = actor.engine.module.model.layers
    if len(layers) != 28:
        raise ValueError("This probe records the 28-layer Qwen3 actor layout")
    if not all(layer.gradient_checkpointing for layer in layers):
        raise ValueError("Selective probe requires the checkpoint-enabled actor baseline")
    for index, layer in enumerate(layers):
        layer.gradient_checkpointing = index % 2 == 0
    return actor


if __name__ == "__main__":
    probe.worker = partial_checkpoint_worker
    probe.main()
    if os.environ["RANK"] == "0":
        path = Path(sys.argv[sys.argv.index("--output") + 1])
        record = json.loads(path.read_text())
        record["selective_checkpoint"] = dict(
            enabled_layers=list(range(0, 28, 2)), total_layers=28, scope="instance-local native HF checkpoint flags"
        )
        path.write_text(json.dumps(record, indent=2) + "\n")
