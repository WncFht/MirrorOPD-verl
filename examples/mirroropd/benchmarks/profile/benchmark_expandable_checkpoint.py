# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Test expandable segments with selective checkpointing in a fresh process."""

import json
import os
import runpy
import sys
from pathlib import Path

if __name__ == "__main__":
    os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    runpy.run_module("examples.mirroropd.benchmarks.profile.benchmark_partial_checkpoint", run_name="__main__")
    if os.environ["RANK"] == "0":
        path = Path(sys.argv[sys.argv.index("--output") + 1])
        result = json.loads(path.read_text())
        result["allocator"] = os.environ["PYTORCH_ALLOC_CONF"]
        path.write_text(json.dumps(result, indent=2) + "\n")
