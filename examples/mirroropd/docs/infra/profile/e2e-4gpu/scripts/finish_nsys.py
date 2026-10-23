# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Wait for this worker's finalized Nsight report before the queue cleans up Ray.

Invoked as a profiler finish hook. No GPU work, uploads, or process termination.
The explicit single-range stop-shutdown configuration is required for this study.
"""

import json
import os
import shutil
import time
from pathlib import Path


def main():
    pid = int(os.environ["VERL_PROFILE_PID"])
    rank = int(os.environ["VERL_PROFILE_RANK"])
    source = Path(os.environ["VERL_PROFILE_RAY_NSIGHT_DIR"])
    destination = Path(os.environ["VERL_PROFILE_SAVE_PATH"])
    destination.mkdir(parents=True, exist_ok=True)
    end = time.monotonic() + 180
    previous = {}
    stable = 0
    while time.monotonic() < end:
        files = [p for p in source.glob(f"worker_process_{pid}.*") if p.suffix == ".nsys-rep"]
        files += [p for p in destination.glob(f"*worker_process_{pid}.*") if p.suffix == ".nsys-rep"]
        sizes = {p: p.stat().st_size for p in files if p.is_file() and p.stat().st_size > 0}
        stable = stable + 1 if sizes and sizes == previous else 0
        if stable >= 3:
            saved = []
            for file in sizes:
                target = destination / f"nsys-rank{rank}-pid{pid}-{file.name}"
                if file.parent != destination:
                    shutil.copy2(file, target)
                else:
                    target = file
                saved.append(str(target))
            record = dict(rank=rank, pid=pid, source=str(source), files=saved, status="finalized")
            (destination / f"capture-rank{rank}.json").write_text(json.dumps(record, indent=2) + "\n")
            print(json.dumps(record), flush=True)
            return
        previous = sizes
        time.sleep(1)
    record = dict(rank=rank, pid=pid, source=str(source), status="missing_report")
    (destination / f"capture-rank{rank}.json").write_text(json.dumps(record, indent=2) + "\n")
    raise RuntimeError(f"No finalized Nsight report after 180 seconds: {record}")


if __name__ == "__main__":
    main()
