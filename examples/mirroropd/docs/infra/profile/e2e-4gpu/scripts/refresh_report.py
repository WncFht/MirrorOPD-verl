# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""CPU-only postprocessing of an existing queue; never starts training or agent turns.

One lock per report directory. Refresh at queue transitions and once after completion.
The resulting report is a measurement snapshot, not an automated performance verdict.
"""

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--style-dir", type=Path)
    parser.add_argument("--base-root", type=Path, help="Merge baseline/Torch evidence with this queue's Nsight retry")
    parser.add_argument("--review", type=Path, help="Embed the existing reviewed analysis in the offline snapshot")
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    scripts = Path(__file__).resolve().parent
    last = None
    with (root / "report.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("A report worker already owns this output directory") from None
        while True:
            status = json.loads((root / "status.json").read_text())
            signature = (status.get("state"), status.get("current_job"), json.dumps(status.get("jobs"), sort_keys=True))
            if signature != last:
                report_status = {
                    "pid": os.getpid(),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "queue_state": status.get("state"),
                    "review": "pending",
                }
                try:
                    if args.base_root and status.get("jobs", {}).get("nsys-profile", {}).get("returncode") == 0:
                        check = subprocess.run(
                            [sys.executable, str(scripts / "inspect_nsys.py"), "--job", str(root / "nsys-profile")],
                            check=False,
                        )
                        report_status["nsys_artifacts_accepted"] = check.returncode == 0
                    collect_command = [
                        sys.executable,
                        str(scripts / "collect.py"),
                        "--root",
                        str(args.base_root.resolve() if args.base_root else root),
                        "--output",
                        str(output / "data.json"),
                    ]
                    if args.base_root:
                        collect_command += ["--nsys-retry-root", str(root)]
                    subprocess.run(collect_command, check=True)
                    render_command = [
                        sys.executable,
                        str(scripts / "render.py"),
                        "--data",
                        str(output / "data.json"),
                        "--output",
                        str(output / "dashboard.html"),
                    ]
                    if args.review:
                        render_command += ["--review", str(args.review.resolve())]
                    subprocess.run(render_command, check=True)
                    if status.get("jobs", {}).get("baseline", {}).get("returncode") == 0:
                        plot_command = [
                            sys.executable,
                            str(scripts / "plot.py"),
                            "--data",
                            str(output / "data.json"),
                            "--output",
                            str(output / "figures/baseline-phases"),
                        ]
                        if args.style_dir:
                            plot_command += ["--style-dir", str(args.style_dir)]
                        subprocess.run(plot_command, check=True)
                    report_status["state"] = "snapshot_generated"
                except subprocess.CalledProcessError as exc:
                    report_status.update(state="failed", error=str(exc))
                    (root / "report-status.json").write_text(json.dumps(report_status, indent=2) + "\n")
                    raise
                (root / "report-status.json").write_text(json.dumps(report_status, indent=2) + "\n")
                last = signature
            if (
                not args.watch
                or status.get("state") in {"completed", "completed_with_failures", "stopped", "failed"}
                or (root / "STOP").exists()
            ):
                return
            time.sleep(30)


if __name__ == "__main__":
    main()
