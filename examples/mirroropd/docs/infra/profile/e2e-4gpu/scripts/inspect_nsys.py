# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Verify finalized four-rank Nsight artifacts contain real CUDA kernel records."""

import argparse
import json
import sqlite3
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for rank in range(4):
        row = {"rank": rank, "accepted": False}
        marker = args.job / "traces" / f"capture-rank{rank}.json"
        if not marker.exists():
            rows.append(row | {"error": "Missing per-rank capture marker"})
            continue
        capture = json.loads(marker.read_text())
        files = [Path(p) for p in capture.get("files", [])]
        if capture.get("status") != "finalized" or not files:
            rows.append(row | {"error": "Capture did not finalize"})
            continue
        row["reports"] = []
        for file in files:
            if not file.is_file():
                row["error"] = f"Missing original Nsight report: {file}"
                continue
            database = file.with_suffix(".sqlite")
            if not database.exists():
                command = ["nsys", "export", "--type=sqlite", "--output=" + str(database), str(file)]
                result = subprocess.run(command, capture_output=True, text=True, timeout=180)
                file.with_suffix(".export.log").write_text(result.stdout + result.stderr)
                if result.returncode:
                    row["error"] = f"Nsight export failed: {file}"
                    continue
            with sqlite3.connect(database) as con:
                tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                kernel_table = next(
                    (n for n in ("CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL") if n in tables),
                    None,
                )
                count, work_ns = (
                    con.execute(f'SELECT count(*), coalesce(sum(end-start),0) FROM "{kernel_table}"').fetchone()
                    if kernel_table
                    else (0, 0)
                )
                nvtx = con.execute("SELECT count(*) FROM NVTX_EVENTS").fetchone()[0] if "NVTX_EVENTS" in tables else 0
            row["reports"].append(
                {"file": str(file), "kernel_count": count, "kernel_work_seconds": work_ns / 1e9, "nvtx_events": nvtx}
            )
        row["accepted"] = (
            bool(row["reports"])
            and not row.get("error")
            and all(r["kernel_count"] > 0 and r["nvtx_events"] > 0 for r in row["reports"])
        )
        rows.append(row)
    result = {
        "accepted": all(r["accepted"] for r in rows),
        "ranks": rows,
        "note": "Artifact coverage only; kernel sums are work, not elapsed critical-path time.",
    }
    (args.job / "nsys-review.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["accepted"] else 1)


if __name__ == "__main__":
    main()
