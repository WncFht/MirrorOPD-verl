# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Synthetic files exercise artifact checks; no Nsight or GPU collection is run."""

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "examples/mirroropd/docs/infra/profile/e2e-4gpu/scripts"


@pytest.mark.parametrize("missing_gpu_rank", [None, 3])
def test_artifact_acceptance_requires_cuda_and_nvtx_on_all_ranks(tmp_path, missing_gpu_rank):
    traces = tmp_path / "traces"
    traces.mkdir()
    for rank in range(4):
        file = traces / f"synthetic-{rank}.nsys-rep"
        file.write_text("synthetic fixture, not a real Nsight report")
        with sqlite3.connect(file.with_suffix(".sqlite")) as con:
            con.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER, end INTEGER)")
            con.execute("CREATE TABLE NVTX_EVENTS (start INTEGER, end INTEGER)")
            if rank != missing_gpu_rank:
                con.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (100, 300)")
            con.execute("INSERT INTO NVTX_EVENTS VALUES (0, 400)")
        (traces / f"capture-rank{rank}.json").write_text(json.dumps({"status": "finalized", "files": [str(file)]}))
    result = subprocess.run(
        [sys.executable, str(SCRIPTS / "inspect_nsys.py"), "--job", str(tmp_path)], capture_output=True, text=True
    )
    assert result.returncode == (0 if missing_gpu_rank is None else 1)
    data = json.loads((tmp_path / "nsys-review.json").read_text())
    assert data["accepted"] is (missing_gpu_rank is None)


def test_finish_hook_copies_only_exact_pid_reports(tmp_path):
    source, destination = tmp_path / "ray", tmp_path / "output"
    source.mkdir()
    (source / "worker_process_1234.nsys-rep").write_text("synthetic fixture")
    (source / "worker_process_12345.nsys-rep").write_text("another synthetic worker")
    env = os.environ | {
        "VERL_PROFILE_PID": "1234",
        "VERL_PROFILE_RANK": "0",
        "VERL_PROFILE_RAY_NSIGHT_DIR": str(source),
        "VERL_PROFILE_SAVE_PATH": str(destination),
    }
    subprocess.run(
        [sys.executable, str(SCRIPTS / "finish_nsys.py")],
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    reports = list(destination.glob("*.nsys-rep"))
    assert len(reports) == 1
    assert reports[0].name.endswith("worker_process_1234.nsys-rep")
    assert (source / "worker_process_12345.nsys-rep").exists()
