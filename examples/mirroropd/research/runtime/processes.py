# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""PID/start-time identities shared by research worker controllers."""

import importlib.util
from pathlib import Path

from examples.mirroropd.research.core.paths import workspace


def identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return [int(pid), fields[19]] if fields[0] not in ("Z", "X") else None
    except (FileNotFoundError, ProcessLookupError):
        return None


def tagged(tag, variable):
    result = []
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            if f"{variable}={tag}".encode() in (path / "environ").read_bytes().split(b"\0"):
                value = identity(int(path.name))
                if value:
                    result.append(value)
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return result


def send(record, sig):
    spec = importlib.util.spec_from_file_location(
        "study_occupy_control", workspace() / "cluster_tools/gpu_occupy_ctl.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    expected = module.identity(record[0])
    if expected and expected["start_ticks"] == record[1]:
        module.signal_process({"identity": expected}, sig)
