# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Shared fixed-trajectory inputs, task selection and cache preparation."""

import argparse
import json
from pathlib import Path

import torch

from .inputs import load_records, normalize_records
from .prefixes import branch_tasks
from .runtime import teacher_cache


def panel(task, max_groups=None):
    """Keep original group order and cache identity while selecting Math rows."""
    rows = normalize_records(load_records(task["records"]), cap=task["cap"], bank_id=task["bank"])
    math = [row for row in rows if row["domain"] == "math"]
    groups = {}
    for row in math:
        key = (row["split"], row.get("diagnostic_group", row.get("group")))
        groups.setdefault(key, []).append(row)
    selected = list(groups.items())[:max_groups]
    sizes = [len(batch) for _, batch in selected]
    if not sizes or min(sizes) != max(sizes):
        raise ValueError("Trajectory measurements require nonempty equal-size groups")
    return rows, math, selected


def request_cache(task, initial_path, root, rows, math):
    """Use complete source panels so earlier teacher/initial caches are reused."""
    teacher = teacher_cache(root / "cache", task["teacher"], rows)
    initial = teacher_cache(root / "cache", initial_path, math, student=True)
    for value in (teacher, initial):
        value["head"] = value["head"].to("cuda")
    return teacher, initial


def parser(description):
    result = argparse.ArgumentParser(description=description)
    result.add_argument("--manifest", type=Path, required=True)
    result.add_argument("--out", type=Path, required=True)
    result.add_argument("--branch", required=True)
    result.add_argument("--model", default="r1")
    result.add_argument("--stages", type=int, nargs="+", required=True)
    result.add_argument("--max-groups", type=int)
    result.add_argument("--token-chunk", type=int, default=128)
    return result


def tasks(args):
    if args.token_chunk <= 0 or any(s < 0 or s > 320 for s in args.stages):
        raise ValueError("positive chunk and stages in [0,320] required")
    if args.max_groups is not None and args.max_groups < 1:
        raise ValueError("max_groups must be positive")
    torch.set_num_threads(4)
    items = [t for t in json.loads(args.manifest.read_text())["tasks"] if t["model"] == args.model]
    initial_paths = {t["student"] for t in items if t["stage"] == 0}
    if len(initial_paths) != 1:
        raise ValueError("A trajectory comparison requires one shared initial student")
    initial = initial_paths.pop()
    selected = [t for t in branch_tasks(items, args.branch) if t["stage"] in args.stages]
    if {t["stage"] for t in selected} != set(args.stages):
        raise ValueError("Requested branch/stages are absent from manifest")
    return initial, selected
