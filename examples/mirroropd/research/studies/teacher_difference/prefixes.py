# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Shared fixed-prefix selection and input-addressed initial teacher requests."""

import hashlib
from pathlib import Path

import torch

from .inputs import load_records, normalize_records, source_fingerprint
from .runtime import checkpoint_identity, collect_memory, load_model, prefix_logits, teacher_cache, write_tensors
from .signals import position_signal


def select(rows, *, per_stratum=4, seed=20261109, domains=None, position_mode="three"):
    if position_mode not in {"three", "uniform"}:
        raise ValueError("Unknown position sampling mode")
    strata = {}
    for row in rows:
        if domains is not None and row["domain"] not in domains:
            continue
        strata.setdefault((row.get("split"), row["domain"]), []).append(row)
    selected = []
    for key, members in sorted(strata.items()):
        # Distinct trajectories per stratum, sorted by deterministic hash.
        members = sorted(members, key=lambda r: hashlib.sha256(f"{r['row_id']}:{seed}".encode()).hexdigest())[
            :per_stratum
        ]
        for row in members:
            length = len(row["response_ids"])
            if position_mode == "uniform":
                import random

                positions = [random.Random(f"{seed}:position:{row['row_id']}").randrange(length)]
            else:
                positions = sorted(set([0, length // 2, length - 1]))
            for pos in positions:
                selected.append((row, pos))
    return selected


def branch_tasks(tasks, branch):
    """Select a training path, excluding legacy target-swap measurement tasks."""
    return [
        task
        for task in tasks
        if task.get("branch", task["target"]) == branch and ("branch" in task or "-target-" not in task["id"])
    ]


def prepare_panel(
    manifest, root, *, model, bank=None, out, per_stratum=4, seed=20261109, domains=None, position_mode="three"
):
    """Reuse exactly the same token inputs, teachers, and initial requests."""
    candidates = [t for t in manifest["tasks"] if t["model"] == model]
    bank = bank or next(t["bank"] for t in candidates if t["stage"] == 0)
    tasks = [t for t in candidates if t["bank"] == bank]
    initial = [t for t in tasks if t["stage"] == 0]
    if len({t["student"] for t in initial}) != 1:
        raise ValueError("Prefix comparison requires one shared initial student per model/bank")
    base = initial[0]
    rows = normalize_records(load_records(base["records"]), cap=base["cap"], bank_id=base["bank"])
    selected = select(rows, per_stratum=per_stratum, seed=seed, domains=domains, position_mode=position_mode)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    requests_file = out / "requests.pt"
    request_identity = dict(
        student=checkpoint_identity(base["student"]),
        records=source_fingerprint(rows),
        teachers={t["target"]: checkpoint_identity(t["teacher"]) for t in initial},
        prefixes=[(r["row_id"], pos) for r, pos in selected],
        protocol="native-bf16-fp32-head-v1",
        torch_version=str(torch.__version__),
    )
    if requests_file.exists():
        payload = torch.load(requests_file, map_location="cpu", weights_only=True)
        if payload.get("identity") != request_identity:
            raise ValueError(f"Frozen-request cache identity mismatch: {requests_file}")
        requests = payload["requests"]
    else:
        student = load_model(base["student"], checkpointing=base.get("checkpointing", False))
        with torch.no_grad():
            logits0 = [prefix_logits(student, row, pos).cpu() for row, pos in selected]
        del student
        collect_memory()
        requests = {}
        for task in initial:
            cache = teacher_cache(root / "cache", task["teacher"], rows)
            head = cache["head"].to("cuda")
            values = []
            with torch.no_grad():
                for (row, pos), z0 in zip(selected, logits0, strict=True):
                    zq = torch.nn.functional.linear(cache["hidden"][row["row_id"]][pos].to("cuda"), head)
                    _, b0 = position_signal(z0.to("cuda"), zq)
                    values.append(dict(B0=b0.cpu(), q_logits=zq.cpu()))
            requests[task["target"]] = values
            del head, cache
            collect_memory()
        write_tensors(requests_file, dict(identity=request_identity, requests=requests))
    return tasks, rows, selected, requests, request_identity
