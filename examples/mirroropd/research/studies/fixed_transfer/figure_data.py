# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Validated run-level figure data; no model loading or GPU work.

All risks use sum / (prompts * horizon), matching the experiment objective.
Position diagnostics instead divide by observed positions and expose coverage.
Frozen identities are checked across snapshots; unavailable panels stay absent.
"""

import csv
import hashlib
import math
from pathlib import Path

from examples.mirroropd.research.core.artifacts import file_hash
from examples.mirroropd.research.core.io import read
from examples.mirroropd.research.studies.fixed_transfer.display_policy import MAX_DISPLAY_UPDATES

GROUPS = list("ABCDEFGHIJKLMNOPQRST")
DOMAINS = {
    "Math": ["M01", "M02", "M03", "M04"],
    "Code": ["Code01", "Code02"],
    "IF": ["IF01", "IF02"],
}
HELD = sum(DOMAINS.values(), [])


def registry(directory):
    """Use the overview's curated source selection, never choose runs by their score."""
    result = []
    for filename, scope in [("risks.csv", "main"), ("appendix-risks.csv", "supplement")]:
        path = directory / filename
        if not path.exists():
            continue
        unique = {}
        for row in csv.DictReader(path.open()):
            key = (row["student"], row["teacher"], row["arm"])
            identity = tuple(row[k] for k in ["steps", "support", "initial_source", "final_source"])
            if key in unique and unique[key][0] != identity:
                raise ValueError(f"Conflicting run sources: {key}")
            unique[key] = (identity, row)
        for _, row in unique.values():
            row = dict(row, scope=scope)
            row["budget"] = min(int(row["steps"]), MAX_DISPLAY_UPDATES)
            origin = str(Path(row["final_source"]).parent)
            digest = hashlib.sha256(origin.encode()).hexdigest()[:8]
            row["run_id"] = f"{row['student']}--{row['teacher']}--{digest}"
            result.append(row)
    return result


def mean(rows, groups, horizon, metric="rkl", band="all", observed=False):
    selected = [r for r in rows if r["group"] in groups]
    if not selected or any(metric not in r["stats"].get(band, {}) for r in selected):
        return None
    denominator = sum(r["stats"][band]["tokens"] for r in selected) if observed else len(selected) * horizon
    return sum(r["stats"][band][metric] for r in selected) / denominator if denominator else None


def load_run(spec):
    root = Path(spec["final_source"]).parent
    protocol_path = root / "protocol.json"
    protocol = read(protocol_path) if protocol_path.exists() else {}
    settings = protocol.get("inputs", {}).get("training", protocol)
    horizon = int(settings.get("horizon", 1024))
    if horizon != 1024:
        raise ValueError("Current position schema requires H1024")
    reuse = int(settings.get("reuse", 16))
    batch_size = int(settings.get("batch_size", settings.get("effective_batch", 16)))
    budget = spec["budget"]
    if budget > MAX_DISPLAY_UPDATES or budget % reuse or budget // reuse > len(GROUPS):
        raise ValueError("Unsupported budget / repeated-batch schedule")
    sources, snapshots, identity = [], {}, None
    files = {int(p.stem.split("-")[1]): p for p in root.glob("fixed-*.json") if int(p.stem.split("-")[1]) <= budget}
    files[0] = Path(spec["initial_source"])
    if budget not in files:
        raise ValueError("Missing endpoint snapshot")
    for step, path in sorted(files.items()):
        rows = sorted(read(path), key=lambda r: r["id"])
        current = [(r["id"], r["group"], r["source"], r["length"]) for r in rows]
        if len({r["id"] for r in rows}) != len(rows):
            raise ValueError("Duplicate prompt IDs")
        if any(r["step"] != step for r in rows):
            raise ValueError("Snapshot step mismatch")
        if identity is not None and identity != current:
            raise ValueError("Frozen-state identity changed")
        identity = current
        for r in rows:
            for values in r["stats"].values():
                if any(not math.isfinite(v) for v in values.values()):
                    raise ValueError("Non-finite scored metric")
        snapshots[step] = rows
        sources.append(dict(kind="snapshot", step=step, path=str(path.resolve()), sha256=file_hash(path)))
    available = {r["group"] for r in snapshots[0]}
    for group in available:
        if sum(r["group"] == group for r in snapshots[0]) != batch_size:
            raise ValueError(f"Incomplete group: {group}")
    panels = {
        "Training scored": [g for g in GROUPS if g in available],
        "Old A-D": list("ABCD"),
        **DOMAINS,
        "All held-out": HELD,
    }
    panels = {k: v for k, v in panels.items() if v and set(v) <= available}
    training_path = root / "training.json"
    training = [r for r in read(training_path) if 0 < r["step"] <= budget] if training_path.exists() else []
    if len({r["step"] for r in training}) != len(training):
        raise ValueError("Duplicate training updates")
    for r in training:
        if r["group"] != GROUPS[(r["step"] - 1) // reuse]:
            raise ValueError("Training log does not follow the declared first-pass batch order")
    for path, kind in [(training_path, "training"), (protocol_path, "protocol")]:
        if path.exists():
            sources.append(dict(kind=kind, path=str(path.resolve()), sha256=file_hash(path)))
    curves = []
    for step, rows in snapshots.items():
        for panel, groups in panels.items():
            for metric in [
                "rkl",
                "js",
                "tv",
                "hp",
                "hq",
                "extra_teacher_mass",
                "extra_student_mass",
                "student_tail_mass",
                "teacher_tail_mass",
            ]:
                value = mean(rows, groups, horizon, metric)
                initial = mean(snapshots[0], groups, horizon, metric)
                if value is not None:
                    curves.append(
                        dict(
                            step=step,
                            panel=panel,
                            metric=metric,
                            value=value,
                            initial=initial,
                            reduction=100 * (1 - value / initial) if initial and initial > 0 else None,
                        )
                    )
    effects, forgetting = [], []
    for stop in range(reuse, budget + 1, reuse):
        if stop not in snapshots or stop - reuse not in snapshots:
            continue
        group = GROUPS[stop // reuse - 1]
        targets = {g: [g] for g in GROUPS if g in available} | {
            k: v for k, v in panels.items() if k in [*DOMAINS, "All held-out"]
        }
        for target, groups in targets.items():
            before = mean(snapshots[stop - reuse], groups, horizon)
            after = mean(snapshots[stop], groups, horizon)
            relation = (
                "held_out"
                if target not in GROUPS
                else "current"
                if target == group
                else "seen"
                if target < group
                else "not_yet_seen"
            )
            effects.append(
                dict(
                    trained=group,
                    target=target,
                    before_step=stop - reuse,
                    step=stop,
                    relation=relation,
                    before=before,
                    after=after,
                    decrease=before - after,
                )
            )
        if group in available:
            initial = mean(snapshots[0], [group], horizon)
            own = mean(snapshots[stop], [group], horizon)
            final = mean(snapshots[budget], [group], horizon)
            forgetting.append(
                dict(group=group, own_step=stop, initial=initial, own=own, final=final, forgetting=final - own)
            )
    questions = []
    for a, b in zip(snapshots[0], snapshots[budget], strict=True):
        questions.append(
            dict(
                id=a["id"],
                group=a["group"],
                length=a["length"],
                initial=a["stats"]["all"]["rkl"] / horizon,
                final=b["stats"]["all"]["rkl"] / horizon,
                decrease=(a["stats"]["all"]["rkl"] - b["stats"]["all"]["rkl"]) / horizon,
            )
        )
    positions = []
    for step in [0, budget]:
        for panel, groups in panels.items():
            for band in ["early", "middle", "late"]:
                chosen = [r for r in snapshots[step] if r["group"] in groups]
                value = mean(chosen, groups, horizon, band=band, observed=True)
                if value is not None:
                    positions.append(
                        dict(
                            step=step,
                            panel=panel,
                            band=band,
                            rkl=value,
                            tokens=sum(r["stats"][band]["tokens"] for r in chosen),
                        )
                    )
    return dict(
        spec=spec,
        horizon=horizon,
        reuse=reuse,
        snapshots=snapshots,
        panels=panels,
        sources=sources,
        training=training,
        curves=curves,
        effects=effects,
        forgetting=forgetting,
        questions=questions,
        positions=positions,
    )
