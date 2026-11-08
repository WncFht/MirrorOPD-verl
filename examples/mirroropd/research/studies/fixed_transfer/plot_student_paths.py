# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""All-teacher path atlas. Semantic/geometry ledger follows plot_absorption_paths.

Color is teacher identity, stable across scopes. In same-size views dashed means
an added SFT stage; solid means RL without an added SFT stage. Base/OPD and
unknown public histories are dotted controls, never forced into an RL class.
In all-size views circle/diamond means same/cross size. Dots are dependent times.
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from examples.mirroropd.research.core.plotting import load_style
from examples.mirroropd.research.studies.fixed_transfer.plot_absorption_paths import BASE, load, render
from examples.mirroropd.research.studies.fixed_transfer.plot_correlations import (
    SOURCE,
    STUDENTS,
    load_points,
    write_csv,
)

OUT = BASE / "fixed-state-results/figures/student-paths"
SFT = {"minimath", "light", "sft-gsm8k", "sft-deepmath", "arithmetic-full", "Q1", "Q2", "L1", "L2", "L3"}
RL = {
    "justrl",
    "deepscaler",
    "nemotron",
    "brorl",
    "deepcoder",
    "e3",
    "grpo-kl",
    "gt",
    "coreward-i",
    "coreward-ii",
    "base-gt-math",
    "base-grpo-4",
    "skywork",
    "L4",
}


def stage(teacher):
    if teacher in SFT:
        return "added SFT"
    if teacher in RL:
        return "RL; no added SFT"
    return "Base/OPD/public-history control"


def main():
    import matplotlib
    from matplotlib.colors import to_rgb

    matplotlib.use("Agg")
    fv = load_style()
    fv.apply_rc("paper", fontsize=10)
    OUT.mkdir(parents=True, exist_ok=True)
    groups = defaultdict(list)
    for p in load_points(SOURCE):
        groups[p["student"]].append(p)
    allrows, sources, inventory = [], [], []
    for student in STUDENTS:
        points = sorted(groups[student], key=lambda p: p["teacher_label"])
        rows = []
        colors = {}
        for i, p in enumerate(points):
            label = p["teacher_label"]
            values, source = load(student, p["teacher"], label, p["step"])
            rows.extend(values)
            allrows.extend(values)
            sources.append(source)
            color = tuple(
                np.array(to_rgb(fv.HUES[["vteal", "vorange", "vcoral", "vviolet"][i % 4]])) * (1 - 0.17 * (i // 4))
            )
            colors[p["teacher"]] = color
            inventory.append(
                dict(
                    student=student,
                    teacher=p["teacher"],
                    label=label,
                    same_size=p["same_size"],
                    added_stage=stage(p["teacher"]),
                    budget=p["step"],
                    run_id=p["run_id"],
                )
            )
        fv.register_colors(*colors.values())
        shared = {}
        for source in sources:
            if source["spec"]["student"] != student:
                continue
            raw = json.loads(Path(source["spec"]["initial_source"]).read_text())
            for r in raw:
                identity = (r["group"], r["source"], r["length"])
                if r["id"] in shared and shared[r["id"]] != identity:
                    raise ValueError(f"Conflicting shared frozen state: {student}/{r['id']}")
                shared[r["id"]] = identity
        for scope in ("same-size", "all-size"):
            chosen = [p for p in points if scope == "all-size" or p["same_size"]]
            if not chosen:
                continue
            target = OUT / scope
            target.mkdir(exist_ok=True)
            pairs, styles = [], {}
            for p in chosen:
                teacher = p["teacher"]
                category = stage(teacher) if scope == "same-size" else "same size" if p["same_size"] else "cross size"
                tag = {
                    "added SFT": "SFT",
                    "RL; no added SFT": "RL",
                    "Base/OPD/public-history control": "other",
                    "same size": "same",
                    "cross size": "cross",
                }[category]
                pairs.append((teacher, f"{teacher} [{p['step']}; {tag}]"))
                styles[teacher] = (
                    {"added SFT": "--", "RL; no added SFT": "-", "Base/OPD/public-history control": ":"}[category]
                    if scope == "same-size"
                    else "-"
                )
            selected = [r for r in rows if r["teacher"] in styles]
            caption = (
                "Same size: solid = RL, dashed = added SFT, dotted = other controls"
                if scope == "same-size"
                else "All teachers: circles = same size, diamonds = cross size; all lines solid"
            )
            for metric in ("log_absorption", "rkl", "time"):
                render(
                    student,
                    selected,
                    metric,
                    fv,
                    pairs=pairs,
                    colors=[colors[t] for t, _ in pairs],
                    styles=styles,
                    point_markers={p["teacher"]: "o" if p["same_size"] else "D" for p in chosen}
                    if scope == "all-size"
                    else None,
                    output=target,
                    scope=caption,
                )
    write_csv(OUT / "snapshots.csv", allrows)
    write_csv(OUT / "inventory.csv", inventory)
    figures = len(list(OUT.glob("*/*.png")))
    (OUT / "provenance.json").write_text(
        json.dumps(
            dict(
                runs=len(sources),
                figures=figures,
                verified_raw_snapshots=sum(s["verified_raw_snapshots"] for s in sources),
                sources=sources,
            ),
            indent=2,
        )
        + "\n"
    )
    print(f"{len(sources)} runs; {figures} figures; all raw snapshots verified.")


if __name__ == "__main__":
    main()
