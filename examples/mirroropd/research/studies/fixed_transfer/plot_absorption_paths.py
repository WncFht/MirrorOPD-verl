# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Fixed-state absorption paths, independently checked against frozen snapshots.

Semantic ledger: a point is one saved update of one run, not a new independent
run. Color identifies teacher; lines join chronological snapshots only. Square,
circle and star identify updates 0, 64 and 320. Risks use prompts times H1024.
Geometry ledger: phase planes use equal numeric ranges on both axes; final-risk
planes are log-log, absorption planes use log10(K0/Kt) on linear axes. Time plots
use optimizer updates and log RKL. No convergence threshold or fit is imposed.
"""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from examples.mirroropd.research.core.paths import REPO
from examples.mirroropd.research.core.plotting import load_style
from examples.mirroropd.research.studies.fixed_transfer.figure_data import DOMAINS, GROUPS, HELD
from examples.mirroropd.research.studies.fixed_transfer.plot_correlations import PANELS, STUDENTS, write_csv

BASE = REPO / "examples/mirroropd/docs/experiments/teacher-transfer"
ATLAS = BASE / "transfer-mechanism/appendix/data/clean-diagnostics"
OUT = BASE / "fixed-state-results/figures/absorption-paths"
SELECTION = {
    "r1": [("justrl", "JustRL (math RL)"), ("minimath", "MiniMath (math SFT)"), ("deepscaler", "DeepScaleR (math RL)")],
    "llama3-3b-off": [("gt", "GT-MATH (RL)"), ("arithmetic-full", "arithmetic-full (SFT)")],
}
GROUP_MAP = {"Training scored": GROUPS, **DOMAINS, "All held-out": HELD}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(student, teacher, label, budget=320):
    paths = list(ATLAS.glob(f"{student}--{teacher}--*/manifest.json"))
    if len(paths) != 1:
        raise ValueError(f"Ambiguous run: {student}/{teacher}")
    path = paths[0]
    manifest = json.loads(path.read_text())
    spec = manifest["spec"]
    metrics = path.parent / "metrics.csv"
    with metrics.open() as stream:
        rows = [r for r in csv.DictReader(stream) if r["band"] == "all" and r["metric"] == "rkl"]
    keyed = {(int(r["step"]), r["panel"]): r for r in rows}
    expected = {(t, p) for t in range(0, budget + 1, 16) for p in PANELS}
    if len(keyed) != len(rows) or set(keyed) != expected:
        raise ValueError(f"Incomplete first-pass grid: {path}")
    snapshots = [s for s in manifest["sources"] if s["kind"] == "snapshot"]
    if sorted(s["step"] for s in snapshots) != list(range(0, budget + 1, 16)):
        raise ValueError("Incomplete source snapshots")
    identity = None
    for source in snapshots:
        raw_path = Path(source["path"])
        if digest(raw_path) != source["sha256"]:
            raise ValueError(f"Source hash changed: {raw_path}")
        raw = json.loads(raw_path.read_text())
        current = sorted((r["id"], r["group"], r["source"], r["length"]) for r in raw)
        if identity is not None and current != identity:
            raise ValueError("Frozen states changed")
        identity = current
        for panel, groups in GROUP_MAP.items():
            selected = [r for r in raw if r["group"] in groups]
            value = sum(r["stats"]["all"]["rkl"] for r in selected) / (len(selected) * 1024)
            saved = float(keyed[source["step"], panel]["fixed_h"])
            if value <= 0 or not np.isclose(value, saved, rtol=1e-12, atol=1e-14):
                raise ValueError(f"Raw risk mismatch: {raw_path}/{panel}")
    values = []
    for (step, panel), r in sorted(keyed.items()):
        initial = float(keyed[0, panel]["fixed_h"])
        risk = float(r["fixed_h"])
        values.append(
            dict(
                run_id=spec["run_id"],
                student=student,
                teacher=teacher,
                label=label,
                step=step,
                panel=panel,
                prompts=int(r["prompts"]),
                initial=initial,
                rkl=risk,
                absolute=initial - risk,
                relative=100 * (1 - risk / initial),
                log_absorption=float(np.log10(initial / risk)),
            )
        )
    provenance = dict(
        spec=spec,
        manifest=str(path.relative_to(REPO)),
        manifest_sha256=digest(path),
        metrics_sha256=digest(metrics),
        sources=manifest["sources"],
        verified_raw_snapshots=len(snapshots),
        frozen_identity_sha256=hashlib.sha256(json.dumps(identity).encode()).hexdigest(),
    )
    return values, provenance


def render(
    student, rows, metric, fv, *, pairs=None, colors=None, styles=None, point_markers=None, output=None, scope=""
):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    time = metric == "time"
    log = metric in ("rkl", "time")
    titles = {
        "rkl": "RKL along training (log-log)",
        "log_absorption": "Absorption: log10(K0 / Kt)",
        "relative": "Relative decrease (%)",
        "absolute": "Absolute decrease (nats)",
        "time": "Fixed-state RKL vs. optimizer updates",
    }
    panels = PANELS if time else PANELS[1:]
    fig, grid = plt.subplots(2, 3 if time else 2, figsize=(16 if time else 14, 10))
    axes = list(grid.flat)
    pairs = pairs if pairs is not None else SELECTION[student]
    bottom = 0.35 if len(pairs) > 5 else 0.22
    fig.subplots_adjust(left=0.09, right=0.97, top=0.84, bottom=bottom, wspace=0.36, hspace=0.65)
    fig.suptitle(f"{STUDENTS[student]} | {titles[metric]}\n{scope}", y=0.96)
    colors = colors if colors is not None else [fv.HUES[k] for k in ("vteal", "vorange", "vviolet")]
    styles = styles or dict.fromkeys((t for t, _ in pairs), "-")
    point_markers = point_markers or {}
    field = "rkl" if time else metric
    values = [r[field] for r in rows]
    bounds = np.log10(values) if log else values
    lo, hi = min(bounds) if log else min(0, min(bounds)), max(bounds)
    span = max(hi - lo, 1e-5)
    limits = lo - 0.08 * span, hi + 0.12 * span
    if log:
        limits = tuple(10**v for v in limits)
    for index, (panel, ax) in enumerate(zip(panels, axes, strict=False)):
        name = "Training" if panel == "Training scored" else "Avg" if panel == "All held-out" else panel
        ax.set(
            title=f"({chr(97 + index)}) {name}",
            xlabel="Optimizer updates" if time else "Training " + titles[metric],
            ylabel="RKL (nats / H1024)" if time else name + " " + titles[metric],
            ylim=limits,
        )
        if log:
            ax.set_yscale("log")
            if not time:
                ax.set_xscale("log")
        if not time:
            ax.set_xlim(limits)
            ax.plot(limits, limits, linestyle="--", color=fv.NEUTRALS["baseline"], linewidth=0.7, label="Equality")
        for (teacher, label), color in zip(pairs, colors, strict=False):
            chosen = sorted(
                [r for r in rows if r["teacher"] == teacher and r["panel"] == panel], key=lambda r: r["step"]
            )
            train = {r["step"]: r for r in rows if r["teacher"] == teacher and r["panel"] == "Training scored"}
            x = [r["step"] if time else train[r["step"]][field] for r in chosen]
            y = [r[field] for r in chosen]
            ax.plot(
                x,
                y,
                color=color,
                linestyle=styles[teacher],
                marker=point_markers.get(teacher, "."),
                markersize=4,
                linewidth=1,
                label=label,
            )
            for step, marker, size in [(0, "s", 7), (64, "o", 7), (320, "*", 12)]:
                if step not in train:
                    continue
                if teacher in point_markers and step > 0:
                    marker = point_markers[teacher]
                    size = 8 if step == 320 else 6
                at = next(i for i, r in enumerate(chosen) if r["step"] == step)
                ax.plot(
                    x[at],
                    y[at],
                    linestyle="none",
                    marker=marker,
                    markersize=size,
                    markerfacecolor=fv.NEUTRALS["paper"] if step == 0 else color,
                    color=color,
                )
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[len(panels) :]:
        ax.set_visible(False)
    handles = [
        Line2D([], [], color=c, linestyle=styles[t], marker=point_markers.get(t, "."), label=label)
        for (t, label), c in zip(pairs, colors, strict=False)
    ]
    if not time:
        handles.append(Line2D([], [], linestyle="--", color=fv.NEUTRALS["baseline"], label="Equality"))
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.10),
        ncol=2 if len(handles) > 5 else len(handles),
        frameon=False,
        fontsize=9,
    )
    fig.text(
        0.5,
        0.065,
        (
            "Each marker = one snapshot; square: 0; enlarged: 64 / 320; circles: same size; diamonds: cross size."
            if point_markers
            else "Each dot = one snapshot, every 16 updates; square: 0; large circle: 64; star: 320."
        ),
        ha="center",
        color=fv.NEUTRALS["baseline"],
        fontsize=9,
    )
    fig.text(
        0.5,
        0.035,
        "Lines follow time within one run; snapshots are dependent, so no cross-run correlation is estimated.",
        ha="center",
        color=fv.NEUTRALS["baseline"],
        fontsize=9,
    )
    fig.canvas.draw()
    issues = fv.audit(fig, print_slot=fig.get_figwidth())
    stem = (output or OUT) / f"{student}-{metric}"
    Path(str(stem) + ".audit.json").write_text(json.dumps([str(p) for p in issues], indent=2) + "\n")
    if any(fv.problem_severity(p) == "error" for p in issues):
        raise RuntimeError(str(issues))
    for ext in ("png", "pdf"):
        fig.savefig(Path(str(stem) + "." + ext), dpi=160)
    plt.close(fig)


def main():
    import matplotlib

    matplotlib.use("Agg")
    fv = load_style()
    fv.apply_rc("paper", fontsize=10)
    OUT.mkdir(parents=True, exist_ok=True)
    rows, sources = [], []
    for student, teachers in SELECTION.items():
        for teacher, label in teachers:
            values, source = load(student, teacher, label)
            rows.extend(values)
            sources.append(source)
        identities = {s["frozen_identity_sha256"] for s in sources if s["spec"]["student"] == student}
        if len(identities) != 1:
            raise ValueError(f"Different frozen banks within student group: {student}")
        selected = [r for r in rows if r["student"] == student]
        for metric in ("log_absorption", "rkl", "relative", "absolute", "time"):
            render(student, selected, metric, fv)
    write_csv(OUT / "snapshots.csv", rows)
    write_csv(OUT / "endpoints.csv", [r for r in rows if r["step"] in (0, 64, 256, 320)])
    context, context_sources = r1_context()
    write_csv(OUT / "r1-context.csv", context)
    (OUT / "provenance.json").write_text(
        json.dumps(dict(runs=len(sources), figures=10, sources=sources, r1_context_sources=context_sources), indent=2)
        + "\n"
    )
    print(f"Verified {sum(s['verified_raw_snapshots'] for s in sources)} raw snapshots; wrote 10 figures.")


def r1_context():
    """Publish common-step context, including the runs available only to step64."""
    rows, sources = [], []
    for path in sorted(ATLAS.glob("r1--*/metrics.csv")):
        with path.open() as stream:
            selected = [r for r in csv.DictReader(stream) if r["band"] == "all" and r["metric"] == "rkl"]
        initial = {r["panel"]: float(r["fixed_h"]) for r in selected if int(r["step"]) == 0}
        for r in selected:
            if int(r["step"]) not in (0, 64, 256, 320):
                continue
            risk, start = float(r["fixed_h"]), initial[r["panel"]]
            rows.append(
                dict(
                    run_id=r["run_id"],
                    teacher=r["teacher"],
                    step=int(r["step"]),
                    panel=r["panel"],
                    initial=start,
                    rkl=risk,
                    relative=100 * (1 - risk / start),
                )
            )
        sources.append(dict(path=str(path.relative_to(REPO)), sha256=digest(path)))
    return rows, sources


if __name__ == "__main__":
    main()
