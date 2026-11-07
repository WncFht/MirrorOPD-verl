#!/usr/bin/env python3
"""Render separate MiMo and Polaris training/position-RKL diagnostics.

Semantic ledger: a row is one logged pre-optimizer forward on the current
rollout batch. A position series is a token-mean native full-vocabulary RKL
in one relative response decile; it is not a fixed-bank position readout.
First/last select the first/last logged observation in EACH outer, retaining
its actual optimizer update ID. Last precedes the final optimizer operation.
Geometry ledger: heatmap columns are equally spaced outer IDs (or updates
for the all-update map), rows are ten equally sized relative-position bins.
Every heatmap figure uses one color range across all of its panels. MiMo and
Polaris remain separate figures because their KL magnitudes are different.
Line facets share a log y range within an arm; late zoom keeps the last five
outers with no smoothing. MiMo and Polaris never share a figure.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[6]
STYLE = ROOT.parent / "dotfiles/.agents/skills/figure-viz/assets/mpl/fvstyle.py"
spec = importlib.util.spec_from_file_location("fvstyle", STYLE)
fv = importlib.util.module_from_spec(spec)

sys.modules[spec.name] = fv
spec.loader.exec_module(fv)
DOCS = ROOT / "examples/mirroropd/docs/experiments/teacher-transfer"
RUNS = ROOT / "runs/mimo-stage-transfer-20261101/formal"
OUT = DOCS / "figures/mimo-position-kl-20261106"
POUT = DOCS / "figures/polaris-position-kl-20261106"
DATA_OUT = DOCS / "data/mimo-position-kl-20261106"
PDATA_OUT = DOCS / "data/polaris-position-kl-20261106"
ARMS = {
    "E71": ("MiMo-7B-RL", "SFT+RL", 32, 25),
    "E72": ("MiMo-7B-RL-Zero", "Direct RL", 16, 20),
    "E73": ("MiMo-7B-SFT", "SFT", 16, 20),
}
POLARIS_HISTORY = DOCS / "cross-origin/polaris/data/polaris-complete-20261027/run/history.jsonl"
BINS = [f"p{i:02d}_{i + 10:02d}" for i in range(0, 100, 10)]
BIN_LABELS = [f"{i}–{i + 10}%" for i in range(0, 100, 10)]
AUDITS = {}


def title(arm):
    if arm == "E48":
        return "E48 · Teacher: Polaris-7B-Preview"
    teacher, stage, _, _ = ARMS[arm]
    return f"{arm} · Teacher: {teacher} ({stage})"


def read_history(path, nested):
    rows = []
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if nested:
            row = row["data"]
        if "step/update" in row and "distill/full_kl" in row:
            rows.append(row)
    rows.sort(key=lambda r: int(r["step/update"]))
    return rows


def group_outer(rows, reuse, count):
    assert [int(r["step/update"]) for r in rows] == list(range(1, reuse * count + 1))
    groups = defaultdict(list)
    for row in rows:
        outer = int(row["step/outer"])
        groups[outer].append(row)
        assert outer == (int(row["step/update"]) - 1) // reuse + 1
        for bucket in BINS:
            assert np.isfinite(float(row[f"distill/full_kl_position/{bucket}/mean"]))
    assert sorted(groups) == list(range(1, count + 1))
    assert all(len(group) == reuse for group in groups.values())
    return groups


def finish(fig, directory, stem, caption):
    fig.text(0.5, 0.012, caption, ha="center", va="bottom", fontsize=8, color=fv.NEUTRALS["baseline"])
    fig.canvas.draw()
    problems = fv.audit(fig, print_slot=fig.get_figwidth())
    lines = fv.format_audit_lines(problems)
    AUDITS[f"{directory.name}/{stem}"] = lines
    (directory / f"{stem}.audit.txt").write_text(
        "\n".join(lines) + "\n" if lines else "PASS: fv.audit(fig) returned no issues.\n"
    )
    if any(fv.problem_severity(p) == "error" for p in problems):
        raise RuntimeError(f"Figure audit failed for {stem}: {lines}")
    for ext in ("png", "pdf", "svg"):
        fig.savefig(directory / f"{stem}.{ext}", dpi=180, facecolor=fv.NEUTRALS["paper"])
    plt.close(fig)


def figure(rows, cols, height=2.65):
    fig, axes = plt.subplots(rows, cols, figsize=(12 if cols == 2 else 10.5, height * rows + 1.5), squeeze=False)
    fig.subplots_adjust(
        left=0.09,
        right=0.93,
        bottom=0.20 if rows == 1 else 0.12,
        top=0.82 if rows == 1 else 0.88,
        hspace=0.52,
        wspace=0.24,
    )
    return fig, axes


def matrix(rows):
    return np.array([[float(row[f"distill/full_kl_position/{bucket}/mean"]) for row in rows] for bucket in BINS])


def outer_views(groups, late):
    outers = sorted(groups)
    if late:
        outers = outers[-5:]
    return outers, [[groups[o][index] for o in outers] for index in (0, -1)]


def heatmaps(data, directory, prefix, late=False, all_updates=False):
    fig, axes = figure(len(data), 1 if all_updates else 2)
    snapshots = {}
    all_values = []
    for arm, groups in data.items():
        if all_updates:
            views = [[r for group in groups.values() for r in group]]
        else:
            _, views = outer_views(groups, late)
        mats = [matrix(view) for view in views]
        snapshots[arm] = (views, mats)
        all_values.extend(float(mat.max()) for mat in mats)
    shared_vmax = max(all_values)
    image = None
    for axrow, (arm, groups) in zip(axes, data.items(), strict=True):
        if all_updates:
            views = snapshots[arm][0]
            xs = [int(r["step/update"]) for r in views[0]]
        else:
            xs, views = outer_views(groups, late)
        mats = snapshots[arm][1]
        for col, (ax, mat) in enumerate(zip(axrow, mats, strict=True)):
            image = ax.imshow(
                mat,
                origin="lower",
                aspect="auto",
                interpolation="nearest",
                cmap="viridis",
                vmin=0,
                vmax=shared_vmax,
                extent=(xs[0] - 0.5, xs[-1] + 0.5, -0.5, 9.5),
            )
            ax.set_yticks(range(10), BIN_LABELS, fontsize=8)
            ticks = xs if len(xs) <= 8 else sorted(set(round(x) for x in np.linspace(xs[0], xs[-1], 6)))
            ax.set_xticks(ticks)
            ax.set_xlabel("Optimizer update" if all_updates else "Outer (one observation per rollout batch)")
            ax.set_ylabel("Relative response position")
            which = (
                "All pre-update observations"
                if all_updates
                else ("First update observation" if col == 0 else "Last update observation")
            )
            ax.set_title(f"{title(arm)}\n{which}", fontsize=10, loc="left")
            if late:
                ax.set_xticks(np.arange(xs[0] - 0.5, xs[-1] + 1, 1), minor=True)
                ax.set_yticks(np.arange(-0.5, 10, 1), minor=True)
                ax.grid(which="minor", color=fv.NEUTRALS["paper"], lw=0.5, alpha=0.35)
                ax.tick_params(which="minor", length=0)
    assert image is not None
    cb = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.85, pad=0.02)
    cb.set_label("Position-bin RKL (shared within figure)", fontsize=9)
    qualifier = "all optimizer updates" if all_updates else ("last 5 outers" if late else "all outers")
    fig.suptitle(f"{'MiMo' if prefix == 'mimo' else 'Polaris'} position RKL · {qualifier}", y=0.99)
    stem = (
        f"{prefix}-position-kl-heatmap"
        if all_updates
        else f"{prefix}-position-kl-first-last-heatmap{'-late' if late else ''}"
    )
    finish(
        fig,
        directory,
        stem,
        "Token-mean RKL on current rollout states; one color scale is shared by every panel in this figure. "
        "Last is BEFORE the final optimizer operation.",
    )


def single_snapshot_figures(data, directory, prefix):
    fig, axes = figure(len(data), 2)
    snapshots = {}
    all_values = []
    for arm, groups in data.items():
        first = groups[min(groups)][0]
        last = groups[max(groups)][-1]
        profiles = [matrix([first])[:, 0], matrix([last])[:, 0]]
        snapshots[arm] = (first, last, profiles)
        all_values.extend(float(values.max()) for values in profiles)
    shared_vmax = max(all_values)
    image = None
    for axrow, (arm, groups) in zip(axes, data.items(), strict=True):
        first, last, profiles = snapshots[arm]
        for ax, row, values, label in zip(
            axrow,
            (first, last),
            profiles,
            ("First outer", "Final outer"),
            strict=True,
        ):
            image = ax.imshow(
                values[None, :],
                origin="lower",
                aspect="auto",
                interpolation="nearest",
                cmap="viridis",
                vmin=0,
                vmax=shared_vmax,
            )
            ax.set_xticks(range(10), BIN_LABELS, rotation=35, ha="right", fontsize=8)
            ax.set_yticks([])
            ax.set_title(f"{title(arm)}\n{label} · optimizer update {row['step/update']}", fontsize=10, loc="left")
    assert image is not None
    cb = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.85, pad=0.02)
    cb.set_label("Position-bin RKL (shared within figure)", fontsize=9)
    fig.suptitle(f"{'MiMo' if prefix == 'mimo' else 'Polaris'} · single-update position RKL", y=0.99)
    finish(
        fig,
        directory,
        f"{prefix}-position-kl-single-update-heatmap",
        "Rows are relative response-position deciles. One color scale is shared by every panel in this figure; "
        "final is pre-update.",
    )

    fig, axes = figure(len(data), 2)
    for axrow, (arm, groups) in zip(axes, data.items(), strict=True):
        first = groups[min(groups)][0]
        last = groups[max(groups)][-1]
        profiles = [matrix([first])[:, 0], matrix([last])[:, 0]]
        ymin = min(float(values.min()) for values in profiles) * 0.85
        ymax = max(float(values.max()) for values in profiles) * 1.15
        for ax, row, values, label in zip(
            axrow,
            (first, last),
            profiles,
            ("First outer", "Final outer"),
            strict=True,
        ):
            ax.plot(range(10), values, marker="o", lw=1.4, color=fv.HUES["vteal"])
            ax.set_yscale("log")
            ax.set_ylim(ymin, ymax)
            ax.set_xticks(range(10), BIN_LABELS, rotation=35, ha="right", fontsize=8)
            ax.set_ylabel("Position-bin RKL (log scale)")
            ax.set_title(f"{title(arm)}\n{label} · optimizer update {row['step/update']}", fontsize=10, loc="left")
            ax.grid(True, which="major", alpha=0.2)
    fig.suptitle(f"{'MiMo' if prefix == 'mimo' else 'Polaris'} · single-update position profile", y=0.99)
    finish(
        fig,
        directory,
        f"{prefix}-position-kl-single-update-lines",
        "Each point is one relative response-position decile on that update's current rollout batch. "
        "Final is pre-update.",
    )


def position_update_lines(data, directory, prefix):
    fig, axes = figure(len(data), 1)
    hues = [fv.HUES[c] for c in ("vteal", "vorange", "vviolet", "vblue")]
    styles = [(hues[i % 4], ("-", "--", ":")[i // 4], ("o", "s", "^")[i // 4]) for i in range(10)]
    for ax, (arm, groups) in zip(axes[:, 0], data.items(), strict=True):
        rows = [row for outer in sorted(groups) for row in groups[outer]]
        for bucket, label, (color, ls, marker) in zip(BINS, BIN_LABELS, styles, strict=True):
            ax.plot(
                [int(row["step/update"]) for row in rows],
                [float(row[f"distill/full_kl_position/{bucket}/mean"]) for row in rows],
                color=color,
                ls=ls,
                marker=marker,
                markevery=max(1, len(rows) // 24),
                ms=2.8,
                lw=1.0,
                label=label,
            )
        ax.set_yscale("log")
        ax.set(
            xlabel="Optimizer update (pre-update observation)",
            ylabel="Position-bin RKL (log scale)",
            title=title(arm),
        )
        ax.grid(True, which="major", alpha=0.2)
    handles = [
        Line2D([], [], color=color, ls=ls, marker=marker, ms=4, label=label)
        for (color, ls, marker), label in zip(styles, BIN_LABELS, strict=True)
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=5, frameon=False, fontsize=9)
    finish(
        fig,
        directory,
        f"{prefix}-position-kl-by-decile-updates",
        "Ten relative response-position deciles are separate curves. "
        "Each value is measured on that update's rollout batch.",
    )


def position_lines(data, directory, prefix, late=False):
    fig, axes = figure(len(data), 2)
    hues = [fv.HUES[c] for c in ("vteal", "vorange", "vviolet", "vblue")]
    styles = [(hues[i % 4], ("-", "--", ":")[i // 4], ("o", "s", "^")[i // 4]) for i in range(10)]
    for axrow, (arm, groups) in zip(axes, data.items(), strict=True):
        xs, views = outer_views(groups, late)
        mats = [matrix(view) for view in views]
        ymin = min(float(m.min()) for m in mats) * 0.85
        ymax = max(float(m.max()) for m in mats) * 1.18
        assert ymin > 0
        for col, (ax, mat) in enumerate(zip(axrow, mats, strict=True)):
            for values, (color, ls, marker), label in zip(mat, styles, BIN_LABELS, strict=True):
                ax.plot(xs, values, color=color, ls=ls, marker=marker, ms=3.2, lw=1.15, label=label)
            ax.set_yscale("log")
            ax.set_ylim(ymin, ymax)
            ax.set_ylabel("Position-bin RKL (log scale)")
            ax.set_xlabel("Outer (one observation per rollout batch)")
            ax.set_xticks(xs if late else sorted(set(round(x) for x in np.linspace(xs[0], xs[-1], 6))))
            which = "First update observation" if col == 0 else "Last update observation"
            ax.set_title(f"{title(arm)}\n{which}", fontsize=10, loc="left")
            ax.grid(True, which="major", alpha=0.2)
    handles = [
        Line2D([], [], color=color, ls=ls, marker=marker, ms=4, label=label)
        for (color, ls, marker), label in zip(styles, BIN_LABELS, strict=True)
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=5, frameon=False, fontsize=9)
    stem = f"{prefix}-position-kl-first-last-lines{'-late' if late else ''}"
    finish(
        fig,
        directory,
        stem,
        "One curve per relative-position decile; first/last share a y range within each arm. "
        "Last is BEFORE the final optimizer operation.",
    )


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def export(data, directory, paths):
    positions, losses, endpoints = [], [], []
    reconstruction = []
    for arm, groups in data.items():
        for outer, rows in groups.items():
            for index, row in enumerate(rows):
                meta = dict(
                    run=arm,
                    teacher=ARMS[arm][0] if arm in ARMS else "Polaris-7B-Preview",
                    outer=outer,
                    update=int(row["step/update"]),
                    update_in_outer=index + 1,
                )
                losses.append(meta | {"loss": float(row["distill/full_kl"])})
                for bucket in BINS:
                    positions.append(
                        meta | dict(bucket=bucket, rkl_mean=float(row[f"distill/full_kl_position/{bucket}/mean"]))
                    )
                total = sum(float(row[f"distill/full_kl_position/{b}/sum"]) for b in BINS)
                count = sum(float(row[f"distill/full_kl_position/{b}/token_count"]) for b in BINS)
                reconstruction.append(abs(total / count - float(row["distill/full_kl"])))
            for selection, row in (("first", rows[0]), ("last", rows[-1])):
                endpoints.append(
                    dict(
                        run=arm,
                        outer=outer,
                        selection=selection,
                        update=int(row["step/update"]),
                        loss=float(row["distill/full_kl"]),
                    )
                    | {b: float(row[f"distill/full_kl_position/{b}/mean"]) for b in BINS}
                )
    assert max(reconstruction) < 1e-6
    write_csv(directory / "loss-before-update.csv", losses)
    write_csv(directory / "position-kl.csv", positions)
    write_csv(directory / "outer-first-last.csv", endpoints)
    manifest = dict(
        sources={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        selection="first/last logged pre-update forward within each actual outer",
        late_zoom="last five actual outers",
        max_position_reconstruction_error=max(reconstruction),
        counts={arm: dict(outers=len(groups), updates=sum(map(len, groups.values()))) for arm, groups in data.items()},
    )
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return endpoints


def training_figures(data, directory, prefix):
    fig, axes = figure(len(data), 1, height=2.3)
    for ax, (arm, groups) in zip(axes[:, 0], data.items(), strict=True):
        rows = [r for group in groups.values() for r in group]
        ax.plot(
            [int(r["step/update"]) for r in rows],
            [float(r["distill/full_kl"]) for r in rows],
            color=fv.HUES["vteal"],
            lw=0.9,
        )
        ax.set(
            xlabel="Optimizer update (pre-update observation)", ylabel="Training RKL", yscale="log", title=title(arm)
        )
        ax.grid(alpha=0.2)
    finish(
        fig,
        directory,
        "loss-before-update",
        "All logged pre-update forwards are shown; each outer reuses one fixed rollout batch.",
    )


def math_figures(data):
    fig, axes = figure(3, 1)
    exported = []
    for ax, (arm, groups) in zip(axes[:, 0], data.items(), strict=True):
        teacher, _, reuse, _ = ARMS[arm]
        xs = sorted(groups)
        for selection, index, color, ls in (("Train first", 0, "vteal", "-"), ("Train last", -1, "vorange", "--")):
            ax.plot(
                xs,
                [float(groups[o][index]["distill/full_kl"]) for o in xs],
                color=fv.HUES[color],
                ls=ls,
                lw=1.2,
                label=selection,
            )
        path = DOCS / f"data/mimo-stage-20261104-final/{arm}/fixed-bank-rkl.csv"
        with path.open() as stream:
            rows = list(csv.DictReader(stream))
        for bank, label, color, marker in (
            ("public_initial", "Fixed Base Math", "vblue", "s"),
            ("public_teacher", "Fixed teacher Math", "vviolet", "o"),
        ):
            own = sorted(
                [r for r in rows if r["bank"] == bank and r["domain"] == "math"],
                key=lambda r: int(r["optimizer_updates"]),
            )
            ax.plot(
                [int(r["optimizer_updates"]) / reuse for r in own],
                [float(r["rkl"]) for r in own],
                color=fv.HUES[color],
                marker=marker,
                ms=4,
                lw=1.2,
                label=label,
            )
            exported.extend(
                dict(
                    run=arm,
                    teacher=teacher,
                    bank=bank,
                    update=int(r["optimizer_updates"]),
                    outer=int(r["optimizer_updates"]) // reuse,
                    math_rkl=float(r["rkl"]),
                )
                for r in own
            )
        ax.set(
            xlabel="Outer (fixed banks: completed outers; train: within outer)",
            ylabel="RKL (log scale)",
            yscale="log",
            title=title(arm),
        )
        ax.grid(alpha=0.2)
    fig.legend(
        *axes[0, 0].get_legend_handles_labels(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=4,
        frameon=False,
        fontsize=9,
    )
    finish(
        fig,
        OUT,
        "math-bank-rkl",
        "Training: token mean on changing rollout states. Fixed Math banks: question mean, "
        "scored after the marked completed outer; no interpolation of missing readouts.",
    )
    write_csv(DATA_OUT / "math-bank-rkl.csv", exported)


def main():
    fv.apply_rc("paper", fontsize=10)
    for path in (OUT, POUT, DATA_OUT, PDATA_OUT):
        path.mkdir(parents=True, exist_ok=True)
    mimo = {
        arm: group_outer(read_history(RUNS / arm / "metrics.jsonl", True), reuse, count)
        for arm, (_, _, reuse, count) in ARMS.items()
    }
    polaris = {"E48": group_outer(read_history(POLARIS_HISTORY, False), 16, 256)}
    paths = [RUNS / arm / "metrics.jsonl" for arm in ARMS] + [
        DOCS / f"data/mimo-stage-20261104-final/{arm}/fixed-bank-rkl.csv" for arm in ARMS
    ]
    export(mimo, DATA_OUT, paths)
    export(polaris, PDATA_OUT, [POLARIS_HISTORY])
    for data, directory, prefix in ((mimo, OUT, "mimo"), (polaris, POUT, "polaris")):
        training_figures(data, directory, prefix)
        heatmaps(data, directory, prefix, all_updates=True)
        single_snapshot_figures(data, directory, prefix)
        position_update_lines(data, directory, prefix)
        for late in (False, True):
            heatmaps(data, directory, prefix, late=late)
            position_lines(data, directory, prefix, late=late)
    math_figures(mimo)
    for directory in (OUT, POUT):
        (directory / "figure-audit.json").write_text(
            json.dumps({k: v for k, v in AUDITS.items() if k.startswith(directory.name + "/")}, indent=2) + "\n"
        )
    print(f"Rendered {len(AUDITS)} figures (PNG/PDF/SVG); data and position reconstruction checks passed.")


if __name__ == "__main__":
    main()
