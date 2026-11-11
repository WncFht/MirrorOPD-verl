#!/usr/bin/env python3
"""Render separate MiMo, Qwen and R1-1.5B position-RKL figures.

Semantic ledger: each heatmap cell is one native response-decile token-mean
RKL on the current rollout batch, observed before an optimizer operation.
First/last select the first/last observed update in each outer, not an extra
post-update forward. E49 logs raw RKL although its objective is conditional
FKL. Each decile has one fixed color, line style and marker across figures.
Geometry ledger: heatmap columns are equally spaced actual outer IDs; rows
are ten equally sized relative-position bins. All panels within a heatmap
share one linear color scale. Line panels share a log y range within an arm;
different arms use their own range to preserve small residual differences.
Early/late select five actual outers without smoothing or interpolation.
Qwen E39/E41/E43 each get their own figure; no Qwen figure mixes runs.
"""

from __future__ import annotations

import argparse
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
WORKSPACE = ROOT.parent
DOCS = ROOT / "examples/mirroropd/docs/experiments/teacher-transfer"
E49_HISTORY = WORKSPACE / "wandb-logs/derived/MirrorOPD-verl/offline-run-20261027_175635-zyru8wdf/history.jsonl"
STYLE = WORKSPACE / "dotfiles/.agents/skills/figure-viz/assets/mpl/fvstyle.py"
spec = importlib.util.spec_from_file_location("fvstyle", STYLE)
fv = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = fv
spec.loader.exec_module(fv)

BINS = [f"p{i:02d}_{i + 10:02d}" for i in range(0, 100, 10)]
LABELS = [f"{i}–{i + 10}%" for i in range(0, 100, 10)]
AUDITS: dict[str, list[str]] = {}


def read_json_rows(path: Path, nested: bool = False) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if nested:
            row = row["data"]
        if "distill/full_kl_position/p00_10/mean" in row:
            rows.append(row)
    return rows


def position_row(outer: int, reuse: int, means: dict[str, float], full_kl: float | None = None) -> dict:
    row = {"step/outer": outer, "step/update": (outer - 1) * 16 + reuse}
    for bucket, value in means.items():
        row[f"distill/full_kl_position/{bucket}/mean"] = value
    if full_kl is None:
        full_kl = float(np.mean(list(means.values())))
    row["distill/full_kl"] = full_kl
    return row


def from_endpoint_csv(path: Path) -> dict[int, list[dict]]:
    groups: dict[int, list[dict]] = defaultdict(list)
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    for outer in sorted({int(r["outer"]) for r in rows}):
        for endpoint in ("first", "last"):
            selected = [r for r in rows if int(r["outer"]) == outer and r["endpoint"] == endpoint]
            means = {r["bin"]: float(r["mean"]) for r in selected}
            total = sum(float(r["sum"]) for r in selected)
            count = sum(float(r["token_count"]) for r in selected)
            groups[outer].append(position_row(outer, 1 if endpoint == "first" else 16, means, total / count))
    return groups


def from_detail_csv(path: Path, experiment: str) -> dict[int, list[dict]]:
    groups: dict[int, list[dict]] = defaultdict(list)
    with path.open(newline="") as stream:
        rows = [r for r in csv.DictReader(stream) if r["experiment"] == experiment]
    for outer in sorted({int(r["outer"]) for r in rows}):
        for reuse in (1, 16):
            selected = [r for r in rows if int(r["outer"]) == outer and int(r["reuse"]) == reuse]
            means = {r["bin"]: float(r["mean"]) for r in selected}
            total = sum(float(r["total"]) for r in selected)
            count = sum(float(r["count"]) for r in selected)
            groups[outer].append(position_row(outer, reuse, means, total / count))
    return groups


def from_history(path: Path, nested: bool, reuse: int, count: int) -> dict[int, list[dict]]:
    rows = read_json_rows(path, nested)
    rows.sort(key=lambda r: int(r.get("step/update", r.get("step/optimizer", 0))))
    groups: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        outer = int(row.get("step/outer", (int(row["step/update"]) - 1) // reuse + 1))
        groups[outer].append(row)
    assert sorted(groups) == list(range(1, count + 1))
    assert all(len(rows) == reuse for rows in groups.values())
    return groups


def matrix(rows: list[dict]) -> np.ndarray:
    return np.array([[float(row[f"distill/full_kl_position/{bucket}/mean"]) for row in rows] for bucket in BINS])


def views(groups: dict[int, list[dict]], late: bool, early: bool = False) -> tuple[list[int], list[list[dict]]]:
    outers = sorted(groups)
    if late:
        outers = outers[-5:]
    elif early:
        outers = outers[:5]
    return outers, [[groups[o][index] for o in outers] for index in (0, -1)]


def label(case: str, arm: str) -> str:
    teachers = {
        "E71": "MiMo-7B-RL",
        "E72": "MiMo-7B-RL-Zero",
        "E73": "MiMo-7B-SFT",
        "E91": "MiMo-7B-RL-Zero → SFT student",
        "E93": "MiMo-7B-RL → SFT student",
        "E35": "JustRL-DeepSeek-1.5B",
        "E39": "Qwen3-8B",
        "E41": "Qwen3-4B",
        "E43": "Qwen3-4B-Instruct-2507",
        "E48": "Polaris-7B-Preview (RKL)",
        "E49": "Polaris-7B-Preview (conditional FKL; raw RKL diagnostic)",
    }
    return f"{arm} · Teacher: {teachers[arm]}"


def finish(fig: plt.Figure, out: Path, stem: str, caption: str) -> None:
    fig.text(0.5, 0.012, caption, ha="center", va="bottom", fontsize=8, color=fv.NEUTRALS["baseline"])
    fig.canvas.draw()
    problems = fv.audit(fig, print_slot=fig.get_figwidth())
    lines = fv.format_audit_lines(problems)
    AUDITS[f"{out.name}/{stem}"] = lines
    (out / f"{stem}.audit.txt").write_text(
        "\n".join(lines) + "\n" if lines else "PASS: fv.audit(fig) returned no issues.\n"
    )
    if any(fv.problem_severity(problem) == "error" for problem in problems):
        raise RuntimeError(f"figure audit failed: {stem}: {lines}")
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out / f"{stem}.{ext}", dpi=180, facecolor=fv.NEUTRALS["paper"])
    plt.close(fig)


def render_group(case: str, data: dict[str, dict[int, list[dict]]], out: Path, prefix: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    arm_items = list(data.items())
    for zoom_name, late, early in (("all", False, False), ("early", False, True), ("late", True, False)):
        selected = {arm: views(groups, late, early) for arm, groups in arm_items}
        global_vmax = max(
            float(mat.max())
            for _, (_, panel_views) in selected.items()
            for panel in panel_views
            for mat in [matrix(panel)]
        )
        fig, axes = plt.subplots(len(arm_items), 2, figsize=(14, max(3.5 * len(arm_items), 5.2)), squeeze=False)
        fig.subplots_adjust(
            left=0.10,
            right=0.91,
            bottom=0.20,
            top=0.80 if len(arm_items) == 1 else 0.91,
            hspace=0.52,
            wspace=0.25,
        )
        image = None
        for row_index, (arm, (xs, panel_views)) in enumerate(selected.items()):
            for col, (ax, panel_rows) in enumerate(zip(axes[row_index], panel_views, strict=True)):
                image = ax.imshow(
                    matrix(panel_rows),
                    origin="lower",
                    aspect="auto",
                    interpolation="nearest",
                    cmap="viridis",
                    vmin=0,
                    vmax=global_vmax,
                    extent=(xs[0] - 0.5, xs[-1] + 0.5, -0.5, 9.5),
                )
                ax.set_yticks(range(10), LABELS, fontsize=8)
                ax.set_xticks(xs if len(xs) <= 8 else sorted(set(round(x) for x in np.linspace(xs[0], xs[-1], 6))))
                ax.set_xlabel("Outer (one observation per rollout batch)")
                ax.set_ylabel("Relative response position")
                ax.set_title(
                    f"{label(case, arm)}\n{'First update observation' if col == 0 else 'Last update observation'}",
                    fontsize=10,
                    loc="left",
                )
                if zoom_name != "all":
                    ax.set_xticks(np.arange(xs[0] - 0.5, xs[-1] + 1, 1), minor=True)
                    ax.set_yticks(np.arange(-0.5, 10, 1), minor=True)
                    ax.grid(which="minor", color=fv.NEUTRALS["paper"], lw=0.5, alpha=0.35)
                    ax.tick_params(which="minor", length=0)
        assert image is not None
        cb = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.86, pad=0.02)
        cb.set_label("Position-bin RKL (shared within figure)", fontsize=9)
        fig.suptitle(f"{case} position RKL · {zoom_name} outer window", y=0.98)
        finish(
            fig,
            out,
            f"{prefix}-position-kl-first-last-heatmap{'-' + zoom_name if zoom_name != 'all' else ''}",
            "Token-mean native position RKL; first/last are pre-update observations "
            "and share one figure-level color scale.",
        )

        fig, axes = plt.subplots(len(arm_items), 2, figsize=(20, max(5.4 * len(arm_items), 6.5)), squeeze=False)
        fig.subplots_adjust(left=0.08, right=0.92, bottom=0.20, top=0.84, hspace=0.70, wspace=0.22)
        hues = [fv.HUES[name] for name in ("vteal", "vorange", "vviolet", "vblue")]
        styles = [(hues[i % 4], ("-", "--", ":")[i // 4], ("o", "s", "^")[i // 4]) for i in range(10)]
        for row_index, (arm, (xs, panel_views)) in enumerate(selected.items()):
            mats = [matrix(panel_rows) for panel_rows in panel_views]
            ymin = max(min(float(mat.min()) for mat in mats) * 0.75, 1e-8)
            ymax = max(float(mat.max()) for mat in mats) * 1.25
            for col, (ax, mat) in enumerate(zip(axes[row_index], mats, strict=True)):
                for bucket_index, (values, (color, linestyle, marker)) in enumerate(zip(mat, styles, strict=True)):
                    ax.plot(
                        xs,
                        values,
                        linestyle=linestyle,
                        marker=marker,
                        color=color,
                        ms=4,
                        lw=1.25,
                        markevery=max(1, len(xs) // 24),
                        label=LABELS[bucket_index],
                    )
                ax.set_yscale("log")
                ax.set_ylim(ymin, ymax)
                ax.set_xticks(xs if len(xs) <= 8 else sorted(set(round(x) for x in np.linspace(xs[0], xs[-1], 6))))
                ax.set_xlabel("Outer")
                ax.set_ylabel("Position-bin RKL (log scale)")
                ax.set_title(
                    f"{label(case, arm)}\n{'First update observation' if col == 0 else 'Last update observation'}",
                    fontsize=10,
                    loc="left",
                )
                ax.grid(True, which="major", alpha=0.2)
        handles = [
            Line2D([], [], color=color, linestyle=linestyle, marker=marker, lw=1.2, ms=4, label=LABELS[i])
            for i, (color, linestyle, marker) in enumerate(styles)
        ]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=5, frameon=False, fontsize=8)
        finish(
            fig,
            out,
            f"{prefix}-position-kl-first-last-lines{'-' + zoom_name if zoom_name != 'all' else ''}",
            "Each line is one relative-response decile; the larger canvas keeps close profiles inspectable.",
        )

    fig, axes = plt.subplots(len(arm_items), 2, figsize=(12, max(3.1 * len(arm_items), 5.0)), squeeze=False)
    fig.subplots_adjust(
        left=0.08,
        right=0.92,
        bottom=0.20,
        top=0.78 if len(arm_items) == 1 else 0.89,
        hspace=0.58,
        wspace=0.23,
    )
    for row_index, (arm, groups) in enumerate(arm_items):
        first = groups[min(groups)][0]
        last = groups[max(groups)][-1]
        profiles = matrix([first, last])
        ymin, ymax = float(profiles.min()) * 0.8, float(profiles.max()) * 1.25
        for col, (ax, row, which) in enumerate(
            zip(axes[row_index], (first, last), ("First outer", "Final outer"), strict=True)
        ):
            vals = matrix([row])[:, 0]
            ax.plot(range(10), vals, marker="o", lw=1.4, color=fv.HUES["vteal"])
            ax.set_yscale("log")
            ax.set_ylim(ymin, ymax)
            ax.set_xticks(range(10), LABELS, rotation=35, ha="right", fontsize=8)
            ax.set_ylabel("Position-bin RKL (log scale)")
            ax.set_title(
                f"{label(case, arm)}\n{which} · optimizer update {row['step/update']}", fontsize=10, loc="left"
            )
            ax.grid(True, which="major", alpha=0.2)
    fig.suptitle(f"{case} single-update position profiles", y=0.98)
    finish(
        fig,
        out,
        f"{prefix}-position-kl-single-update-lines",
        "Each point is one relative-response decile; final is the last logged pre-update observation.",
    )


def collect() -> tuple[dict[str, dict[int, list[dict]]], dict[str, dict[int, list[dict]]], list[Path]]:
    qwen_detail = WORKSPACE / "wandb-logs/MirrorOPD-verl/analysis/qwen-e41-e43-final-20261028/position-kl-detail.csv"
    e39_csv = WORKSPACE / "wandb-logs/MirrorOPD-verl/analysis/e39-qwen3base-final-20261027/position-kl.csv"
    qwen = {
        "E39": from_endpoint_csv(e39_csv),
        "E41": from_detail_csv(qwen_detail, "E41"),
        "E43": from_detail_csv(qwen_detail, "E43"),
    }
    e35 = {
        "E35": from_history(
            DOCS.parent.parent / "archive/experiments/polaris/data/polaris-grcuotmi-20261026/E35/history.jsonl",
            False,
            16,
            250,
        )
    }
    return qwen, e35, [qwen_detail, e39_csv]


def export(groups: dict[str, dict[int, list[dict]]], family: str) -> None:
    directory = DOCS / "data/case-position-kl-20261107"
    directory.mkdir(parents=True, exist_ok=True)
    rows, summary = [], []
    for arm, outers in groups.items():
        for outer in sorted(outers):
            for endpoint, row in (("first", outers[outer][0]), ("last", outers[outer][-1])):
                values = matrix([row])[:, 0]
                assert np.isfinite(values).all() and (values >= 0).all()
                rows.append(
                    {
                        "experiment": arm,
                        "outer": outer,
                        "endpoint": endpoint,
                        "update": int(row["step/update"]),
                        "full_kl": float(row["distill/full_kl"]),
                        **dict(zip(BINS, values, strict=True)),
                    }
                )
        first, last = outers[min(outers)][0], outers[max(outers)][-1]
        key = "distill/full_kl_position/p00_10/mean"
        late = sorted(outers)[-5:]
        summary.append(
            {
                "experiment": arm,
                "outers": len(outers),
                "updates": int(last["step/update"]),
                "initial_p00_10": float(first[key]),
                "final_p00_10": float(last[key]),
                "late5_first_p00_10": float(np.mean([outers[o][0][key] for o in late])),
                "late5_last_p00_10": float(np.mean([outers[o][-1][key] for o in late])),
            }
        )
    for name, entries in ((family + "-endpoints", rows), (family + "-summary", summary)):
        with (directory / f"{name}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(entries[0]))
            writer.writeheader()
            writer.writerows(entries)


def render_qwen(data: dict[str, dict[int, list[dict]]], out: Path, paths: list[Path]) -> None:
    for arm, groups in data.items():
        render_group(f"Qwen / {arm}", {arm: groups}, out, arm.lower())
    (out / "figure-audit.json").write_text(
        json.dumps({k: v for k, v in AUDITS.items() if k.startswith(out.name + "/")}, indent=2) + "\n"
    )
    manifest = {
        "source_sha256": {
            str(path.relative_to(WORKSPACE)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
        },
        "renderer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "experiments": list(data),
        "layout": "one experiment per figure; first and last observations in separate panels",
        "selection": "all outers, first five outers, last five outers; pre-update observations",
        "heatmap_scale": "one shared linear color scale across first/last panels within each figure",
        "figures": sorted(k.split("/", 1)[1] for k in AUDITS if k.startswith(out.name + "/")),
    }
    (DOCS / "data/case-position-kl-20261107/qwen-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    # Retire only the previous combined Qwen figures after all replacements pass audit.
    for path in out.glob("qwen-position-kl-*"):
        if path.suffix in {".png", ".pdf", ".svg", ".txt"}:
            path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--qwen-only", action="store_true", help="Render Qwen experiments separately; keep other families."
    )
    parser.add_argument("--mimo-sft-only", action="store_true", help="Render completed E91/E93 SFT-student position curves.")
    args = parser.parse_args()
    fv.apply_rc("paper", fontsize=10)
    if args.mimo_sft_only:
        paths = [ROOT / f"runs/{run}/formal/{arm}/metrics.jsonl" for arm, run in (
            ("E91", "e91-mimo-rlzero-sft-20261107"), ("E93", "e93-mimo-rl-sft-20261108"))]
        data = {arm: from_history(path, True, 16, 20) for arm, path in zip(("E91", "E93"), paths, strict=True)}
        out = DOCS / "figures/mimo-position-kl-20261107/sft-students"
        render_group("MiMo SFT students", data, out, "mimo-sft")
        (out / "provenance.json").write_text(json.dumps({
            "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
            "renderer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "selection": "first/last pre-update observations; reuse16; 20 outer; E91/E93 start from SFT",
        }, indent=2) + "\n")
        return
    qwen, e35, paths = collect()
    qout = DOCS / "figures/qwen-position-kl-20261107"
    if args.qwen_only:
        render_qwen(qwen, qout, paths)
        print("Rendered 21 separate Qwen figures: E39, E41 and E43; other families unchanged.")
        return
    e49_metadata = E49_HISTORY.with_name("run.json")
    assert json.loads(e49_metadata.read_text())["run"]["run_id"] == "zyru8wdf"
    # The E48 and E49 logs are distinct directories but share the same schema.
    r1 = {
        "E48": from_history(
            DOCS / "cross-origin/polaris/data/polaris-complete-20261027/run/history.jsonl", False, 16, 256
        ),
        "E49": from_history(E49_HISTORY, False, 16, 256),
    }
    assert "distill/full_forward_kl" in r1["E49"][1][0]
    assert r1["E48"][1][0]["distill/full_kl"] != r1["E49"][1][0]["distill/full_kl"]
    eout = DOCS / "figures/r1-1.5b-position-kl-20261107"
    mout = DOCS / "figures/mimo-position-kl-20261107"
    mimo_paths = [
        ROOT / f"runs/mimo-stage-transfer-20261101/formal/{arm}/metrics.jsonl" for arm in ("E71", "E72", "E73")
    ]
    mimo = {
        arm: from_history(path, True, reuse, count)
        for arm, path, reuse, count in zip(("E71", "E72", "E73"), mimo_paths, (32, 16, 16), (25, 20, 20), strict=True)
    }
    render_group("MiMo", mimo, mout, "mimo")
    render_qwen(qwen, qout, paths)
    render_group("R1-1.5B / E35", e35, eout, "e35")
    render_group("R1-1.5B / E48–E49", r1, eout, "e48-e49")
    for family, groups in (("mimo", mimo), ("qwen", qwen), ("r1-1.5b", e35 | r1)):
        export(groups, family)
    paths += mimo_paths + [
        DOCS.parent.parent / "archive/experiments/polaris/data/polaris-grcuotmi-20261026/E35/history.jsonl",
        DOCS / "cross-origin/polaris/data/polaris-complete-20261027/run/history.jsonl",
        E49_HISTORY,
        e49_metadata,
    ]
    manifest = {
        "source_sha256": {
            str(path.relative_to(WORKSPACE)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
        },
        "renderer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "selection": "first and last pre-update observations per actual outer; early/late = five outers",
        "heatmap_scale": "one shared linear color scale per entire figure",
        "experiments": ["E71", "E72", "E73", "E35", "E48", "E49", "E39", "E41", "E43"],
    }
    (DOCS / "data/case-position-kl-20261107/manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for out in (mout, qout, eout):
        (out / "figure-audit.json").write_text(
            json.dumps({k: v for k, v in AUDITS.items() if k.startswith(out.name + "/")}, indent=2) + "\n"
        )
    print("Rendered MiMo E71/E72/E73, Qwen E39/E41/E43 and R1-1.5B E35/E48/E49 position-KL figures.")


if __name__ == "__main__":
    main()
