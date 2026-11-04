# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Compare E57 logged evaluations with original TOP-D Figure 5 vector curves.

Semantic ledger: each E57 point is a completed in-memory evaluation (avg@16).
Each TOP-D point is a published polyline vertex (avg@32), not a recovered log.
Response presentations count each optimizer use; fresh responses count generation
once, without deduplicating prompts or answer strings. Neither measures FLOPs.
Geometry ledger: columns are AIME24/25/26, rows full budget and early-budget zoom.
All panels share a linear percent y-axis; response counts are in thousands,
optimizer updates are unscaled counts, with one x scale per row. Lines connect
observed points; there is no smoothing or extrapolation.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from examples.mirroropd.research.core.plotting import load_style

BENCHMARKS = ("AIME24", "AIME25", "AIME26")
COLORS = {(249, 142, 113): "TOP-D alpha=0.1", (224, 102, 84): "TOP-D alpha=0.2"}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract_topd(source):
    """Calibrate original vector coordinates using the printed x/y grid ticks."""
    import pymupdf

    result = []
    with pymupdf.open(source) as document:
        page = document[0]
        words, drawings = page.get_text("words"), page.get_drawings()
        for panel, benchmark in enumerate(BENCHMARKS):
            left = panel * 231.78666
            ticks = {
                int(w[4]): (w[0] + w[2]) / 2
                for w in words
                if w[4] in ("0", "100", "200", "300", "400") and 128 < w[1] < 130 and left + 35 < w[0] < left + 205
            }
            xzero, xscale = ticks[0], (ticks[400] - ticks[0]) / 400
            grids = sorted(
                {
                    d["rect"].y0
                    for d in drawings
                    if d["color"]
                    and abs(d["color"][0] - 0.73327) < 0.001
                    and len(d["items"]) == 1
                    and abs(d["rect"].x0 - (left + 39.6756)) < 0.01
                    and d["rect"].width > 150
                }
            )
            candidates = [w for w in words if w[4].isdigit() and left + 10 < w[0] < left + 35 and w[1] < 125]
            yticks = [(y, int(min(candidates, key=lambda w: abs((w[1] + w[3]) / 2 - y))[4])) for y in grids]
            (y0, v0), (y1, v1) = yticks[0], yticks[-1]
            for drawing in drawings:
                if not drawing["color"] or len(drawing["items"]) < 15:
                    continue
                if not left + 40 < drawing["rect"].x0 < left + 55:
                    continue
                label = COLORS.get(tuple(round(v * 255) for v in drawing["color"]))
                if label is None:
                    continue
                points = [drawing["items"][0][1]] + [item[-1] for item in drawing["items"]]
                for point in points:
                    step = (point.x - xzero) / xscale
                    assert abs(step - round(step)) < 0.01, step
                    outer = round(step)
                    result.append(
                        dict(
                            series=label,
                            benchmark=benchmark,
                            outer=outer,
                            optimizer_updates=outer * 16,
                            fresh_responses=outer * 4096,
                            response_presentations=outer * 4096,
                            accuracy_percent=v0 + (point.y - y0) * (v1 - v0) / (y1 - y0),
                        )
                    )
    assert len(result) == 3 * 2 * 42
    return result


def extract_e57(source):
    run = next(r for r in json.loads(source.read_text())["runs"] if r["name"] == "E57")
    assert run["source_status"] == "finished" and run["exit_code"] == 0
    assert all(v == 0 for v in run["bad_flags"].values())
    assert [u["step/update"] for u in run["updates"]] == list(range(1, 1601))
    assert [o["step"] for o in run["outer"]] == list(range(1, 101))
    assert all(o["responses"] == 64 for o in run["outer"])
    rows = []
    for evaluation in run["validation"]:
        benchmark = evaluation["dataset"].upper()
        if benchmark not in BENCHMARKS:
            continue
        outer = evaluation["outer"]
        updates = [u for u in run["updates"] if u["step/outer"] <= outer]
        assert len(updates) == outer * 16
        rows.append(
            dict(
                series="E57",
                benchmark=benchmark,
                outer=outer,
                optimizer_updates=len(updates),
                fresh_responses=sum(o["responses"] for o in run["outer"] if o["step"] <= outer),
                response_presentations=len(updates) * 64,
                accuracy_percent=evaluation["accuracy"] * 100,
            )
        )
    assert len(rows) == 30
    return rows, {k: run[k] for k in ("run_id", "url", "raw_source", "sha256", "config", "code_commit")}


def render(rows, out, metric, fv):
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5), sharey=True)
    styles = [
        ("E57", "vteal", "o", "-"),
        ("TOP-D alpha=0.1", "vorange", ".", "-"),
        ("TOP-D alpha=0.2", "vcoral", ".", "--"),
    ]
    updates_axis = metric == "optimizer_updates"
    scale = 1 if updates_axis else 1000
    endpoint = max(r[metric] for r in rows if r["series"] == "E57") / scale
    for column, benchmark in enumerate(BENCHMARKS):
        for row_index in range(2):
            ax = axes[row_index, column]
            limit = (7000 if updates_axis else 1750) if row_index == 0 else endpoint * 1.08
            for label, hue, marker, linestyle in styles:
                points = sorted(
                    (r for r in rows if r["series"] == label and r["benchmark"] == benchmark), key=lambda r: r[metric]
                )
                # Include the next observed point to show the connecting segment at the zoom edge.
                visible = [r for r in points if r[metric] / scale <= limit]
                following = next((r for r in points if r[metric] / scale > limit), None)
                if len(visible) < 2:
                    following = None
                if following is not None:
                    visible.append(following)
                ax.plot(
                    [r[metric] / scale for r in visible],
                    [r["accuracy_percent"] for r in visible],
                    color=fv.HUES[hue],
                    marker=marker,
                    markevery=list(range(len(visible) - int(following is not None))),
                    ms=4 if label == "E57" else 3,
                    ls=linestyle,
                    lw=1.6,
                    label=label,
                )
            ax.set(xlim=(-0.015 * limit, limit), ylim=(0, 22), yticks=(0, 5, 10, 15, 20))
            ax.grid(axis="y", alpha=0.2)
            ax.text(0.03, 0.93, f"({chr(97 + row_index * 3 + column)})", transform=ax.transAxes, va="top")
            if row_index == 0:
                ax.set_title(benchmark)
                ax.set_xticks((0, 2000, 4000, 6000) if updates_axis else (0, 500, 1000, 1500))
            else:
                ticks = {
                    "response_presentations": (0, 25, 50, 75, 100),
                    "fresh_responses": (0, 2, 4, 6),
                    "optimizer_updates": (0, 400, 800, 1200, 1600),
                }
                ax.set_xticks(ticks[metric])
            if column == 0:
                ax.set_ylabel(("Full budget" if row_index == 0 else "Early budget") + "\nAccuracy (%)")
    fig.legend(
        *axes[0, 0].get_legend_handles_labels(), loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.99)
    )
    title = {
        "response_presentations": "Cumulative training response presentations (thousands)",
        "fresh_responses": "Cumulative newly generated training responses (thousands)",
        "optimizer_updates": "Cumulative optimizer updates",
    }[metric]
    fig.supxlabel(title, y=0.095, fontsize=11)
    fig.text(
        0.5,
        0.035,
        "E57: logged avg@16; TOP-D: published Figure 5 avg@32. Evaluation protocols differ; no compute equivalence.",
        ha="center",
        fontsize=9,
        color=fv.NEUTRALS["baseline"],
    )
    fig.subplots_adjust(top=0.87, bottom=0.18, left=0.08, right=0.985, hspace=0.28, wspace=0.16)
    fig.canvas.draw()
    # Zoom deliberately clips curves to its window, including one subsequent vertex.
    problems = fv.audit(
        fig, print_slot=fig.get_figwidth(), scoped_ignore=[("data-clipped-outside-limits", ax) for ax in axes[1]]
    )
    audit = fv.format_audit_lines(problems)
    (out / f"{metric}.audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    assert not any(fv.problem_severity(p) == "error" for p in problems), audit
    for extension in ("png", "pdf"):
        fig.savefig(out / f"{metric}.{extension}", dpi=180, facecolor=fv.NEUTRALS["paper"])
    plt.close(fig)
    print(metric, audit)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", required=True, type=Path, help="New evidence directory, or frozen directory for replot"
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--e57-data", type=Path, help="Original frozen full log extraction; omit when replotting")
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=("response_presentations", "fresh_responses", "optimizer_updates"),
        default=("response_presentations", "fresh_responses", "optimizer_updates"),
        help="Horizontal axes to render",
    )
    args = parser.parse_args()
    if args.e57_data:
        e57, provenance = extract_e57(args.e57_data)
        provenance["extracted_log_sha256"] = sha256(args.e57_data)
        provenance["topd_figure_sha256"] = sha256(args.data / "topd-figure5.pdf")
        provenance["topd_source"] = "https://arxiv.org/src/2607.04751v2; figures/ablation_curves.pdf"
        provenance["topd_budget"] = (
            "Inferred outer counter: 512 prompts * 8 responses, one epoch, 16 minibatches * 256 responses."
        )
        provenance["method"] = (
            "Original PDF vector paths calibrated against printed ticks; rounded step residual <0.01; "
            "no extra smoothing. Published curves may already be smoothed."
        )
        rows = e57 + extract_topd(args.data / "topd-figure5.pdf")
        (args.data / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        (args.data / "curves.json").write_text(json.dumps(rows, indent=2) + "\n")
        with (args.data / "curves.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    else:
        rows = json.loads((args.data / "curves.json").read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    fv = load_style()
    fv.apply_rc("paper", fontsize=11)
    for metric in args.metrics:
        render(rows, args.out, metric, fv)
    (args.out / "render-provenance.json").write_text(
        json.dumps(
            dict(
                script_sha256=sha256(Path(__file__)),
                curves_sha256=sha256(args.data / "curves.json"),
                matplotlib_version=matplotlib.__version__,
                metrics=args.metrics,
                audit_exception="Zoom axes intentionally clip the connecting segment to the next observed vertex.",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
