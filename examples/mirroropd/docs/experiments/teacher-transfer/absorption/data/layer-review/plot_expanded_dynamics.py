"""Freeze accepted CPU expansion results and plot per-module subspace overlap.

Ledger: columns are model conditions; rows are rank16/rank128; each cell is the
right-subspace checkpoint-to-final overlap adjusted by the isotropic k/d null.
Gray cells are unmeasured MLP layers, never zero. Three MLP modules are measured
only at the middle layer. Conditions have different early progress fractions.
One cell is one tensor, not a population mean. Wide standalone PDF keeps 14in.
"""

import argparse
import gzip
import json
import os
import re
import sys
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.environ["FIGURE_VIZ_MPL"])
import fvstyle as fv

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent.parent / "figures/layer-review"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--results", type=Path)
parser.add_argument("--correction", type=Path)
parser.add_argument("--supplement", type=Path)
args = parser.parse_args()
if args.results:
    status = json.loads((args.results / "exit.json").read_text())
    if status["failed"] or status["complete"] != status["total"]:
        raise ValueError(f"Incomplete expansion: {status}")
    records = [json.loads(p.read_text()) for p in sorted(args.results.glob("matrix-*.json"))]
    assert len(records) == status["total"]
    protocol = json.loads((args.results / "protocol.json").read_text())
    if args.correction:
        correction_status = json.loads((args.correction / "exit.json").read_text())
        if correction_status["failed"] or correction_status["complete"] != correction_status["total"]:
            raise ValueError("Correction is incomplete")
        corrected = [json.loads(p.read_text()) for p in sorted(args.correction.glob("matrix-*.json"))]
        ids = {r["task"]["job_id"] for r in corrected}
        assert len(corrected) == sum(r["task"]["job_id"] in ids for r in records)
        records = [r for r in records if r["task"]["job_id"] not in ids] + corrected
        protocol["correction"] = json.loads((args.correction / "protocol.json").read_text())["correction"]
        for item in protocol["coverage"]:
            if item["job_id"] in ids:
                item["points"] = next(r["task"]["points"] for r in corrected if r["task"]["job_id"] == item["job_id"])
    if args.supplement:
        extra_status = json.loads((args.supplement / "exit.json").read_text())
        if extra_status["failed"] or extra_status["complete"] != extra_status["total"]:
            raise ValueError("Supplement is incomplete")
        extra = [json.loads(p.read_text()) for p in sorted(args.supplement.glob("matrix-*.json"))]
        records += extra
        keys = [(r["task"]["job_id"], r["task"]["tensor"]) for r in records]
        assert len(set(keys)) == len(keys), "Duplicate tensor conditions"
        protocol["supplement"] = "All 28 k/v layers, five q/o depth samples, middle-layer MLP only"
    data = dict(protocol=protocol, records=records)
    (ROOT / "expanded-update-dynamics.json.gz").write_bytes(
        gzip.compress(json.dumps(data, separators=(",", ":")).encode(), mtime=0)
    )
else:
    data = json.loads(gzip.decompress((ROOT / "expanded-update-dynamics.json.gz").read_bytes()))
rows = []
for record in data["records"]:
    for r in record["rows"]:
        if not -1e-8 <= r["final_overlap"] <= 1 + 1e-8:
            raise ValueError("Invalid subspace overlap")
        if r["label"] == "endpoint" and abs(r["final_overlap"] - 1) > 1e-8:
            raise ValueError("Final self-overlap identity failed")
        rows.append(
            dict(
                r,
                job_id=record["task"]["job_id"],
                layer=int(re.search(r"layers\.(\d+)", record["task"]["tensor"])[1]),
                module=record["task"]["tensor"].split(".")[-2],
            )
        )
summary = []
for job in [7, 8, 2, 6]:
    for module in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]:
        for side in ["left", "right"]:
            for rank in [16, 64, 128, 256, 512]:
                group = [
                    r
                    for r in rows
                    if r["job_id"] == job
                    and r["module"] == module
                    and r["side"] == side
                    and r["requested_rank"] == rank
                    and r["label"] not in ["endpoint"]
                ]
                # Compare first available stage separately from later JustRL checkpoints.
                label = min({r["label"] for r in group}, key=lambda x: int(re.search(r"\d+", x)[0]))
                group = [r for r in group if r["label"] == label]
                summary.append(
                    dict(
                        job_id=job,
                        module=module,
                        side=side,
                        rank=rank,
                        label=label,
                        layers=len(group),
                        raw=median(r["final_overlap"] for r in group),
                        adjusted=median(
                            r["adjusted_final_overlap"] for r in group if r["adjusted_final_overlap"] is not None
                        )
                        if any(r["adjusted_final_overlap"] is not None for r in group)
                        else None,
                    )
                )
(ROOT / "expanded-update-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
fv.apply_rc("paper")
modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
fig, axes = plt.subplots(2, 4, figsize=(14, 7.5), layout="constrained")
names = {7: "Qwen GRPO: 64/320", 8: "Qwen SFT: 64/320", 2: "JustRL: 16/320", 6: "R1-7B Light: 32/64"}
cmap = fv.ramp_cmap("seq")
cmap.set_bad(fv.NEUTRALS["vmist"])
for col, job in enumerate([7, 8, 2, 6]):
    group = [r for r in rows if r["job_id"] == job]
    early = min({r["label"] for r in group if r["label"] != "endpoint"}, key=lambda x: int(re.search(r"\d+", x)[0]))
    layers = next(x["layers"] for x in data["protocol"]["coverage"] if x["job_id"] == job)
    for i, rank in enumerate([16, 128]):
        grid = np.full((len(modules), len(layers)), np.nan)
        for r in group:
            if r["label"] == early and r["requested_rank"] == rank and r["side"] == "right" and r["layer"] in layers:
                grid[modules.index(r["module"]), layers.index(r["layer"])] = r["adjusted_final_overlap"]
        mesh = axes[i, col].imshow(grid, vmin=0, vmax=1, cmap=cmap, aspect="auto")
        axes[i, col].set(
            xticks=range(len(layers)),
            xticklabels=layers,
            yticks=range(len(modules)),
            yticklabels=modules,
            xlabel="Layer index",
            ylabel=f"Right rank {rank}",
            title=names[job],
        )
bar = fig.colorbar(mesh, ax=axes, shrink=0.8)
bar.set_label("Adjusted early-to-final overlap")
fig.get_layout_engine().set(rect=(0, 0.05, 1, 0.92))
fig.text(
    0.5,
    0.005,
    "Exact FP64 delta SVD; gray = unmeasured; MLP middle layer only; early fractions differ across conditions.",
    ha="center",
    fontsize=8,
    color=fv.NEUTRALS["baseline"],
)
fig.canvas.draw()
audit = [str(x) for x in fv.audit(fig, ignore=["print-fit"])]
fig.savefig(OUT / "expanded-module-early-final.png", dpi=180)
fig.savefig(OUT / "expanded-module-early-final.pdf")
(ROOT / "expanded-module-figure-audit.json").write_text(json.dumps(audit, indent=2) + "\n")
if data["protocol"].get("supplement"):
    fig, axes = plt.subplots(2, 4, figsize=(14, 7.5), layout="constrained", sharey=True)
    colors = fv.series_colors("paper", 4)
    for col, job in enumerate([7, 8, 2, 6]):
        available = [r for r in rows if r["job_id"] == job and r["label"] != "endpoint"]
        early = min({r["label"] for r in available}, key=lambda x: int(re.search(r"\d+", x)[0]))
        for ri, module in enumerate(["k_proj", "v_proj"]):
            ax = axes[ri, col]
            for color, (side, rank) in zip(
                colors, [("left", 16), ("right", 16), ("left", 128), ("right", 128)], strict=True
            ):
                group = sorted(
                    [
                        r
                        for r in available
                        if r["label"] == early
                        and r["module"] == module
                        and r["side"] == side
                        and r["requested_rank"] == rank
                    ],
                    key=lambda x: x["layer"],
                )
                assert len(group) == 28
                ax.plot(
                    [r["layer"] for r in group],
                    [r["adjusted_final_overlap"] for r in group],
                    color=color,
                    label=f"{side} k={rank}",
                    lw=1.5,
                )
            ax.set(xlabel="Layer index", ylabel=module + " adjusted overlap", title=names[job], ylim=(-0.05, 1.05))
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside upper center", ncol=4)
    fig.get_layout_engine().set(rect=(0, 0.05, 1, 0.93))
    fig.text(
        0.5,
        0.005,
        "All 28 k/v layers; exact delta subspaces; (overlap - k/d)/(1 - k/d); early fractions differ.",
        ha="center",
        fontsize=8,
        color=fv.NEUTRALS["baseline"],
    )
    fig.canvas.draw()
    extra_audit = [str(x) for x in fv.audit(fig, ignore=["print-fit"])]
    fig.savefig(OUT / "all-layer-kv-early-final.png", dpi=180)
    fig.savefig(OUT / "all-layer-kv-early-final.pdf")
    (ROOT / "all-layer-kv-figure-audit.json").write_text(json.dumps(extra_audit, indent=2) + "\n")
    print("all layer audit", extra_audit)
print("audit", audit)
for job in [7, 8, 2, 6]:
    print(
        job, [(r["module"], r["side"], round(r["raw"], 3)) for r in summary if r["job_id"] == job and r["rank"] == 16]
    )
