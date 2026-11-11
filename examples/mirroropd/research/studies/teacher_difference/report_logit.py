# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Logit experiment tables and figures.

Semantic ledger: color identifies native training branch, row is a fixed
prefix, stages are actual checkpoint updates. Cross responses use distinct
questions, with train sources and heldout receivers. Geometry ledger: figures
show median site-level quantities or mean cross responses, not full-bank
estimates; Ritz residuals remain visible in the numeric tables.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .report_stages import DEFAULT_STYLE, frame
from .results import write_csv
from .runtime import write_json


def collect(root):
    sites, cross, sources = [], [], {}
    records = []
    for path in sorted(root.glob("*-s*.json")):
        data = json.loads(path.read_text())
        artifact = torch.load(path.with_suffix(".pt"), map_location="cpu", weights_only=True)
        if json.loads(json.dumps(artifact["signature"])) != data["signature"]:
            raise ValueError(f"JSON/tensor identity mismatch: {path}")
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        sources[str(path.with_suffix(".pt"))] = hashlib.sha256(path.with_suffix(".pt").read_bytes()).hexdigest()
        records.append((data, artifact))
    baseline = next((pair for pair in records if pair[0]["task"]["stage"] == 0), None)
    if baseline:
        measured_initial = {
            data["task"].get("branch", data["task"]["target"]) for data, _ in records if data["task"]["stage"] == 0
        }
        branches = {data["task"].get("branch", data["task"]["target"]) for data, _ in records}
        for branch in sorted(branches - measured_initial):
            data, artifact = baseline
            if not all(f"{branch}/live" in site["fields"] for site in data["sites"]):
                raise ValueError(f"Shared initialization lacks request field: {branch}")
            records.append((dict(data, task=dict(data["task"], branch=branch, target=branch)), artifact))
    for data, artifact in records:
        branch, stage = data["task"].get("branch", data["task"]["target"]), data["task"]["stage"]
        for i, site in enumerate(data["sites"]):
            spectrum = artifact["spectra"][i]
            leading = spectrum["eigenvalues"][:3]
            geometry = dict(
                leading_strength=float(leading.sum()),
                max_ritz_residual=float(spectrum["residual_relative"].max()),
                projected_asymmetry=spectrum["projected_asymmetry"],
            )
            if baseline:
                base_data, base_artifact = baseline
                base_site = base_data["sites"][i]
                if (site["row_id"], site["position"]) != (base_site["row_id"], base_site["position"]):
                    raise ValueError("Baseline sites do not match")
                base_spectrum = base_artifact["spectra"][i]
                overlap = base_spectrum["vectors"].double().T @ spectrum["vectors"].double()
                geometry.update(
                    G3=float(leading.sum() / base_spectrum["eigenvalues"][:3].sum()),
                    O3=float(overlap.square().sum() / 3),
                )
            for field, metrics in site["fields"].items():
                target, mode = field.split("/")
                profile = site["profiles"][target]
                row = dict(
                    branch=branch,
                    stage=stage,
                    site=i,
                    row_id=site["row_id"],
                    position=site["position"],
                    split=site["split"],
                    target=target,
                    mode=mode,
                    loss=profile["loss"],
                    native_support_floor=profile["native_support_floor"],
                    **geometry,
                    **{k: v for k, v in metrics.items() if isinstance(v, int | float)},
                )
                row["top3_projection"] = sum(metrics["leading_request_energy"])
                row["top3_accessibility"] = sum(metrics["leading_accessibility_contribution"])
                row["rayleigh_duality_relative"] = abs(
                    metrics["gradient_norm_sq"] - metrics["request_response_inner"]
                ) / max(metrics["gradient_norm_sq"], 1e-30)
                for j, bin_ in enumerate(profile["bins"]):
                    row[f"live_pressure_fraction_bin{j}"] = bin_["pressure"] / max(profile["pressure"], 1e-30)
                if baseline:
                    base_metric = baseline[0]["sites"][i]["fields"][field]
                    row["projection_gain"] = row["top3_projection"] - sum(base_metric["leading_request_energy"])
                sites.append(row)
        for item in data["cross"]:
            source_site, receiver_site = data["sites"][item["source"]], data["sites"][item["receiver"]]
            source_energy = source_site["fields"][item["field"]]["gradient_norm_sq"]
            receiver_energy = receiver_site["fields"][item["field"]]["gradient_norm_sq"]
            cross.append(
                {
                    **item,
                    "branch": branch,
                    "stage": stage,
                    "source_split": source_site["split"],
                    "receiver_split": receiver_site["split"],
                    "source_row_id": source_site["row_id"],
                    "receiver_row_id": receiver_site["row_id"],
                    "duality_normalized_error": abs(item["gradient_inner"] - item["request_response_inner"])
                    / max((source_energy * receiver_energy) ** 0.5, 1e-30),
                }
            )
    return sites, cross, sources


def panel_summaries(sites, cross):
    """Exact finite-panel moments; off-diagonal Gram retains negative values."""
    from .metrics import moment_summary

    groups = {}
    for row in sites:
        key = (row["branch"], row["stage"], row["target"], row["mode"], row["split"])
        groups.setdefault(key, []).append(row)
    output = []
    for (branch, stage, target, mode, split), group in sorted(groups.items()):
        ids = {r["site"] for r in group}
        edges = [
            r
            for r in cross
            if r["branch"] == branch
            and r["stage"] == stage
            and r["field"] == f"{target}/{mode}"
            and r["source"] in ids
            and r["receiver"] in ids
        ]
        n = len(group)
        if len(edges) != n * (n - 1):
            raise ValueError("Incomplete cross panel")
        diagonal = sum(r["gradient_norm_sq"] for r in group)
        off = sum(r["gradient_inner"] for r in edges)
        moments = moment_summary(
            pressure=np.mean([r["pressure"] for r in group]),
            second=diagonal / n,
            mean_norm=(diagonal + off) / (n * n),
            samples=n,
        )
        output.append(
            dict(
                branch=branch,
                stage=stage,
                target=target,
                mode=mode,
                split=split,
                loss_mean=np.mean([r["loss"] for r in group]),
                **moments,
                G3_median=np.median([r.get("G3", np.nan) for r in group]),
                O3_median=np.median([r.get("O3", np.nan) for r in group]),
                request_projection_median=np.median([r["top3_projection"] for r in group]),
                max_ritz_residual=max(r["max_ritz_residual"] for r in group),
            )
        )
    transfer = []
    keys = sorted({(r["branch"], r["stage"], r["field"]) for r in cross})
    for branch, stage, field in keys:
        edges = [
            r
            for r in cross
            if (r["branch"], r["stage"], r["field"]) == (branch, stage, field)
            and r["source_split"] == "train"
            and r["receiver_split"] == "heldout"
            and r["source_row_id"] != r["receiver_row_id"]
        ]
        if not edges:
            continue
        transfer.append(
            dict(
                branch=branch,
                stage=stage,
                field=field,
                pairs=len(edges),
                response_norm=np.mean([r["response_norm"] for r in edges]),
                request_response_cosine=np.mean([r["request_response_cosine"] for r in edges]),
                request_response_inner=np.mean([r["request_response_inner"] for r in edges]),
                negative_fraction=np.mean([r["request_response_inner"] < 0 for r in edges]),
            )
        )
    return output, transfer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--style", type=Path, default=DEFAULT_STYLE)
    args = parser.parse_args()
    sites, cross, sources = collect(args.root)
    if not sites:
        raise ValueError("No completed logit measurements")
    out = args.root / "analysis"
    out.mkdir(exist_ok=True)
    write_csv(out / "sites.csv", sites)
    write_csv(out / "cross.csv", cross)
    panels, transfer_summary = panel_summaries(sites, cross)
    write_csv(out / "panels.csv", panels)
    if transfer_summary:
        write_csv(out / "transfer.csv", transfer_summary)
    sys.path.insert(0, str(args.style))
    import fvstyle as fv

    fv.apply_rc("paper")
    colors = dict(justrl=fv.HUES["vteal"], polaris=fv.HUES["vorange"])
    audits = {}

    def plot(name, metrics, rows):
        fig, axes = plt.subplots(1, len(metrics), figsize=(2.6 * len(metrics), 3.5), squeeze=False)
        fig.subplots_adjust(left=0.09, right=0.97, bottom=0.26, top=0.91, wspace=0.55)
        for ax, (key, label) in zip(axes[0], metrics, strict=True):
            for branch in sorted({r["branch"] for r in rows}):
                selected = [r for r in rows if r["branch"] == branch and key in r]
                stages = sorted({r["stage"] for r in selected})
                values = [float(np.median([r[key] for r in selected if r["stage"] == stage])) for stage in stages]
                if stages:
                    ax.plot(stages, values, "o-", color=colors[branch], label=branch, markersize=4)
            frame(ax, "Optimizer updates", label)
        axes[0, 0].legend(fontsize=10, loc="best")
        fig.text(
            0.5,
            0.055,
            "Fixed Math prefixes; native teacher; site medians / panel and transfer means.",
            ha="center",
            fontsize=10,
            color=fv.NEUTRALS["baseline"],
        )
        fig.canvas.draw()
        audits[name] = [str(issue) for issue in fv.audit(fig)]
        for ext in ["png", "pdf"]:
            fig.savefig(out / f"{name}.{ext}", dpi=200)
        plt.close(fig)

    native = [r for r in sites if r["branch"] == r["target"]]
    frozen = [r for r in native if r["mode"] == "frozen_B0"]
    for split in sorted({r["split"] for r in native}):
        plot(
            f"kernel_geometry_{split}",
            [
                ("G3", "Leading strength / initial"),
                ("O3", "Leading subspace overlap"),
                ("projection_gain", "Frozen request projection gain"),
            ],
            [r for r in frozen if r["split"] == split],
        )
        plot(
            f"request_response_{split}",
            [("loss", "Prefix reverse KL"), ("pressure", "Request energy"), ("rayleigh", "Request accessibility")],
            [r for r in native if r["mode"] == "live" and r["split"] == split],
        )
        plot(
            f"pac_{split}",
            [
                ("P", "Mean request pressure"),
                ("A", "Prefix accessibility"),
                ("cross_record_normalized", "Cross-position coherence"),
            ],
            [r for r in panels if r["branch"] == r["target"] and r["mode"] == "live" and r["split"] == split],
        )
    transfer = [r for r in transfer_summary if r["field"] == f"{r['branch']}/live"]
    if transfer:
        plot(
            "cross_response",
            [
                ("response_norm", "Receiver logit response norm"),
                ("request_response_cosine", "Receiver request alignment"),
                ("request_response_inner", "Cross first-order signal"),
            ],
            transfer,
        )
    write_json(
        out / "audit.json",
        dict(
            sources=sources,
            figure_audits=audits,
            max_duality_normalized_error=max((r["duality_normalized_error"] for r in cross), default=0),
            max_ritz_residual=max(r["max_ritz_residual"] for r in sites),
            note="Ritz estimates require residual checks; fixed small panel does not estimate full-bank uncertainty.",
        ),
    )


if __name__ == "__main__":
    main()
