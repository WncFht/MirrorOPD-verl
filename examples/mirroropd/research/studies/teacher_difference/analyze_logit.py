# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Question-balanced logit transfer and paired Krylov stability analysis.

Each question has equal weight; positions within a question are averaged.
Delete-one ranges describe this finite panel, not population confidence bands.
"""

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .metrics import moment_summary
from .report_logit import collect
from .report_stages import DEFAULT_STYLE, frame
from .results import write_csv
from .runtime import write_json


def question_panel(sites, edges):
    """Return position-averaged question Gram and equal-question moments."""
    questions = sorted({r["row_id"] for r in sites})
    members = [[r for r in sites if r["row_id"] == q] for q in questions]
    gram = {(r["source"], r["receiver"]): r["gradient_inner"] for r in edges}
    gram.update({(r["site"], r["site"]): r["gradient_norm_sq"] for r in sites})
    matrix = np.array(
        [[np.mean([gram[a["site"], b["site"]] for a in left for b in right]) for right in members] for left in members]
    )
    pressure = np.mean([np.mean([r["pressure"] for r in group]) for group in members])
    loss = np.mean([np.mean([r["loss"] for r in group]) for group in members])
    moments = moment_summary(
        pressure=pressure, second=np.trace(matrix) / len(matrix), mean_norm=matrix.mean(), samples=len(matrix)
    )
    return (
        questions,
        matrix,
        dict(
            loss=loss,
            PXi=moments["mean_gradient_norm_sq"],
            PXi_over_L=moments["mean_gradient_norm_sq"] / loss if loss > 0 else float("nan"),
            **moments,
        ),
    )


def transfer_moments(gram, losses, source, receiver):
    energy_s = gram[np.ix_(source, source)].mean()
    energy_r = gram[np.ix_(receiver, receiver)].mean()
    inner = gram[np.ix_(source, receiver)].mean()
    held_loss = np.mean(losses[receiver])
    return dict(
        cross_inner=inner,
        cross_cosine=inner / max(energy_s * energy_r, 1e-30) ** 0.5,
        cross_over_source=inner / max(energy_s, 1e-30),
        cross_over_loss=inner / max(held_loss, 1e-30),
    )


def question_summaries(sites, cross, *, panel):
    keys = sorted({(r["branch"], r["stage"], r["target"], r["mode"]) for r in sites})
    summaries, sensitivity = [], []
    for branch, stage, target, mode in keys:
        local = [r for r in sites if (r["branch"], r["stage"], r["target"], r["mode"]) == (branch, stage, target, mode)]
        edges = [r for r in cross if (r["branch"], r["stage"], r["field"]) == (branch, stage, f"{target}/{mode}")]
        ids, gram, _ = question_panel(local, edges)
        splits = {r["row_id"]: r["split"] for r in local}
        source = [i for i, q in enumerate(ids) if splits[q] == "train"]
        receiver = [i for i, q in enumerate(ids) if splits[q] == "heldout"]
        if not source or not receiver:
            continue
        metadata = dict(panel=panel, branch=branch, stage=stage, target=target, mode=mode)

        losses = np.array([np.mean([r["loss"] for r in local if r["row_id"] == q]) for q in ids])

        for split in ["train", "heldout"]:
            _, _, moments = question_panel([r for r in local if r["split"] == split], edges)
            summaries.append(dict(**metadata, split=split, **moments))
        full = transfer_moments(gram, losses, source, receiver)
        deletions = []
        for label, group in [("train", source), ("heldout", receiver)]:
            if len(group) < 2:
                continue
            for index in group:
                src = [i for i in source if i != index] if label == "train" else source
                dst = [i for i in receiver if i != index] if label == "heldout" else receiver
                value = transfer_moments(gram, losses, src, dst)
                deletions.append(value)
                sensitivity.append(dict(**metadata, removed_split=label, removed_question=ids[index], **value))
        for key in list(full):
            values = [r[key] for r in deletions] or [full[key]]
            full[f"{key}_delete_min"], full[f"{key}_delete_max"] = min(values), max(values)
        summaries.append(
            dict(
                **metadata,
                split="train-to-heldout",
                source_questions=len(source),
                receiver_questions=len(receiver),
                **full,
            )
        )
    return summaries, sensitivity


def spectrum_stability(reference, comparison):
    """Compare same-site Ritz outputs, preserving per-direction residuals."""
    rows = []
    for path in sorted(comparison.glob("*-s*.pt")):
        other = reference / path.name
        if not other.exists():
            continue
        left, right = [torch.load(p, map_location="cpu", weights_only=True) for p in [other, path]]
        if left["signature"]["request"]["prefixes"] != right["signature"]["request"]["prefixes"]:
            raise ValueError(f"Spectrum sites differ: {path.name}")
        for i, (a, b) in enumerate(zip(left["spectra"], right["spectra"], strict=True)):
            rows.append(
                dict(
                    checkpoint=path.stem,
                    site=i,
                    reference_iterations=a["iterations"],
                    comparison_iterations=b["iterations"],
                    max_eigenvalue_relative_change=float(
                        ((a["eigenvalues"][:3] - b["eigenvalues"][:3]) / b["eigenvalues"][:3]).abs().max()
                    ),
                    subspace_overlap=float((a["vectors"].double().T @ b["vectors"].double()).square().sum() / 3),
                    reference_residual=float(a["residual_relative"].max()),
                    comparison_residual=float(b["residual_relative"].max()),
                )
            )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", type=Path, nargs="+", required=True)
    parser.add_argument("--convergence", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--style", type=Path, default=DEFAULT_STYLE)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    summaries, sensitivity, sources = [], [], {}
    for root in args.roots:
        sites, cross, hashes = collect(root)
        if not sites:
            raise ValueError(f"No completed measurements: {root}")
        rows, deleted = question_summaries(sites, cross, panel=root.name)
        summaries.extend(rows)
        sensitivity.extend(deleted)
        sources.update(hashes)
    write_csv(args.out / "questions.csv", summaries)
    write_csv(args.out / "delete-one.csv", sensitivity)
    sys.path.insert(0, str(args.style))
    import fvstyle as fv

    fv.apply_rc("paper")
    colors = dict(justrl=fv.HUES["vteal"], polaris=fv.HUES["vorange"])
    audits = {}

    def save(fig, name):
        fig.canvas.draw()
        audits[name] = [str(issue) for issue in fv.audit(fig)]
        for ext in ["png", "pdf"]:
            fig.savefig(args.out / f"{name}.{ext}", dpi=200)
        plt.close(fig)

    # Same color has the same branch meaning; dashed lines freeze the request.
    panels = [r.name for r in args.roots]
    fig, axes = plt.subplots(len(panels), 3, figsize=(10.5, 3.2 * len(panels)), squeeze=False)
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.17, top=0.93, wspace=0.45, hspace=0.55)
    for i, panel in enumerate(panels):
        for j, (key, label) in enumerate(
            [
                ("cross_cosine", "Mean-gradient cosine"),
                ("cross_over_source", "Cross / source self energy"),
                ("cross_over_loss", "Cross signal / heldout loss"),
            ]
        ):
            ax = axes[i, j]
            for branch in colors:
                for mode, linestyle in [("live", "-"), ("frozen_B0", "--")]:
                    rows = sorted(
                        [
                            r
                            for r in summaries
                            if r["panel"] == panel
                            and r["branch"] == r["target"] == branch
                            and r["mode"] == mode
                            and r["split"] == "train-to-heldout"
                        ],
                        key=lambda r: r["stage"],
                    )
                    if rows:
                        ax.plot(
                            [r["stage"] for r in rows],
                            [r[key] for r in rows],
                            linestyle,
                            marker="o",
                            markersize=3,
                            color=colors[branch],
                            label=f"{branch} {mode}",
                        )
            frame(ax, "Optimizer updates", label)
            ax.set_title(panel, fontsize=11)
    for column in range(3):
        limits = [axes[row, column].get_ylim() for row in range(len(panels))]
        for row in range(len(panels)):
            axes[row, column].set_ylim(min(v[0] for v in limits), max(v[1] for v in limits))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=10)
    save(fig, "question_transfer")

    if args.convergence:
        stability = spectrum_stability(args.roots[0], args.convergence)
        write_csv(args.out / "spectrum-stability.csv", stability)
        for path in args.convergence.glob("*-s*.pt"):
            import hashlib

            sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
        fig.subplots_adjust(left=0.1, right=0.97, bottom=0.23, top=0.92, wspace=0.5)
        for ax, (key, label) in zip(
            axes,
            [
                ("max_eigenvalue_relative_change", "Top-3 relative eigenvalue change"),
                ("subspace_overlap", "24 / 64-step subspace overlap"),
                ("comparison_residual", "64-step maximum Ritz residual"),
            ],
            strict=True,
        ):
            for branch in colors:
                rows = [r for r in stability if r["checkpoint"].startswith(branch)]
                ax.scatter(range(len(rows)), [r[key] for r in rows], s=14, color=colors[branch], label=branch)
            frame(ax, "Paired site index", label)
        axes[0].legend(fontsize=10)
        save(fig, "spectrum_stability")
    write_json(
        args.out / "audit.json",
        dict(
            sources=sources,
            figure_audits=audits,
            question_weighting="equal questions; mean positions within question",
            delete_one="remove one source OR one receiver question; min/max are sensitivity ranges, not CIs",
        ),
    )


if __name__ == "__main__":
    main()
