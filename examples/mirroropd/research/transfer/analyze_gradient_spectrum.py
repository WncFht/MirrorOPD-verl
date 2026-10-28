#!/usr/bin/env python3
# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""CPU spectral diagnostics of previously measured full-parameter gradients.

This analyzes target-conditioned loss-gradient Gram matrices, NOT a teacher-
independent output NTK or an Adam-preconditioned kernel. No models are updated.
Plot ledger: hue = teacher; dashed/solid = initial/own warm128 checkpoint;
each point is one empirical eigenmode, or one subset-size diagnostic.
All spectra use the same 32 disjoint B16 groups; Held64 never selects modes.
Held64 projection is an oracle geometric diagnostic, not a usable update rule.
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from ..common.io import dump, read


def eigensystem(matrix):
    """Return a checked PSD eigensystem, descending and in float64."""
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not np.isfinite(matrix).all():
        raise ValueError("Expected a finite square Gram matrix")
    scale = max(float(np.linalg.norm(matrix, 2)), 1e-30)
    asymmetry = float(np.max(np.abs(matrix - matrix.T)))
    if asymmetry > 1e-6 * scale:
        raise ValueError("Gram matrix is not numerically symmetric")
    values, vectors = np.linalg.eigh((matrix + matrix.T) / 2)
    if values[0] < -1e-6 * scale:
        raise ValueError("Gram matrix has a material negative eigenvalue")
    return np.maximum(values[::-1], 0), vectors[:, ::-1]


def projection_curve(gram, cross, held_norm_sq):
    """Compute squared orthogonal-projection fraction into top gradient modes."""
    values, vectors = eigensystem(gram)
    cross = np.asarray(cross, dtype=np.float64)
    if cross.shape != (len(values),) or held_norm_sq <= 0:
        raise ValueError("Invalid cross-Gram vector or held-out norm")
    active = values > max(float(values[0]) * 1e-10, 1e-30)
    mass = np.zeros_like(values)
    mass[active] = (vectors[:, active].T @ cross) ** 2 / values[active] / held_norm_sq
    curve = np.cumsum(mass)
    if curve[-1] > 1 + 1e-5:
        raise ValueError("Projection exceeds held-out gradient energy")
    return curve, values, vectors


def spectral_summary(gram, cross, held_norm_sq):
    """Summarize spectrum, pool-mean alignment and held-out projection."""
    curve, values, vectors = projection_curve(gram, cross, held_norm_sq)
    trace = float(values.sum())
    if trace <= 0:
        raise ValueError("Zero training gradients")
    fractions = values / trace
    n = len(values)
    mean_norm_sq = float(np.sum(gram) / n**2)
    mean_cross = float(np.sum(cross) / n)
    return {
        "eigenvalues": values.tolist(),
        "eigenvalue_fractions": fractions.tolist(),
        "cumulative_held_projection": curve.tolist(),
        "top1_trace_fraction": float(fractions[0]),
        "top4_trace_fraction": float(fractions[:4].sum()),
        "participation_rank": float(1 / np.sum(fractions**2)),
        "entropy_rank": float(np.exp(-np.sum(fractions[fractions > 0] * np.log(fractions[fractions > 0])))),
        "held_projection_top1": float(curve[0]),
        "held_projection_top4": float(curve[min(3, n - 1)]),
        "held_projection_all": float(curve[-1]),
        "held_norm_sq": float(held_norm_sq),
        "mean_gradient_norm_sq": mean_norm_sq,
        "pool_mean_held_cosine": mean_cross / np.sqrt(mean_norm_sq * held_norm_sq),
        "mean_energy_fraction": float(mean_norm_sq / (trace / n)),
        "top_mode_uniform_alignment": float((vectors[:, 0].sum() / np.sqrt(n)) ** 2),
        "min_eigenvalue": float(values[-1]),
    }


def analyze_case(obj, subsets, seed):
    """Analyze one source case without changing source normalization."""
    n = obj["training_microbatches"]
    full = np.asarray(obj["gram"], dtype=np.float64)
    held = obj["groups"].index("Held64")
    if full.shape != (len(obj["groups"]), len(obj["groups"])) or n != 32:
        raise ValueError("Unexpected source groups")
    eigensystem(full)
    ids = obj["prompt_ids"]
    flat = [x for group in ids[:n] for x in group]
    if len(flat) != len(set(flat)) or set(flat) & set(ids[held]):
        raise ValueError("Training groups overlap each other or Held64")
    gram, cross, held_sq = full[:n, :n], full[:n, held], float(full[held, held])
    raw = spectral_summary(gram, cross, held_sq)
    norms = np.sqrt(np.diag(gram))
    normalized = spectral_summary(gram / np.outer(norms, norms), cross / norms, held_sq)
    center = np.eye(n) - np.ones((n, n)) / n
    centered_eig, _ = eigensystem(center @ gram @ center)
    centered_frac = centered_eig / centered_eig.sum()
    rng = np.random.default_rng(seed)
    sensitivity = []
    for size in [4, 8, 16, 24, 32]:
        samples = []
        for _ in range(subsets if size < n else 1):
            take = np.sort(rng.choice(n, size, replace=False))
            curve, _, _ = projection_curve(gram[np.ix_(take, take)], cross[take], held_sq)
            samples.append(float(curve[-1]))
        sensitivity.append(
            {
                "size": size,
                "draws": len(samples),
                "q05": float(np.quantile(samples, 0.05)),
                "median": float(np.median(samples)),
                "q95": float(np.quantile(samples, 0.95)),
            }
        )
    reconstruction = abs(float(np.trace(gram)) - sum(raw["eigenvalues"]))
    direct = float(cross @ np.linalg.pinv(gram, rcond=1e-10) @ cross / held_sq)
    if not np.isclose(raw["held_projection_all"], direct, rtol=1e-9, atol=1e-10):
        raise ValueError("Independent pseudoinverse projection disagrees")
    return {
        "teacher": obj["teacher"],
        "phase": obj["phase"],
        "parameter_count": obj["parameter_count"],
        "train_prompts": len(flat),
        "held_prompts": len(ids[held]),
        "raw": raw,
        "unit_gradient": normalized,
        "centered": {
            "eigenvalue_fractions": centered_frac.tolist(),
            "participation_rank": float(1 / np.sum(centered_frac**2)),
            "top1_trace_fraction": float(centered_frac[0]),
        },
        "subset_sensitivity": sensitivity,
        "checks": {"trace_reconstruction_error": reconstruction, "pseudoinverse_projection": direct},
    }


def plot_cases(cases, out):
    """Render data figures with a shared palette and explicit audit output."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from ..common.plotting import load_style

    fv = load_style()
    fv.apply_rc("paper", fontsize=8)
    out.mkdir(parents=True, exist_ok=True)
    colors = {"justrl": fv.HUES["vteal"], "polaris": fv.HUES["vorange"]}
    audit = {}

    def save(fig, stem):
        fig.canvas.draw()
        audit[stem] = [str(x) for x in fv.audit(fig)]
        for extension in ["png", "pdf"]:
            fig.savefig(out / f"{stem}.{extension}", dpi=220)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 3.15))
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.30, top=0.90, wspace=0.43)
    for row in cases:
        label = ("JustRL" if row["teacher"] == "justrl" else "Polaris") + (
            " / initial" if row["phase"] == "initial" else " / warm128"
        )
        style = "--" if row["phase"] == "initial" else "-"
        for ax, key in zip(axes, ["eigenvalue_fractions", "cumulative_held_projection"], strict=True):
            ax.plot(np.arange(1, 33), row["raw"][key], label=label, color=colors[row["teacher"]], linestyle=style)
            ax.set_xlabel("Number of leading modes")
            ax.set_xticks([1, 8, 16, 24, 32])
    axes[0].set(title="(a) Gradient Gram spectrum", yscale="log", ylabel="Eigenvalue / trace", ylim=(0.001, 1.1))
    axes[1].set(title="(b) Held64 projection", ylabel="Gradient energy fraction", ylim=(0, 1.02))
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, 0.06), ncol=2)
    fig.text(
        0.5, 0.015, "32 full-parameter loss gradients; not an output NTK.", ha="center", color=fv.NEUTRALS["baseline"]
    )
    save(fig, "gradient-spectrum")

    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    fig.subplots_adjust(left=0.12, right=0.97, bottom=0.26, top=0.88)
    for row in cases:
        if row["phase"] != "warm128":
            continue
        s = row["subset_sensitivity"]
        x = [r["size"] for r in s]
        color = colors[row["teacher"]]
        ax.plot(x, [r["median"] for r in s], color=color, marker="o", label=row["teacher"].title())
        ax.fill_between(x, [r["q05"] for r in s], [r["q95"] for r in s], color=color, alpha=0.18)
    ax.set(
        title="Warm128: subset-span sensitivity",
        xlabel="Number of B16 gradients in the span",
        ylabel="Held64 gradient energy captured",
        ylim=(0, 1.02),
        xticks=[4, 8, 16, 24, 32],
    )
    ax.legend(loc="center right")
    fig.text(
        0.5,
        0.025,
        "Bands: subset 5–95% range; not a training-seed confidence interval.",
        ha="center",
        color=fv.NEUTRALS["baseline"],
    )
    save(fig, "gradient-subsets")
    dump(out / "figure-audit.json", audit)


def main():
    """Read frozen matrices and write only to a separate analysis directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--figures", type=Path)
    parser.add_argument("--subsets", type=int, default=256)
    args = parser.parse_args()
    if args.data.resolve() == args.out.resolve() or args.subsets < 1:
        raise ValueError("Use a separate output directory and positive subset count")
    args.out.mkdir(parents=True, exist_ok=True)
    cases, reference_ids = [], None
    for teacher in ["justrl", "polaris"]:
        for phase in ["initial", "warm128"]:
            source = args.data / f"noise-{teacher}-{phase}.json"
            obj = read(source)
            if reference_ids is None:
                reference_ids = obj["prompt_ids"]
            elif obj["prompt_ids"] != reference_ids:
                raise ValueError("Input groups differ across cases")
            case = analyze_case(obj, args.subsets, seed=20261028)
            case["source"] = str(source.resolve())
            cases.append(case)
    dump(args.out / "spectral-results.json", cases)
    dump(
        args.out / "protocol.json",
        {
            "measurement": "target-conditioned raw loss-gradient Gram, not NTK",
            "source": str(args.data.resolve()),
            "cases": 4,
            "training_batches": 32,
            "prompts_per_batch": 16,
            "heldout_prompts": 64,
            "parameters": 1777088000,
            "source_horizon": 1024,
            "new_gpu_work": False,
            "source_is_production_e20_e48_checkpoint": False,
            "projection": "held-out squared norm captured by training-gradient span; oracle diagnostic",
            "subset_draws": args.subsets,
            "seed": 20261028,
        },
    )
    columns = [
        "teacher",
        "phase",
        "top1_trace_fraction",
        "participation_rank",
        "held_projection_top1",
        "held_projection_all",
        "pool_mean_held_cosine",
    ]
    with (args.out / "spectral-summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for case in cases:
            row = {k: case[k] if k in ["teacher", "phase"] else case["raw"][k] for k in columns}
            writer.writerow(row)
            print(row)
    if args.figures:
        plot_cases(cases, args.figures)


if __name__ == "__main__":
    main()
