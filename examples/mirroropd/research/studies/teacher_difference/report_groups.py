# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Large-panel PAC, clustered bootstrap transfer and local-spectrum figures.

Resampling units are original B8 groups with basic bootstrap intervals. Source and receiver groups are drawn
independently, with shared draws across stages and live/frozen fields. Bands
measure fixed-bank sampling variability, not training-seed variability.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch
from scipy.stats import t as student_t

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .metrics import moment_summary
from .report_stages import DEFAULT_STYLE, frame
from .results import write_csv
from .runtime import write_json


def bootstrap_weights(n, replicates, generator):
    draws = generator.integers(n, size=(replicates, n))
    return np.eye(n)[draws].mean(1)


def basic_interval(point, samples):
    """Reflected bootstrap interval; account for bias of squared-mean norms."""
    low, high = np.quantile(samples, [0.025, 0.975])
    return 2 * point - high, 2 * point - low


def transfer_bootstrap(gram, source, receiver, losses, *, replicates=2000, seed=20261109, return_samples=False):
    generator = np.random.default_rng(seed)
    ws = bootstrap_weights(len(source), replicates, generator)
    wr = bootstrap_weights(len(receiver), replicates, generator)
    ss, rr, sr = gram[np.ix_(source, source)], gram[np.ix_(receiver, receiver)], gram[np.ix_(source, receiver)]
    src = np.einsum("bi,ij,bj->b", ws, ss, ws)
    dst = np.einsum("bi,ij,bj->b", wr, rr, wr)
    cross = np.einsum("bi,ij,bj->b", ws, sr, wr)
    risk = wr @ losses[receiver]
    values = dict(
        cross_inner=cross,
        cross_cosine=cross / np.sqrt(np.maximum(src * dst, 1e-30)),
        cross_over_source=cross / np.maximum(src, 1e-30),
        cross_over_loss=cross / np.maximum(risk, 1e-30),
    )
    point = dict(
        cross_inner=sr.mean(),
        cross_cosine=sr.mean() / max(ss.mean() * rr.mean(), 1e-30) ** 0.5,
        cross_over_source=sr.mean() / max(ss.mean(), 1e-30),
        cross_over_loss=sr.mean() / max(losses[receiver].mean(), 1e-30),
    )
    for key, samples in values.items():
        point[f"{key}_low"], point[f"{key}_high"] = basic_interval(point[key], samples)

    # Squared mean norms acquire large diagonal bias in naive resampling.
    # Use independent-source/receiver delete-one errors for normalized angles.
    def normalized(src, dst):
        a = gram[np.ix_(src, src)].mean()
        b = gram[np.ix_(dst, dst)].mean()
        c = gram[np.ix_(src, dst)].mean()
        return dict(cross_cosine=c / max(a * b, 1e-30) ** 0.5, cross_over_source=c / max(a, 1e-30))

    left = [normalized([i for i in source if i != removed], receiver) for removed in source]
    right = [normalized(source, [i for i in receiver if i != removed]) for removed in receiver]
    critical = student_t.ppf(0.975, min(len(source), len(receiver)) - 1)
    for key in ["cross_cosine", "cross_over_source"]:
        variance = (len(source) - 1) * np.var([r[key] for r in left]) + (len(receiver) - 1) * np.var(
            [r[key] for r in right]
        )
        error = critical * variance**0.5
        point[f"{key}_low"], point[f"{key}_high"] = point[key] - error, point[key] + error
    return (point, values) if return_samples else point


def summarize(data, artifact, branch, *, replicates):
    sites, groups = data["sites"], artifact["groups"]
    batch_indices = {}
    if "records" in data["task"]:
        bank = json.loads(Path(data["task"]["records"]).read_text())
        batch_indices = {r["group"]: r["batch_index"] for r in bank if r["split"] == "train"}
    rows, transfers, spectral = [], [], []
    for field, tensor in artifact["grams"].items():
        target, mode = field.split("/")
        gram = tensor.numpy()
        group_losses = np.array([np.mean([sites[i]["profiles"][target]["loss"] for i in g["sites"]]) for g in groups])
        source = [i for i, g in enumerate(groups) if g["split"] == "train"]
        receiver = [i for i, g in enumerate(groups) if g["split"] == "heldout"]
        meta = dict(branch=branch, stage=data["task"]["stage"], target=target, mode=mode)
        transfers.append(
            dict(
                **meta,
                source_role="all-train",
                source_groups=len(source),
                receiver_groups=len(receiver),
                **transfer_bootstrap(gram, source, receiver, group_losses, replicates=replicates),
            )
        )
        future = [
            i
            for i in source
            if groups[i]["group"] in batch_indices and batch_indices[groups[i]["group"]] >= data["task"]["stage"] // 16
        ]
        if future:
            transfers.append(
                dict(
                    **meta,
                    source_role="future-train",
                    source_groups=len(future),
                    receiver_groups=len(receiver),
                    **transfer_bootstrap(gram, future, receiver, group_losses, replicates=replicates),
                )
            )
        for split, indices in [("train", source), ("heldout", receiver)]:
            block = gram[np.ix_(indices, indices)]
            members = [i for g in indices for i in groups[g]["sites"]]
            group_pressure = np.array(
                [np.mean([sites[i]["fields"][field]["pressure"] for i in groups[g]["sites"]]) for g in indices]
            )
            pressure = group_pressure.mean()
            second = np.mean([sites[i]["fields"][field]["gradient_norm_sq"] for i in members])
            question_pressure = np.array([sites[i]["fields"][field]["pressure"] for i in members])
            loss = group_losses[indices].mean()
            moments = moment_summary(
                pressure=pressure, second=np.trace(block) / len(block), mean_norm=block.mean(), samples=len(block)
            )
            row = dict(
                **meta,
                split=split,
                questions=len(members),
                groups=len(indices),
                loss=loss,
                P_over_L=pressure / max(loss, 1e-30),
                A_prefix=second / max(pressure, 1e-30),
                largest_question_pressure_share=question_pressure.max() / max(question_pressure.sum(), 1e-30),
                pressure_effective_questions=question_pressure.sum() ** 2 / max((question_pressure**2).sum(), 1e-30),
                PXi=block.mean(),
                PXi_over_L=block.mean() / max(loss, 1e-30),
                **moments,
            )
            # Same unit as PAC: B8 group. Retain off-diagonal negatives.
            generator = np.random.default_rng(20261109)
            weights = bootstrap_weights(len(block), replicates, generator)
            energies = np.einsum("bi,ij,bj->b", weights, block, weights)
            risk = weights @ group_losses[indices]
            sampled_pressure = weights @ group_pressure
            diagonal = weights @ np.diag(block)
            for key, values in [
                ("PXi", energies),
                ("PXi_over_L", energies / np.maximum(risk, 1e-30)),
                ("loss", risk),
                ("P", sampled_pressure),
                ("A", diagonal / np.maximum(sampled_pressure, 1e-30)),
                ("C", energies / np.maximum(diagonal, 1e-30)),
                ("Xi", energies / np.maximum(sampled_pressure, 1e-30)),
            ]:
                row[f"{key}_low"], row[f"{key}_high"] = basic_interval(row[key], values)
            jackknife = []
            for removed in range(len(block)) if len(block) > 2 else []:
                keep = [i for i in range(len(block)) if i != removed]
                reduced = block[np.ix_(keep, keep)]
                diagonal_mean = np.diag(reduced).mean()
                off = (reduced.sum() - np.trace(reduced)) / (len(reduced) * (len(reduced) - 1))
                jackknife.append(off / max(diagonal_mean, 1e-30))
            se = ((len(block) - 1) * np.var(jackknife)) ** 0.5 if jackknife else float("nan")
            row["cross_record_normalized_low"] = row["cross_record_normalized"] - 1.96 * se
            row["cross_record_normalized_high"] = row["cross_record_normalized"] + 1.96 * se
            rows.append(row)
    for i, spectrum in artifact["spectra"].items():
        for field, metrics in sites[i]["fields"].items():
            target, mode = field.split("/")
            if target != branch:
                continue
            spectral.append(
                dict(
                    branch=branch,
                    stage=data["task"]["stage"],
                    site=i,
                    row_id=sites[i]["row_id"],
                    split=sites[i]["split"],
                    mode=mode,
                    leading_strength=float(spectrum["eigenvalues"][:3].sum()),
                    top3_projection=sum(metrics["leading_request_energy"]),
                    top3_accessibility=sum(metrics["leading_accessibility_contribution"]),
                    total_accessibility=metrics["rayleigh"],
                    top3_accessibility_fraction=sum(metrics["leading_accessibility_contribution"])
                    / max(metrics["rayleigh"], 1e-30),
                    max_residual=float(spectrum["residual_relative"].max()),
                )
            )
    return rows, transfers, spectral


def paired_contrasts(records, *, replicates):
    samples = {}
    points = {}
    for data, artifact in records:
        groups = artifact["groups"]
        source = [i for i, g in enumerate(groups) if g["split"] == "train"]
        receiver = [i for i, g in enumerate(groups) if g["split"] == "heldout"]
        for field, gram in artifact["grams"].items():
            target, mode = field.split("/")
            losses = np.array(
                [np.mean([data["sites"][i]["profiles"][target]["loss"] for i in g["sites"]]) for g in groups]
            )
            point, values = transfer_bootstrap(
                gram.numpy(), source, receiver, losses, replicates=replicates, return_samples=True
            )
            key = (target, data["task"]["stage"], mode)
            samples[key], points[key] = values, point
    rows = []
    pairs = []
    for stage in [80, 240]:
        for mode in ["live", "frozen_B0"]:
            pairs.append((("justrl", stage, mode), ("polaris", stage, mode), "justrl-minus-polaris", mode))
        for branch in ["justrl", "polaris"]:
            pairs.append(((branch, stage, "live"), (branch, stage, "frozen_B0"), "live-minus-frozen", branch))
    for left, right, contrast, label in pairs:
        if left not in samples or right not in samples:
            continue
        for metric in ["cross_inner", "cross_over_loss"]:
            difference = samples[left][metric] - samples[right][metric]
            point = points[left][metric] - points[right][metric]
            lower, upper = basic_interval(point, difference)
            rows.append(
                dict(
                    stage=left[1], mode=label, metric=metric, contrast=contrast, difference=point, low=lower, high=upper
                )
            )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--style", type=Path, default=DEFAULT_STYLE)
    args = parser.parse_args()
    torch.set_num_threads(4)
    rows, transfers, spectral, sources = [], [], [], {}
    gram_checks = []
    records = []
    for path in sorted(args.root.glob("*-s*.json")):
        data = json.loads(path.read_text())
        artifact = torch.load(path.with_suffix(".pt"), weights_only=True)
        if json.loads(json.dumps(artifact["signature"])) != data["signature"]:
            raise ValueError(f"Identity mismatch: {path}")
        inputs = [path, path.with_suffix(".pt")]
        if "records" in data["task"]:
            inputs.append(Path(data["task"]["records"]))
        for p in inputs:
            sources[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
        for field, gram in artifact["grams"].items():
            eigenvalues = torch.linalg.eigvalsh(gram.double())
            scale = max(float(eigenvalues.abs().max()), 1e-30)
            error = float(eigenvalues.min()) / scale
            if error < -1e-8:
                raise ValueError(f"Non-PSD group Gram: {path}, {field}, {error}")
            gram_checks.append(dict(checkpoint=path.stem, field=field, minimum_relative_eigenvalue=error))
        records.append((data, artifact))
    if not records:
        raise ValueError("No completed group measurements")
    first_request = records[0][0]["signature"]["request"]
    first_groups = records[0][1]["groups"]
    for data, artifact in records:
        if data["signature"]["request"] != first_request or artifact["groups"] != first_groups:
            raise ValueError("Group report requires identical requests and question groups across stages")
    for data, artifact in records:
        branches = (
            sorted({f.split("/")[0] for f in artifact["grams"]})
            if data["task"]["stage"] == 0
            else [data["task"]["target"]]
        )
        for branch in branches:
            a, b, c = summarize(data, artifact, branch, replicates=args.replicates)
            rows.extend(a)
            transfers.extend(b)
            spectral.extend(c)
    initial_artifact = next(artifact for data, artifact in records if data["task"]["stage"] == 0)
    artifacts = {
        (data["task"].get("branch", data["task"]["target"]), data["task"]["stage"]): artifact
        for data, artifact in records
    }
    baseline = {(r["branch"], r["site"], r["mode"]): r for r in spectral if r["stage"] == 0}
    for r in spectral:
        base = baseline[r["branch"], r["site"], r["mode"]]
        r["G3"] = r["leading_strength"] / base["leading_strength"]
        r["projection_gain"] = r["top3_projection"] - base["top3_projection"]
        current = initial_artifact if r["stage"] == 0 else artifacts[r["branch"], r["stage"]]
        u0 = initial_artifact["spectra"][r["site"]]["vectors"].double()
        u = current["spectra"][r["site"]]["vectors"].double()
        r["O3"] = float((u0.T @ u).square().sum() / 3)
    spectral_summary = []
    keys = sorted({(r["branch"], r["stage"], r["mode"], r["split"]) for r in spectral})
    for branch, stage, mode, split in keys:
        group = [
            r for r in spectral if (r["branch"], r["stage"], r["mode"], r["split"]) == (branch, stage, mode, split)
        ]
        row = dict(branch=branch, stage=stage, mode=mode, split=split, questions=len(group))
        draws = np.random.default_rng(20261109).integers(len(group), size=(args.replicates, len(group)))
        for key in [
            "G3",
            "O3",
            "top3_projection",
            "projection_gain",
            "top3_accessibility_fraction",
            "total_accessibility",
            "max_residual",
        ]:
            values = np.array([r[key] for r in group])
            row[key] = np.median(values)
            row[f"{key}_low"], row[f"{key}_high"] = np.quantile(np.median(values[draws], axis=1), [0.025, 0.975])
        spectral_summary.append(row)
    out = args.root / "analysis"
    out.mkdir(exist_ok=True)
    write_csv(out / "panels.csv", rows)
    write_csv(out / "transfer.csv", transfers)
    contrasts = paired_contrasts(records, replicates=args.replicates)
    if contrasts:
        write_csv(out / "paired-contrasts.csv", contrasts)
    if spectral:
        write_csv(out / "spectra.csv", spectral)
        write_csv(out / "spectral-summary.csv", spectral_summary)
    sys.path.insert(0, str(args.style))
    import fvstyle as fv

    fv.apply_rc("paper")
    colors = dict(justrl=fv.HUES["vteal"], polaris=fv.HUES["vorange"])
    audits = {}

    def plot(name, metrics, records):
        fig, axes = plt.subplots(1, len(metrics), figsize=(3.0 * len(metrics), 4.2), squeeze=False)
        fig.subplots_adjust(left=0.09, right=0.98, bottom=0.34, top=0.93, wspace=0.5)
        for ax, (key, label) in zip(axes[0], metrics, strict=True):
            for branch in colors:
                for mode, linestyle in [("live", "-"), ("frozen_B0", "--")]:
                    chosen = [
                        r
                        for r in records
                        if r["branch"] == branch and r["mode"] == mode and r.get("target", branch) == branch
                    ]
                    stages = sorted({r["stage"] for r in chosen})
                    if not stages:
                        continue
                    values = [np.median([r[key] for r in chosen if r["stage"] == s]) for s in stages]
                    ax.plot(
                        stages,
                        values,
                        linestyle,
                        color=colors[branch],
                        marker="o",
                        markersize=3,
                        label=f"{branch} {mode}",
                    )
                    if chosen and key + "_low" in chosen[0]:
                        ax.fill_between(
                            stages,
                            [next(r[key + "_low"] for r in chosen if r["stage"] == s) for s in stages],
                            [next(r[key + "_high"] for r in chosen if r["stage"] == s) for s in stages],
                            color=colors[branch],
                            alpha=0.12,
                        )
            frame(ax, "Optimizer updates", label)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=10, frameon=False)
        fig.canvas.draw()
        audits[name] = [str(issue) for issue in fv.audit(fig)]
        for ext in ["png", "pdf"]:
            fig.savefig(out / f"{name}.{ext}", dpi=200)
        plt.close(fig)

    plot(
        "transfer",
        [
            ("cross_cosine", "Train / heldout mean-gradient cosine"),
            ("cross_over_source", "Cross / source self energy"),
            ("cross_over_loss", "Cross signal / heldout loss"),
        ],
        [r for r in transfers if r["source_role"] == "all-train"],
    )
    future_rows = [r for r in transfers if r["source_role"] == "future-train"]
    if future_rows:
        plot(
            "future_transfer",
            [
                ("cross_cosine", "Future / heldout gradient cosine"),
                ("cross_inner", "Future / heldout cross inner product"),
                ("cross_over_loss", "Future signal / heldout loss"),
            ],
            future_rows,
        )
    for split in ["train", "heldout"]:
        subset = [r for r in rows if r["split"] == split]
        plot(
            f"pac_{split}",
            [
                ("P", "Mean request energy"),
                ("A", "B8 accessibility"),
                ("cross_record_normalized", "Distinct B8 coherence"),
            ],
            subset,
        )
        plot(
            f"loss_signal_{split}",
            [
                ("loss", "Prefix reverse KL"),
                ("PXi", "Mean-gradient energy"),
                ("PXi_over_L", "Mean-gradient energy / loss"),
            ],
            [r for r in subset if r["mode"] == "live"],
        )
        if spectral_summary:
            plot(
                f"spectrum_{split}",
                [
                    ("G3", "Top-3 strength / initial"),
                    ("O3", "Top-3 subspace overlap"),
                    ("projection_gain", "Top-3 projection gain"),
                ],
                [r for r in spectral_summary if r["split"] == split],
            )
            plot(
                f"request_spectrum_{split}",
                [
                    ("top3_projection", "Request energy in top 3"),
                    ("top3_accessibility_fraction", "Top-3 share of accessibility"),
                    ("total_accessibility", "Local request accessibility"),
                ],
                [r for r in spectral_summary if r["split"] == split],
            )
    write_json(
        out / "audit.json",
        dict(
            sources=sources,
            gram_checks=gram_checks,
            figure_audits=audits,
            bootstrap_replicates=args.replicates,
            bootstrap_unit=(
                "original B8 groups; independent source/receiver draws; paired across stages; basic intervals"
            ),
            max_ritz_residual=max((r["max_residual"] for r in spectral), default=None),
            note=(
                "Intervals describe fixed-bank variability; cosine/source ratios use two-sample jackknife-t. "
                "PXi resampling includes diagonal finite-sample energy."
            ),
        ),
    )


if __name__ == "__main__":
    main()
