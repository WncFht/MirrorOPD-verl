# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Render stage diagnostics and finite-span bounds from auditable gradient Grams.

Semantic ledger: colors identify teachers (JustRL/Polaris, RL-Zero/SFT).
Line styles identify uniform gradient versus finite-span oracle bound.
Every stage point uses its own frozen parameters and one unchanged input bank.
Local-gradient span spectra are NOT Fisher/self-kernel eigenframes.
Geometry ledger: stage axes are optimizer updates; heatmap cells are B8 gradient
pairs; finite-span bounds use the measured training span and held-out mean.
Historical panels explicitly identify N3 and L as existing frozen evidence.
"""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .geometry import calibrated_reweighting, cross_batch_jackknife, cross_receiver_projection, span_diagnostics
from .results import loss_signal, summary_rows, write_csv
from .stages import subset_summary

ROOT = Path(__file__).resolve().parents[5]
DEFAULT_STYLE = ROOT.parent / "dotfiles/.agents/skills/figure-viz/assets/mpl"
COLORS = {}
AUDITS = {}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(fig, path, caption, fv, *, caption_fontsize=6.5):
    fig.text(0.5, 0.015, caption, ha="center", va="bottom", color=fv.NEUTRALS["baseline"], fontsize=caption_fontsize)
    fig.canvas.draw()
    AUDITS[path.name] = [str(x) for x in fv.audit(fig)]
    fig.savefig(path.with_suffix(".png"), dpi=220)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def frame(ax, xlabel, ylabel):
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.spines[["top", "right"]].set_visible(False)


def historical(out, fv):
    directory = ROOT.parent / "wandb-logs/MirrorOPD-verl/analysis/batch-size-stage-20261027"
    rows, sources = [], {}
    for target in ["justrl", "polaris"]:
        for stage in ["initial", "warm128"]:
            path = directory / f"noise-{target}-{stage}.json"
            data = json.loads(path.read_text())
            G = np.array(data["gram"])
            weights = np.zeros(len(G))
            weights[data["groups"].index("Held64")] = 1
            result = span_diagnostics(G, list(range(32)), weights)
            result.update(target=target, stage=stage, C_batch=float(G[:32, :32].mean() / G[:32, :32].diagonal().mean()))
            rows.append(result)
            sources[str(path)] = digest(path)
    (out / "historical-span.json").write_text(json.dumps(dict(rows=rows, sources=sources), indent=2))
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.4))
    fig.subplots_adjust(bottom=0.35, top=0.94, wspace=0.65)
    labels = ["J/0", "J/128", "P/0", "P/128"]
    x = np.arange(4)
    for ax, key, ylabel in zip(
        axes[:2], ["C_batch", "effective_rank"], ["Batch coherence", "Effective rank"], strict=True
    ):
        ax.bar(x, [r[key] for r in rows], color=[COLORS[r["target"]] for r in rows])
        ax.set_xticks(x, labels, rotation=35)
        frame(ax, "Teacher / updates (N3)", ylabel)
    axes[2].plot(x, [r["uniform_cosine"] for r in rows], "o-", color=fv.HUES["vteal"], label="Uniform mean")
    axes[2].plot(x, [r["span_cosine_upper"] for r in rows], "s--", color=fv.HUES["vorange"], label="Oracle span bound")
    axes[2].set_xticks(x, labels, rotation=35)
    axes[2].set_ylim(0, 1.05)
    axes[2].legend(loc="lower left", fontsize=6.5)
    frame(axes[2], "Teacher / updates (N3)", "Cosine with Held64")
    save(
        fig,
        out / "historical_span",
        "Existing N3: 32 B16 gradients, fixed 512-prompt pool; oracle is an upper bound.",
        fv,
    )
    directory = ROOT.parent / "wandb-logs/MirrorOPD-verl/analysis/twenty-batch-pass-20261027"
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.4))
    fig.subplots_adjust(bottom=0.29, top=0.9, wspace=0.6)
    for target in ["justrl", "polaris"]:
        data = json.loads((directory / target / "update-probes.json").read_text())
        x = [r["training_batch"] for r in data]
        for ax, key in zip(axes, ["raw_train_held_cosine", "descent_alignment", "held_gradient_norm"], strict=True):
            ax.plot(x, [r[key] for r in data], "o-", markersize=2.5, color=COLORS[target], label=target)
    for ax, ylabel in zip(
        axes, ["Raw-gradient cosine", "Adam displacement alignment", "Held-gradient norm"], strict=True
    ):
        frame(ax, "Incoming batch (L)", ylabel)
        ax.set_xticks([1, 5, 10, 15, 20])
    axes[0].legend(fontsize=6.5)
    save(
        fig, out / "historical_adam", "Existing L: first update of each incoming batch; FP32 master, BF16 backbone.", fv
    )


def extract_span(data):
    G = np.array(data["geometry"]["gram"])
    train = [i for i, g in enumerate(data["groups"]) if g["split"] == "train" and g["domain"] == "math"]
    held = [i for i, g in enumerate(data["groups"]) if g["split"] == "heldout" and g["domain"] == "math"]
    if not train or not held:
        return None
    weights = np.zeros(len(G))
    weights[held] = 1 / len(held)
    result = span_diagnostics(G, train, weights)
    result["receiver_corrected_projection"] = cross_receiver_projection(G, train, held)
    result["train_cross_batch_jackknife"] = cross_batch_jackknife(G[np.ix_(train, train)])
    result["held_cross_batch_jackknife"] = cross_batch_jackknife(G[np.ix_(held, held)])
    rng = np.random.default_rng(20261109)
    trials = []
    for _ in range(100):
        order = rng.permutation(held)
        cal, test = np.zeros(len(G)), np.zeros(len(G))
        cal[order[: len(held) // 2]] = 2 / len(held)
        test[order[len(held) // 2 :]] = 2 / len(held)
        trials.append(calibrated_reweighting(G, train, cal, test))
    comparison = []
    rng = np.random.default_rng(20261109)
    for _ in range(100):
        shuffled = rng.permutation(held)
        source, receiver = shuffled[: len(held) // 2], shuffled[len(held) // 2 :]
        matched_train = rng.choice(train, size=len(source), replace=False)
        receiver_weight = np.zeros(len(G))
        receiver_weight[receiver] = 1 / len(receiver)
        from_train = span_diagnostics(G, matched_train, receiver_weight)
        from_new = span_diagnostics(G, source, receiver_weight)
        comparison.append(
            dict(
                train_panel_cosine=from_train["uniform_cosine"],
                independent_panel_cosine=from_new["uniform_cosine"],
                train_panel_bound=from_train["span_cosine_upper"],
                independent_panel_bound=from_new["span_cosine_upper"],
            )
        )
    result["matched_source_comparison"] = {
        key: dict(
            mean=float(np.mean([r[key] for r in comparison])),
            q025=float(np.quantile([r[key] for r in comparison], 0.025)),
            q975=float(np.quantile([r[key] for r in comparison], 0.975)),
        )
        for key in comparison[0]
    }
    result["matched_source_note"] = (
        "Both sources contain 6 B8 gradients; same disjoint 6-B8 receiver; 100 deterministic panel splits."
    )
    result["calibration_test"] = {
        key: dict(
            mean=float(np.mean([r[key] for r in trials])),
            q025=float(np.quantile([r[key] for r in trials], 0.025)),
            q975=float(np.quantile([r[key] for r in trials], 0.975)),
        )
        for key in trials[0]
    }
    return result


def r1_figures(results, out, fv):
    native = [(t, d) for t, d in results if t["model"] == "r1" and "-target-" not in t["id"]]
    if not native:
        return
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.5))
    fig.subplots_adjust(bottom=0.20, top=0.94, hspace=0.6, wspace=0.65)
    metrics = [
        ("loss_mean", "RKL / fixed H"),
        ("P", "Pressure P"),
        ("A", "Trajectory accessibility A"),
        ("C", "Finite-bank coherence"),
        ("cross_record_normalized", "Distinct-record coherence"),
        ("Xi", "Collective gain Xi"),
        ("PXi", "Gradient energy P Xi"),
        ("PXi_over_L", "Relative gradient rate P Xi / L"),
    ]
    for target in ["justrl", "polaris"]:
        pairs = sorted([(t, d) for t, d in native if t["target"] == target], key=lambda pair: pair[0]["stage"])
        for ax, (metric, ylabel) in zip(axes.flat, metrics, strict=True):
            summaries = [d["summaries"]["heldout/math"] for t, d in pairs]
            values = [(s | loss_signal(s))[metric] for s in summaries]
            ax.plot([t["stage"] for t, d in pairs], values, "o-", color=COLORS[target], label=target, markersize=3)
            frame(ax, "Optimizer updates", ylabel)
            ax.set_xticks([0, 80, 160, 240, 320])
            if metric in ["loss_mean", "P", "A", "Xi", "PXi", "PXi_over_L"]:
                ax.set_yscale("log")
    axes[0, 0].legend(fontsize=6.5)
    save(
        fig,
        out / "r1_stage_pac",
        "L checkpoints: same 96 held-out math trajectories, H1024; A/C use trajectory gradients.",
        fv,
    )
    span_rows = []
    for task, data in native:
        span = extract_span(data)
        span_rows.append(dict(task=task["id"], target=task["target"], stage=task["stage"], **span))
    (out / "r1-span.json").write_text(json.dumps(span_rows, indent=2))
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.8))
    fig.subplots_adjust(bottom=0.28, top=0.92, wspace=0.65)
    for target in ["justrl", "polaris"]:
        selected = sorted([r for r in span_rows if r["target"] == target], key=lambda r: r["stage"])
        x = [r["stage"] for r in selected]
        corrected = [r["receiver_corrected_projection"] for r in selected]
        axes[0].errorbar(
            x,
            [r["projected_shared_fraction"] for r in corrected],
            yerr=[1.96 * r["fraction_jackknife_se"] for r in corrected],
            fmt="o-",
            color=COLORS[target],
            label=target,
            markersize=3,
        )
        for source, style in [("train_panel", "-"), ("independent_panel", "--")]:
            axes[1].plot(
                x,
                [r["matched_source_comparison"][f"{source}_bound"]["mean"] for r in selected],
                style,
                color=COLORS[target],
                label=f"{target}: {source.split('_')[0]}",
            )
        for mode, style in [("uniform", "-"), ("positive_calibrated", "--"), ("signed_calibrated", ":")]:
            axes[2].plot(
                x,
                [r["calibration_test"][mode]["mean"] for r in selected],
                style,
                color=COLORS[target],
                label=f"{target}: {mode.split('_')[0]}",
            )
    for ax, label in zip(
        axes, ["Shared energy captured", "Matched-source oracle cosine", "Disjoint-test cosine"], strict=True
    ):
        frame(ax, "Optimizer updates", label)
        ax.set_xticks([0, 160, 320])
        ax.legend(fontsize=5.2)
    save(
        fig,
        out / "r1_propagation_controls",
        "Left: receiver self-energy removed, jackknife 95% interval. Middle/right: 100 fixed-panel splits.",
        fv,
    )
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.7))
    fig.subplots_adjust(bottom=0.28, top=0.92, wspace=0.65)
    for target in ["justrl", "polaris"]:
        selected = sorted([r for r in span_rows if r["target"] == target], key=lambda r: r["stage"])
        x = [r["stage"] for r in selected]
        axes[0].plot(
            x,
            [r["uniform_cosine"] for r in selected],
            "o-",
            color=COLORS[target],
            label=f"{target}: mean",
            markersize=3,
        )
        axes[0].plot(
            x, [r["span_cosine_upper"] for r in selected], "--", color=COLORS[target], label=f"{target}: bound"
        )
        axes[1].plot(
            x, [r["outside_span_energy"] for r in selected], "o-", color=COLORS[target], label=target, markersize=3
        )
    frame(axes[0], "Optimizer updates", "Cosine with held-out math")
    frame(axes[1], "Optimizer updates", "Held energy outside train span")
    axes[0].set_ylim(-0.05, 1.05)
    axes[1].set_ylim(0, 1)
    axes[0].legend(fontsize=6)
    save(
        fig,
        out / "r1_stage_span",
        "Finite span of 20 B8 train gradients; target is mean of 12 disjoint held-out B8 gradients.",
        fv,
    )
    endpoints = {(t["target"], t["stage"]): d for t, d in native if t["stage"] in [0, 320]}
    if len(endpoints) == 4:
        gram_grid(endpoints, ["justrl", "polaris"], out / "r1_gram", fv)


def gram_grid(endpoints, targets, path, fv):
    fig, axes = plt.subplots(2, 2, figsize=(5.5, 5))
    fig.subplots_adjust(bottom=0.18, top=0.94, hspace=0.5, wspace=0.35, right=0.84)
    for row, target in enumerate(targets):
        for col, stage in enumerate([0, 320]):
            data = endpoints[(target, stage)]
            G = np.array(data["geometry"]["cosine"])
            image = axes[row, col].imshow(G, vmin=-1, vmax=1, cmap=fv.ramp_cmap("div"), rasterized=True)
            axes[row, col].set_title(f"{target} / {stage} updates")
            frame(axes[row, col], "Batch index", "Batch index")
    bar = fig.colorbar(image, cax=fig.add_axes([0.88, 0.22, 0.025, 0.63]))
    bar.set_label("Gradient cosine")
    save(fig, path, "Exact B8 gradient cosine; ordering and group labels are recorded in the result JSON.", fv)


def panel_span(data):
    """A/B diagnostic split of public panels, not formal-training membership."""
    G = np.asarray(data["geometry"]["gram"])
    bydomain = {}
    for i, group in enumerate(data["groups"]):
        bydomain.setdefault(group["domain"], []).append(i)
    sources, receivers = [], []
    for indices in bydomain.values():
        midpoint = len(indices) // 2
        sources.extend(indices[:midpoint])
        receivers.extend(indices[midpoint:])
    weights = np.zeros(len(G))
    weights[receivers] = 1 / len(receivers)
    result = span_diagnostics(G, sources, weights)
    result["receiver_corrected_projection"] = cross_receiver_projection(G, sources, receivers)
    result["source_cross_batch_jackknife"] = cross_batch_jackknife(G[np.ix_(sources, sources)])
    result["receiver_cross_batch_jackknife"] = cross_batch_jackknife(G[np.ix_(receivers, receivers)])
    result["source_indices"] = sources
    result["receiver_indices"] = receivers
    result["split_note"] = "Within-public-panel diagnostic split; not formal-training versus prompt-heldout."
    return result


def mimo_figures(results, out, fv):
    native = [(t, d) for t, d in results if t["model"] == "mimo" and "-target-" not in t["id"]]
    if not native:
        return
    fig, axes = plt.subplots(2, 3, figsize=(6.8, 4.5))
    fig.subplots_adjust(bottom=0.2, top=0.94, hspace=0.65, wspace=0.65)
    for row, role in enumerate(["initial", "teacher"]):
        for target in ["rlzero", "sft"]:
            pairs = sorted(
                [(t, d) for t, d in native if t["target"] == target and f"-{role}-" in t["id"]],
                key=lambda p: p[0]["stage"],
            )
            if not pairs:
                continue
            for ax, metric, label in zip(
                axes[row],
                ["P", "cross_record_normalized", "Xi"],
                ["Pressure P", "Distinct-record coherence", "Collective gain Xi"],
                strict=True,
            ):
                ax.plot(
                    [t["stage"] for t, d in pairs],
                    [d["summaries"]["all"][metric] for t, d in pairs],
                    "o-",
                    color=COLORS[target],
                    label=target,
                    markersize=3,
                )
                frame(ax, f"Updates / {role} bank", label)
                ax.set_xticks([0, 320])
                if metric in ["P", "Xi"]:
                    ax.set_yscale("log")
    axes[0, 0].legend(fontsize=6.5)
    save(
        fig,
        out / "mimo_stage_pac",
        "MiMo: same 128 questions, 32 per domain; fixed H1024, initial and teacher banks kept separate.",
        fv,
    )
    spans = [
        dict(task=t["id"], target=t["target"], stage=t["stage"], bank=t["bank"], **panel_span(d)) for t, d in native
    ]
    (out / "mimo-span.json").write_text(json.dumps(spans, indent=2))
    endpoints = {(t["target"], t["stage"]): d for t, d in native if "-initial-" in t["id"]}
    if len(endpoints) == 4:
        gram_grid(endpoints, ["rlzero", "sft"], out / "mimo_gram", fv)
    for metric, ylabel in [
        ("P", "Pressure P"),
        ("cross_record_normalized", "Distinct-record coherence"),
        ("Xi", "Collective gain Xi"),
    ]:
        fig, axes = plt.subplots(2, 2, figsize=(5.5, 4.2))
        fig.subplots_adjust(bottom=0.20, top=0.92, hspace=0.65, wspace=0.6)
        for ax, domain in zip(axes.flat, ["math", "code", "science", "if"], strict=True):
            for target in ["rlzero", "sft"]:
                pairs = sorted(
                    [(t, d) for t, d in native if t["target"] == target and "-initial-" in t["id"]],
                    key=lambda p: p[0]["stage"],
                )
                if not pairs:
                    continue
                ax.plot(
                    [t["stage"] for t, d in pairs],
                    [d["summaries"][f"heldout/{domain}"][metric] for t, d in pairs],
                    "o-",
                    color=COLORS[target],
                    label=target,
                    markersize=3,
                )
            frame(ax, "Optimizer updates", f"{domain}: {ylabel}")
            ax.set_xticks([0, 320])
            if metric in ["P", "Xi"]:
                ax.set_yscale("log")
        axes[0, 0].legend(fontsize=6.5)
        save(
            fig,
            out / f"mimo_domains_{metric}",
            "Initial-Base-prefix bank; 32 questions per domain, fixed response H1024.",
            fv,
        )


def point_figures(run, out, fv):
    rows = []
    diagnostics = []
    for path in sorted((run / "point-probe").glob("*/s*.json")):
        data = json.loads(path.read_text())
        reference = data.get("teacher_copy_reference", [])
        for field, value in data["fields"].items():
            target, mode = field.split("/")
            if mode == "live":
                diagnostics.append(
                    dict(
                        model=data["model"],
                        branch=data["branch"],
                        stage=data["stage"],
                        target=target,
                        live_rkl=float(np.mean(value["live_kl"])),
                        copy_rkl=float(np.mean([r["rkl"] for r in reference])) if reference else None,
                        copy_pressure=float(np.mean([r["pressure"] for r in reference])) if reference else None,
                        fisher={
                            k: float(np.mean([r[k] for r in value["live_fisher"]])) for k in value["live_fisher"][0]
                        },
                    )
                )
            rows.append(
                dict(
                    model=data["model"],
                    branch=data["branch"],
                    stage=data["stage"],
                    target=target,
                    field=mode,
                    **value["summary"],
                )
            )
    if not rows:
        return
    (out / "prefix-fisher.json").write_text(json.dumps(diagnostics, indent=2))
    with (out / "prefix-pac.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for model, branches in [("r1", ["justrl", "polaris"]), ("mimo", ["rlzero", "sft"])]:
        selected = [r for r in rows if r["model"] == model]
        if not selected:
            continue
        fig, axes = plt.subplots(2, 2, figsize=(5.5, 4.3))
        fig.subplots_adjust(bottom=0.20, top=0.92, hspace=0.65, wspace=0.6)
        for col, branch in enumerate(branches):
            for target in branches:
                frozen = sorted(
                    [
                        r
                        for r in selected
                        if r["branch"] == branch and r["target"] == target and r["field"] == "frozen_B0"
                    ],
                    key=lambda r: r["stage"],
                )
                live = sorted(
                    [r for r in selected if r["branch"] == branch and r["target"] == target and r["field"] == "live"],
                    key=lambda r: r["stage"],
                )
                if not frozen or frozen[0]["stage"] != 0 or not live:
                    continue
                x = [r["stage"] for r in frozen]
                axes[0, col].plot(
                    x, [r["A"] / frozen[0]["A"] for r in frozen], "o-", color=COLORS[target], label=target, markersize=3
                )
                axes[1, col].plot(
                    [r["stage"] for r in live],
                    [r["P"] / live[0]["P"] for r in live],
                    "o-",
                    color=COLORS[target],
                    label=target,
                    markersize=3,
                )
            axes[0, col].set_title(f"{model}: trained on {branch}")
            frame(axes[0, col], "Optimizer updates", "Frozen-request A / initial A")
            frame(axes[1, col], "Optimizer updates", "Live P / initial P")
            axes[0, col].axhline(1, color=fv.NEUTRALS["baseline"], ls=":", linewidth=0.8)
            axes[0, col].set_yscale("log")
            axes[1, col].set_yscale("log")
            axes[0, col].legend(fontsize=6.5)
        save(
            fig,
            out / f"{model}_frozen_request",
            "Same sampled prefixes: 4 trajectories per split/domain, first/middle/last response positions.",
            fv,
        )


def e91_figures(results, out, fv):
    controls = [(t, d) for t, d in results if t["model"] == "mimo-control"]
    if not controls:
        return
    fig, axes = plt.subplots(2, 3, figsize=(6.8, 4.5))
    fig.subplots_adjust(bottom=0.25, top=0.94, hspace=0.65, wspace=0.65)
    metrics = [
        ("loss_mean", "RKL / fixed H"),
        ("P", "Pressure P"),
        ("A", "Trajectory accessibility A"),
        ("C", "Finite-bank coherence"),
        ("cross_record_normalized", "Distinct-record coherence"),
        ("Xi", "Collective gain Xi"),
    ]
    spans = []
    for target in ["rlzero", "sft"]:
        pairs = [
            (0, t, d)
            for t, d in results
            if t["model"] == "mimo" and t["target"] == target and t["bank"] == "mimo-initial" and t["stage"] == 0
        ]
        pairs += [(1 if t["stage"] == 0 else 2, t, d) for t, d in controls if t["target"] == target]
        pairs.sort(key=lambda item: item[0])
        for ax, (metric, ylabel) in zip(axes.flat, metrics, strict=True):
            ax.plot(
                [x for x, t, d in pairs],
                [d["summaries"]["all"][metric] for x, t, d in pairs],
                "o-",
                color=COLORS[target],
                label=target,
                markersize=3,
            )
            frame(ax, "Student checkpoint", ylabel)
            ax.set_xticks([0, 1, 2], ["Base", "SFT", "E91 / 320"], rotation=20)
            if metric in ["loss_mean", "P", "A", "Xi"]:
                ax.set_yscale("log")
        spans += [dict(task=t["id"], **panel_span(d)) for x, t, d in pairs]
    axes[0, 0].legend(fontsize=6.5)
    save(
        fig,
        out / "mimo_e91_control",
        "Same Base-prefix panel and both teacher targets. E91 trained SFT toward RL-Zero; not E72/E73 endpoints.",
        fv,
    )
    (out / "mimo-e91-span.json").write_text(json.dumps(spans, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--style", type=Path, default=DEFAULT_STYLE)
    args = parser.parse_args()
    sys.path.insert(0, str(args.style))
    import fvstyle as fv

    fv.apply_rc("paper")
    COLORS.update(justrl=fv.HUES["vteal"], polaris=fv.HUES["vorange"], rlzero=fv.HUES["vteal"], sft=fv.HUES["vorange"])
    out = args.run / "analysis"
    out.mkdir(exist_ok=True)
    historical(out, fv)
    manifest = json.loads((args.run / "manifest.json").read_text())
    results = []
    rows = []
    sources = {}
    task_paths = [(task, args.run) for task in manifest["tasks"]]
    control_run = args.run / "mimo-e91-control"
    if (control_run / "manifest.json").exists():
        control_manifest = json.loads((control_run / "manifest.json").read_text())
        task_paths += [(task, control_run) for task in control_manifest["tasks"]]
    for task, task_root in task_paths:
        path = task_root / "results" / task["id"] / "result.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        if task["model"] == "r1":
            train = [i for i, g in enumerate(data["groups"]) if g["split"] == "train"]
            seen = task["stage"] // 16
            G = np.asarray(data["geometry"]["gram"])
            for key, indices in [("seen/math", train[:seen]), ("future/math", train[seen:])]:
                if indices:
                    data["summaries"][key] = subset_summary(data["groups"], G, indices)
        results.append((task, data))
        sources[str(path)] = digest(path)
        rows.extend(summary_rows(task, data))
    write_csv(out / "pac.csv", rows)
    r1_figures(results, out, fv)
    mimo_figures(results, out, fv)
    point_figures(args.run, out, fv)
    e91_figures(results, out, fv)
    (out / "audit.json").write_text(
        json.dumps(
            dict(
                completed=len(results),
                planned=len(task_paths),
                unavailable=manifest.get("unavailable_tasks", []),
                sources=sources,
                figures=AUDITS,
            ),
            indent=2,
        )
    )
    print(json.dumps(dict(completed=len(results), planned=len(manifest["tasks"]), out=str(out))))


if __name__ == "__main__":
    main()
