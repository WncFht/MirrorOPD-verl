# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Plot native four-domain evaluation artifacts, keeping legacy 0–320 reports separate.

Ledger: points are completed outer readouts or committed update records. Colors
identify domains; line styles identify trajectory producers. History cells are
question means within an observed training batch, not all historical questions.
Missing observations stay absent. No horizon normalization or interpolation.
"""

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from examples.mirroropd.research.core.io import dump, write_csv
from examples.mirroropd.research.core.plotting import load_style, save_figure
from verl.trainer.evaluation.artifacts import ArtifactStore, file_sha256
from verl.trainer.evaluation.metrics import summarize_rkl


def load_plot_snapshot(path):
    """Replay published scalar evidence without access to models or rollout tokens."""
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    for name in ("plot-inputs.json", "evaluation-metrics.csv", "question-rkl.csv"):
        if file_sha256(path / name) != manifest["artifacts"][name]:
            raise ValueError(f"Plot input checksum differs: {name}")
    data = json.loads((path / "plot-inputs.json").read_text())
    events = {e["outer"]: dict(e, metrics={}) for e in data["events"]}
    with (path / "evaluation-metrics.csv").open() as stream:
        for row in csv.DictReader(stream):
            events[int(row["outer"])]["metrics"][row["metric"]] = float(row["value"])
    data["events"] = list(events.values())
    integers = {"outer", "tokens", "history_batch", "sample_index"}
    reals = {"rkl_sum", "uncoarsened_rkl_sum", "student_excluded_mass_sum", "teacher_excluded_mass_sum"}
    with (path / "question-rkl.csv").open() as stream:
        data["scores"] = [
            {k: None if v == "" else int(v) if k in integers else float(v) if k in reals else v for k, v in row.items()}
            for row in csv.DictReader(stream)
        ]
    updates = {}
    if data.pop("has_updates"):
        if file_sha256(path / "updates.csv") != manifest["artifacts"]["updates.csv"]:
            raise ValueError("Plot update checksum differs")
        with (path / "updates.csv").open() as stream:
            for row in csv.DictReader(stream):
                step = int(row["step"])
                updates.setdefault(step, {"step": step})[row["metric"]] = float(row["value"])
    data["updates"] = list(updates.values())
    return data


def load_run(path):
    if (Path(path) / "plot-inputs.json").exists():
        return load_plot_snapshot(path)
    store = ArtifactStore(path)
    manifest = store.manifest()
    if manifest["kind"] != "unified_evaluation_run":
        raise ValueError("This reader accepts only native unified evaluation artifacts")
    events, scores, updates, sources, incomplete = [], [], [], {}, []
    for directory in sorted((store.root / "events").glob("outer-*")):
        event = ArtifactStore(directory)
        value = event.manifest()
        if value["status"] != "complete":
            incomplete.append(directory.name)
            continue
        metadata = value["metadata"]
        outer = metadata["plan"]["outer"]
        events.append({"outer": outer, **metadata})
        scores.extend(dict(row, outer=outer) for row in event.read_rows("rkl.parquet"))
        sources[str(event.manifest_path)] = file_sha256(event.manifest_path)
    for name in sorted(manifest["files"]):
        if name.startswith("updates/"):
            updates.extend(dict(step=r["step"], **dict(r["metrics"])) for r in store.iter_rows(name))
            sources[str(store.root / name)] = manifest["files"][name]["sha256"]
    if not events:
        raise ValueError("No completed evaluation events to plot")
    bootstrap = Path(manifest["metadata"]["bootstrap"])
    source_enabled = manifest["metadata"]["configuration"].get("source_enabled", (bootstrap / "source").exists())
    source_bank = ArtifactStore(bootstrap / ("source" if source_enabled else "initial"))
    source_manifest = source_bank.manifest(complete=True)
    thinking = {
        row["thinking_open"]
        for name in source_manifest["files"]
        for row in source_bank.iter_rows(name, columns=["thinking_open"])
    }
    input_mode = "thinking" if thinking == {True} else "non-thinking" if thinking == {False} else "mixed"
    sources[str(source_bank.manifest_path)] = file_sha256(source_bank.manifest_path)
    measures = {row.get("measure_id") for row in scores}
    if len(measures) > 1:
        raise ValueError("A plot cannot combine different RKL probability spaces")
    distribution = next((e["timing"].get("distribution") for e in events if e["timing"].get("distribution")), None)
    models = {}
    for name in ("run.yaml", "acceptance.yaml"):
        candidate = store.root.parent / name
        if candidate.exists():
            config = OmegaConf.load(candidate)
            for role, key in (
                ("student", "actor_rollout_ref.model.path"),
                ("teacher", "distillation.teacher_models.teacher_model.model_path"),
            ):
                model = Path(OmegaConf.select(config, key))
                models[role] = next(
                    (part.split("--", 1)[1] for part in reversed(model.parts) if "--" in part), model.name
                )
            sources[str(candidate)] = file_sha256(candidate)
            break
    return dict(
        events=events,
        scores=scores,
        updates=updates,
        sources=sources,
        status=manifest["status"],
        incomplete=incomplete,
        config=manifest["metadata"]["configuration"],
        models=models,
        distribution=distribution,
        input_mode=input_mode,
        source_run=str(Path(path).resolve()),
    )


def render(path, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = load_run(path)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        existing = json.loads((output / "manifest.json").read_text())
        if existing["source_run"] != data["source_run"]:
            raise ValueError("Each scientific run must own a separate figure directory")
    style = load_style()
    style.apply_rc("paper", fontsize=10)
    colors = {
        d: style.HUES[h]
        for d, h in zip(("math", "code", "science", "if"), ("vteal", "vviolet", "vblue", "vorange"), strict=True)
    }
    neutral = style.NEUTRALS["vink"]
    training_color = colors[data["config"].get("training_domain", "math")]
    sizes = [
        re.search(r"(\d+(?:\.\d+)?)B(?:-|$)", data["models"].get(role, ""), re.I) for role in ("student", "teacher")
    ]
    same_size = all(sizes) and sizes[0].group(1) == sizes[1].group(1)
    marker = "o" if same_size else "^"
    figures = []

    def save(fig, name, caption):
        if data["models"]:
            fig.suptitle(
                f"{data['models']['teacher']} → {data['models']['student']}\n"
                f"Training: {data['config'].get('training_domain', 'math').capitalize()} | {data['input_mode']} input",
                fontsize=11,
                y=0.98,
            )
        fig.canvas.draw()
        fig.tight_layout(rect=(0, 0.10, 1, 0.90))
        if data["distribution"]:
            definition = data["distribution"]
            mode = definition["spec"]["mode"]
            if mode != "native" or definition["termination_ids"]:
                count = definition["spec"]["common_vocab_size"] or len(definition["spec"]["student_ids"])
                caption += f" Space: {mode}, C={count}, EOS actions={len(definition['termination_ids'])}."
        save_figure(fig, output / name, caption)
        figures.append(name)

    def series(axis, key, label, color, linestyle="-"):
        points = [(event["outer"], event["metrics"][key]) for event in data["events"] if key in event["metrics"]]
        if points:
            axis.plot(
                *zip(*points, strict=True),
                marker=marker,
                markerfacecolor=color if same_size else "none",
                label=label,
                color=color,
                linestyle=linestyle,
            )

    def training_boundary_series(axis):
        batches = defaultdict(list)
        for row in data["updates"]:
            if "step/outer" in row and "distill/full_kl" in row:
                batches[int(row["step/outer"])].append(row)
        for offset, label, linestyle in [(0, "First-update training RKL", "-"), (-1, "Last-update training RKL", "--")]:
            points = [
                (outer, sorted(rows, key=lambda row: row["step"])[offset]["distill/full_kl"])
                for outer, rows in sorted(batches.items())
            ]
            if points:
                axis.plot(*zip(*points, strict=True), label=label, color=training_color, linestyle=linestyle)

    def common_limits(axes):
        axes = list(axes)
        values = [float(y) for axis in axes for line in axis.lines for y in line.get_ydata() if math.isfinite(y)]
        if values:
            low, high = min(0.0, min(values)), max(values)
            margin = max((high - low) * 0.05, 1e-6)
            for axis in axes:
                axis.set_ylim(low - margin, high + margin)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for bank, label, color, suffix in [
        ("source_heldout", "Source heldout: initial student", training_color, "question_mean"),
        ("history", "Training history: batch equal", neutral, "equal_batch_question_mean"),
    ]:
        series(axes[0], f"eval/rkl/{bank}/{suffix}", label, color)
    has_source_readouts = bool(axes[0].lines)
    if not has_source_readouts:
        training_boundary_series(axes[0])
    for bank, label, linestyle in [
        ("public_initial", "Initial-student trajectories", "-"),
        ("public_teacher", "Teacher trajectories", "--"),
        ("public_current", "Current-student trajectories", ":"),
    ]:
        series(axes[1], f"eval/rkl/{bank}/all/question_mean", label, neutral, linestyle)
    for axis, title in zip(
        axes,
        ("Source dataset" if has_source_readouts else "Training batch: token mean", "Public benchmarks: domain equal"),
        strict=True,
    ):
        axis.set(xlabel="Completed outer steps", ylabel="RKL (nats)", title=title)
        axis.legend(fontsize=8)
    common_limits(axes)
    save(fig, "rkl-by-trajectory", "Within-question token mean; history batch equal; public domains equal.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for bank, label, color, suffix in [
        ("source_heldout", "Source heldout", training_color, "question_mean"),
        ("history", "Training history", neutral, "equal_batch_question_mean"),
    ]:
        series(axes[0], f"eval/rkl/{bank}/{suffix}", label, color)
    if not axes[0].lines:
        training_boundary_series(axes[0])
    for domain, color in colors.items():
        series(axes[1], f"eval/rkl/public_initial/{domain}/question_mean", domain.capitalize(), color)
    for axis, title in zip(
        axes,
        (
            "Source and actual training history" if has_source_readouts else "Training batch: token mean",
            "Fixed initial-student public bank",
        ),
        strict=True,
    ):
        axis.set(xlabel="Completed outer steps", ylabel="RKL (nats; log scale)", yscale="log", title=title)
        axis.legend(fontsize=8)
    positive = [y for ax in axes for line in ax.lines for y in line.get_ydata() if y > 0]
    if positive:
        for ax in axes:
            ax.set_ylim(min(positive) / 1.5, max(positive) * 1.5)
    save(fig, "rkl-log-detail", "Fixed source/public trajectories; history membership changes after each outer.")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for (domain, color), axis in zip(colors.items(), axes.flat, strict=True):
        for bank, label, linestyle in [
            ("public_initial", "Initial student", "-"),
            ("public_teacher", "Teacher", "--"),
            ("public_current", "Current student", ":"),
        ]:
            series(axis, f"eval/rkl/{bank}/{domain}/question_mean", f"{domain.capitalize()}: {label}", color, linestyle)
        axis.set(xlabel="Completed outer steps", ylabel="RKL (nats)", title=domain.capitalize())
        axis.legend(fontsize=8)
    common_limits(axes.flat)
    save(fig, "rkl-by-domain", "Each public panel uses its frozen questions and sample index 0.")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for (domain, color), axis in zip(colors.items(), axes.flat, strict=True):
        for bank, label, linestyle in (("public_initial", "Initial student", "-"), ("public_teacher", "Teacher", "--")):
            series(
                axis, f"eval/rkl/{bank}/{domain}/reduction_percent", f"{domain.capitalize()}: {label}", color, linestyle
            )
        axis.set(xlabel="Completed outer steps", ylabel="RKL reduction from initial (%)", title=domain.capitalize())
        axis.legend(fontsize=8)
    common_limits(axes.flat)
    save(fig, "fixed-bank-reduction", "Each curve retains identical trajectories and its own initial RKL baseline.")

    if data["updates"]:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
        for axis, key, ylabel in [
            (axes[0], "distill/full_kl", "Training RKL: token mean (nats)"),
            (axes[1], "optim/actor/grad_norm_preclip", "Gradient norm before clipping"),
        ]:
            points = [
                (r.get("step/update", r["step"]), r[key]) for r in data["updates"] if key in r and math.isfinite(r[key])
            ]
            if points:
                axis.plot(*zip(*points, strict=True), color=training_color)
            else:
                axis.text(0.5, 0.5, "Metric unavailable", transform=axis.transAxes, ha="center")
            axis.set(xlabel="Optimizer update attempts", ylabel=ylabel)
        save(fig, "training-updates", "Actual training-loss records; evaluation RKL uses a different averaging rule.")

        batches = defaultdict(list)
        for row in data["updates"]:
            if "step/outer" in row and "distill/full_kl" in row:
                batches[int(row["step/outer"])].append(row)
        if batches:
            fig, axis = plt.subplots(figsize=(8.5, 4.8))
            for offset, label, linestyle in [(0, "First update loss", "-"), (-1, "Last update loss", "--")]:
                points = [
                    (outer, sorted(rows, key=lambda r: r["step"])[offset]["distill/full_kl"])
                    for outer, rows in sorted(batches.items())
                ]
                axis.plot(
                    *zip(*points, strict=True), label=label, color=training_color, linestyle=linestyle, marker=marker
                )
            series(axis, "eval/rkl/source_heldout/question_mean", "Source heldout after update", neutral, ":")
            axis.set(xlabel="Completed outer steps", ylabel="RKL (nats; log scale)", yscale="log")
            axis.legend(fontsize=8)
            save(
                fig,
                "reuse-profile",
                "First/last losses precede their respective updates; loss token-mean, heldout question-mean.",
            )

    final_outer = max(event["outer"] for event in data["events"])
    final_rows = [r for r in data["scores"] if r["outer"] == final_outer and r["bank"].startswith("public_")]
    if final_rows:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        all_values = [r["rkl_sum"] / r["tokens"] for r in final_rows if r["tokens"] > 0]
        for axis, (domain, color) in zip(axes.flat, colors.items(), strict=True):
            for bank, label, linestyle in [
                ("public_initial", "Initial student", "-"),
                ("public_teacher", "Teacher", "--"),
                ("public_current", "Current student", ":"),
            ]:
                values = sorted(
                    r["rkl_sum"] / r["tokens"]
                    for r in final_rows
                    if r["bank"] == bank and r["domain"] == domain and r["tokens"] > 0
                )
                if values:
                    axis.step(
                        values,
                        np.arange(1, len(values) + 1) / len(values),
                        where="post",
                        color=color,
                        linestyle=linestyle,
                        label=f"{domain.upper() if domain == 'if' else domain.capitalize()}: {label}",
                    )
            axis.set(
                xlabel="Per-question RKL (nats)",
                ylabel="Empirical cumulative fraction",
                title="IF" if domain == "if" else domain.capitalize(),
                ylim=(0, 1.03),
            )
            if all_values and min(all_values) > 0:
                axis.set_xscale("log")
                axis.set_xlim(min(all_values) / 1.2, max(all_values) * 1.2)
            axis.legend(fontsize=8)
        save(
            fig,
            "rkl-tail-distribution",
            "Final per-question distributions include every observed tail; no outlier removal.",
        )

    history = defaultdict(list)
    for row in data["scores"]:
        if row["bank"] == "history":
            history[row["history_batch"], row["outer"]].append(row)
    if history:
        batches, outers = sorted({key[0] for key in history}), sorted({key[1] for key in history})
        matrix = np.full((len(batches), len(outers)), np.nan)
        cells = []
        for (batch, outer), rows in history.items():
            summary = summarize_rkl(rows)
            if "mean" in summary:
                matrix[batches.index(batch), outers.index(outer)] = summary["mean"]
            cells.append({"training_batch": batch, "outer": outer, **summary})
        write_csv(output / "history-cells.csv", cells)
        fig, axis = plt.subplots(figsize=(10, 5))
        artist = axis.imshow(matrix, aspect="auto", interpolation="none", cmap=style.ramp_cmap())
        positions = np.unique(np.linspace(0, len(outers) - 1, min(10, len(outers))).astype(int))
        axis.set_xticks(positions, [outers[i] for i in positions])
        positions = np.unique(np.linspace(0, len(batches) - 1, min(10, len(batches))).astype(int))
        axis.set_yticks(positions, [batches[i] for i in positions])
        axis.set(xlabel="Completed outer steps", ylabel="Original training batch")
        fig.colorbar(artist, ax=axis, label="History question-mean RKL (nats)")
        save(
            fig,
            "training-history",
            "Each cell uses the retained random sample from that batch; blank cells are unobserved.",
        )

    first, last = min(e["outer"] for e in data["events"]), max(e["outer"] for e in data["events"])
    panel_specs = [
        (bank, label)
        for bank, label in (
            ("source_heldout", "Source heldout"),
            ("public_initial", "Public: initial student"),
            ("public_teacher", "Public: teacher"),
        )
        if any(row["bank"] == bank for row in data["scores"])
    ]
    fig, axes = plt.subplots(1, len(panel_specs), figsize=(5 * len(panel_specs), 5))
    axes = np.atleast_1d(axes)
    for axis, (bank, label) in zip(axes, panel_specs, strict=True):
        endpoints = {
            outer: {r["id"]: r for r in data["scores"] if r["outer"] == outer and r["bank"] == bank}
            for outer in (first, last)
        }
        maximum = 0
        for domain, color in colors.items():
            points = []
            for key, before in endpoints[first].items():
                after = endpoints[last].get(key)
                if (
                    after
                    and before["domain"] == domain
                    and before["tokens"] == after["tokens"] > 0
                    and math.isfinite(before["rkl_sum"])
                    and math.isfinite(after["rkl_sum"])
                ):
                    points.append((before["rkl_sum"] / before["tokens"], after["rkl_sum"] / after["tokens"]))
            if points:
                axis.scatter(*zip(*points, strict=True), s=12, alpha=0.65, color=color, label=domain)
                maximum = max(maximum, max(max(p) for p in points))
        if maximum > 0:
            axis.plot(
                [0, maximum],
                [0, maximum],
                color=style.NEUTRALS["baseline"],
                linestyle="--",
                linewidth=1,
                label="Unchanged RKL",
            )
        axis.set(xlabel=f"Question RKL at outer {first}", ylabel=f"Question RKL at outer {last}", title=label)
        axis.set_aspect("equal", adjustable="box")
        axis.legend(fontsize=8)
    bound = max(max(axis.get_xlim()[1], axis.get_ylim()[1]) for axis in axes)
    for axis in axes:
        axis.set(xlim=(-0.02 * bound, bound), ylim=(-0.02 * bound, bound))
    save(
        fig,
        "paired-question-rkl",
        "Identical fixed trajectories at both endpoints; points below the diagonal improved.",
    )

    if history:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
        series(axes[0], "eval/rkl/history/paired_absolute_reduction", "Retained old trajectories", training_color)
        axes[0].axhline(0, color=style.NEUTRALS["baseline"], linestyle="--", linewidth=1)
        axes[0].set(xlabel="Completed outer steps", ylabel="Previous minus current RKL (nats)")
        series(axes[1], "eval/rkl/history/paired_rows", "Paired history rows", neutral)
        axes[1].set(xlabel="Completed outer steps", ylabel="Common trajectory count")
        save(
            fig,
            "paired-history-change",
            "Only trajectories retained from the preceding readout; batch-equal aggregation.",
        )

    fig, axis = plt.subplots(figsize=(10, 4.6))
    series(axis, "perf/timing/evaluation_seconds", "Evaluation callback", neutral)
    axis.plot(
        [e["outer"] for e in data["events"]],
        [e["timing"]["seconds"] for e in data["events"]],
        marker="o",
        color=colors["code"],
        label="Serial RKL windows, including audit",
    )
    axis.set(xlabel="Completed outer steps", ylabel="Wall-clock seconds")
    axis.legend(fontsize=8)
    save(fig, "evaluation-timing", "Reported callback timers exclude subsequent native weight synchronization.")

    # Initial/current accuracy names are explicit in the event metadata.
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for (domain, color), axis in zip(colors.items(), axes.flat, strict=True):
        specs = [d for d in data["config"]["datasets"] if d["domain"] == domain]
        for index, spec in enumerate(specs):
            points = []
            for event in data["events"]:
                prefix = "initial_student/" if event["outer"] == 0 else ""
                key = f"eval/accuracy/{prefix}{spec['name']}/avg_at_n"
                if key in event["metrics"]:
                    points.append((event["outer"], 100 * event["metrics"][key]))
            if points:
                axis.plot(
                    *zip(*points, strict=True),
                    marker=("o", "s", "^", "D", "v")[index % 5],
                    color=color,
                    alpha=1 - index * 0.12,
                    label=f"{spec['name']} avg@{spec['n']}",
                )
        axis.set(xlabel="Completed outer steps", ylabel="Accuracy (%)", title=domain.capitalize(), ylim=(-3, 103))
        axis.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
    save(fig, "accuracy", "Full benchmark questions; per-question mean over the declared number of samples.")

    write_csv(
        output / "evaluation-metrics.csv",
        [{"outer": e["outer"], "metric": k, "value": v} for e in data["events"] for k, v in e["metrics"].items()],
    )
    write_csv(output / "question-rkl.csv", data["scores"])
    if data["updates"]:
        write_csv(
            output / "updates.csv",
            [
                {"step": r["step"], "metric": k, "value": v}
                for r in data["updates"]
                for k, v in r.items()
                if k != "step"
            ],
        )
    snapshot = {k: v for k, v in data.items() if k not in {"events", "scores", "updates"}}
    snapshot["events"] = [{k: v for k, v in e.items() if k != "metrics"} for e in data["events"]]
    snapshot["has_updates"] = bool(data["updates"])
    dump(output / "plot-inputs.json", snapshot)
    dump(
        output / "manifest.json",
        dict(
            source_run=data["source_run"],
            sources=data["sources"],
            status=data["status"],
            incomplete_events=data["incomplete"],
            figures=figures,
            generator_sha256=file_sha256(__file__),
            artifacts={p.name: file_sha256(p) for p in sorted(output.iterdir()) if p.name != "manifest.json"},
        ),
    )
    return figures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    render(args.run, args.output)


if __name__ == "__main__":
    main()
