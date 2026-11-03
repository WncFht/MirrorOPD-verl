# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Freeze and plot a partial native stage-transfer run without touching model execution.

Ledger: update points are committed optimizer forwards (before that update);
outer points are completed batches; RKL points are completed fixed-prefix readouts.
Each domain keeps one color. Time bars partition a measured outer without adding
nested rollout timers. Token passes count reuse, not checkpoint recomputation.
The historical reference is a different model, not a controlled speed comparison.
"""

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from examples.mirroropd.research.core.io import dump, write_csv
from examples.mirroropd.research.core.plotting import load_style, save_figure
from examples.mirroropd.research.studies.pipeline.plot_unified_evaluation import load_run
from verl.trainer.evaluation.artifacts import ArtifactStore, file_sha256
from verl.trainer.evaluation.metrics import summarize_rkl


def means(rows, keys):
    return {key: statistics.mean(row[key] for row in rows) for key in keys}


def collect(args):
    run, destination = Path(args.run), Path(args.data)
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "manifest.json").exists():
        raise FileExistsError("Keep the published snapshot immutable; use a new data directory")
    metrics_path = Path(args.metrics) if getattr(args, "metrics", None) else run / "metrics.jsonl"
    raw = metrics_path.read_bytes()
    # Ignore only a final uncommitted line from the live writer.
    lines = raw[: raw.rfind(b"\n") + 1].decode().splitlines()
    records = [json.loads(line)["data"] for line in lines]
    outer_records = {
        int(row["step/outer"]): row
        for row in records
        if "perf/timing/outer_step_seconds" in row and row["step/outer"] <= args.outer_limit
    }
    last = max(outer_records)
    assert sorted(outer_records) == list(range(1, last + 1))
    config = OmegaConf.load(args.config)
    reuse, gpus = int(config.actor_rollout_ref.actor.ppo_epochs), int(config.trainer.n_gpus_per_node)
    data = load_run(run / "evaluation")
    data["events"] = [event for event in data["events"] if event["outer"] <= last]
    events = {event["outer"]: event for event in data["events"]}
    updates = [row for row in data["updates"] if row["step"] <= last * reuse]
    assert [row["step"] for row in updates] == list(range(1, last * reuse + 1))
    update_rows = []
    for row in updates:
        loss, grad = row["optim/actor/loss"], row["optim/actor/grad_norm_preclip"]
        assert math.isfinite(loss) and math.isfinite(grad)
        update_rows.append(dict(update=row["step"], outer=int(row["step/outer"]), loss=loss, grad_norm=grad))
    trajectory_rows = []
    store = ArtifactStore(run / "evaluation")
    published = store.manifest()
    for outer, row in outer_records.items():
        name = f"trajectories/outer-{outer:03d}.parquet"
        samples = store.read_rows(name, columns=["prompt_tokens", "response_tokens", "valid_tokens"])
        total = sum(s["prompt_tokens"] + s["response_tokens"] for s in samples)
        assert total == row["rollout/total_tokens"] and len(samples) == config.data.train_batch_size
        data["sources"][str(store.root / name)] = published["files"][name]["sha256"]
        evaluation = events.get(outer, {}).get("metrics", {}).get("perf/timing/evaluation_seconds", 0)
        actor, rollout, wall = (
            row[f"perf/timing/{key}_seconds"] for key in ["update_actor", "rollout_pipeline", "outer_step"]
        )
        other = wall - actor - rollout - evaluation
        assert other >= -1e-6
        sync = row.get("perf/timing/update_weights_seconds")
        if sync is None:
            # The terminal outer does not synchronize weights for another rollout.
            assert outer == int(config.trainer.total_training_steps)
            sync = 0.0
        trajectory_rows.append(
            dict(
                outer=outer,
                optimizer_updates=outer * reuse,
                questions=len(samples),
                input_tokens=total,
                response_tokens=sum(s["response_tokens"] for s in samples),
                mean_response_length=row["rollout/response_length/mean"],
                length_finish_fraction=row["rollout/finish_reason/length_count"] / len(samples),
                actor_seconds=actor,
                actor_seconds_per_update=actor / reuse,
                rollout_seconds=rollout,
                evaluation_seconds=evaluation,
                other_seconds=max(0, other),
                outer_seconds=wall,
                teacher_seconds=row["perf/timing/teacher_forward_seconds"]
                + row["perf/timing/teacher_transfer_seconds"],
                sync_seconds=sync,
                token_passes_per_gpu_second=total * reuse / actor / gpus,
                actor_mfu=row["perf/actor/mfu"],
                allocated_peak_gib=row["perf/actor/allocated_peak_gib"],
                balanced_max_min_ratio=row["perf/load_balance/balanced_max"] / row["perf/load_balance/balanced_min"],
            )
        )
    rkl = []
    for outer, event in events.items():
        for bank in ["public_initial", "public_teacher"]:
            for domain in ["all", "math", "code", "science", "if"]:
                selected = [
                    r
                    for r in data["scores"]
                    if r["outer"] == outer and r["bank"] == bank and (domain == "all" or r["domain"] == domain)
                ]
                aggregate = summarize_rkl(selected, equal_groups="domain" if domain == "all" else None)
                measured = event["metrics"][f"eval/rkl/{bank}/{domain}/question_mean"]
                assert aggregate["invalid_rows"] == 0 and math.isclose(aggregate["mean"], measured, rel_tol=1e-12)
                rkl.append(
                    dict(
                        outer=outer,
                        optimizer_updates=outer * reuse,
                        bank=bank,
                        domain=domain,
                        rkl=measured,
                        questions=len(selected),
                    )
                )
    accuracy = []
    bootstrap = Path(config.trainer.evaluation.bootstrap)
    for role in ["initial", "teacher"]:
        path = bootstrap / (role + "-accuracy") / "manifest.json"
        metrics = json.loads(path.read_text())["metadata"]["metrics"]
        data["sources"][str(path)] = file_sha256(path)
        accuracy.extend(
            dict(model=role, dataset=k.split("/")[2], accuracy=v) for k, v in metrics.items() if k.endswith("/avg_at_n")
        )
    reference_path = Path(args.reference)
    reference = json.loads((reference_path / "summary.json").read_text())
    ref_records = [
        json.loads(line)["data"] for line in (Path(reference["run"]) / "metrics.jsonl").read_text().splitlines()
    ]
    with (reference_path / "evaluation-nodes.csv").open() as stream:
        ordinary_ids = {int(row["outer"]) for row in csv.DictReader(stream) if int(row["source_history_off_rows"]) == 0}
    historical = [
        row for row in ref_records if "perf/timing/outer_step_seconds" in row and int(row["step/outer"]) in ordinary_ids
    ]
    historical_actor = sum(row["perf/timing/update_actor_seconds"] for row in historical)
    historical_inputs = sum(row["rollout/total_tokens"] for row in historical)
    ordinary = [row for row in trajectory_rows if row["evaluation_seconds"] == 0]
    stage_keys = [
        "outer_seconds",
        "actor_seconds",
        "rollout_seconds",
        "teacher_seconds",
        "sync_seconds",
        "input_tokens",
        "mean_response_length",
        "actor_mfu",
        "allocated_peak_gib",
        "balanced_max_min_ratio",
    ]
    summary = dict(
        label=args.label,
        outer=last,
        updates=len(updates),
        reuse=reuse,
        gpus=gpus,
        completed_readouts=sorted(events),
        ordinary_outers=[r["outer"] for r in ordinary],
        ordinary=means(ordinary, stage_keys),
        recent5=means(trajectory_rows[-5:], stage_keys),
        actor_share=sum(r["actor_seconds"] for r in ordinary) / sum(r["outer_seconds"] for r in ordinary),
        token_passes_per_gpu_second=sum(r["input_tokens"] for r in ordinary)
        * reuse
        / sum(r["actor_seconds"] for r in ordinary)
        / gpus,
        first_update=update_rows[0],
        last_update=update_rows[-1],
        last_outer=trajectory_rows[-1],
        first_batch_last_update=update_rows[reuse - 1],
        last_batch_first_update=update_rows[-reuse],
        initial_rkl=[r for r in rkl if r["outer"] == 0],
        last_rkl=[r for r in rkl if r["outer"] == max(events)],
        historical=dict(
            label="JustRL / R1 1.5B",
            ordinary_outers=len(historical),
            actor_seconds=historical_actor / len(historical),
            input_tokens=historical_inputs / len(historical),
            rollout_seconds=statistics.mean(r["perf/timing/rollout_pipeline_seconds"] for r in historical),
            outer_seconds=statistics.mean(r["perf/timing/outer_step_seconds"] for r in historical),
            source_history_seconds=statistics.mean(r["perf/timing/evaluation_seconds"] for r in historical),
            token_passes_per_gpu_second=historical_inputs
            * reference["reuse"]
            / historical_actor
            / len(reference["gpus"]),
        ),
        setting={
            key: OmegaConf.to_container(value, resolve=False) if OmegaConf.is_config(value) else value
            for key in [
                "actor_rollout_ref.model.path",
                "distillation.teacher_models.teacher_model.model_path",
                "actor_rollout_ref.actor.optim",
                "actor_rollout_ref.actor.fsdp_config",
                "distillation.distillation_loss",
                "actor_rollout_ref.actor.ppo_max_token_len_per_gpu",
                "actor_rollout_ref.rollout.max_num_seqs",
                "actor_rollout_ref.rollout.gpu_memory_utilization",
                "trainer.evaluation.generation_concurrency",
            ]
            for value in [OmegaConf.select(config, key)]
        },
    )
    for name, rows in [
        ("updates", update_rows),
        ("outers", trajectory_rows),
        ("fixed-bank-rkl", rkl),
        ("baseline-accuracy", accuracy),
    ]:
        write_csv(destination / (name + ".csv"), rows)
    dump(destination / "plot-data.json", dict(updates=update_rows, outers=trajectory_rows, rkl=rkl, summary=summary))
    dump(destination / "summary.json", summary)
    data["sources"][str(Path(args.config).resolve())] = file_sha256(args.config)
    data["sources"][str(reference_path / "summary.json")] = file_sha256(reference_path / "summary.json")
    dump(
        destination / "manifest.json",
        dict(
            source_run=str(run.resolve()),
            outer_cutoff=last,
            source_metrics_sha256=hashlib.sha256(raw).hexdigest(),
            source_metrics_path=str(metrics_path.resolve()),
            source_files=data["sources"],
            generator_sha256=file_sha256(__file__),
            artifacts={p.name: file_sha256(p) for p in destination.iterdir() if p.is_file()},
            notes=[
                "Snapshot cutoff is the last completed outer; final accuracy requires a complete event",
                "RKL independently recomputed from per-question rows",
                "Ordinary-outer throughput uses total token visits / total actor time, with matching outer sets",
            ],
        ),
    )
    return json.loads((destination / "plot-data.json").read_text())


def render(data, args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fv = load_style()
    fv.apply_rc("paper", fontsize=8.5)
    color, gray = fv.HUES["vteal"], fv.NEUTRALS["baseline"]
    output, performance = Path(args.figures), Path(args.performance_figures)
    output.mkdir(parents=True, exist_ok=True)
    performance.mkdir(parents=True, exist_ok=True)
    updates, outers, summary = data["updates"], data["outers"], data["summary"]
    x = [row["update"] for row in updates]
    steps = [row["outer"] for row in outers]
    fig, axes = plt.subplots(2, 2, figsize=(5.5, 4.35))
    for ax, key, label, tag in zip(
        axes[0],
        ["loss", "grad_norm"],
        ["Training RKL (nats/token)", "Pre-clip gradient norm"],
        ["(a)", "(b)"],
        strict=True,
    ):
        ax.plot(x, [r[key] for r in updates], color=color, linewidth=0.65)
        ax.set(xlabel="Optimizer update", ylabel=label, yscale="log")
        ax.set_title(tag, loc="left")
    axes[0, 1].axhline(1, color=gray, linestyle=":", linewidth=0.8, label="Clip threshold")
    axes[0, 1].legend(loc="upper right", frameon=False, fontsize=6.5)
    axes[1, 0].plot(steps, [r["mean_response_length"] for r in outers], color=color, marker="o", markersize=2.5)
    axes[1, 0].axhline(7168, color=gray, linestyle=":", linewidth=0.8, label="Response cap")
    axes[1, 0].set(xlabel="Outer", ylabel="Mean response tokens", ylim=(0, 8000))
    axes[1, 0].set_title("(c)", loc="left")
    axes[1, 0].legend(loc="lower right", frameon=False, fontsize=6.5)
    axes[1, 1].plot(steps, [100 * r["length_finish_fraction"] for r in outers], color=color, marker="o", markersize=2.5)
    axes[1, 1].set(xlabel="Outer", ylabel="Length-limit finishes (%)", ylim=(0, 100))
    axes[1, 1].set_title("(d)", loc="left")
    fig.tight_layout(rect=(0, 0.07, 1, 1), h_pad=1.1, w_pad=1)
    save_figure(
        fig,
        output / "training",
        f"{args.label}: {summary['outer']} completed batches, {summary['updates']} updates; unsmoothed.",
        caption_size=6.7,
    )

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.8))
    domains = dict(zip(["math", "code", "science", "if"], ["vteal", "vviolet", "vblue", "vorange"], strict=True))
    for ax, bank, title in zip(
        axes,
        ["public_initial", "public_teacher"],
        ["(a) Fixed Base prefixes", "(b) Fixed teacher prefixes"],
        strict=True,
    ):
        for domain, hue in domains.items():
            rows = [r for r in data["rkl"] if r["bank"] == bank and r["domain"] == domain]
            ax.plot(
                [r["outer"] for r in rows],
                [r["rkl"] for r in rows],
                marker="o",
                markersize=2.5,
                color=fv.HUES[hue],
                label="IF" if domain == "if" else domain.title(),
            )
        ax.set(xlabel="Outer", ylabel="Question-mean RKL (nats/token)", yscale="log", ylim=(0.09, 3.3))
        ax.set_title(title, loc="left")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.065), ncol=4, frameon=False, fontsize=7)
    fig.tight_layout(rect=(0, 0.19, 1, 1), w_pad=1)
    save_figure(
        fig,
        output / "fixed-bank-rkl",
        "Same 128 questions/domain and sample-0 prefixes at every point; lower is closer.",
        caption_size=6.5,
    )

    fig, axes = plt.subplots(2, 1, figsize=(5.5, 4.4))
    bottom = np.zeros(len(outers))
    for key, label, shade in [
        ("actor_seconds", f"Actor ({summary['reuse']} updates)", color),
        ("rollout_seconds", "Rollout", fv.HUES["vorange"]),
        ("evaluation_seconds", "Evaluation", fv.HUES["vviolet"]),
        ("other_seconds", "Teacher / sync / other", gray),
    ]:
        values = np.array([r[key] for r in outers]) / 60
        axes[0].bar(steps, values, bottom=bottom, color=shade, label=label, width=0.8)
        bottom += values
    axes[0].set(xlabel="Outer", ylabel="Minutes / outer", ylim=(0, max(bottom) * 1.1))
    axes[0].set_title("(a) Measured stage costs", loc="left")
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.43), ncol=2, fontsize=6.8, frameon=False)
    axes[1].plot(
        steps,
        [r["token_passes_per_gpu_second"] / 1000 for r in outers],
        color=color,
        marker="o",
        markersize=2.5,
        label="MiMo 7B",
    )
    axes[1].axhline(
        summary["historical"]["token_passes_per_gpu_second"] / 1000,
        color=gray,
        linestyle=":",
        label="JustRL / R1 1.5B reference",
    )
    axes[1].set(xlabel="Outer", ylabel="1k input-token passes/s/GPU", ylim=(0, 25))
    axes[1].set_title("(b) Reuse-normalized actor throughput", loc="left")
    axes[1].legend(loc="center right", frameon=False, fontsize=7)
    fig.subplots_adjust(left=0.15, right=0.98, top=0.81, bottom=0.13, hspace=0.7)
    save_figure(
        fig,
        performance / "mimo-stage-cost",
        "4 H800; input tokens x reuse / actor time / GPUs. Different models, not a causal A/B.",
        caption_size=6.4,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--metrics", help="Explicit metrics JSONL when execution logs live outside the artifact directory"
    )
    parser.add_argument("--outer-limit", type=int, required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--figures", required=True)
    parser.add_argument("--performance-figures", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--label", default="Stage-transfer run")
    parser.add_argument("--replot", action="store_true")
    args = parser.parse_args()
    if args.replot:
        manifest = json.loads((Path(args.data) / "manifest.json").read_text())
        if file_sha256(Path(args.data) / "plot-data.json") != manifest["artifacts"]["plot-data.json"]:
            raise ValueError("Frozen plot input changed")
        data = json.loads((Path(args.data) / "plot-data.json").read_text())
    else:
        data = collect(args)
    render(data, args)
    for directory in [Path(args.figures), Path(args.performance_figures)]:
        dump(
            directory / "render.json",
            {
                "renderer_sha256": file_sha256(__file__),
                "data_manifest": str(Path(args.data).resolve() / "manifest.json"),
                "data_manifest_sha256": file_sha256(Path(args.data) / "manifest.json"),
                "artifacts": {p.name: file_sha256(p) for p in directory.iterdir() if p.suffix in {".png", ".pdf"}},
            },
        )


if __name__ == "__main__":
    main()
