# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Freeze MiMo stage comparisons, preserving each arm's actual budget and endpoint.

Semantic ledger: a series is E71 (SFT+RL), E72 (direct RL), or E73 (SFT).
Training loss is measured on each arm's own states and teacher. Fixed RKL uses
128 identical question IDs per domain but teacher targets/banks differ by arm.
Accuracy points are question means over the recorded sample count, not pass@n.
Geometry ledger: training axes use committed updates or completed outers. Risk
facets share a log scale. Accuracy panels share a 0..100 percent scale. E73 has
no endpoint until its final accuracy artifact is complete; no missing value is zero.
"""

import argparse
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf

from examples.mirroropd.research.core.io import dump, write_csv
from examples.mirroropd.research.core.plotting import load_style
from examples.mirroropd.research.studies.pipeline.plot_stage_progress import collect
from verl.trainer.evaluation.artifacts import ArtifactStore, file_sha256

ARMS = {"E71": ("E71", 25, "SFT+RL"), "E72": ("E72-r16-o20", 20, "Direct RL"), "E73": ("E73-r16-o20", 20, "SFT")}
DOMAINS = ("math", "code", "science", "if")
DATASETS = ("MATH-500", "AMC23", "AIME24", "AIME25", "AIME26", "LCBv5", "GPQA-Diamond", "IFEval")


def checked_accuracy(path, arm, role, sources):
    store = ArtifactStore(path)
    manifest = store.manifest(complete=True)
    sources[str(store.manifest_path)] = file_sha256(store.manifest_path)
    rows = [row for name in manifest["files"] if name.startswith("scores/") for row in store.iter_rows(name)]
    keys = [(r["dataset"], r["source_id"], r["sample_index"]) for r in rows]
    assert len(keys) == len(set(keys))
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        assert row["status"] in {"ok", "missing_final_answer"} and math.isfinite(row["score"])
        groups[row["dataset"]][row["source_id"]].append(row["score"])
    result, prompts = [], []
    for dataset in DATASETS:
        questions = groups[dataset]
        prefix = f"eval/accuracy/{dataset}"
        metrics = manifest["metadata"]["metrics"]
        count, samples = metrics[prefix + "/questions"], metrics[prefix + "/samples"]
        assert len(questions) == count and sum(map(len, questions.values())) == samples
        assert all(len(v) == samples / count for v in questions.values())
        value = math.fsum(math.fsum(v) / len(v) for v in questions.values()) / count
        assert math.isclose(value, metrics[prefix + "/avg_at_n"], abs_tol=1e-12)
        result.append(
            dict(
                arm=arm,
                role=role,
                dataset=dataset,
                accuracy=value,
                questions=count,
                samples=samples,
                n=int(samples / count),
            )
        )
        prompts.extend(
            dict(arm=arm, role=role, dataset=dataset, question=q, accuracy=math.fsum(v) / len(v), n=len(v))
            for q, v in questions.items()
        )
    return result, prompts


def snapshot(args):
    args.data.mkdir(parents=True, exist_ok=False)
    result = dict(captured_at=datetime.now().astimezone().isoformat(timespec="seconds"), arms={}, accuracy=[])
    prompts, sources = [], {}
    for arm, (directory, limit, teacher) in ARMS.items():
        config_path = args.root / "budget-amendment-20261102/configs" / (arm + ".yaml")
        run = args.root / "formal" / directory
        config = OmegaConf.load(config_path)
        metrics = args.root / "formal" / arm / "metrics.jsonl"
        data = collect(
            SimpleNamespace(
                run=run,
                config=config_path,
                metrics=metrics,
                outer_limit=limit,
                data=args.data / arm,
                label=f"MiMo {arm} ({teacher})",
                reference=args.reference,
            )
        )
        summary = data["summary"]
        records = [json.loads(line)["data"] for line in metrics.read_text().splitlines() if line.strip()]
        updates = [
            r for r in records if "optim/actor/update_applied" in r and r["step/optimizer"] <= summary["updates"]
        ]
        assert len(updates) == summary["updates"]
        assert all(
            r["optim/actor/update_applied"] == 1
            and r["optim/actor/update_skipped"] == 0
            and r["optim/actor/loss_nonfinite"] == 0
            for r in updates
        )
        artifacts = ArtifactStore(run / "evaluation")
        manifest = artifacts.manifest()
        complete = manifest["status"] == "complete"
        final_event = run / f"evaluation/events/outer-{limit:03d}"
        complete = complete and (final_event / "accuracy/manifest.json").exists()
        if complete:
            ArtifactStore(final_event).manifest(complete=True)
            ArtifactStore(run / "evaluation/final-model/export-audit").manifest(complete=True)
            assert manifest["metadata"]["optimizer_steps"] == limit * summary["reuse"]
        summary.update(
            teacher_stage=teacher,
            final_complete=complete,
            planned_outer=limit,
            planned_updates=limit * summary["reuse"],
            train_questions=summary["outer"] * 128,
        )
        result["arms"][arm] = data
        paths = {
            role: Path(config.trainer.evaluation.bootstrap) / (role + "-accuracy") for role in ("initial", "teacher")
        }
        if complete:
            paths["student"] = final_event / "accuracy"
        for role, path in paths.items():
            rows, question_rows = checked_accuracy(path, arm, role, sources)
            result["accuracy"].extend(rows)
            prompts.extend(question_rows)
        sources[str(artifacts.manifest_path)] = file_sha256(artifacts.manifest_path)
    write_csv(args.data / "accuracy.csv", result["accuracy"])
    write_csv(args.data / "accuracy-per-question.csv", prompts)
    dump(args.data / "plot-data.json", result)
    dump(
        args.data / "manifest.json",
        dict(
            captured_at=result["captured_at"],
            sources=sources,
            generator_sha256=file_sha256(__file__),
            artifacts={p.name: file_sha256(p) for p in args.data.iterdir() if p.is_file()},
            scope=__doc__,
        ),
    )
    return result


def render(data, args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fv = load_style()
    fv.apply_rc("paper", fontsize=10)
    colors = dict(zip(ARMS, [fv.HUES[c] for c in ("vteal", "vorange", "vviolet")], strict=True))
    args.figures.mkdir(parents=True, exist_ok=False)
    args.performance.mkdir(parents=True, exist_ok=False)
    audits = {}

    def finish(fig, directory, name, caption):
        fig.text(0.5, 0.025, caption, ha="center", fontsize=8, color=fv.NEUTRALS["baseline"])
        fig.canvas.draw()
        problems = fv.audit(fig, print_slot=fig.get_figwidth())
        audits[name] = fv.format_audit_lines(problems)
        assert not any(fv.problem_severity(p) == "error" for p in problems), problems
        for extension in ("png", "pdf", "svg"):
            fig.savefig(directory / f"{name}.{extension}", dpi=180, facecolor=fv.NEUTRALS["paper"])
        plt.close(fig)

    handles = [Line2D([], [], color=colors[a], label=f"{a}: {v[2]}") for a, v in ARMS.items()]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for arm, own in data["arms"].items():
        for ax, key in zip(axes[0], ("loss", "grad_norm"), strict=True):
            ax.plot([r["update"] for r in own["updates"]], [r[key] for r in own["updates"]], color=colors[arm], lw=0.6)
        for ax, key, factor in [(axes[1, 0], "mean_response_length", 1), (axes[1, 1], "length_finish_fraction", 100)]:
            ax.plot(
                [r["outer"] for r in own["outers"]],
                [r[key] * factor for r in own["outers"]],
                color=colors[arm],
                marker="o",
                ms=3,
                lw=1.2,
            )
    for ax, ylabel in zip(axes[0], ("Own-state training RKL", "Pre-clip gradient norm"), strict=True):
        ax.set(xlabel="Optimizer update", ylabel=ylabel, yscale="log")
    axes[1, 0].set(xlabel="Completed outer", ylabel="Mean response tokens", ylim=(0, 7600))
    axes[1, 1].set(xlabel="Completed outer", ylabel="Length-limit finishes (%)", ylim=(0, 100))
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False)
    fig.subplots_adjust(top=0.87, bottom=0.14, hspace=0.4, wspace=0.3)
    finish(fig, args.figures, "training", "E71: reuse32; E72/E73: reuse16. Own-state loss is not a capability ranking.")

    fig, axes = plt.subplots(3, 2, figsize=(10, 9))
    domain_colors = dict(zip(DOMAINS, [fv.HUES[c] for c in ("vteal", "vviolet", "vblue", "vorange")], strict=True))
    risk_values = [r["rkl"] for own in data["arms"].values() for r in own["rkl"] if r["domain"] != "all"]
    assert min(risk_values) > 0
    for index, (arm, own) in enumerate(data["arms"].items()):
        for col, bank in enumerate(("public_initial", "public_teacher")):
            ax = axes[index, col]
            for domain in DOMAINS:
                rows = [r for r in own["rkl"] if r["domain"] == domain and r["bank"] == bank]
                ax.plot(
                    [r["outer"] for r in rows],
                    [r["rkl"] for r in rows],
                    color=domain_colors[domain],
                    marker="o",
                    ms=3,
                    lw=1.3,
                    label=domain.upper(),
                )
            ax.set(
                xlabel="Completed outer",
                ylabel="Question-mean RKL",
                yscale="log",
                ylim=(min(risk_values) * 0.8, max(risk_values) * 1.25),
                title=f"{arm}: {'Base prefixes' if col == 0 else 'Own teacher prefixes'}",
            )
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", ncol=4, frameon=False)
    fig.subplots_adjust(top=0.90, bottom=0.11, hspace=0.60, wspace=0.3)
    finish(
        fig,
        args.figures,
        "fixed-bank-rkl",
        "128 questions/domain; different teacher targets across arms. Missing readouts stay absent.",
    )

    completed = [a for a in ARMS if data["arms"][a]["summary"]["final_complete"]]
    fig, axes = plt.subplots(1, len(completed), figsize=(6 * len(completed), 5.1), squeeze=False)
    roles = ("initial", "teacher", "student")
    for ax, arm in zip(axes[0], completed, strict=True):
        for j, role in enumerate(roles):
            rows = [r for r in data["accuracy"] if (r["arm"], r["role"]) == (arm, role)]
            ax.plot(
                np.arange(8) + (j - 1) * 0.17,
                [100 * r["accuracy"] for r in rows],
                ls="none",
                marker=("s", "^", "o")[j],
                ms=6,
                color=[fv.NEUTRALS["baseline"], fv.HUES["vorange"], fv.HUES["vteal"]][j],
                label=role.title(),
            )
        summary = data["arms"][arm]["summary"]
        ax.set(
            xticks=range(8),
            xticklabels=["MATH500", "AMC23", "AIME24", "AIME25", "AIME26", "LCBv5", "GPQA", "IFEval"],
            ylim=(0, 100),
            xlabel="Benchmark",
            ylabel="Question-mean avg@n (%)",
            title=f"{arm}: outer{summary['outer']}, {summary['updates']} updates",
        )
        ax.tick_params(axis="x", labelrotation=45)
        ax.grid(axis="y", alpha=0.2)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", ncol=3, frameon=False)
    fig.subplots_adjust(top=0.82, bottom=0.31, left=0.07, right=0.98, wspace=0.25)
    finish(
        fig,
        args.figures,
        "accuracy",
        "Unequal training budgets; one training seed. "
        + ("All three endpoints are complete." if len(completed) == len(ARMS) else "Unfinished endpoints stay absent."),
    )

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.7))
    cost_limit = max(r["outer_seconds"] / 60 for own in data["arms"].values() for r in own["outers"]) * 1.08
    for ax, (arm, own) in zip(axes, data["arms"].items(), strict=True):
        bottom = np.zeros(len(own["outers"]))
        for key, label, color in [
            ("actor_seconds", "Actor", fv.HUES["vteal"]),
            ("rollout_seconds", "Rollout", fv.HUES["vorange"]),
            ("evaluation_seconds", "Evaluation", fv.HUES["vviolet"]),
            ("other_seconds", "Other", fv.NEUTRALS["baseline"]),
        ]:
            values = np.array([r[key] / 60 for r in own["outers"]])
            ax.bar([r["outer"] for r in own["outers"]], values, bottom=bottom, color=color, label=label)
            bottom += values
        ax.set(
            xlabel="Completed outer",
            ylabel="Minutes / outer",
            ylim=(0, cost_limit),
            title=f"{arm}: reuse{own['summary']['reuse']}",
        )
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=4, frameon=False)
    fig.subplots_adjust(top=0.81, bottom=0.19, left=0.07, right=0.98, wspace=0.34)
    finish(
        fig,
        args.performance,
        "stage-cost",
        "4 H800 per run; disjoint stage costs. Final evaluation is included in its completed outer.",
    )
    for destination in (args.figures, args.performance):
        dump(destination / "figure-audit.json", audits)
        dump(
            destination / "render.json",
            dict(
                renderer_sha256=file_sha256(__file__),
                data_manifest_sha256=file_sha256(args.data / "manifest.json"),
                artifacts={
                    p.name: file_sha256(p) for p in destination.iterdir() if p.suffix in {".png", ".pdf", ".svg"}
                },
            ),
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--figures", type=Path, required=True)
    parser.add_argument("--performance", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--replot", action="store_true")
    args = parser.parse_args()
    if args.replot:
        manifest = json.loads((args.data / "manifest.json").read_text())
        assert file_sha256(args.data / "plot-data.json") == manifest["artifacts"]["plot-data.json"]
        data = json.loads((args.data / "plot-data.json").read_text())
    else:
        data = snapshot(args)
    render(data, args)


if __name__ == "__main__":
    main()
