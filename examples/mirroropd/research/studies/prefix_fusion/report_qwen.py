# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Audit and plot completed Qwen prefix arms without importing a live study profile.

Semantic ledger: LM/PM denote learner endpoints trained on distinct rollout states;
initial/teacher are original model controls. Fixed risk uses identical frozen
questions/states and sum(RKL)/1024 per question. Accuracy first averages four
answers per prompt, then gives each of five datasets equal weight. Bootstrap
resamples prompts within datasets with shared indices across model endpoints.
Geometry ledger: loss/risk panels use outer coordinates; speed bars show disjoint
wall-clock components (seconds or hours), never nested timer sums. Color identifies
one endpoint consistently. Ordinary rounds exclude outer 1 and multiples of five.
"""

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ARMS = ("LM", "PM")
ENDPOINTS = ("initial", "LM", "PM", "teacher")
DATASETS = ("MATH-500", "AMC23", "AIME24", "AIME25", "AIME26")
PHASES = {
    "Rollout": "perf/timing/rollout_policy_batch_seconds",
    "Teacher scoring": None,
    "Actor update": "perf/timing/update_actor_seconds",
    "Checkpoint": "perf/timing/save_checkpoint_seconds",
    "Readouts": "science/readout_seconds",
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def table(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class Evidence:
    def __init__(self, repo):
        self.repo, self.files = repo, {}

    def track(self, path):
        self.files[str(path.relative_to(self.repo))] = sha256(path)

    def read(self, path):
        self.track(path)
        return json.loads(path.read_text())

    def lines(self, path):
        self.track(path)
        with path.open() as stream:
            yield from (json.loads(line) for line in stream if line.strip())


def accuracy(evidence, paths, *, endpoints=ENDPOINTS, comparisons=None):
    if comparisons is None:
        comparisons = [("LM", "PM"), ("LM", "initial"), ("PM", "initial")]
    scores, stats, prompt_rows = {}, {}, []
    for endpoint, path in paths.items():
        receipt = evidence.read(path / "complete.json")
        assert receipt["questions"] == 630 and receipt["records"] == 2520
        assert sha256(path / "records.jsonl") == receipt["records_sha256"]
        groups, lengths, caps = defaultdict(dict), [], 0
        for row in evidence.lines(path / "records.jsonl"):
            key = (row["data_source"], row["prompt_id"])
            assert row["status"] == "ok" and row["sample_index"] not in groups[key]
            groups[key][row["sample_index"]] = float(row["acc"])
            lengths.append(row["response_length"])
            caps += row["finish_reason"] == "length"
        assert len(groups) == 630 and all(set(g) == set(range(4)) for g in groups.values())
        scores[endpoint] = {k: float(np.mean(list(v.values()))) for k, v in groups.items()}
        by_dataset = {
            d: float(np.mean([v for (name, _), v in scores[endpoint].items() if name == d])) for d in DATASETS
        }
        official = evidence.read(path / "metrics.json")
        for name, value in by_dataset.items():
            assert abs(official[f"eval/{name.lower().replace('-', '')}/acc/mean"] - value) < 1e-12
        stats[endpoint] = dict(
            macro5=float(np.mean(list(by_dataset.values()))),
            by_dataset=by_dataset,
            seconds=receipt["elapsed_seconds"],
            mean_length=float(np.mean(lengths)),
            p95_length=float(np.quantile(lengths, 0.95)),
            cap_hits=int(caps),
            tokens=sum(lengths),
            effective_tokens_per_second=sum(lengths) / receipt["elapsed_seconds"],
        )
        prompt_rows.extend(
            dict(endpoint=endpoint, dataset=k[0], prompt_id=k[1], avg4=v) for k, v in scores[endpoint].items()
        )
    assert all(set(s) == set(scores["initial"]) for s in scores.values())
    rng = np.random.default_rng(20261101)
    draws = np.zeros((10000, len(endpoints)))
    for dataset in DATASETS:
        keys = sorted(k for k in scores["initial"] if k[0] == dataset)
        values = np.array([[scores[name][k] for name in endpoints] for k in keys])
        for start in range(0, len(draws), 500):
            indices = rng.integers(len(keys), size=(500, len(keys)))
            draws[start : start + 500] += values[indices].mean(axis=1) / len(DATASETS)
    for index, name in enumerate(endpoints):
        stats[name]["macro5_ci95"] = np.quantile(draws[:, index], [0.025, 0.975]).tolist()
    differences = {}
    for left, right in comparisons:
        differences[f"{left}-{right}"] = dict(
            estimate=stats[left]["macro5"] - stats[right]["macro5"],
            ci95=np.quantile(
                draws[:, endpoints.index(left)] - draws[:, endpoints.index(right)], [0.025, 0.975]
            ).tolist(),
        )
    return stats, differences, prompt_rows


def collect(root, out):
    repo = root.parents[1]
    evidence = Evidence(repo)
    recovery = root / "recovery-20261101"
    manifest = evidence.read(recovery / "manifest.json")
    result = dict(
        revision=manifest["revision"],
        arms={},
        paired_accuracy={},
        statistical_scope=(
            "10000 stratified paired prompt bootstrap draws; one training seed; no training-seed interval"
        ),
    )
    updates_all, outer_all, risks = [], [], []
    paths = {name: root / "baseline-accuracy" / key for name, key in [("initial", "S0"), ("teacher", "Q")]}
    for arm in ARMS:
        run = recovery / "training" / f"prefix-science-Q-{arm}-qwenfix-v2"
        audit = evidence.read(recovery / "stages" / f"science-{arm}-audit.json")
        assert audit["passed"] and audit["applied_updates"] == 640
        receipt = evidence.read(recovery / "stages" / f"science-{arm}.json")
        assert receipt["exit_code"] == 0
        evidence.track(run / "resolved_config.yaml")
        records = [v["data"] for v in evidence.lines(run / "metrics.jsonl")]
        updates = [v for v in records if "optim/actor/update_applied" in v]
        assert [d["step/optimizer"] for d in updates] == list(range(1, 641))
        assert all(v["optim/actor/update_applied"] == 1 and v["optim/actor/loss_nonfinite"] == 0 for v in updates)
        updates_all.extend(
            dict(
                arm=arm,
                outer=v["step/outer"],
                update=v["step/optimizer"],
                reuse=(v["step/optimizer"] - 1) % 16 + 1,
                loss=v["optim/actor/distillation_loss"],
            )
            for v in updates
        )
        rows = [v for v in records if "rollout/response_length/mean" in v and "optim/actor/update_applied" not in v]
        assert [v["step/outer"] for v in rows] == list(range(1, 41))
        phases = defaultdict(float)
        for row in rows:
            parts = {
                label: row.get(key, 0.0)
                if key
                else row["perf/timing/teacher_transfer_seconds"] + row["perf/timing/teacher_forward_seconds"]
                for label, key in PHASES.items()
            }
            total = row["perf/timing/outer_step_seconds"]
            residual = total - sum(parts.values())
            assert residual > -0.05, (arm, row["step/outer"], residual)
            for key, val in parts.items():
                phases[key] += val
            phases["Other"] += residual
            tokens = row["rollout/response_length/mean"] * 64
            generation = row["perf/timing/rollout_policy_fusion_generate_seconds"]
            outer_all.append(
                dict(
                    arm=arm,
                    outer=row["step/outer"],
                    seconds=total,
                    response_tokens=tokens,
                    mean_length=tokens / 64,
                    generation_seconds=generation,
                    generation_tps=tokens / generation,
                    update_ms_per_unique_token=1000 * parts["Actor update"] / tokens,
                    refit_seconds=row["perf/timing/fusion_current_refit_seconds"],
                    other_seconds=residual,
                    **{k.lower().replace(" ", "_") + "_seconds": v for k, v in parts.items()},
                )
            )
        readout_details = []
        for outer in range(0, 41, 5):
            dest = run / "science" / f"outer-{outer:03d}"
            readout = evidence.read(dest / "complete.json")
            assert readout["optimizer_step"] == outer * 16
            for key, value in readout["metrics"].items():
                if key.endswith("/rkl"):
                    _, _, bank, domain, _ = key.split("/")
                    risks.append(dict(arm=arm, outer=outer, bank=bank, domain=domain, rkl=value))
            acc_seconds = 0.0
            if outer in [20, 40]:
                acc_seconds = evidence.read(dest / "accuracy/complete.json")["elapsed_seconds"]
            readout_details.append(
                dict(outer=outer, total_seconds=readout["elapsed_seconds"], accuracy_seconds=acc_seconds)
            )
        phases["Readouts"] += readout_details[0]["total_seconds"]
        full = receipt["finished_at"] - receipt["started_at"]
        phases["Other"] += full - sum(phases.values())
        assert phases["Other"] >= 0
        own = [r for r in outer_all if r["arm"] == arm]
        ordinary = [r for r in own if r["outer"] > 1 and r["outer"] % 5]
        memory, rss, metadata = 0, 0, None
        for sample in evidence.lines(run / "resources.jsonl"):
            if sample["kind"] == "metadata":
                metadata = sample
            if sample["kind"] == "sample":
                memory = max(memory, *(g["memory_used"] for g in sample["gpus"]))
                rss = max(rss, sample["process_tree_rss_sum"])
        totals = {k: float(v) for k, v in phases.items()}
        result["arms"][arm] = dict(
            full_seconds=full,
            total_phases=totals,
            readouts=readout_details,
            ordinary_rounds=len(ordinary),
            ordinary_median_seconds=float(np.median([r["seconds"] for r in ordinary])),
            ordinary_mean_phases={
                k: float(np.mean([r[k] for r in ordinary]))
                for k in ["rollout_seconds", "teacher_scoring_seconds", "actor_update_seconds", "other_seconds"]
            },
            ordinary_median_generation_tps=float(np.median([r["generation_tps"] for r in ordinary])),
            ordinary_median_update_ms_per_unique_token=float(
                np.median([r["update_ms_per_unique_token"] for r in ordinary])
            ),
            median_refit_seconds=float(np.median([r["refit_seconds"] for r in own[1:]])),
            tokens=sum(r["response_tokens"] for r in own),
            nvml_peak_gib=memory / 2**30,
            process_tree_rss_sum_peak_gib=rss / 2**30,
            resource_metadata=metadata,
        )
        paths[arm] = run / "science/outer-040/accuracy"
    result["accuracy"], result["paired_accuracy"], prompts = accuracy(evidence, paths)
    baseline = evidence.read(repo / "runs/token-fusion-tuning-20261031/summary.json")
    result["rollout_pilots"] = [
        dict(
            name=r["name"],
            seqs=r["max_num_seqs"],
            kv_gib=r["kv_gib"],
            peak_gib=r["sampled_process_peak_gib"],
            **r["measurements"][0],
        )
        for r in baseline["cases"]
        if r["pair"] == "Q"
    ]
    table(out / "updates.csv", updates_all)
    table(out / "outers.csv", outer_all)
    table(out / "fixed-risk.csv", risks)
    table(out / "accuracy-per-prompt.csv", prompts)
    dump(out / "summary.json", result)
    dump(out / "sources.json", dict(files=evidence.files, script_sha256=sha256(Path(__file__))))
    return result, updates_all, outer_all, risks


def figures(summary, updates, outers, risks, destination, style):
    sys.path.insert(0, str(style))
    import fvstyle as fv
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fv.apply_rc("paper")
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )
    colors = {
        "LM": fv.HUES["vteal"],
        "PM": fv.HUES["vorange"],
        "initial": fv.NEUTRALS["baseline"],
        "teacher": fv.HUES["vviolet"],
    }
    audits = {}

    def finish(fig, name, caption, scoped_ignore=()):
        fig.text(0.5, 0.015, caption, ha="center", fontsize=7, color=fv.NEUTRALS["baseline"])
        fig.canvas.draw()
        problems = fv.audit(fig, print_slot=fig.get_figwidth(), scoped_ignore=scoped_ignore)
        audits[name] = fv.format_audit_lines(problems)
        for ext in ["png", "pdf", "svg"]:
            fig.savefig(destination / f"{name}.{ext}", dpi=200, facecolor=fv.NEUTRALS["paper"])
        plt.close(fig)
        errors = [p for p in problems if fv.problem_severity(p) == "error"]
        if errors:
            raise ValueError(errors)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for arm in ARMS:
        for reuse, linestyle in [(1, "-"), (16, "--")]:
            rows = [r for r in updates if r["arm"] == arm and r["reuse"] == reuse]
            axes[0].plot(
                [r["outer"] for r in rows],
                [r["loss"] for r in rows],
                color=colors[arm],
                ls=linestyle,
                lw=1.4,
                label=arm if reuse == 1 else "_nolegend_",
            )
        rows = [r for r in outers if r["arm"] == arm]
        axes[1].plot([r["outer"] for r in rows], [r["mean_length"] for r in rows], color=colors[arm], label=arm)
    axes[0].set(xlabel="Outer", ylabel="Recorded training RKL loss", title="(a) Loss on each arm's own rollout states")
    axes[1].set(xlabel="Outer", ylabel="Mean response tokens", title="(b) Natural training response length")
    for ax in axes:
        ax.set_xlim(1, 40)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    handles += [Line2D([], [], color=fv.NEUTRALS["vink"], ls=ls) for ls in ["-", "--"]]
    labels += ["Loss: reuse 1", "Loss: reuse 16"]
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1), ncol=4, frameon=False)
    fig.subplots_adjust(top=0.77, bottom=0.19, wspace=0.32, left=0.08, right=0.98)
    finish(fig, "training", "40 outers x reuse16; first/last recorded loss in each batch, not a common-state ranking.")

    fig, axes = plt.subplots(2, 3, figsize=(9, 5.1), sharex=True, sharey=True)
    for row, bank in enumerate(["S0", "Q"]):
        for col, domain in enumerate(["math", "code", "if"]):
            ax = axes[row, col]
            for arm in ARMS:
                records = [r for r in risks if r["arm"] == arm and r["bank"] == bank and r["domain"] == domain]
                ax.plot(
                    [r["outer"] for r in records],
                    [r["rkl"] for r in records],
                    color=colors[arm],
                    marker="o",
                    ms=3,
                    label=arm,
                )
            domain_label = domain.upper() if domain == "if" else domain.capitalize()
            bank_label = "initial" if bank == "S0" else "teacher"
            ax.set_title(f"{domain_label} / {bank_label} states")
            ax.set_ylim(bottom=0)
            ax.grid(axis="y", alpha=0.2)
            if row == 1:
                ax.set_xlabel("Outer")
            if col == 0:
                ax.set_ylabel("Fixed-state RKL / H")
    axes[0, 0].set_ylim(0, max(r["rkl"] for r in risks) * 1.1)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", ncol=2, frameon=False)
    fig.subplots_adjust(top=0.84, bottom=0.14, left=0.08, right=0.98, wspace=0.22, hspace=0.42)
    finish(
        fig,
        "heldout",
        "Frozen common states; H=1024 per prompt; Math64 / Code32 / IF32; one training seed.",
        # Shared matrix axis labels are intentionally confined to the left edge.
        scoped_ignore=[("missing-axis-label", i) for i in [1, 2, 4, 5]],
    )

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.7), gridspec_kw={"width_ratios": [1, 1.8]})
    acc = summary["accuracy"]
    for i, name in enumerate(ENDPOINTS):
        value = 100 * acc[name]["macro5"]
        lo, hi = np.array(acc[name]["macro5_ci95"]) * 100
        axes[0].bar(i, value, color=colors[name], label=name, width=0.65)
        interval = axes[0].errorbar(
            i, value, yerr=[[value - lo], [hi - value]], fmt="none", color=fv.NEUTRALS["vink"], capsize=3
        )
        for artist in [*interval.lines[1], *interval.lines[2]]:
            fv.mark_neutral(artist)
        axes[1].bar(
            np.arange(5) + (i - 1.5) * 0.19,
            [100 * acc[name]["by_dataset"][d] for d in DATASETS],
            width=0.18,
            color=colors[name],
            label=name,
        )
    axes[0].set_xticks(range(4), ["Initial", "LM", "PM", "Teacher"])
    axes[1].set_xticks(range(5), ["MATH500", "AMC23", "AIME24", "AIME25", "AIME26"])
    axes[0].set(title="(a) Equal-weight macro5", ylabel="Accuracy (%)", xlabel="Endpoint")
    axes[1].set(title="(b) Per-dataset avg@4", ylabel="Accuracy (%)", xlabel="Dataset")
    for ax in axes:
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    fig.legend(*axes[1].get_legend_handles_labels(), loc="upper center", ncol=4, frameon=False)
    fig.subplots_adjust(top=0.78, bottom=0.22, left=0.07, right=0.98, wspace=0.28)
    finish(
        fig,
        "accuracy",
        "630 prompts x 4 responses; bars are avg@4, not pass@4; macro5 error bars: paired prompt bootstrap 95% CI.",
    )

    phase_colors = {
        "Rollout": fv.HUES["vteal"],
        "Teacher scoring": fv.HUES["vgold"],
        "Actor update": fv.HUES["vorange"],
        "Readouts": fv.HUES["vviolet"],
        "Other": fv.NEUTRALS["baseline"],
    }
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for i, arm in enumerate(ARMS):
        a = summary["arms"][arm]
        left = 0
        for label in ["Rollout", "Teacher scoring", "Actor update", "Readouts", "Other"]:
            value = a["total_phases"][label] + (a["total_phases"]["Checkpoint"] if label == "Other" else 0)
            axes[0].barh(
                i, value / 3600, left=left, color=phase_colors[label], label=label if i == 0 else None, height=0.55
            )
            left += value / 3600
        left = 0
        for label, key in [
            ("Rollout", "rollout_seconds"),
            ("Teacher scoring", "teacher_scoring_seconds"),
            ("Actor update", "actor_update_seconds"),
            ("Other", "other_seconds"),
        ]:
            value = a["ordinary_mean_phases"][key]
            axes[1].barh(i, value, left=left, color=phase_colors[label], height=0.55, label=label if i == 0 else None)
            left += value
    axes[0].set(title="(a) Entire job, including startup", xlabel="Hours", ylabel="Arm")
    axes[1].set(title="(b) Ordinary outer: mean phase cost", xlabel="Seconds", ylabel="Arm")
    for ax in axes:
        ax.set_yticks([0, 1], ARMS)
        ax.invert_yaxis()
        ax.set_xlim(left=0)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=5, frameon=False)
    fig.subplots_adjust(top=0.77, bottom=0.21, left=0.07, right=0.98, wspace=0.28)
    finish(
        fig,
        "timing",
        "Disjoint timers; actor update includes KL precompute + reuse16; ordinary outers exclude 1 and multiples of 5.",
    )
    dump(destination / "figure-audit.json", audits)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--figures", type=Path, required=True)
    parser.add_argument("--style", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.figures.mkdir(parents=True, exist_ok=True)
    values = collect(args.root.resolve(), args.out)
    figures(*values, args.figures, args.style)
    print(json.dumps(values[0], ensure_ascii=False))


if __name__ == "__main__":
    main()
