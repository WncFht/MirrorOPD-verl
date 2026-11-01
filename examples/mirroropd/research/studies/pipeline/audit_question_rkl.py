# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Bench and fixed-trajectory audit from immutable scalar readouts.

Ledger: one observation is one question's sample-0 trajectory under a declared
producer, scored at one student checkpoint. Fixed before/after pairs retain the
same trajectory and token count. Current trajectories are never paired as fixed.
Accuracy uses all declared samples, independently of the smaller RKL panel.
"""

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from examples.mirroropd.research.core.io import dump, read, write_csv
from examples.mirroropd.research.core.plotting import load_style, save_figure
from verl.trainer.evaluation.artifacts import file_sha256

BENCHES = ("MATH-500", "AMC23", "AIME24", "AIME25", "AIME26", "LCBv5", "GPQA-Diamond", "IFEval")
BANKS = ("public_initial", "public_teacher", "public_current")


def accuracy_rows(root):
    manifest = read(root / "manifest.json")
    assert manifest["status"] == "complete"
    rows = []
    for name in sorted(manifest["files"]):
        rows.extend(
            pq.read_table(
                root / name, columns=["id", "source_id", "dataset", "sample_index", "score", "status"]
            ).to_pylist()
        )
    assert len({r["id"] for r in rows}) == len(rows)
    return rows, manifest


def audit(run, output):
    run, output = Path(run).resolve(), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifests, scores, sources = {}, {}, {}
    for p in sorted((run / "evaluation/events").glob("outer-*/manifest.json")):
        m = read(p)
        assert m["status"] == "complete"
        outer = m["metadata"]["plan"]["outer"]
        rows = pq.read_table(p.parent / "rkl.parquet").to_pylist()
        assert len({r["id"] for r in rows}) == len(rows)
        assert all(r["tokens"] > 0 and np.isfinite(r["rkl_sum"]) and r["rkl_sum"] >= 0 for r in rows)
        assert all(r["sample_index"] == 0 for r in rows if r["bank"].startswith("public_"))
        manifests[outer], scores[outer] = m, rows
        sources[str(p)] = file_sha256(p)
        sources[str(p.parent / "rkl.parquet")] = file_sha256(p.parent / "rkl.parquet")
    first, last = min(scores), max(scores)
    reference = {r["id"]: r for r in scores[first]}
    stats, pairs, curves, contrasts = [], [], [], []
    for outer, rows in scores.items():
        grouped = defaultdict(list)
        for row in rows:
            if row["bank"] in BANKS:
                grouped[row["bank"], row["dataset"]].append(row)
        for (bank, bench), group in grouped.items():
            values = np.array([r["rkl_sum"] / r["tokens"] for r in group])
            curves.append(
                dict(outer=outer, bank=bank, benchmark=bench, questions=len(group), mean=float(values.mean()))
            )
            if bank != "public_current":
                expected = {r["id"] for r in reference.values() if r["bank"] == bank and r["dataset"] == bench}
                assert {r["id"] for r in group} == expected
                for row in group:
                    old = reference[row["id"]]
                    assert all(
                        old[k] == row[k] for k in ["trajectory_id", "source_id", "dataset", "tokens", "sample_index"]
                    )
            if outer != last:
                continue
            total = float(values.sum())
            record = dict(
                bank=bank,
                benchmark=bench,
                questions=len(group),
                after=float(values.mean()),
                median=float(np.median(values)),
                p90=float(np.quantile(values, 0.9)),
                maximum=float(values.max()),
                top1_share_percent=100 * float(values.max()) / total,
                token_weighted_after=sum(r["rkl_sum"] for r in group) / sum(r["tokens"] for r in group),
            )
            if bank != "public_current":
                before = np.array([reference[r["id"]]["rkl_sum"] / r["tokens"] for r in group])
                assert np.all(before > 0)
                reduction = 100 * (1 - values / before)
                record.update(
                    before=float(before.mean()),
                    reduction_percent=100 * (1 - float(values.mean() / before.mean())),
                    median_question_reduction_percent=float(np.median(reduction)),
                    mean_question_reduction_percent=float(reduction.mean()),
                    improved_questions=int((values < before).sum()),
                )
                for row, a, b, drop in zip(group, before, values, reduction, strict=True):
                    pairs.append(
                        dict(
                            bank=bank,
                            benchmark=bench,
                            domain=row["domain"],
                            question_id=row["source_id"],
                            trajectory_id=row["trajectory_id"],
                            tokens=row["tokens"],
                            before=float(a),
                            after=float(b),
                            reduction_percent=float(drop),
                            contribution_percent=100 * float(b) / total,
                        )
                    )
            stats.append(record)
    # Independently reconstruct all three bank/domain means; do not trust plotted aggregates.
    errors = []
    for outer, rows in scores.items():
        metrics = manifests[outer]["metadata"]["metrics"]
        for bank in BANKS:
            selected = [r for r in rows if r["bank"] == bank]
            if not selected:
                continue
            means = []
            for domain in ["math", "code", "science", "if"]:
                q = [r for r in selected if r["domain"] == domain]
                value = float(np.mean([r["rkl_sum"] / r["tokens"] for r in q]))
                means.append(value)
                errors.append(abs(value - metrics[f"eval/rkl/{bank}/{domain}/question_mean"]))
            errors.append(abs(float(np.mean(means)) - metrics[f"eval/rkl/{bank}/all/question_mean"]))
    assert max(errors) < 1e-12
    # Matched questions with different answers: a cross-bank comparison, not a learning delta.
    for bench in BENCHES:
        bankrows = {
            b: {r["source_id"]: r for r in scores[last] if r["bank"] == b and r["dataset"] == bench} for b in BANKS
        }
        assert bankrows[BANKS[0]].keys() == bankrows[BANKS[1]].keys() == bankrows[BANKS[2]].keys()
        for qid in bankrows[BANKS[0]]:
            contrasts.append(
                dict(
                    benchmark=bench,
                    question_id=qid,
                    **{b: bankrows[b][qid]["rkl_sum"] / bankrows[b][qid]["tokens"] for b in BANKS},
                )
            )
    accuracy, perquestion, accuracy_error = [], {}, []
    statuses = {}
    locations = {
        "initial": run / "bootstrap/initial-accuracy",
        "teacher": run / "bootstrap/teacher-accuracy",
        "outer25": run / "evaluation/events/outer-025/accuracy",
        "outer50": run / "evaluation/events/outer-050/accuracy",
    }
    for role, root in locations.items():
        rows, m = accuracy_rows(root)
        sources[str(root / "manifest.json")] = file_sha256(root / "manifest.json")
        statuses[role] = dict(Counter(r["status"] for r in rows))
        for name in m["files"]:
            sources[str(root / name)] = file_sha256(root / name)
        bybench = defaultdict(lambda: defaultdict(list))
        for row in rows:
            assert row["status"] in {"ok", "missing_final_answer"}
            if row["status"] == "missing_final_answer":
                assert row["score"] == 0
            bybench[row["dataset"]][row["source_id"]].append(row)
        for bench, questions in bybench.items():
            ns = {len(x) for x in questions.values()}
            assert len(ns) == 1
            n = ns.pop()
            for qid, items in questions.items():
                assert sorted(r["sample_index"] for r in items) == list(range(n))
                perquestion[role, bench, qid] = dict(
                    avg=float(np.mean([r["score"] for r in items])),
                    sample0=next(r["score"] for r in items if r["sample_index"] == 0),
                )
            mean = float(np.mean([perquestion[role, bench, qid]["avg"] for qid in questions]))
            error = abs(mean - m["metadata"]["metrics"][f"eval/accuracy/{bench}/avg_at_n"])
            accuracy_error.append(error)
            assert error < 1e-12
            accuracy.append(
                dict(
                    role=role, benchmark=bench, questions=len(questions), n=n, samples=n * len(questions), accuracy=mean
                )
            )
    case_ids = {
        "bedddb1b13e29ddb3175f77d106b36bc21d12632ebd7751701ba8d77708cf48b",
        "3cef5d4c42fe63b3b79e25dfda4ebe34840683ac5ebbec4f7514cf30b5146b09",
    }
    case_scores = [
        dict(role=role, benchmark=bench, question_id=qid, **values)
        for (role, bench, qid), values in perquestion.items()
        if qid in case_ids
    ]
    write_csv(output / "bench-rkl.csv", stats)
    write_csv(output / "bench-curves.csv", curves)
    write_csv(output / "paired-questions.csv", pairs)
    write_csv(output / "same-question-different-trajectories.csv", contrasts)
    write_csv(output / "bench-accuracy.csv", accuracy)
    write_csv(output / "case-accuracy.csv", case_scores)
    dump(
        output / "audit.json",
        dict(
            first_outer=first,
            last_outer=last,
            events=len(scores),
            fixed_public_trajectories=len(pairs),
            matched_questions_across_banks=len(contrasts),
            accuracy_status_counts=statuses,
            max_domain_mean_reconstruction_error=max(errors),
            max_accuracy_reconstruction_error=max(accuracy_error),
            protocol="question token mean; fixed trajectories paired; current trajectories not paired",
            sources=sources,
            generator_sha256=file_sha256(__file__),
        ),
    )
    return stats, pairs, curves, accuracy, contrasts


def plot(run, output, figures):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stats, pairs, curves, accuracy, contrasts = audit(run, output)
    figures = Path(figures)
    figures.mkdir(parents=True, exist_ok=True)
    fv = load_style()
    fv.apply_rc("paper", fontsize=10)
    palette = fv.series_colors("paper", 4)
    saved = []

    def save(fig, name, caption):
        fig.tight_layout(rect=(0, 0.1, 1, 1))
        save_figure(fig, figures / name, caption)
        saved.append(name)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), sharex=True, sharey=True)
    for axis, bank, title in zip(
        axes, BANKS[:2], ["Fixed initial-student answers", "Fixed teacher answers"], strict=True
    ):
        lookup = {r["benchmark"]: r for r in stats if r["bank"] == bank}
        axis.hlines(
            range(len(BENCHES)),
            [lookup[b]["after"] for b in BENCHES],
            [lookup[b]["before"] for b in BENCHES],
            color=fv.NEUTRALS["baseline"],
            linewidth=1,
            label="Same questions",
        )
        axis.scatter(
            [lookup[b]["before"] for b in BENCHES],
            range(len(BENCHES)),
            marker="o",
            facecolors="none",
            edgecolors=palette[0],
            label="Before",
        )
        axis.scatter(
            [lookup[b]["after"] for b in BENCHES],
            range(len(BENCHES)),
            marker="o",
            color=palette[1],
            label="After 50 outer",
        )
        axis.set_yticks(range(len(BENCHES)), [f"{b} (N={lookup[b]['questions']})" for b in BENCHES])
        axis.set(xscale="log", xlabel="Question-mean RKL (nats)", title=title)
        axis.legend(loc="lower center", bbox_to_anchor=(0.5, 1.08), ncol=3, frameon=False)
    axes[0].invert_yaxis()
    axes[1].tick_params(labelleft=True)
    save(fig, "bench-rkl", "RKL uses one sample per panel question; N differs from full accuracy evaluation.")
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), sharex=True, sharey=True)
    for axis, domain in zip(axes.flat, ["math", "code", "science", "if"], strict=True):
        for bank, color, linestyle, label in zip(
            BANKS[:2], palette[:2], ["-", "--"], ["Initial-student answers", "Teacher answers"], strict=True
        ):
            rows = [r for r in pairs if r["domain"] == domain and r["bank"] == bank]
            x = sorted(r["reduction_percent"] for r in rows)
            axis.step(x, np.arange(1, len(x) + 1) / len(x), where="post", color=color, linestyle=linestyle, label=label)
        axis.set(
            xlabel="Per-question RKL reduction (%)",
            ylabel="Empirical cumulative fraction",
            title=domain.upper() if domain == "if" else domain.capitalize(),
            xlim=(min(-2, min(r["reduction_percent"] for r in pairs) - 2), 102),
            ylim=(0, 1.03),
        )
        axis.legend(loc="upper left", fontsize=8)
    save(
        fig,
        "paired-question-reduction",
        "Each percentage pairs the same trajectory at outer 0 and 50; no current-bank pairing.",
    )
    # Separate typical behavior and tail concentration; both retain all questions.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.3))
    for bank, color, label in zip(BANKS[:2], palette[:2], ["Initial-student answers", "Teacher answers"], strict=True):
        rows = [r for r in pairs if r["bank"] == bank]
        x = sorted(r["after"] for r in rows)
        axes[0].step(x, np.arange(1, len(x) + 1) / len(x), where="post", color=color, label=label)
        descending = np.array(x[::-1])
        axes[1].plot(np.arange(1, len(x) + 1), 100 * np.cumsum(descending) / descending.sum(), color=color, label=label)
    axes[0].set(xscale="log", xlabel="Final question RKL (nats)", ylabel="Empirical cumulative fraction")
    axes[1].set(
        xscale="log",
        xlabel="Highest-RKL trajectories retained (count)",
        ylabel="Share of total final RKL (%)",
        ylim=(0, 103),
    )
    for axis in axes:
        axis.legend(fontsize=8)
    save(
        fig,
        "question-tail-concentration",
        "512 questions per bank, 128 per domain; tail curves include every observation.",
    )
    dump(
        figures / "manifest.json",
        dict(
            source_run=str(Path(run).resolve()),
            figures=saved,
            generator_sha256=file_sha256(__file__),
            artifacts={p.name: file_sha256(p) for p in figures.iterdir() if p.name != "manifest.json"},
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--figures", required=True)
    args = parser.parse_args()
    plot(args.run, args.output, args.figures)
