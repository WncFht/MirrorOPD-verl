# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Summarize completed native runs without changing their banks or training budgets."""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from examples.mirroropd.research.core.io import dump, write_csv
from examples.mirroropd.research.studies.pipeline.plot_unified_evaluation import load_run
from verl.trainer.evaluation.artifacts import file_sha256
from verl.trainer.evaluation.metrics import summarize_rkl


def summarize(run, output):
    data = load_run(run)
    if data["status"] != "complete" or data["incomplete"]:
        raise ValueError("A final report requires all published events to be complete")
    events = {e["outer"]: e["metrics"] for e in data["events"]}
    first, last = min(events), max(events)
    groups = defaultdict(list)
    for row in data["scores"]:
        if row["outer"] in {first, last}:
            if row["tokens"] <= 0:
                raise ValueError("Do not summarize a partial question panel")
            groups[row["bank"], row["domain"], row["outer"]].append(row)
    panels = []
    for (bank, domain, outer), rows in sorted(groups.items()):
        if outer != last:
            continue
        values = np.asarray([r["rkl_sum"] / r["tokens"] for r in rows])
        initial = groups.get((bank, domain, first), [])
        record = dict(
            bank=bank,
            domain=domain,
            questions=len(rows),
            final_mean=float(values.mean()),
            final_median=float(np.median(values)),
            final_p90=float(np.quantile(values, 0.9)),
            final_max=float(values.max()),
            top1_share_percent=float(100 * values.max() / values.sum()),
            aggregation="equal_batch_question_mean" if bank == "history" else "question_mean",
        )
        if bank == "history":
            record["final_mean"] = summarize_rkl(rows, equal_groups="history_batch")["mean"]
        if initial and bank != "history":
            before = np.mean([r["rkl_sum"] / r["tokens"] for r in initial])
            record.update(
                initial_mean=float(before),
                reduction_percent=float(100 * (1 - values.mean() / before)),
                fixed_trajectories=bank != "public_current",
            )
            lookup = {r["id"]: r["rkl_sum"] / r["tokens"] for r in initial}
            if bank != "public_current":
                record["improved_questions"] = sum(r["rkl_sum"] / r["tokens"] < lookup[r["id"]] for r in rows)
            if bank != "public_current" and {r["id"] for r in initial} != {r["id"] for r in rows}:
                raise ValueError("Fixed panel changed trajectory identities")
        panels.append(record)
    batches = defaultdict(list)
    for row in data["updates"]:
        batches[int(row["step/outer"])].append(row)
    reuse = []
    for outer, rows in sorted(batches.items()):
        rows.sort(key=lambda r: r["step"])
        a, b = rows[0]["distill/full_kl"], rows[-1]["distill/full_kl"]
        reuse.append(
            dict(
                outer=outer,
                updates=len(rows),
                first_preupdate_loss=a,
                last_preupdate_loss=b,
                reduction_before_last_update_percent=100 * (1 - b / a),
            )
        )
    late = []
    for bank in ("source_heldout", "public_initial", "public_teacher"):
        for domain in [None] if bank == "source_heldout" else ["math", "code", "science", "if"]:
            key = f"eval/rkl/{bank}/" + (domain + "/" if domain else "") + "question_mean"
            if 25 in events and key in events[25] and key in events[last]:
                a, b = events[25][key], events[last][key]
                late.append(
                    dict(
                        bank=bank,
                        domain=domain,
                        start_outer=25,
                        end_outer=last,
                        before=a,
                        after=b,
                        absolute_reduction=a - b,
                        remaining_risk_reduction_percent=100 * (1 - b / a),
                    )
                )
    accuracy = []
    for spec in data["config"]["datasets"]:
        name = spec["name"]
        key = f"eval/accuracy/{name}/avg_at_n"
        if key in events[first] and key in events[last]:
            accuracy.append(
                dict(
                    benchmark=name,
                    n=spec["n"],
                    initial=events[first][key],
                    teacher=events[first][f"eval/accuracy/teacher/{name}/avg_at_n"],
                    final=events[last][key],
                    gap_recovery_percent=events[last].get(f"eval/accuracy/{name}/gap_recovery_percent"),
                )
            )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if accuracy:
        write_csv(output / "accuracy.csv", accuracy)
    write_csv(output / "panels.csv", panels)
    write_csv(output / "reuse.csv", reuse)
    write_csv(output / "late-progress.csv", late)
    dump(
        output / "summary.json",
        dict(
            models=data["models"],
            input_mode=data["input_mode"],
            final_outer=last,
            events=len(events),
            updates=len(data["updates"]),
            panels=panels,
            late_progress=late,
            median_within_batch_preupdate_reduction_percent=float(
                np.median([r["reduction_before_last_update_percent"] for r in reuse])
            ),
            sources=data["sources"],
            generator_sha256=file_sha256(__file__),
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    summarize(args.run, args.output)
