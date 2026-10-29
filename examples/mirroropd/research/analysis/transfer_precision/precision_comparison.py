# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Recompute E50–E53 precision/target comparisons from immutable local snapshots.

Each batch has sixteen pre-update observations, spanning fifteen attempts.
Window reductions are ratios of means; checkpoint deltas are descriptive,
not independent seeds. No training, cloud requests or source-log writes.
"""

import argparse
import hashlib
import json
import statistics
from pathlib import Path

from examples.mirroropd.research.analysis.transfer_precision.analyze import flatten, write_csv

RUNS = {"E50": "rsi5krqd", "E51": "6vv5o9b8", "E52": "7rwc1ioz", "E53": "k7db6qxl"}
DATASETS = ("math500", "amc23", "aime25")
TOL = 1e-10


def analyze(logs, output):
    output.mkdir(parents=True, exist_ok=True)
    summaries, configs, cohorts, evaluations, costs, batches = {}, {}, [], [], [], []
    for label, rid in RUNS.items():
        sources = list((logs / "derived/MirrorOPD-verl").glob("*" + rid))
        if len(sources) != 1:
            raise ValueError(f"Expected one decoded source for {rid}")
        source = sources[0]
        metadata = json.loads((source / "run.json").read_text())
        key = metadata["cache_key"]
        raw = Path(key["source"])
        stamp = raw.stat()
        if (stamp.st_size, stamp.st_mtime_ns) != (key["bytes"], key["mtime_ns"]):
            raise ValueError(f"Stale decode for {rid}")
        if metadata["status"] not in ("finished", "snapshot", "partial_snapshot"):
            raise ValueError(f"Invalid decode for {rid}: {metadata['status']}")
        config = json.loads((source / "config.json").read_text())
        configs[label] = flatten(config)
        history = [json.loads(line) for line in (source / "history.jsonl").read_text().splitlines()]
        updates, outer = {}, {}
        for row in history:
            step = row.get("step/outer")
            if "optim/actor/loss" in row:
                updates.setdefault(step, []).append(row)
            if "rollout/response_tokens" in row:
                outer[step] = row
            for dataset in DATASETS:
                prefix = f"eval/{dataset}/acc/"
                if prefix + "mean" in row:
                    evaluations.append(
                        {
                            "run": label,
                            "outer": step,
                            "dataset": dataset,
                            **{
                                k: row.get(prefix + k)
                                for k in ("mean", "mean_ci_low", "mean_ci_high", "pass@16", "majority@16")
                            },
                        }
                    )
        if any(len(rows) != 16 for rows in updates.values()):
            raise ValueError(f"Unexpected reuse count in {label}")
        primary = "distill/target_kl" if label in ("E51", "E53") else "distill/full_kl"
        teacher = "distill/teacher_kl" if label in ("E51", "E53") else "distill/full_kl"
        for step, rows in updates.items():
            first, second, last = (rows[i][primary] for i in (0, 1, -1))
            batch = {
                "run": label,
                "outer": step,
                "first": first,
                "second": second,
                "last": last,
                "net_gain": first - last,
                "first_step_rise": second > first + TOL,
                "end_worse": last > first + TOL,
                "skipped": sum(r.get("optim/actor/update_skipped", 0) for r in rows),
            }
            if primary == "distill/target_kl":
                batch.update(
                    teacher_gain=rows[0][teacher] - rows[-1][teacher],
                    anchor_cost=(7 / 3) * (rows[-1]["distill/anchor_kl"] - rows[0]["distill/anchor_kl"]),
                )
            batches.append(batch)
        summaries[label] = {
            "run_id": rid,
            "decode_status": metadata["status"],
            "exit_code": metadata["exit_code"],
            "raw_source": str(raw.relative_to(logs)),
            "sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
            "recorded_git": metadata["run"].get("git"),
            "history_rows": len(history),
            "outer": max(updates),
            "successful_updates": sum(
                r.get("optim/actor/update_applied", 0) for rows in updates.values() for r in rows
            ),
            "skipped_updates": sum(r.get("optim/actor/update_skipped", 0) for rows in updates.values() for r in rows),
            "precision": configs[label]["actor_rollout_ref.actor.fsdp_config.forward_precision"],
            "eta": configs[label]["distillation.distillation_loss.mirror.eta"],
            "worse_batches": [b["outer"] for b in batches if b["run"] == label and b["end_worse"]],
        }
        environment = [json.loads(line) for line in (source / "environment.jsonl").read_text().splitlines()]
        summaries[label]["hardware"] = {
            key: row[key] for row in environment for key in ("gpu_type", "gpu_count", "cpu_count") if key in row
        }
        if primary == "distill/target_kl":
            summaries[label].update(
                max_log_z_drift=max(
                    max(r["distill/target_log_z"] for r in rows) - min(r["distill/target_log_z"] for r in rows)
                    for rows in updates.values()
                ),
                max_objective_residual=max(
                    abs(r["distill/mdpo/objective"] - r[teacher] - (7 / 3) * r["distill/anchor_kl"])
                    for rows in updates.values()
                    for r in rows
                ),
            )
        windows = [
            ("all", 1, max(updates)),
            ("common96", 1, 96),
            ("early", 1, 32),
            ("middle", 65, 96),
            ("late", 225, 256),
        ]
        for window, lo, hi in windows:
            if hi > max(updates):
                continue
            selected = [rows for step, rows in updates.items() if lo <= step <= hi]
            values = [b for b in batches if b["run"] == label and lo <= b["outer"] <= hi]
            first = statistics.mean(b["first"] for b in values)
            last = statistics.mean(b["last"] for b in values)
            cohort = {
                "run": label,
                "window": window,
                "start": lo,
                "end": hi,
                "batches": len(values),
                "primary": primary,
                "first_mean": first,
                "second_mean": statistics.mean(b["second"] for b in values),
                "last_mean": last,
                "reduction_pct": 100 * (1 - last / first),
                "first_step_rises": sum(b["first_step_rise"] for b in values),
                "end_worse": sum(b["end_worse"] for b in values),
                "adjacent_rises": sum(
                    b[primary] > a[primary] + TOL for rows in selected for a, b in zip(rows, rows[1:], strict=False)
                ),
                "teacher_first": statistics.mean(rows[0][teacher] for rows in selected),
                "teacher_last": statistics.mean(rows[-1][teacher] for rows in selected),
            }
            if primary == "distill/target_kl":
                cohort.update(
                    teacher_gain=statistics.mean(b["teacher_gain"] for b in values),
                    anchor_cost=statistics.mean(b["anchor_cost"] for b in values),
                )
                cohort["anchor_cost_over_teacher_gain"] = cohort["anchor_cost"] / cohort["teacher_gain"]
            cohorts.append(cohort)
            selected_outer = [row for step, row in outer.items() if lo <= step <= hi]
            if len(selected_outer) != hi - lo + 1:
                raise ValueError(f"Missing outer records for {label}/{window}")
            costs.append(
                {
                    "run": label,
                    "window": window,
                    "batches": len(selected_outer),
                    "response_tokens": sum(r["rollout/response_tokens"] for r in selected_outer),
                    "length_stop_fraction": sum(r["rollout/finish_reason/length_count"] for r in selected_outer)
                    / sum(r["rollout/response_count"] for r in selected_outer),
                    "actor_seconds": sum(r["perf/timing/update_actor_seconds"] for r in selected_outer),
                    "outer_seconds": sum(r["perf/timing/outer_step_seconds"] for r in selected_outer),
                    "actor_peak_gib": max(r["perf/actor/allocated_peak_gib"] for r in selected_outer),
                }
            )
    differences = {
        key: {label: config.get(key) for label, config in configs.items()}
        for key in sorted(set().union(*(set(c) for c in configs.values())))
        if len({json.dumps(c.get(key), sort_keys=True) for c in configs.values()}) > 1
    }
    for name, rows in (("cohorts", cohorts), ("batches", batches), ("evaluations", evaluations), ("costs", costs)):
        write_csv(output / (name + ".csv"), rows)
    comparisons = []
    for a, b, end in (("E51", "E50", 256), ("E52", "E50", 256), ("E53", "E51", 96), ("E53", "E52", 96)):
        for dataset in DATASETS:
            left = {r["outer"]: r["mean"] for r in evaluations if r["run"] == a and r["dataset"] == dataset}
            right = {r["outer"]: r["mean"] for r in evaluations if r["run"] == b and r["dataset"] == dataset}
            steps = sorted(s for s in left.keys() & right.keys() if s <= end)
            comparisons.append(
                {
                    "comparison": f"{a}-{b}",
                    "dataset": dataset,
                    "last_common_outer": steps[-1],
                    "checkpoints": len(steps),
                    "last_delta_pp": 100 * (left[steps[-1]] - right[steps[-1]]),
                    "mean_checkpoint_delta_pp": statistics.mean(100 * (left[s] - right[s]) for s in steps),
                }
            )
    write_csv(output / "comparisons.csv", comparisons)
    for name, value in (("summary", summaries), ("config-differences", differences)):
        (output / (name + ".json")).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summaries, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs-repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.logs_repo.resolve(), args.out.resolve())


if __name__ == "__main__":
    main()
