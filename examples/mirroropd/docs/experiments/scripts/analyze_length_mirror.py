#!/usr/bin/env python3
# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Audit long-window Full-KL and mirror runs from local, unsampled exports.

All loss observations precede their optimizer update. Reuse windows contain
complete batches; evaluation comparisons use exact common checkpoints only.
No cloud calls, training, interpolation, missing-value imputation or seed tests.
"""

import argparse
import csv
import gzip
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

from analyze_policy_run import finite, read_rows
from compare_policy_runs import analyze, cost

RUNS = {"E17": "cix5i2x2", "E20": "lwgb5eex", "E35": "ntzdb4o7", "E45": "625vw4ob", "E47": "e7pkdbtg"}
DATASETS = ("math500", "amc23", "aime25")


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def average(rows, key):
    xs = [r[key] for r in rows if finite(r.get(key))]
    return st.mean(xs) if xs else None


def reuse_window(groups, lo, hi, metrics, prefix):
    selected = [g for step, g in groups.items() if lo <= step <= hi and len(g) == 16]
    result = {"lo": lo, "hi": hi, "batches": len(selected), "metrics": {}, "position": {}}
    if not selected:
        return result
    for metric in metrics:
        if not all(metric in row for group in selected for row in group):
            continue
        profiles = [[r[metric] for r in group] for group in selected]
        result["metrics"][metric] = {
            "first": st.mean(p[0] for p in profiles),
            "last": st.mean(p[-1] for p in profiles),
            "mean_relative_drop": st.mean(1 - p[-1] / p[0] for p in profiles if p[0] > 1e-12)
            if metric not in ("distill/anchor_kl", "distill/target_log_z") and any(p[0] > 1e-12 for p in profiles)
            else None,
            "last_above_first_fraction": st.mean(p[-1] > p[0] for p in profiles),
            "second_above_first_fraction": st.mean(p[1] > p[0] for p in profiles),
            "argmin_observation_counts": {str(i + 1): sum(p.index(min(p)) == i for p in profiles) for i in range(16)},
            "upward_transitions": sum(b > a for p in profiles for a, b in zip(p[:-1], p[1:], strict=True)),
            "transitions": len(profiles) * 15,
            "profile": [st.mean(p[i] for p in profiles) for i in range(16)],
        }
    if prefix and prefix + "/p00_10/sum" in selected[0][0]:
        for index, phase in ((0, "first"), (-1, "last")):
            bins = []
            for start in range(0, 100, 10):
                stem = f"{prefix}/p{start:02d}_{start + 10:02d}"
                total = sum(g[index][stem + "/sum"] for g in selected)
                count = sum(g[index][stem + "/token_count"] for g in selected)
                bins.append({"sum": total, "count": count, "mean": total / count if count else None})
            result["position"][phase] = bins
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exports", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--logs-commit", required=True, help="Raw-log commit used when creating the exports")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    result = {"runs": {}, "comparisons": {}, "provenance": {}, "windows": {}}
    all_series, all_catalog, configs, all_rows = {}, [], {}, {}
    for label, rid in RUNS.items():
        directory = args.exports / rid
        for name in ("history.jsonl", "metric-definitions.jsonl"):
            (directory / (name + ".gz")).write_bytes(gzip.compress((directory / name).read_bytes(), mtime=0))
        base, series, catalog, config, _ = analyze(directory, label)
        base.pop("selected_metrics")
        base.pop("reuse")
        # Keep the report compact; every other evaluation metric remains in series.json.gz.
        report_metrics = {
            "acc/mean",
            "acc/mean_ci_low",
            "acc/mean_ci_high",
            "acc/pass@16",
            "acc/majority@16",
            "acc/all_correct@16",
            "acc/prediction_missing_count",
            "format/mean",
            "response_count",
            "observed_response_count",
            "materialized_response_count",
            "failed_count",
            "nonfinite_score_count",
            "generation/response_length/mean",
            "generation/finish_reason/length_count",
            "generation/finish_reason/stop_count",
        }
        for evaluation in base["evaluation"]:
            evaluation["metrics"] = {
                key: value for key, value in evaluation["metrics"].items() if key.split("/", 2)[-1] in report_metrics
            }
        configs[label] = config
        all_series[label] = series
        for metric, stats in catalog.items():
            all_catalog.append({"run": label, "metric": metric, "axis": series[metric]["axis"], **stats})
        rows = read_rows(directory / "history.jsonl")
        all_rows[label] = rows
        updates = [r for r in rows if "optim/actor/update_applied" in r]
        outers = [r for r in rows if "perf/timing/outer_step_seconds" in r]
        groups = defaultdict(list)
        for row in updates:
            groups[int(row["step/outer"])].append(row)
        for group in groups.values():
            group.sort(key=lambda r: r["step/update"])
        objective = "distill/mdpo/objective" if label in ("E45", "E47") else "distill/full_kl"
        teacher = "distill/teacher_kl" if label in ("E45", "E47") else objective
        metrics = [
            objective,
            teacher,
            "distill/target_kl",
            "distill/anchor_kl",
            "distill/target_log_z",
            "optim/actor/grad_norm_preclip",
        ]
        prefix = "distill/mdpo/objective_position" if label in ("E45", "E47") else "distill/full_kl_position"
        last_complete = max(k for k, v in groups.items() if len(v) == 16)
        windows = [(1, 20), (81, 100), (141, 160), (181, 200), (205, 224), (last_complete - 19, last_complete)]
        result["windows"][label] = {
            f"{lo}-{hi}": reuse_window(groups, lo, hi, metrics, prefix) for lo, hi in windows if hi <= last_complete
        }
        checks = {"position_rows": 0, "position_global_max_abs_error": None, "mirror_identity_max_abs_error": None}
        position_errors, identity_errors, geometric_errors, target_variation = [], [], [], []
        for row in updates:
            if prefix + "/p00_10/sum" in row:
                sums = sum(row[f"{prefix}/p{i:02d}_{i + 10:02d}/sum"] for i in range(0, 100, 10))
                count = sum(row[f"{prefix}/p{i:02d}_{i + 10:02d}/token_count"] for i in range(0, 100, 10))
                position_errors.append(abs(sums / count - row[objective]))
            if "distill/target_kl" in row:
                eta = config["distillation.distillation_loss.mirror.eta"]
                identity_errors.append(
                    abs(eta * row[objective] - (row["distill/target_kl"] - row["distill/target_log_z"]))
                )
                if label == "E45":
                    geometric_errors.append(
                        abs(row[objective] - (row[teacher] + (1 / eta - 1) * row["distill/anchor_kl"]))
                    )
        for group in groups.values():
            if "distill/target_log_z" in group[0]:
                zs = [r["distill/target_log_z"] for r in group]
                target_variation.append(max(zs) - min(zs))
        checks.update(
            position_rows=len(position_errors),
            position_global_max_abs_error=max(position_errors, default=None),
            mirror_identity_max_abs_error=max(identity_errors, default=None),
            geometric_identity_max_abs_error=max(geometric_errors, default=None),
            frozen_log_z_max_range=max(target_variation, default=None),
        )
        checks["all_evaluations_complete"] = all(
            all(
                r.get(f"eval/{d}/response_count") == n
                and r.get(f"eval/{d}/failed_count") == 0
                and r.get(f"eval/{d}/observed_response_count") == n
                and r.get(f"eval/{d}/materialized_response_count") == n
                and r.get(f"eval/{d}/nonfinite_score_count") == 0
                for d, n in [("math500", 8000), ("amc23", 640), ("aime25", 480)]
            )
            for r in rows
            if "eval/math500/acc/mean" in r
        )
        base["checks"] = checks
        base["batch_curves"] = []
        for outer, group in groups.items():
            if len(group) != 16:
                continue
            base["batch_curves"].append(
                {
                    "outer": outer,
                    "teacher_first": group[0][teacher],
                    "teacher_last": group[-1][teacher],
                    "objective_first": group[0][objective],
                    "objective_last": group[-1][objective],
                    "gradient_first": group[0]["optim/actor/grad_norm_preclip"],
                    "gradient_last": group[-1]["optim/actor/grad_norm_preclip"],
                }
            )
        base["window_rollout"] = {}
        for lo, hi in windows:
            rs = [r for r in outers if lo <= r["step/outer"] <= hi]
            if len(rs) != hi - lo + 1:
                continue
            responses = sum(r["rollout/response_count"] for r in rs)
            base["window_rollout"][f"{lo}-{hi}"] = {
                "length": average(rs, "rollout/response_length/mean"),
                "score": average(rs, "rollout/score/mean"),
                "length_stop_fraction": sum(r["rollout/finish_reason/length_count"] for r in rs) / responses,
                "stop_fraction": sum(r["rollout/finish_reason/stop_count"] for r in rs) / responses
                if all("rollout/finish_reason/stop_count" in r for r in rs)
                else None,
                "response_tokens": sum(r["rollout/response_tokens"] for r in rs),
            }
        base["recorded_code"] = json.loads((directory / "hardware.json").read_text()).get("git")
        result["runs"][label] = base
        print(
            label,
            base["source"]["decoder_status"],
            base["source"]["exit_code"],
            base["coverage"]["updates"],
            base["coverage"]["complete_outer_steps"],
            base["hardware"].get("gpu"),
            flush=True,
        )
    for name, labels in [
        ("length", ["E17", "E35"]),
        ("mirror", ["E20", "E45", "E47"]),
        ("length_equal_budget", ["E20", "E35"]),
    ]:
        evs = {label: {e["outer"]: e for e in result["runs"][label]["evaluation"]} for label in labels}
        common = sorted(set.intersection(*(set(e) for e in evs.values())))
        points = []
        for outer in common:
            points.append(
                {"outer": outer, "updates": outer * 16, "runs": {label: evs[label][outer] for label in labels}}
            )
        costs = {}
        for outer in sorted(
            set(common + [min(result["runs"][run_label]["coverage"]["complete_outer_steps"] for run_label in labels)])
        ):
            costs[str(outer)] = {}
            for label in labels:
                rows = [
                    r for r in all_rows[label] if "perf/timing/outer_step_seconds" in r and r["step/outer"] <= outer
                ]
                costs[str(outer)][label] = {
                    "timing": cost(rows, outer),
                    "response_tokens": sum(r["rollout/response_tokens"] for r in rows),
                    "responses": sum(r["rollout/response_count"] for r in rows),
                    "length_stop_fraction": sum(r["rollout/finish_reason/length_count"] for r in rows)
                    / sum(r["rollout/response_count"] for r in rows),
                }
        differences = {
            k: {run_label: configs[run_label].get(k) for run_label in labels}
            for k in sorted(set.union(*(set(configs[run_label]) for run_label in labels)))
            if len({json.dumps(configs[run_label].get(k), sort_keys=True) for run_label in labels}) > 1
        }
        result["comparisons"][name] = {
            "labels": labels,
            "common_evaluations": points,
            "costs": costs,
            "config_differences": differences,
        }
    result["provenance"] = {
        "log_merge_commit": args.logs_commit,
        "exports": str(args.exports.resolve()),
        "scope": "Unsampled local histories; no cloud state used as exit evidence.",
        "limitations": [
            "Single seed",
            "On-policy prefixes differ",
            "No interpolation between evaluation checkpoints",
            "Position bins describe each run objective, not a shared teacher-KL probe",
            "No held-out fixed-prefix KL or cross-run paired per-question records available",
        ],
    }
    dump(args.output / "analysis.json", result)
    dump(args.output / "configurations.json", configs)
    with (args.output / "metric-catalog.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_catalog[0]))
        writer.writeheader()
        writer.writerows(all_catalog)
    (args.output / "series.json.gz").write_bytes(
        gzip.compress(json.dumps(all_series, ensure_ascii=False, allow_nan=False).encode(), mtime=0)
    )
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
