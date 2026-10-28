#!/usr/bin/env python3
# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Analyze equal-update Full-KL runs from the shared decoded archive.

No training or cloud calls. Raw logs stay in wandb-logs; this emits derived
statistics and all metric series for a reproducible report. Every loss is a
pre-update observation. Comparisons use actual optimizer evaluation coordinates,
not W&B event indices, interpolated checkpoints, or equal outer-step budgets.
"""

import argparse
import csv
import gzip
import json
import math
import re
import statistics as st
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from examples.mirroropd.research.analysis.analyze_policy_run import normalize, read_rows, stats
from examples.mirroropd.research.analysis.compare_policy_runs import (
    cost,
    describe,
    flatten,
    get,
    recorded_axes,
    reuse,
    telemetry,
)

RUNS = {"E18": "9hx3uegm", "E19": "d6jhh378", "E20": "lwgb5eex"}
DATASETS = ("math500", "amc23", "aime25")


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def csv_rows(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def budget(outer, updates, epochs, through):
    selected = [r for r in outer if r["step/optimizer"] <= through]
    us = [r for r in updates if r["step/optimizer"] <= through]
    counts = sum(r["rollout/response_count"] for r in selected)
    tokens = sum(r["rollout/response_tokens"] for r in selected)
    times = cost(selected, through // epochs)
    return {
        "update": through,
        "complete_outer": len(selected),
        "updates": len(us),
        "responses": counts,
        "response_tokens": tokens,
        "reused_response_tokens": tokens * epochs,
        "mean_response_length": tokens / counts,
        "train_score": sum(r["rollout/score/mean"] * r["rollout/response_count"] for r in selected) / counts,
        "length_stop_fraction": sum(r["rollout/finish_reason/length_count"] for r in selected) / counts,
        "time": times,
        "actor_seconds_per_update": times["sums"]["perf/timing/update_actor_seconds"] / len(us),
    }


def reuse_window(batches, updates, low, high):
    selected = [b for b in batches if low <= b["first_update"] and b["last_update"] <= high]
    us = [u for u in updates if low <= u["step/update"] <= high]
    by_outer = {b["outer"]: b for b in batches}
    rebounds = [
        by_outer[b["outer"] + 1]["first"] / b["last"]
        for b in selected
        if b["outer"] + 1 in by_outer and by_outer[b["outer"] + 1]["first_update"] <= high
    ]
    return {
        "update_from": low,
        "update_through": high,
        "batches": len(selected),
        "first_kl": describe([b["first"] for b in selected]),
        "last_preupdate_kl": describe([b["last"] for b in selected]),
        "within_drop": describe([b["relative_drop"] for b in selected]),
        "last_transition_drop": describe([1 - b["values"][-1] / b["values"][-2] for b in selected]),
        "next_batch_rebound": describe(rebounds),
        "grad_norm": describe([u["optim/actor/grad_norm_preclip"] for u in us]),
        "adjacent_increases": sum(b["increases"] for b in selected),
        "adjacent_transitions": sum(len(b["values"]) - 1 for b in selected),
        "normalized_profile": [
            st.mean(b["values"][i] / b["first"] for b in selected) for i in range(len(selected[0]["values"]))
        ],
        "prefix_drop": {
            str(n): st.mean(1 - b["values"][n - 1] / b["first"] for b in selected)
            for n in [2, 4, 8, 16, 32, 64]
            if n <= len(selected[0]["values"])
        },
    }


def audit_run(meta, config, rows):
    epochs = get(config, "actor_rollout_ref.actor.ppo_epochs")
    updates = [r for r in rows if "optim/actor/update_applied" in r]
    outer = [r for r in rows if "perf/timing/outer_step_seconds" in r]
    evals = [r for r in rows if "eval/math500/acc/mean" in r]
    assert [r["step/update"] for r in updates] == list(range(1, len(updates) + 1))
    for r in updates:
        assert r["step/update"] == r["step/optimizer"]
        assert r["step/outer"] == (r["step/update"] - 1) // epochs + 1
        assert r["rollout/data_reuse/update_index"] == (r["step/update"] - 1) % epochs + 1
        assert r["optim/actor/update_applied"] == 1 and r["optim/actor/update_skipped"] == 0
        assert math.isfinite(r["distill/full_kl"]) and math.isfinite(r["optim/actor/grad_norm_preclip"])
    assert [r["step/outer"] for r in outer] == list(range(1, len(outer) + 1))
    assert len(updates) == len(outer) * epochs
    assert all(r["step/optimizer"] == r["step/outer"] * epochs for r in outer + evals)
    for row in evals:
        for ds, prompts in zip(DATASETS, [500, 40, 30], strict=True):
            pre = f"eval/{ds}/"
            assert row[pre + "prompt_count"] == prompts
            assert all(
                row[pre + k] == prompts * 16
                for k in ["response_count", "observed_response_count", "materialized_response_count"]
            )
            assert row[pre + "samples_per_prompt_min"] == row[pre + "samples_per_prompt_max"] == 16
            assert row[pre + "failed_count"] == row[pre + "nonfinite_score_count"] == 0
    if meta["exit_code"] == 0:
        assert len(outer) == get(config, "trainer.total_training_steps")
        assert evals[-1]["step/optimizer"] == len(updates)
    return updates, outer, evals


def analyze(logs, label, run_id):
    candidates = list((logs / "derived/MirrorOPD-verl").glob(f"*-{run_id}"))
    assert len(candidates) == 1, candidates
    path = candidates[0]
    meta = json.loads((path / "run.json").read_text())
    assert meta["status"] in ("finished", "snapshot", "partial_snapshot")
    config = json.loads((path / "config.json").read_text())
    raw = logs / "MirrorOPD-verl/wandb" / path.name / f"run-{run_id}.wandb"
    stamp = raw.stat()
    assert (stamp.st_size, stamp.st_mtime_ns) == (meta["cache_key"]["bytes"], meta["cache_key"]["mtime_ns"]), (
        "Decode first"
    )
    rows = read_rows(path / "history.jsonl")
    axes = recorded_axes(rows, read_rows(path / "metric-definitions.jsonl"))
    series = normalize(rows, axes)
    catalog = {key: stats(series, key) for key in series}
    updates, outer, evals = audit_run(meta, config, rows)
    epochs = get(config, "actor_rollout_ref.actor.ppo_epochs")
    batches = reuse(updates, epochs)
    env = read_rows(path / "environment.jsonl")
    gpu = next(e for e in env if e.get("gpu_count"))
    system = [
        {
            "_timestamp": datetime.fromisoformat(
                re.sub(r"(\.\d{6})\d+", r"\1", r["timestamp"]).replace("Z", "+00:00")
            ).timestamp(),
            **{"system." + k: v for k, v in r["metrics"].items()},
        }
        for r in read_rows(path / "system.jsonl.gz")
    ]
    budgets = sorted({512, 1024, 1536, 2048, 2560, len(updates)} | {r["step/optimizer"] for r in evals})
    eval_entries = []
    for row in evals:
        metrics = {k: v for k, v in row.items() if k.startswith("eval/")}
        eval_entries.append(
            {
                "outer": row["step/outer"],
                "optimizer": row["step/optimizer"],
                "accounted_hours": cost(outer, row["step/outer"])["accounted_hours"],
                "metrics": metrics,
            }
        )
    common_windows = [(i + 1, i + 512) for i in range(0, len(updates) - 511, 512)]
    common_windows += [(1025, 2560), (2049, 2560), (len(updates) - 511, len(updates))]
    windows = {f"{lo}-{hi}": reuse_window(batches, updates, lo, hi) for lo, hi in common_windows}
    clip = get(config, "actor_rollout_ref.actor.optim.clip_grad")
    clipped = [r["step/update"] for r in updates if r["optim/actor/grad_norm_preclip"] > clip]
    coord = {axis: {} for axis in ["step/update", "step/optimizer", "step/outer"]}
    for row in rows:
        for axis in coord:
            if axis in row:
                coord[axis][row[axis]] = {"outer": row["step/outer"], "optimizer": row["step/optimizer"]}
    outcome = {
        "id": run_id,
        "epochs": epochs,
        "status": meta["status"],
        "exit_code": meta["exit_code"],
        "history_rows": len(rows),
        "updates": len(updates),
        "complete_outer": len(outer),
        "last_eval_optimizer": evals[-1]["step/optimizer"],
        "last_eval_outer": evals[-1]["step/outer"],
        "metric_families": dict(Counter(k.split("/")[0] for k in series)),
        "metric_count": len(series),
        "constant_metrics": sum(x["constant"] for x in catalog.values()),
        "position_metrics": [k for k in series if "full_kl_position/" in k],
        "policy_metrics": [k for k in series if k.startswith("policy/")],
        "hardware": {"gpus": gpu["gpu_count"], "gpu": gpu["gpu_type"], "writer_id": gpu["writer_id"]},
        "source": {
            "logs_directory": str(path.relative_to(logs)),
            "cache_key": meta["cache_key"],
            "recorded_git": meta["run"].get("git"),
            "start_time_utc": meta["run"]["start_time"],
            "record_counts": meta["record_counts"],
            "error": meta["error"],
        },
        "budgets": {str(n): budget(outer, updates, epochs, n) for n in budgets},
        "evaluation": eval_entries,
        "reuse": batches,
        "reuse_windows": windows,
        "gradient_clip": {"count": len(clipped), "last_update": max(clipped, default=None), "updates": clipped},
        "loss_identity_max_abs_error": max(abs(r["optim/actor/loss"] - r["distill/full_kl"]) for r in updates),
        "last_kl": updates[-1]["distill/full_kl"],
        "last_grad_norm": updates[-1]["optim/actor/grad_norm_preclip"],
        "nonfinite_flags": {k: v["max"] for k, v in catalog.items() if "nonfinite" in k},
        "system": telemetry(system, outer, gpu["gpu_count"]),
        "recorded_wall_hours": (
            outer[-1]["_timestamp"]
            - datetime.fromisoformat(meta["run"]["start_time"].replace("Z", "+00:00")).timestamp()
        )
        / 3600,
        "outer_series": [{k: v for k, v in r.items() if not k.startswith("eval/")} for r in outer],
    }
    return outcome, series, catalog, flatten(config), coord


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs-repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    results, series, catalogs, configs, coordinates = {}, {}, {}, {}, {}
    for label, rid in RUNS.items():
        results[label], series[label], catalogs[label], configs[label], coordinates[label] = analyze(
            args.logs_repo.resolve(), label, rid
        )
    common = sorted(set.intersection(*[{e["optimizer"] for e in r["evaluation"]} for r in results.values()]))
    differences = {
        k: {label: cfg.get(k) for label, cfg in configs.items()}
        for k in sorted(set.union(*(set(c) for c in configs.values())))
        if len({json.dumps(c.get(k), sort_keys=True) for c in configs.values()}) > 1
    }
    analysis = {
        "runs": results,
        "common_evaluation_updates": common,
        "logs_commit": subprocess.check_output(
            ["git", "-C", str(args.logs_repo), "rev-parse", "HEAD"], text=True
        ).strip(),
        "analysis_base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "analysis_revision": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    evaluation_rows = []
    for label, r in results.items():
        for ev in r["evaluation"]:
            for ds in DATASETS:
                pre = f"eval/{ds}/"
                evaluation_rows.append(
                    {
                        "run": label,
                        "dataset": ds,
                        "outer": ev["outer"],
                        "update": ev["optimizer"],
                        "hours": ev["accounted_hours"],
                        **{
                            k: ev["metrics"][pre + k]
                            for k in [
                                "acc/mean",
                                "acc/mean_ci_low",
                                "acc/mean_ci_high",
                                "acc/pass@16",
                                "acc/majority@16",
                                "acc/all_correct@16",
                                "generation/response_length/mean",
                                "generation/response_length/p95",
                                "generation/finish_reason/length_count",
                                "acc/prediction_missing_count",
                                "format/mean",
                                "response_count",
                            ]
                        },
                    }
                )
    write(args.out / "analysis.json", analysis)
    write(args.out / "config-differences.json", differences)
    write(args.out / "configurations.json", configs)
    csv_rows(args.out / "evaluation.csv", evaluation_rows)
    csv_rows(
        args.out / "metric-catalog.csv",
        [{"run": label, "metric": k, **v} for label, cat in catalogs.items() for k, v in cat.items()],
    )
    union = sorted(set.union(*(set(s) for s in series.values())))
    csv_rows(
        args.out / "metric-coverage.csv",
        [{"metric": k, **{label: catalogs[label].get(k, {}).get("count", "missing") for label in RUNS}} for k in union],
    )
    bundle = {"analysis": analysis, "series": series, "coordinates": coordinates, "config_differences": differences}
    (args.out / "comparison.json.gz").write_bytes(gzip.compress(json.dumps(bundle, allow_nan=False).encode(), mtime=0))
    print(
        json.dumps(
            {
                k: {
                    f: v[f]
                    for f in ["updates", "complete_outer", "metric_count", "last_eval_optimizer", "gradient_clip"]
                }
                for k, v in results.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
