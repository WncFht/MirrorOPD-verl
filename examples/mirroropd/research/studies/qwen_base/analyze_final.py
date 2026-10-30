#!/usr/bin/env python3
# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Freeze and analyze completed E39 metrics and validation samples on CPU.

Read W&B history decoded by wandb-logs. No API upload or training is performed.
Repetition uses one fixed sample_index=0 per question at steps 0/10/100/200,
re-tokenized from saved decoded text, rather than original generation IDs.
"""

import argparse
import collections
import csv
import gzip
import json
import statistics as st
from pathlib import Path

import numpy as np

from examples.mirroropd.assets.paths import resolve_location

DATASETS = [("math500", "MATH-500"), ("amc23", "AMC23"), ("aime25", "AIME25")]
PANEL_STEPS = [0, 10, 100, 200]
BINS = [f"p{i:02d}_{i + 10:02d}" for i in range(0, 100, 10)]


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def csvsave(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def freeze_responses(run, data):
    prompts = {}
    panel = []
    with gzip.open(data / "validation-responses.jsonl.gz", "wt") as compact:
        for step in range(0, 201, 10):
            count = 0
            with (run / f"training/validation/records/{step}.jsonl").open() as stream:
                for line in stream:
                    row = json.loads(line)
                    key = row["prompt_id"]
                    prompt = {"dataset": row["data_source"], "input": row["input"], "ground_truth": row["gts"]}
                    if key in prompts:
                        assert prompts[key] == prompt, key
                    else:
                        prompts[key] = prompt
                    small = {
                        k: row.get(k)
                        for k in [
                            "step",
                            "data_source",
                            "prompt_id",
                            "sample_index",
                            "acc",
                            "response_length",
                            "finish_reason",
                            "format_score",
                            "pred",
                        ]
                    }
                    compact.write(json.dumps(small, ensure_ascii=False) + "\n")
                    if step in PANEL_STEPS and row["sample_index"] == 0:
                        panel.append({**small, "output": row["output"]})
                    count += 1
            assert count == 9120, (step, count)
            print("Archived validation", step, count, flush=True)
    assert len(prompts) == 570 and len(panel) == 2280
    save(data / "prompts.json", prompts)
    with gzip.open(data / "sample-panel.jsonl.gz", "wt") as stream:
        for row in panel:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    timers = []
    for path in (run / "timeline").glob("events-*.jsonl"):
        with path.open() as stream:
            first = stream.readline()
            if '"role": "driver"' not in first:
                continue
            for line in [first, *stream]:
                if '"name": "timer/testing"' in line:
                    timers.append(json.loads(line))
    save(data / "validation-timers.json", sorted(timers, key=lambda x: x["start_ns"]))


def repeat_metrics(ids):
    grams = max(0, len(ids) - 7)
    unique = len(set(zip(*(ids[i:] for i in range(8)), strict=False))) if grams else 0
    array = np.asarray(ids, dtype=np.int64)
    span = 0
    best_period = 0
    for period in range(1, min(128, len(ids) // 4) + 1):
        eq = array[period:] == array[:-period]
        edges = np.diff(np.concatenate(([False], eq, [False])).astype(np.int8))
        starts = np.flatnonzero(edges == 1)
        ends = np.flatnonzero(edges == -1)
        if len(starts):
            run = int((ends - starts).max())
            if run >= 3 * period and run + period >= 128 and run + period > span:
                span, best_period = run + period, period
    return {
        "retokenized_length": len(ids),
        "repeat8_text_tokens": 1 - unique / grams if grams else 0.0,
        "long_loop": span >= 128,
        "loop_span": span,
        "loop_period": best_period,
    }


def analyze(data):
    history = [json.loads(line) for line in (data / "history.jsonl").read_text().splitlines()]
    updates = [d for d in history if "distill/termination_equiv_kl" in d]
    outer = [d for d in history if "perf/timing/outer_step_seconds" in d]
    evals = [d for d in history if "eval/math500/acc/mean" in d]
    assert len(updates) == 3200 and len(outer) == 200 and len(evals) == 21
    assert [int(d["step/update"]) for d in updates] == list(range(1, 3201))
    assert [int(d["step/optimizer"]) for d in updates] == list(range(1, 3201))
    assert [int(d["step/outer"]) for d in outer] == list(range(1, 201))
    assert [int(d["step/outer"]) for d in evals] == list(range(0, 201, 10))
    by_step = collections.defaultdict(list)
    for d in updates:
        by_step[int(d["step/outer"])].append(d)
    assert all(len(v) == 16 for v in by_step.values())
    evaluation = []
    for d in evals:
        for slug, name in DATASETS:
            p = f"eval/{slug}/"
            n = d[p + "response_count"]
            evaluation.append(
                {
                    "outer": int(d["step/outer"]),
                    "dataset": name,
                    "slug": slug,
                    "responses": int(n),
                    "accuracy": d[p + "acc/mean"],
                    "accuracy_ci_low": d[p + "acc/mean_ci_low"],
                    "accuracy_ci_high": d[p + "acc/mean_ci_high"],
                    "pass16": d[p + "acc/pass@16"],
                    "majority16": d[p + "acc/majority@16"],
                    "mean_length": d[p + "generation/response_length/mean"],
                    "median_length": d[p + "generation/response_length/p50"],
                    "p95_length": d[p + "generation/response_length/p95"],
                    "cap_fraction": d[p + "generation/finish_reason/length_count"] / n,
                    "format_score": d.get(p + "format/mean"),
                    "prediction_missing_count": d.get(p + "acc/prediction_missing_count"),
                    "validation_seconds": d.get("perf/timing/validation_seconds"),
                }
            )
    batches = []
    positions = []
    for d in outer:
        step = int(d["step/outer"])
        u = by_step[step]
        a, b = u[0], u[-1]
        row = {
            "outer": step,
            "kl_first": a["distill/termination_equiv_kl"],
            "kl_last": b["distill/termination_equiv_kl"],
            "raw_kl_first": a["distill/full_kl"],
            "raw_kl_last": b["distill/full_kl"],
            "identity_first": a["distill/eos_identity_kl"],
            "identity_last": b["distill/eos_identity_kl"],
            "inner_relative_drop": 1 - b["distill/termination_equiv_kl"] / a["distill/termination_equiv_kl"],
            "mean_length": d["rollout/response_length/mean"],
            "train_score": d["rollout/score/mean"],
            "cap_fraction": d["rollout/finish_reason/length_count"] / d["rollout/response_count"],
            "grad_first": a["optim/actor/grad_norm_preclip"],
            "grad_last": b["optim/actor/grad_norm_preclip"],
            "train_seconds": d["perf/timing/outer_step_seconds"],
            "actor_seconds": d["perf/timing/update_actor_seconds"],
            "teacher_seconds": d["perf/timing/teacher_forward_seconds"],
            "rollout_seconds": d["perf/timing/rollout_pipeline_seconds"],
            "fsdp_peak_gib": d.get("perf/memory/fsdp_allocated_peak_gib"),
        }
        batches.append(row)
        for endpoint, urow in [("first", a), ("last", b)]:
            for bin_name in BINS:
                stem = f"distill/full_kl_position/{bin_name}/"
                positions.append(
                    {
                        "outer": step,
                        "endpoint": endpoint,
                        "bin": bin_name,
                        "sum": urow[stem + "sum"],
                        "token_count": urow[stem + "token_count"],
                        "mean": urow[stem + "mean"],
                    }
                )
    csvsave(data / "validation.csv", evaluation)
    csvsave(data / "training.csv", batches)
    csvsave(data / "position-kl.csv", positions)
    q = collections.defaultdict(lambda: collections.defaultdict(list))
    full_stats = collections.defaultdict(list)
    with gzip.open(data / "validation-responses.jsonl.gz", "rt") as stream:
        for line in stream:
            x = json.loads(line)
            q[x["step"]][x["prompt_id"]].append(x)
            full_stats[(x["step"], x["data_source"])].append(x)
    assert all(len(xs) == 16 for groups in q.values() for xs in groups.values())
    assert all(set(groups) == set(q[0]) for groups in q.values())
    extras = []
    for row in evaluation:
        xs = full_stats[(row["outer"], row["dataset"])]
        assert len(xs) == row["responses"]
        assert abs(st.mean(x["acc"] for x in xs) - row["accuracy"]) < 1e-10
        assert abs(st.mean(x["response_length"] for x in xs) - row["mean_length"]) < 1e-7
        extras.append(
            {
                "outer": row["outer"],
                "dataset": row["dataset"],
                "short_under32_fraction": st.mean(x["response_length"] < 32 for x in xs),
                "correct_length": st.mean(x["response_length"] for x in xs if x["acc"])
                if any(x["acc"] for x in xs)
                else None,
                "wrong_length": st.mean(x["response_length"] for x in xs if not x["acc"])
                if any(not x["acc"] for x in xs)
                else None,
            }
        )
    csvsave(data / "validation-output-diagnostics.csv", extras)
    paired = []
    rng = np.random.default_rng(42)
    for left, right in [(0, 200), (10, 200), (100, 200)]:
        for _, name in DATASETS:
            ids = sorted(k for k, v in q[left].items() if v[0]["data_source"] == name)
            delta = np.array(
                [st.mean(x["acc"] for x in q[right][key]) - st.mean(x["acc"] for x in q[left][key]) for key in ids]
            )
            boot = delta[rng.integers(0, len(ids), size=(5000, len(ids)))].mean(axis=1)
            paired.append(
                {
                    "from": left,
                    "to": right,
                    "dataset": name,
                    "delta": float(delta.mean()),
                    "ci_low": float(np.quantile(boot, 0.025)),
                    "ci_high": float(np.quantile(boot, 0.975)),
                    "questions": len(ids),
                }
            )
    save(data / "paired-accuracy.json", paired)
    windows = []
    for lo, hi in [(1, 10), (11, 30), (81, 100), (121, 140), (181, 200)]:
        bb = [b for b in batches if lo <= b["outer"] <= hi]
        windows.append(
            {
                "from": lo,
                "to": hi,
                **{
                    k: st.mean(b[k] for b in bb)
                    for k in [
                        "kl_first",
                        "kl_last",
                        "raw_kl_first",
                        "raw_kl_last",
                        "identity_first",
                        "identity_last",
                        "inner_relative_drop",
                        "mean_length",
                        "train_score",
                        "cap_fraction",
                        "grad_first",
                        "grad_last",
                    ]
                },
            }
        )
    timings = {
        k: sum(d.get(k, 0) for d in history)
        for k in [
            "perf/timing/validation_seconds",
            "perf/timing/outer_step_seconds",
            "perf/timing/teacher_forward_seconds",
            "perf/timing/update_actor_seconds",
            "perf/timing/rollout_pipeline_seconds",
            "perf/timing/save_checkpoint_seconds",
        ]
    }
    started = json.loads((data / "started.json").read_text())["time"]
    exit_record = json.loads((data / "exit.json").read_text())
    ended = exit_record["time"]
    assert exit_record["exit_code"] == 0 and json.loads((data / "recovered.json").read_text())["physical_gpus"] == [
        4,
        5,
        6,
        7,
    ]
    summary = {
        "completed": True,
        "run_id": "f0e44ec7",
        "source_commit": (data / "source_commit.txt").read_text().strip(),
        "started_at": started,
        "ended_at": ended,
        "wall_hours": (ended - started) / 3600,
        "updates": len(updates),
        "outer_steps": len(outer),
        "validations": len(evals),
        "validation_responses": sum(x["responses"] for x in evaluation),
        "gradient_nonfinite_updates": sum(d["optim/actor/grad_norm_preclip_nonfinite"] for d in updates),
        "loss_nonfinite_updates": sum(d["optim/actor/loss_nonfinite"] for d in updates),
        "skipped_updates": sum(d["optim/actor/update_skipped"] for d in updates),
        "max_grad_norm": max(d["optim/actor/grad_norm_preclip"] for d in updates),
        "max_fsdp_peak_gib": max(x["fsdp_peak_gib"] for x in batches),
        "timing_seconds": timings,
        "initial_validation_seconds": evals[0]["perf/timing/validation_seconds"],
        "windows": windows,
        "paired_accuracy": paired,
        "last_update": {k: v for k, v in updates[-1].items() if "position" not in k},
        "kv_preemption_available": False,
    }
    timers = json.loads((data / "validation-timers.json").read_text())
    samples = [json.loads(line) for line in (data / "gpu-samples.jsonl").read_text().splitlines()]
    resources = []
    assert len(timers) == 21
    for step, timer in zip(range(0, 201, 10), timers, strict=False):
        end = timer["wall_ns"] / 1e9
        start = end - (timer["end_ns"] - timer["start_ns"]) / 1e9
        measurements = []
        count = 0
        for sample in samples:
            if start <= sample["time"] <= end and sample.get("returncode") == 0:
                for gpu in sample["gpus"]:
                    _, util, memory = [float(value.strip()) for value in gpu.split(",")]
                    measurements.append((util, memory / 1024))
                count += 1
        resources.append(
            {
                "outer": step,
                "minutes": (end - start) / 60,
                "samples": count,
                "mean_gpu_util_pct": st.mean(x[0] for x in measurements),
                "mean_mem_gib": st.mean(x[1] for x in measurements),
                "max_mem_gib": max(x[1] for x in measurements),
            }
        )
    save(data / "resources.json", resources)
    save(data / "analysis.json", summary)
    print(
        "SUMMARY",
        json.dumps({k: v for k, v in summary.items() if k not in ["last_update", "paired_accuracy", "windows"]}),
        flush=True,
    )
    print("PAIRED", paired, flush=True)


def analyze_panel(data):
    from transformers import AutoTokenizer

    config = json.loads((data / "config.json").read_text())
    tok = AutoTokenizer.from_pretrained(
        resolve_location(config["actor_rollout_ref"]["model"]["path"]), local_files_only=True
    )
    rows = []
    texts = {}
    with gzip.open(data / "sample-panel.jsonl.gz", "rt") as stream:
        for line in stream:
            x = json.loads(line)
            text = x.pop("output")
            texts[(x["step"], x["prompt_id"])] = text
            x.update(repeat_metrics(tok.encode(text, add_special_tokens=False)))
            rows.append(x)
            if len(rows) % 570 == 0:
                print("Repetition panel processed", len(rows), flush=True)
    save(data / "sample-panel-metrics.json", rows)
    result = []
    for step in PANEL_STEPS:
        for group in ["all", *(n for _, n in DATASETS)]:
            xs = [r for r in rows if r["step"] == step and (group == "all" or r["data_source"] == group)]
            result.append(
                {
                    "step": step,
                    "dataset": group,
                    "responses": len(xs),
                    "acc": st.mean(x["acc"] for x in xs),
                    "mean_length": st.mean(x["response_length"] for x in xs),
                    "loop_fraction": st.mean(x["long_loop"] for x in xs),
                    "repeat8": st.mean(x["repeat8_text_tokens"] for x in xs),
                    "short_fraction": st.mean(x["response_length"] < 32 for x in xs),
                }
            )
    csvsave(data / "sample-panel-summary.csv", result)
    prompts = json.loads((data / "prompts.json").read_text())
    lookup = {(x["step"], x["prompt_id"]): x for x in rows}
    keys = sorted(prompts)
    categories = {
        "loop_to_correct": [
            k
            for k in keys
            if lookup[(0, k)]["long_loop"]
            and not lookup[(200, k)]["long_loop"]
            and lookup[(200, k)]["acc"]
            and not lookup[(0, k)]["acc"]
        ],
        "short_to_correct": [k for k in keys if lookup[(0, k)]["response_length"] < 32 and lookup[(200, k)]["acc"]],
        "remaining_loop": [k for k in keys if lookup[(200, k)]["long_loop"] and not lookup[(200, k)]["acc"]],
        "wrong_without_loop": [
            k
            for k in keys
            if not lookup[(200, k)]["long_loop"]
            and not lookup[(200, k)]["acc"]
            and lookup[(200, k)]["pred"] is not None
            and prompts[k]["dataset"] == "AIME25"
        ],
    }
    examples = []
    for category, candidates in categories.items():
        if not candidates:
            continue
        key = candidates[0]
        examples.append(
            {
                "category": category,
                "prompt_id": key,
                "prompt": prompts[key],
                "before": {**lookup[(0, key)], "text": texts[(0, key)]},
                "after": {**lookup[(200, key)], "text": texts[(200, key)]},
            }
        )
    with gzip.open(data / "sample-examples.json.gz", "wt") as stream:
        json.dump(examples, stream, ensure_ascii=False)
    print("PANEL", result, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--prepare-run", type=Path)
    parser.add_argument("--panel", action="store_true")
    args = parser.parse_args()
    if args.prepare_run:
        freeze_responses(args.prepare_run, args.data)
    analyze(args.data)
    if args.panel:
        analyze_panel(args.data)


if __name__ == "__main__":
    main()
