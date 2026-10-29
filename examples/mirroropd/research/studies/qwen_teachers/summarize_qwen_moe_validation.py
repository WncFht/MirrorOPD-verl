# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Audit complete MoE reference runs and the paired AIME25 template comparison."""

import argparse
import csv
import gzip
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from examples.mirroropd.research.core.io import dump


def read_run(directory):
    protocol = json.loads((directory / "protocol.json").read_text())
    metrics = json.loads((directory / "metrics.json").read_text())
    assert json.loads((directory / "exit.json").read_text())["exit_code"] == 0
    rows = []
    with (directory / "records.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            assert row["status"] == "ok"
            assert 0 <= row["sample_index"] < 16
            ids = row["response_ids"]
            assert len(ids) == row["response_length"] <= protocol["response_cap"]
            assert not any(t in protocol["stop_ids"] for t in ids[:-1])
            if row["finish_reason"] == "stop":
                assert ids[-1] in protocol["stop_ids"]
            rows.append(
                {
                    k: row[k]
                    for k in [
                        "data_source",
                        "prompt_id",
                        "sample_index",
                        "seed",
                        "input",
                        "score",
                        "format_score",
                        "response_length",
                        "finish_reason",
                        "repeat8",
                        "attempts",
                    ]
                }
                | {
                    "last_token": ids[-1],
                    "generated_think_marker": any(t in (151667, 151668) for t in ids),
                    "generated_role_marker": 151644 in ids,
                }
            )
    assert len(rows) == protocol["response_count"]
    keys = {(r["prompt_id"], r["sample_index"]) for r in rows}
    assert len(keys) == len(rows)
    counts = Counter(r["prompt_id"] for r in rows)
    assert len(counts) == protocol["prompt_count"] and set(counts.values()) == {16}
    return protocol, metrics, rows


def resources(directory):
    with (directory / "gpu-samples.jsonl").open() as stream:
        gpu = [json.loads(line) for line in stream]
    with (directory / "vllm-metrics.jsonl").open() as stream:
        telemetry = [json.loads(line) for line in stream]
    active = []
    counters = defaultdict(list)
    for row in telemetry:
        samples = row.get("samples", [])
        for item in samples:
            if "num_preemptions_total" in item["series"]:
                counters[item["series"]].append(item["value"])
        if sum(item["value"] for item in samples if "num_requests_running" in item["series"]) > 0:
            active.append(row)
    assert active
    begin, end = active[0]["time"], active[-1]["time"]
    gpu_values = []
    for row in gpu:
        if begin <= row["time"] <= end and row.get("returncode") == 0:
            for value in row["gpus"]:
                index, util, memory = [float(x.strip()) for x in value.split(",")]
                gpu_values.append((index, util, memory / 1024))
    kv = [s["value"] for r in active for s in r["samples"] if "kv_cache_usage_perc" in s["series"]]
    assert len(counters) == 4
    for values in counters.values():
        assert all(b >= a for a, b in zip(values, values[1:], strict=False))
    return dict(
        active_sample_start=begin,
        active_sample_end=end,
        gpu_observations=len(gpu_values),
        gpu_util_mean=float(np.mean([r[1] for r in gpu_values])),
        gpu_memory_peak_gib=max(r[2] for r in gpu_values),
        kv_peak_pct=max(kv) * 100,
        kv_p95_pct=float(np.percentile(kv, 95)) * 100,
        preemptions=sum(v[-1] - v[0] for v in counters.values()),
    )


def archive(source, destination):
    destination.mkdir(parents=True, exist_ok=True)
    for name in [
        "protocol.json",
        "prompts.json",
        "metrics.json",
        "exit.json",
        "wandb.json",
        "stop-smoke.json",
        "server-command.json",
        "backend-parity.json",
        "progress.json",
        "grader-cleanup.json",
    ]:
        if (source / name).exists():
            shutil.copy2(source / name, destination / name)
    for name in ["records.jsonl", "gpu-samples.jsonl", "vllm-metrics.jsonl", "driver.log", "server.log"]:
        with (source / name).open("rb") as inp, gzip.open(destination / (name + ".gz"), "wb", compresslevel=3) as out:
            shutil.copyfileobj(inp, out)
    shutil.copy2(source.parent / "recovered.json", destination / "recovered.json")
    shutil.copy2(source.parent / "source/evaluate_qwen_moe.py", destination / "evaluator.py")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--legacy", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    all_runs = {}
    tables = []
    for label, directory in [("native", args.native), ("legacy", args.legacy)]:
        protocol, metrics, rows = read_run(directory)
        all_runs[label] = (protocol, metrics, rows)
        resource = resources(directory)
        progress = json.loads((directory / "progress.json").read_text())
        resource["generation_and_grading_seconds"] = progress["elapsed_seconds"]
        resource["evaluation_including_finalization_seconds"] = metrics["perf/validation_seconds"]
        resource["effective_response_tokens_per_second"] = (
            sum(r["response_length"] for r in rows) / progress["elapsed_seconds"]
        )
        resource["completed_responses"] = len(rows)
        resource["retry_attempts"] = sum(r["attempts"] - 1 for r in rows)
        resource["generated_think_marker_responses"] = sum(r["generated_think_marker"] for r in rows)
        resource["generated_role_marker_responses"] = sum(r["generated_role_marker"] for r in rows)
        resource["stop_token_counts"] = dict(Counter(r["last_token"] for r in rows if r["finish_reason"] == "stop"))
        dump(args.out / f"{label}-resources.json", resource)
        for source in sorted({r["data_source"] for r in rows}):
            slug = {"MATH-500": "math500", "AMC23": "amc23"}.get(source, source.lower())
            prefix = f"eval/{slug}/"
            selected = [r for r in rows if r["data_source"] == source]
            mean = np.mean([r["score"] for r in selected])
            assert abs(mean - metrics[prefix + "score/mean"]) < 1e-12
            tables.append(
                dict(
                    run=label,
                    benchmark=source,
                    responses=len(selected),
                    accuracy_pct=100 * mean,
                    ci_low_pct=100 * metrics[prefix + "score/mean_ci_low"],
                    ci_high_pct=100 * metrics[prefix + "score/mean_ci_high"],
                    pass16_pct=100 * metrics[prefix + "acc/pass@16"],
                    majority16_pct=100 * metrics[prefix + "acc/majority@16"],
                    mean_length=metrics[prefix + "generation/response_length/mean"],
                    p95_length=metrics[prefix + "generation/response_length/p95"],
                    cap_pct=100 * np.mean([r["finish_reason"] == "length" for r in selected]),
                    format_pct=100 * np.mean([r["format_score"] for r in selected]),
                    repeat8_pct=100 * np.mean([r["repeat8"] for r in selected]),
                )
            )
        archive(directory, args.out / label)
    native = {(r["prompt_id"], r["sample_index"]): r for r in all_runs["native"][2] if r["data_source"] == "AIME25"}
    legacy = {(r["prompt_id"], r["sample_index"]): r for r in all_runs["legacy"][2]}
    assert native.keys() == legacy.keys() and len(native) == 480
    for key, a in native.items():
        b = legacy[key]
        assert a["seed"] == b["seed"]
        assert b["input"] == a["input"] + "<think>\n\n</think>\n\n"
    prompt_ids = sorted({k[0] for k in native})
    deltas = []
    for pid in prompt_ids:
        a = [native[(pid, i)] for i in range(16)]
        b = [legacy[(pid, i)] for i in range(16)]
        deltas.append(
            [
                np.mean([r["score"] for r in a]) - np.mean([r["score"] for r in b]),
                np.mean([r["response_length"] for r in a]) - np.mean([r["response_length"] for r in b]),
            ]
        )
    values = np.asarray(deltas)
    rng = np.random.default_rng(42)
    bootstrap = values[rng.integers(30, size=(10000, 30))].mean(axis=1)
    pair = dict(
        direction="native minus legacy",
        prompts=30,
        responses_per_arm=480,
        matched_seed_and_prompt=True,
        accuracy_delta_pp=float(values[:, 0].mean() * 100),
        accuracy_delta_ci_pp=(np.quantile(bootstrap[:, 0], [0.025, 0.975]) * 100).tolist(),
        length_delta_tokens=float(values[:, 1].mean()),
        length_delta_ci_tokens=np.quantile(bootstrap[:, 1], [0.025, 0.975]).tolist(),
        scope="Paired prompt bootstrap for one evaluation seed configuration; not training-seed uncertainty",
    )
    dump(args.out / "paired-template.json", pair)
    with (args.out / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(tables[0]))
        writer.writeheader()
        writer.writerows(tables)
    dump(args.out / "results.json", tables)
    print(json.dumps({"results": tables, "paired_template": pair}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
