# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Validate complete coordinated topology batches and report global wall time."""

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


def summarize(root):
    config = json.loads((root / "config.json").read_text())
    prompts_raw = (root / "prompts.json").read_bytes()
    prompts = json.loads(prompts_raw)
    if config["tp"] * config["replicas"] != 2 or len(prompts) != config["batch"] or config["repeats"] < 1:
        raise ValueError("Invalid protected topology or workload")
    records = json.loads((root / "results.json").read_text())["batches"]
    if [row["repeat"] for row in records] != list(range(-1, config["repeats"])):
        raise ValueError("Incomplete or duplicate global batches")
    previous = [0] * config["replicas"]
    batches = []
    for record in records:
        if len(record["replicas"]) != config["replicas"]:
            raise ValueError("Missing replica")
        if not math.isfinite(record["seconds"]) or record["seconds"] <= 0:
            raise ValueError("Invalid global batch interval")
        release = json.loads((root / f"go-{record['repeat']}.json").read_text())["released_monotonic"]
        if not math.isfinite(release) or any(
            not math.isfinite(r["started_monotonic"])
            or not math.isfinite(r["ended_monotonic"])
            or not release <= r["started_monotonic"] <= r["ended_monotonic"]
            for r in record["replicas"]
        ):
            raise ValueError("Invalid replica timing")
        engine_end = max(r["ended_monotonic"] for r in record["replicas"])
        collected = record["collected_monotonic"]
        if (
            not math.isfinite(collected)
            or collected < engine_end
            or record["seconds"] != collected - release
            or record["engine_completion_seconds"] != engine_end - release
            or record["result_collection_seconds"] != collected - engine_end
        ):
            raise ValueError("Global wall time does not cover the actual release/completion interval")
        responses, preemptions = [], []
        for rank, replica in enumerate(record["replicas"]):
            if replica["replica"] != rank:
                raise ValueError("Replica identity differs")
            expected = list(range(rank, config["batch"], config["replicas"]))
            if [r["global_index"] for r in replica["responses"]] != expected:
                raise ValueError("Replica request routing differs")
            for response in replica["responses"]:
                index = response["global_index"]
                if response["seed"] != 42 + index or response["prompt_ids"] != prompts[index]["prompt_token_ids"]:
                    raise ValueError("Global input or random stream changed")
                tokens, probs = response["token_ids"], response["log_probs"]
                if not 0 < len(tokens) <= config["cap"] or len(tokens) != len(probs):
                    raise ValueError("Incomplete response")
                if any(not math.isfinite(value) for value in probs):
                    raise ValueError("Nonfinite behavior probability")
                reason = response["finish_reason"]
                if reason not in {"stop", "length"} or (reason == "length" and len(tokens) != config["cap"]):
                    raise ValueError("Invalid termination")
            responses.extend(replica["responses"])
            counters = [
                m["value"] for m in replica["backend_metrics_cumulative"] if m["name"] == "vllm:num_preemptions"
            ]
            if len(counters) != 1 or not math.isfinite(counters[0]) or counters[0] < previous[rank]:
                raise ValueError("Missing or regressing preemption counter")
            preemptions.append(counters[0] - previous[rank])
            previous[rank] = counters[0]
        if sorted(r["global_index"] for r in responses) != list(range(config["batch"])):
            raise ValueError("Incomplete global identity coverage")
        lengths = sorted(len(r["token_ids"]) for r in responses)
        tokens = sum(lengths)
        batches.append(
            {
                "repeat": record["repeat"],
                "seconds": record["seconds"],
                "engine_completion_seconds": record["engine_completion_seconds"],
                "result_collection_seconds": record["result_collection_seconds"],
                "effective_tokens": tokens,
                "tokens_per_second": tokens / record["seconds"],
                "gpu_hours_per_million_tokens": record["seconds"] * 2 / 3600 * 1e6 / tokens,
                "launch_skew_seconds": record["launch_skew_seconds"],
                "lengths": lengths,
                "eos": sum(r["finish_reason"] == "stop" for r in responses),
                "cap_hits": sum(r["finish_reason"] == "length" for r in responses),
                "preemptions": sum(preemptions),
            }
        )
    for rank in range(config["replicas"]):
        audit = json.loads((root / f"replica-{rank}-audit.json").read_text())
        if len(audit) != config["tp"] or {r["tp_rank"] for r in audit} != set(range(config["tp"])):
            raise ValueError("Wrong native TP topology")
        if any(r["tensor_parallel_size"] != config["tp"] or r["alpha"] != config["alpha"] for r in audit):
            raise ValueError("Wrong native execution settings")
        if any(r["student_path"] != config["student"] or r["teacher_path"] != config["teacher"] for r in audit):
            raise ValueError("Native model binding differs from the declared pair")
    resources = [json.loads(line) for line in (root / "resources.jsonl").read_text().splitlines()]
    if resources[0].get("physical_gpus") != [6, 7] or resources[-1]["kind"] != "end":
        raise ValueError("Incomplete protected two-GPU resource interval")
    samples = [r for r in resources if r["kind"] == "sample"]
    if any({g["index"] for g in r["gpus"]} != {6, 7} for r in samples):
        raise ValueError("Resource samples differ from the protected allocation")
    measured = batches[1:]
    return {
        "scope": "Frozen-checkpoint generation; not training/refit acceptance. One full global warmup excluded.",
        "config": config,
        "source": json.loads((root / "source.json").read_text()),
        "prompt_sha256": hashlib.sha256(prompts_raw).hexdigest(),
        "batches": batches,
        "formal_generation_complete": config["batch"] == 64 and config["cap"] == 7168 and len(measured) >= 3,
        "median_seconds": statistics.median(r["seconds"] for r in measured),
        "median_tokens_per_second": statistics.median(r["tokens_per_second"] for r in measured),
        "nvml_peak_bytes_by_gpu": {
            str(gpu): max(g["memory_used"] for r in samples for g in r["gpus"] if g["index"] == gpu) for gpu in [6, 7]
        },
        "cpu_rss_sum_peak_bytes": max(r["process_tree_rss_sum"] for r in samples),
        "resource_scope": "One coordinator process tree; do not sum replica RSS peaks",
        "process_wall_seconds": resources[-1]["wall_seconds"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with args.out.open("x") as stream:
        json.dump(summarize(args.run), stream, indent=2)


if __name__ == "__main__":
    main()
