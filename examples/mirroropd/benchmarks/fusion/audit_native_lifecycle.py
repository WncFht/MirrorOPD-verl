# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Audit applied updates, source identity, validation, checkpoint and GPU scope."""

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


def inspect_run(run, outers, alphas, binding, student, commit, save=None):
    errors, witnesses = [], []

    def read(relative, lines=False):
        path = run / relative
        raw = path.read_bytes()
        witnesses.append({"path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest()})
        return [json.loads(line) for line in raw.splitlines()] if lines else json.loads(raw)

    metrics = [row["data"] for row in read("metrics.jsonl", lines=True)]
    updates = [row for row in metrics if "optim/actor/loss" in row]
    if [row.get("step/optimizer") for row in updates] != outers:
        errors.append("Applied optimizer steps do not match the declared outers")
    for row in updates:
        if any(
            not math.isfinite(row.get(key, math.nan)) for key in ["optim/actor/loss", "optim/actor/grad_norm_preclip"]
        ):
            errors.append("Nonfinite optimizer update")
    summaries = []
    for outer, alpha in zip(outers, alphas, strict=True):
        plan = read(f"checkpoints/rollout_plans/step_{outer}.json")
        runtime = read(f"checkpoints/rollout_plans/step_{outer}_runtime.json")
        version = [outer - 1, outer - 1]
        if plan["actor_version"] != version or plan["resolved_control"] != alpha:
            errors.append(f"Outer {outer}: actor coordinate or alpha mismatch")
        identity = plan["student_binding"]
        if identity["kind"] != ("current" if binding == "current" else "checkpoint"):
            errors.append(f"Outer {outer}: student binding changed")
        if Path(identity["path"]).resolve() != Path(student).resolve():
            errors.append(f"Outer {outer}: original student checkpoint changed")
        keys = [(request["uid"], request["response_index"]) for request in plan["requests"]]
        expected = [(uid, index) for uid, indices in plan["expected_sessions"] for index in indices]
        if len(keys) != 2 or len(set(keys)) != 2 or sorted(keys) != sorted(expected):
            errors.append(f"Outer {outer}: request identity mismatch")
        if runtime["metrics"].get("rollout/policy/requests") != 2:
            errors.append(f"Outer {outer}: incomplete generated batch")
        ranks = runtime["ranks"]
        expected_loaded = version if binding == "current" else ["checkpoint", str(Path(student).resolve())]
        if len(ranks) != 1 or ranks[0]["version"] != expected_loaded:
            errors.append(f"Outer {outer}: receiver version mismatch")
        validation = read(f"validation/records/{outer}.jsonl", lines=True)
        sources = Counter(row["data_source"] for row in validation)
        if len(validation) != 10 or len(sources) != 5 or any(count != 2 for count in sources.values()):
            errors.append(f"Outer {outer}: incomplete five-source validation")
        if len({(row["prompt_id"], row["sample_index"]) for row in validation}) != 10:
            errors.append(f"Outer {outer}: duplicate validation identity")
        if any(row["status"] != "ok" or not math.isfinite(row["score"]) for row in validation):
            errors.append(f"Outer {outer}: invalid validation result")
        summaries.append(
            {
                "outer": outer,
                "alpha": alpha,
                "receiver_version": ranks[0]["version"],
                "validation_sources": dict(sources),
            }
        )
    if save is not None:
        state = read(f"checkpoints/global_step_{save}/trainer_state.json")
        if state["global_step"] != save or state["actor_optimizer_step"] != save:
            errors.append("Saved trainer state does not match applied updates")
        for kind in ["model", "optim", "extra_state"]:
            path = run / f"checkpoints/global_step_{save}/actor/{kind}_world_size_1_rank_0.pt"
            if not path.is_file() or path.stat().st_size == 0:
                errors.append(f"Missing checkpoint shard {path.name}")
    source_commit = (run / "source_commit.txt").read_text().strip()
    if not source_commit.startswith(commit):
        errors.append("Source commit mismatch")
    resources = read("resources.jsonl", lines=True)
    samples = [row for row in resources if row["kind"] == "sample"]
    if resources[0].get("physical_gpus") != [7] or resources[-1]["kind"] != "end":
        errors.append("Resource interval is incomplete or includes other GPUs")
    gpu_rows = [gpu for row in samples for gpu in row["gpus"]]
    if any(gpu["index"] != 7 for gpu in gpu_rows):
        errors.append("Unexpected GPU in resource samples")
    return {
        "scope": "Single-GPU short lifecycle; not a long-cap performance or task-accuracy acceptance",
        "passed": not errors,
        "errors": errors,
        "source_commit": source_commit,
        "outers": summaries,
        "applied_updates": [row["step/optimizer"] for row in updates],
        "resources": {
            "physical_gpus": [7],
            "wall_seconds": resources[-1].get("wall_seconds"),
            "nvml_peak_bytes": max(row["memory_used"] for row in gpu_rows),
            "cpu_rss_sum_peak_bytes": max(row["process_tree_rss_sum"] for row in samples),
        },
        "witnesses": witnesses,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--outers", nargs="+", type=int, required=True)
    parser.add_argument("--alphas", nargs="+", type=float, required=True)
    parser.add_argument("--binding", choices=["current", "initial"], required=True)
    parser.add_argument("--student", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--save", type=int)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = inspect_run(args.run, args.outers, args.alphas, args.binding, args.student, args.commit, args.save)
    with args.out.open("x") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps({key: report[key] for key in ["passed", "errors", "applied_updates"]}))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
