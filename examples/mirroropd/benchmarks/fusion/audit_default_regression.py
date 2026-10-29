# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Pool the original three and the predeclared six additional regression batches."""

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

import yaml


def audit(original, additional, exit_code, baseline_identity_file):
    errors, witnesses, records = [], [], {}
    reference_behavior = reference_config = baseline_identity = None
    paired_behavior = {}
    if exit_code != 0:
        errors.append(f"Additional run controller exited with {exit_code}")

    def read(path, kind="jsonl"):
        raw = path.read_bytes()
        witnesses.append({"path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest()})
        if kind == "yaml":
            return yaml.safe_load(raw)
        if kind == "text":
            return raw.decode().strip()
        if kind == "json":
            return json.loads(raw)
        return [json.loads(line) for line in raw.splitlines()]

    identity = read(baseline_identity_file, "json")
    if not identity.get("passed") or identity.get("errors") or identity.get("checked_tracked_blobs", 0) < 1:
        errors.append("Baseline archive identity has not been verified against its declared source and patch")

    def setting(config, path):
        for key in path.split("."):
            config = config[key]
        return config

    paths = [
        "data.train_batch_size",
        "data.shuffle",
        "data.max_prompt_length",
        "data.max_response_length",
        "actor_rollout_ref.model.path",
        "trainer.n_gpus_per_node",
        "trainer.nnodes",
    ] + [
        "actor_rollout_ref.rollout." + key
        for key in (
            "name",
            "tensor_model_parallel_size",
            "temperature",
            "top_p",
            "top_k",
            "seed",
            "n",
            "full_determinism",
            "stop_token_ids",
            "enforce_eager",
            "engine_kwargs",
            "max_model_len",
            "max_num_seqs",
            "max_num_batched_tokens",
            "gpu_memory_utilization",
        )
    ]
    for variant in ("baseline", "current"):
        records[variant] = []
        for group, root, count in (("original", original, 4), ("additional", additional, 7)):
            run = root / variant
            config = read(run / "resolved_config.yaml", "yaml")
            source_commit = read(run / "source_commit.txt", "text")
            patch = read(run / "source_changes.patch", "text")
            if variant == "baseline":
                identity = (source_commit, patch)
                if baseline_identity is None:
                    baseline_identity = identity
                if identity != baseline_identity:
                    errors.append("Frozen baseline source or isolation patch changed")
            selected = {path: setting(config, path) for path in paths}
            if reference_config is None:
                reference_config = selected
            if selected != reference_config:
                errors.append(f"{variant}/{group}: frozen generation settings differ")
            rows = [row["data"] for row in read(run / "metrics.jsonl")]
            if [row.get("step/outer") for row in rows] != list(range(1, count + 1)):
                errors.append(f"{variant}/{group}: incomplete or duplicate batches")
            if any(row.get("step/optimizer") != 0 or row.get("step/update") != 0 for row in rows):
                errors.append(f"{variant}/{group}: unexpected training update")
            generation_files = sorted((run / "generations").glob("*.jsonl"))
            if len(generation_files) != count:
                errors.append(f"{variant}/{group}: wrong generation batch count")
            for step in range(1, count + 1):
                generated = read(run / f"generations/{step}.jsonl")
                behavior = sorted(
                    json.dumps({k: r[k] for k in ("input", "output", "gts", "score")}, sort_keys=True)
                    for r in generated
                )
                fixed_outputs = sorted(
                    json.dumps({k: r[k] for k in ("input", "output", "gts")}, sort_keys=True) for r in generated
                )
                if reference_behavior is None:
                    reference_behavior = fixed_outputs
                if len(behavior) != 64 or fixed_outputs != reference_behavior:
                    errors.append(f"{variant}/{group}/{step}: frozen input/output/answer changed")
                if variant == "baseline":
                    paired_behavior[group, step] = behavior
                elif behavior != paired_behavior[group, step]:
                    errors.append(f"{variant}/{group}/{step}: paired input/output/answer/score changed")
            resources = read(run / "resources.jsonl")
            if resources[-1].get("kind") != "end" or resources[0].get("physical_gpus") not in ([6], [7]):
                errors.append(f"{variant}/{group}: incomplete or unexpected resource interval")
            timing_keys = (
                "rollout_batch",
                "rollout_pipeline",
                "outer_step",
                "teacher_transfer",
                "teacher_forward",
                "update_weights",
            )
            times = {key: [row.get(f"perf/timing/{key}_seconds") for row in rows[1:]] for key in timing_keys}
            for key, values in times.items():
                if any(not isinstance(v, int | float) or not math.isfinite(v) or v < 0 for v in values):
                    errors.append(f"{variant}/{group}: invalid {key} measurement")
                if key == "rollout_batch" and any(v == 0 for v in values):
                    errors.append(f"{variant}/{group}: zero generation interval")
            if any(row.get("rollout/response_tokens") != 45059 for row in rows):
                errors.append(f"{variant}/{group}: changed frozen response token count")
            records[variant].append(
                {
                    "group": group,
                    "source_commit": source_commit,
                    "measured_batches": len(rows) - 1,
                    "seconds": times,
                }
            )
    pooled = {
        variant: [value for group in groups for value in group["seconds"]["rollout_batch"]]
        for variant, groups in records.items()
    }
    if any(len(values) != 9 for values in pooled.values()):
        errors.append("Both variants require exactly the original three plus six additional measured batches")
    relative = None
    if not errors:
        relative = statistics.median(pooled["current"]) / statistics.median(pooled["baseline"]) - 1
    passed = not errors and relative is not None and relative <= 0.05
    return {
        "passed": passed,
        "behavior_and_completeness_passed": not errors,
        "errors": errors,
        "relative_median_change": relative,
        "limit": 0.05,
        "records": records,
        "pooled_seconds": pooled,
        "additional_controller_exit_code": exit_code,
        "witnesses": witnesses,
        "gate_scope": (
            "Original rollout_batch timer: dispatch to final generated token. "
            "Other complete phase costs reported separately."
        ),
        "policy": "Keep every original/additional measured sample; exclude only each process's declared warmup",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--additional", type=Path, required=True)
    parser.add_argument("--additional-exit-code", type=int, required=True)
    parser.add_argument("--baseline-identity", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.original, args.additional, args.additional_exit_code, args.baseline_identity)
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({key: result[key] for key in ("passed", "errors", "relative_median_change")}))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
