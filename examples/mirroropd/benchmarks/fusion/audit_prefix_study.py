# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Verify completed prefix runs from per-update observations and source/version plans."""

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import yaml
from check_prefix_study_config import check


def audit(run, reference, pair, arm, mode, exit_code):
    witnesses = []

    def read(path, kind="json"):
        raw = path.read_bytes()
        witnesses.append({"path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest()})
        if kind == "yaml":
            return yaml.safe_load(raw)
        if kind == "jsonl":
            return [json.loads(line) for line in raw.splitlines()]
        return json.loads(raw)

    config = read(run / "resolved_config.yaml", "yaml")
    scientific = mode in {"science", "science-smoke"}
    smoke = mode in {"smoke", "science-smoke"}
    outers = 40 if mode in {"formal", "science"} else 2 if smoke else config["trainer"]["total_training_steps"]
    report = check(config, reference, pair, arm, outers, smoke=smoke, scientific=scientific)
    errors = report["errors"]
    if exit_code != 0:
        errors.append(f"Run controller reported exit code {exit_code}")
    if not config["trainer"]["experiment_name"].startswith(f"prefix-{mode}-{pair}-{arm}-"):
        errors.append("Run identity/mode differs from the requested audit scope")
    if mode == "formal" and config["global_profiler"]["tool"] is not None:
        errors.append("Profiled timings cannot count as a formal study run")
    metrics = [row["data"] for row in read(run / "metrics.jsonl", "jsonl")]
    updates = [row for row in metrics if "optim/actor/update_applied" in row]
    expected = list(range(1, outers * 16 + 1))
    if [row.get("step/optimizer") for row in updates] != expected or [
        row.get("step/update") for row in updates
    ] != expected:
        errors.append("Missing, duplicate, skipped or unordered optimizer updates")
    for i, row in enumerate(updates):
        if (row.get("step/outer"), row.get("rollout/data_reuse/update_index")) != (i // 16 + 1, i % 16 + 1):
            errors.append(f"Update {i + 1}: wrong outer or reuse coordinate")
        for key, value in [
            ("execution_known", 1),
            ("update_applied", 1),
            ("update_skipped", 0),
            ("skipped_updates", 0),
            ("loss_nonfinite", 0),
            ("grad_norm_preclip_nonfinite", 0),
        ]:
            if row.get("optim/actor/" + key) != value:
                errors.append(f"Update {i + 1}: invalid {key}")
        for key in ["loss", "distillation_loss", "grad_norm_preclip", "lr"]:
            value = row.get("optim/actor/" + key)
            if value is None or not math.isfinite(value):
                errors.append(f"Update {i + 1}: missing/nonfinite {key}")
    if any(any(key.startswith("eval/") for key in row) for row in metrics):
        errors.append("Unexpected validation metrics")
    if not scientific and (
        list((run / "validation").rglob("*.jsonl"))
        or list((run / "runtime").glob("global_step_*"))
        or list(run.rglob("model_world_size_*.pt"))
    ):
        errors.append("Unexpected validation records or model checkpoints")
    if scientific:
        from examples.mirroropd.research.studies.prefix_fusion.audit import audit_readouts

        errors.extend(audit_readouts(run, pair, smoke=smoke))
    batch = config["data"]["train_batch_size"]
    plans, costs = [], []
    expected_sources = {
        "S0": {"student": batch},
        "S": {"student": batch},
        "LM": {"fusion": batch},
        "PM": {"fusion": batch},
        "TM": {"student": batch // 2, "teacher": batch // 2},
        "T": {"teacher": batch},
    }[arm]
    for outer in range(1, outers + 1):
        plan = read(run / f"runtime/rollout_plans/step_{outer}.json")
        runtime = read(run / f"runtime/rollout_plans/step_{outer}_runtime.json")
        version = [outer - 1, (outer - 1) * 16]
        requests = plan["requests"]
        if plan["actor_version"] != version:
            errors.append(f"Outer {outer}: incorrect applied-update version")
        if Counter(row["source"] for row in requests) != expected_sources:
            errors.append(f"Outer {outer}: incorrect global source quota")
        identities = [(row["uid"], row["response_index"]) for row in requests]
        sessions = [(uid, index) for uid, indices in plan["expected_sessions"] for index in indices]
        if len(set(identities)) != batch or sorted(identities) != sorted(sessions):
            errors.append(f"Outer {outer}: incomplete request identities")
        if runtime["metrics"].get("rollout/policy/requests") != batch:
            errors.append(f"Outer {outer}: incomplete finished batch")
        binding = plan["student_binding"]
        if binding["kind"] != ("checkpoint" if arm == "S0" else "current"):
            errors.append(f"Outer {outer}: student binding changed")
        if Path(binding["path"]).resolve() != Path(config["actor_rollout_ref"]["model"]["path"]).resolve():
            errors.append(f"Outer {outer}: original learner identity changed")
        expected_space = "probability" if arm == "PM" else "logit"
        if plan.get("fusion_space", "logit") != expected_space:
            errors.append(f"Outer {outer}: wrong fusion space in source plan")
        if arm in {"LM", "PM"}:
            ranks = runtime["ranks"]
            if any(row.get("fusion_space", "logit") != expected_space for row in ranks):
                errors.append(f"Outer {outer}: receiver fusion space differs from plan")
            world = config["trainer"]["n_gpus_per_node"]
            tp = config["rollout_policy"]["execution"]["tensor_parallel_size"]
            actual = {(row.get("replica"), row.get("tp_rank"), row.get("tensor_parallel_size")) for row in ranks}
            if len(ranks) != world or actual != {(rank // tp, rank % tp, tp) for rank in range(world)}:
                errors.append(f"Outer {outer}: wrong receiver topology")
            if any(row["version"] != version for row in ranks) or dict(plan["sampling"]).get("logprobs") is not True:
                errors.append(f"Outer {outer}: invalid refit/behavior probability contract")
        merged = {k: v for row in metrics if row.get("step/outer") == outer for k, v in row.items()}
        plans.append({"outer": outer, "actor_version": version, "sources": expected_sources})
        costs.append(
            {
                "outer": outer,
                "response_tokens": merged.get("rollout/response_tokens"),
                "seconds": {k: v for k, v in merged.items() if k.startswith("perf/timing/")},
            }
        )
    resources = read(run / "resources.jsonl", "jsonl")
    allocated = resources[0].get("physical_gpus", [])
    if (
        not allocated
        or not set(allocated) <= {6, 7}
        or len(allocated) != config["trainer"]["n_gpus_per_node"]
        or resources[-1]["kind"] != "end"
    ):
        errors.append("Incomplete or unexpected GPU resource interval")
    gpu_rows = [gpu for row in resources if row["kind"] == "sample" for gpu in row["gpus"]]
    if any(gpu["index"] not in allocated for gpu in gpu_rows):
        errors.append("Unexpected GPU in resource samples")
    report.update(
        passed=not errors,
        formal_complete=mode in {"formal", "science"} and not errors,
        mode=mode,
        controller_exit_code=exit_code,
        source_commit=(run / "source_commit.txt").read_text().strip(),
        applied_updates=len(updates),
        plans=plans,
        costs=costs,
        cost_scope="Last value per outer/key; nested timing intervals must not be summed",
        update_curves=[
            {k: v for k, v in row.items() if k.startswith(("step/", "optim/actor/", "rollout/data_reuse/"))}
            for row in updates
        ],
        gpu_peak_bytes={
            str(gpu): max(row["memory_used"] for row in gpu_rows if row["index"] == gpu) for gpu in allocated
        },
        witnesses=witnesses,
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--pair", choices=["J", "P"], required=True)
    parser.add_argument("--arm", choices=["S0", "S", "LM", "PM", "TM", "T"], required=True)
    parser.add_argument("--mode", choices=["formal", "profile", "smoke", "science", "science-smoke"], required=True)
    parser.add_argument(
        "--exit-code", type=int, required=True, help="Observed subprocess result from the run controller"
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.run, json.loads(args.reference.read_text()), args.pair, args.arm, args.mode, args.exit_code)
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({k: result[k] for k in ["passed", "formal_complete", "applied_updates", "errors"]}))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
