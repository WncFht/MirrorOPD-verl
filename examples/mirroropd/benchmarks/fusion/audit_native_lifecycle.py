# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Audit applied updates, source identity, validation, checkpoint and GPU scope."""

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import yaml


def checkpoint_delta(before, after, world_size):
    """Read CPU shard storage without reconstructing distributed collectives."""
    import torch

    records = []
    for rank in range(world_size):
        filename = f"model_world_size_{world_size}_rank_{rank}.pt"
        left = torch.load(before / filename, map_location="cpu", weights_only=False, mmap=True)
        right = torch.load(after / filename, map_location="cpu", weights_only=False, mmap=True)
        if left.keys() != right.keys():
            raise ValueError("Checkpoint parameter keys changed across restore/update")
        changed, changed_bf16, maximum = 0, 0, 0.0
        for name in left:
            a = getattr(left[name], "_local_tensor", left[name])
            b = getattr(right[name], "_local_tensor", right[name])
            if a.shape != b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():
                raise ValueError(f"Invalid checkpoint shard tensor: rank {rank}, {name}")
            changed += int(torch.count_nonzero(a != b))
            changed_bf16 += int(torch.count_nonzero(a.bfloat16() != b.bfloat16()))
            maximum = max(maximum, float((a.float() - b.float()).abs().max()))
        if not changed:
            raise ValueError(f"Rank {rank} checkpoint weights did not change after the resumed update")
        records.append(
            {
                "rank": rank,
                "changed_state_elements": changed,
                "changed_bf16_state_elements": changed_bf16,
                "max_absolute_change": maximum,
            }
        )
    return {"scope": "Checkpoint state entries, including tied aliases", "ranks": records}


def inspect_run(run, outers, alphas, binding, student, commit, save=None, world_size=1, physical_gpus=(7,), requests=2):
    errors, witnesses = [], []
    config_path = run / "resolved_config.yaml"
    raw_config = config_path.read_bytes()
    config = yaml.safe_load(raw_config)
    witnesses.append({"path": str(config_path.resolve()), "sha256": hashlib.sha256(raw_config).hexdigest()})
    tp = int(config["rollout_policy"]["execution"]["tensor_parallel_size"])
    if config["trainer"]["n_gpus_per_node"] != world_size or world_size % tp:
        errors.append("Declared actor world size differs from the launch topology")

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
        identity = dict(plan["bindings"])["left"] if "bindings" in plan else plan["student_binding"]
        if identity["kind"] != ("current" if binding == "current" else "checkpoint"):
            errors.append(f"Outer {outer}: student binding changed")
        if Path(identity["path"]).resolve() != Path(student).resolve():
            errors.append(f"Outer {outer}: original student checkpoint changed")
        keys = [(request["uid"], request["response_index"]) for request in plan["requests"]]
        expected = [(uid, index) for uid, indices in plan["expected_sessions"] for index in indices]
        if len(keys) != requests or len(set(keys)) != requests or sorted(keys) != sorted(expected):
            errors.append(f"Outer {outer}: request identity mismatch")
        if runtime["metrics"].get("rollout/policy/requests") != requests:
            errors.append(f"Outer {outer}: incomplete generated batch")
        ranks = runtime["ranks"]
        expected_loaded = version if binding == "current" else ["checkpoint", str(Path(student).resolve())]
        uses_fusion = any(phase["decoder"] == "fusion" for phase in plan["phases"])
        expected_ranks = world_size if uses_fusion else 0
        if len(ranks) != expected_ranks or any(rank["version"] != expected_loaded for rank in ranks):
            errors.append(f"Outer {outer}: receiver version mismatch")
        if world_size > 1 and uses_fusion:
            expected_receivers = {(rank // tp, rank % tp, tp) for rank in range(world_size)}
            actual_receivers = {
                (rank.get("replica"), rank.get("tp_rank"), rank.get("tensor_parallel_size")) for rank in ranks
            }
            if actual_receivers != expected_receivers:
                errors.append(f"Outer {outer}: receiver replica/TP mapping differs from the launch topology")
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
                "receiver_version": ranks[0]["version"] if ranks else None,
                "receiver_topology": [
                    {key: rank.get(key) for key in ["replica", "tp_rank", "tensor_parallel_size"]} for rank in ranks
                ],
                "validation_sources": dict(sources),
            }
        )
    if save is not None:
        state = read(f"checkpoints/global_step_{save}/trainer_state.json")
        if state["global_step"] != save or state["actor_optimizer_step"] != save:
            errors.append("Saved trainer state does not match applied updates")
        for kind in ["model", "optim", "extra_state"]:
            for rank in range(world_size):
                path = run / f"checkpoints/global_step_{save}/actor/{kind}_world_size_{world_size}_rank_{rank}.pt"
                if not path.is_file() or path.stat().st_size == 0:
                    errors.append(f"Missing checkpoint shard {path.name}")
    source_commit = (run / "source_commit.txt").read_text().strip()
    if not source_commit.startswith(commit):
        errors.append("Source commit mismatch")
    resources = read("resources.jsonl", lines=True)
    samples = [row for row in resources if row["kind"] == "sample"]
    if resources[0].get("physical_gpus") != list(physical_gpus) or resources[-1]["kind"] != "end":
        errors.append("Resource interval is incomplete or includes other GPUs")
    gpu_rows = [gpu for row in samples for gpu in row["gpus"]]
    if any(gpu["index"] not in physical_gpus for gpu in gpu_rows):
        errors.append("Unexpected GPU in resource samples")
    return {
        "scope": "Local lifecycle artifacts; long-cap performance and task accuracy require separate acceptance",
        "passed": not errors,
        "errors": errors,
        "source_commit": source_commit,
        "outers": summaries,
        "applied_updates": [row["step/optimizer"] for row in updates],
        "timing_scope": "Last emitted value per outer/key; nested intervals must not be summed",
        "timing_by_outer": [
            {
                "outer": outer,
                "seconds": {
                    key: value
                    for row in metrics
                    if row.get("step/outer") == outer
                    for key, value in row.items()
                    if key.startswith("perf/timing/")
                },
            }
            for outer in outers
        ],
        "resources": {
            "physical_gpus": list(physical_gpus),
            "wall_seconds": resources[-1].get("wall_seconds"),
            "nvml_peak_bytes": max(row["memory_used"] for row in gpu_rows),
            "nvml_peak_bytes_by_gpu": {
                str(index): max(row["memory_used"] for row in gpu_rows if row["index"] == index)
                for index in physical_gpus
            },
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
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--physical-gpus", nargs="+", type=int, choices=[6, 7], default=[7])
    parser.add_argument("--requests", type=int, default=2)
    parser.add_argument("--previous-actor-checkpoint", type=Path, help="Compare saved actor shard values after resume")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = inspect_run(
        args.run,
        args.outers,
        args.alphas,
        args.binding,
        args.student,
        args.commit,
        args.save,
        args.world_size,
        args.physical_gpus,
        args.requests,
    )
    if args.previous_actor_checkpoint is not None:
        if args.save is None:
            raise ValueError("Checkpoint delta needs --save")
        try:
            report["checkpoint_delta"] = checkpoint_delta(
                args.previous_actor_checkpoint,
                args.run / f"checkpoints/global_step_{args.save}/actor",
                args.world_size,
            )
        except (ValueError, RuntimeError, OSError) as exc:
            report["errors"].append(f"Checkpoint delta failed: {exc}")
            report["passed"] = False
    with args.out.open("x") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps({key: report[key] for key in ["passed", "errors", "applied_updates"]}))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
