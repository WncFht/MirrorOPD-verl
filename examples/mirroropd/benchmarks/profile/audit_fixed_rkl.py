# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Audit native fixed-RKL acceptance artifacts and summarize observed work."""

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import yaml

from verl.trainer.evaluation.artifacts import ArtifactStore
from verl.trainer.evaluation.schedule import Schedule


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def mean_rkl(rows, *, question_mean=False):
    if question_mean:
        return sum(r["rkl_sum"] / r["tokens"] for r in rows) / len(rows)
    return sum(r["rkl_sum"] for r in rows) / sum(r["tokens"] for r in rows)


def audit(run):
    config = yaml.safe_load((run / "resolved_config.yaml").read_text())
    outers = config["trainer"]["total_training_steps"]
    batch = config["data"]["train_batch_size"]
    reuse = config["actor_rollout_ref"]["actor"]["ppo_epochs"]
    options = config["trainer"]["rkl_readout"]
    root = run / "readout"
    modern = (root / "manifest.json").exists()
    store = ArtifactStore(root) if modern else None
    if modern:
        bank_root = Path(store.manifest()["metadata"]["source_bank"])
        bank_store = ArtifactStore(bank_root)
        bank_store.manifest(complete=True)
        bank_path = bank_root / "manifest.json"
        bank = {"rows": bank_store.read_rows("trajectories.parquet")}
    else:
        bank_path = root / "initial-student-bank.json"
        bank = read(bank_path)
    require(len(bank["rows"]) == options["source_count"], "Wrong source held-out count")
    bank_hash = hashlib.sha256(bank_path.read_bytes()).hexdigest()
    held_ids = {r["source_id"] for r in bank["rows"]}
    require(len(held_ids) == len(bank["rows"]), "Duplicate held-out source IDs")
    updates = [json.loads(line)["data"] for line in (run / "metrics.jsonl").read_text().splitlines()]
    losses = [d for d in updates if "optim/actor/loss" in d]
    require(len(losses) == outers * reuse, "Missing optimizer update records")
    for i, row in enumerate(losses, 1):
        require(row["step/optimizer"] == i, "Optimizer counter is not contiguous")
        require(row["optim/actor/update_applied"] == 1, "Skipped optimizer update")
        require(math.isfinite(row["optim/actor/loss"]), "Nonfinite loss")
        require(math.isfinite(row["optim/actor/grad_norm_preclip"]), "Nonfinite gradient norm")
    trajectories, seen, readouts, audits, histories = {}, set(), [], [], {}
    prior_history, revisits, prior_history_outer = {}, [], None
    source_schedule = Schedule(at=(0,), every=options.get("source_every", options.get("every", 1)))
    history_schedule = Schedule(every=options.get("history_every", options.get("every", 1)))
    audit_requested = options.get("audit_state", False)
    for outer in range(outers + 1):
        if outer:
            rows = (
                store.read_rows(f"trajectories/outer-{outer:03d}.parquet")
                if modern
                else read(root / "trajectories" / f"outer-{outer:03d}.json")
            )
            ids = {r["source_id"] for r in rows}
            require(len(rows) == len(ids) == batch, "Wrong training batch count or duplicate IDs")
            require(not ids & (seen | held_ids), "Training repeats or held-out overlap")
            seen.update(ids)
            require(
                {(r["min_global_steps"], r["max_global_steps"]) for r in rows} == {(outer - 1, outer - 1)},
                "Training rollout did not use the expected current student version",
            )
            trajectories.update({r["id"]: r for r in rows})
            history = (
                store.resolve(store.read_rows(f"history/outer-{outer:03d}.parquet"))
                if modern
                else read(root / "history" / f"outer-{outer:03d}.json")
            )
            require(len(history) == options["history_size"], "Wrong history count")
            counts = Counter(r["history_batch"] for r in history)
            require(set(counts) == set(range(1, outer + 1)), "History omitted an observed batch")
            require(max(counts.values()) - min(counts.values()) <= 1, "Unbalanced history quotas")
            for row in history:
                original = trajectories[row["id"]]
                for key in ["prompt_ids", "response_ids", "response_mask"]:
                    require(row[key] == original[key], "History is not the actual training trajectory")
            histories[outer] = history
        if not source_schedule.due(outer, outers) and not history_schedule.due(outer, outers):
            continue
        if modern:
            event = ArtifactStore(root / "scores" / f"outer-{outer:03d}")
            scored = event.manifest(complete=True)["metadata"] | {"rows": event.read_rows("rkl.parquet")}
            scored["initial_bank_sha256"] = scored["source_bank_manifest_sha256"]
        else:
            scored = read(root / "scores" / f"outer-{outer:03d}.json")
        require(scored["optimizer_steps"] == outer * reuse, "Readout used wrong optimizer boundary")
        require(scored["initial_bank_sha256"] == bank_hash, "Initial student bank changed")
        expected = []
        if source_schedule.due(outer, outers):
            expected.extend(bank["rows"])
        if history_schedule.due(outer, outers):
            expected.extend(histories.get(outer, []))
        by_id = {r["id"]: r for r in expected}
        require(len(scored["rows"]) == len(by_id), "Wrong readout sample count")
        require({r["id"] for r in scored["rows"]} == set(by_id), "Readout sample identities differ")
        for row in scored["rows"]:
            require(row["tokens"] == sum(by_id[row["id"]]["response_mask"]), "Readout mask count differs")
            require(math.isfinite(row["rkl_sum"]), "Nonfinite RKL")
        state = scored["audit"]["ranks"] if modern else read(root / "audit" / f"outer-{outer:03d}.json")
        if audit_requested:
            require(
                bool(state) and all(x["unchanged"] and x.get("state_digest") for x in state),
                "Readout lacks a verified model/Adam audit",
            )
        audits.append(state)
        source = [r for r in scored["rows"] if r["bank"] == "source_heldout"]
        if source:
            readouts.append(
                dict(outer=outer, optimizer_steps=outer * reuse, source_rkl=mean_rkl(source, question_mean=modern))
            )
        current_history = {r["id"]: r for r in scored["rows"] if r["bank"] == "history"}
        shared = set(prior_history) & set(current_history)
        for group in sorted({current_history[key]["history_batch"] for key in shared}):
            keys = sorted(key for key in shared if current_history[key]["history_batch"] == group)
            before = mean_rkl([prior_history[key] for key in keys], question_mean=modern)
            after = mean_rkl([current_history[key] for key in keys], question_mean=modern)
            revisits.append(
                dict(
                    from_outer=prior_history_outer,
                    to_outer=outer,
                    history_batch=group,
                    rows=len(keys),
                    **{
                        f"before_{'question' if modern else 'token'}_mean": before,
                        f"after_{'question' if modern else 'token'}_mean": after,
                    },
                )
            )
        if current_history:
            prior_history = current_history
            prior_history_outer = outer
    if audit_requested:
        require(audits[0] != audits[-1], "No model/optimizer state change across training")
    outer_records = [d for d in updates if "perf/timing/outer_step_seconds" in d]
    require(len(outer_records) == outers, "Missing completed outer records")
    return dict(
        complete=True,
        outers=outers,
        updates=outers * reuse,
        unique_training_questions=len(seen),
        source_questions=len(held_ids),
        response_cap=config["data"]["max_response_length"],
        source_response_tokens=sum(len(r["response_ids"]) for r in bank["rows"]),
        training_response_tokens=sum(len(r["response_ids"]) for r in trajectories.values()),
        readouts=readouts,
        matched_history_revisits=revisits,
        aggregation="question_mean" if modern else "legacy_token_mean",
        source_rkl_reduction_percent=(
            100 * (1 - readouts[-1]["source_rkl"] / readouts[0]["source_rkl"])
            if readouts[0]["source_rkl"] > 0
            else None
        ),
        outer_timings=[
            {k: v for k, v in d.items() if k.startswith(("perf/timing/", "rkl_readout/", "eval/rkl/", "step/"))}
            for d in outer_records
        ],
        loss_windows=[
            dict(
                outer=o,
                first_forward=losses[(o - 1) * reuse]["optim/actor/loss"],
                last_forward=losses[o * reuse - 1]["optim/actor/loss"],
            )
            for o in range(1, outers + 1)
        ],
        final_history_batch_counts=dict(Counter(r["history_batch"] for r in histories[outers])),
        initial_bank_sha256=bank_hash,
        all_readouts_preserved_model_and_adam=True if audit_requested else None,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.run)
    (args.run / "acceptance-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
