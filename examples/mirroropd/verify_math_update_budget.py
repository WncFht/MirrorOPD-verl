#!/usr/bin/env python3
# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Check a completed synchronous Full-KL run using exported policy-v3 history.

This reads existing exports; it does not decode .wandb files or contact W&B.
Successful updates, complete outer records and validation coordinates must all
agree. A partial snapshot is rejected even if its final optimizer counter exists.
"""

import argparse
import gzip
import json
import math
from pathlib import Path

from omegaconf import OmegaConf


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(config, rows, expected_updates):
    def get(key, default=None):
        return OmegaConf.select(config, key, default=default)

    outer = get("trainer.total_training_steps")
    epochs = get("actor_rollout_ref.actor.ppo_epochs")
    batch = get("data.train_batch_size")
    test_freq = get("trainer.test_freq")
    require(
        all(type(v) is int and v > 0 for v in (outer, epochs, batch, test_freq, expected_updates)),
        "outer, epochs, batch, test_freq and expected_updates must be positive integers",
    )
    require(get("trainer.v1.trainer_mode") == "sync", "Expected synchronous training")
    require(get("trainer.v1.sync.parameter_sync_step", 1) == 1, "Expected one parameter sync step")
    require(get("trainer.total_epochs") == 1, "Expected one data epoch")
    require(not get("trainer.val_only", False), "val-only is not a training run")
    require(get("trainer.resume_mode") == "disable", "Expected a fresh run without resume")
    require(get("actor_rollout_ref.actor.ppo_mini_batch_size") == batch, "Expected batch == mini-batch")
    require(get("actor_rollout_ref.rollout.n") == 1, "Expected one training response per prompt")
    require(not get("algorithm.filter_groups.enable", False), "Group filtering changes prompt consumption")
    require(not get("trainer.v1.sampler.sync_refill_failed_groups", False), "Refill changes prompt consumption")
    require(get("distillation.enabled") is True, "Expected enabled distillation")
    require(get("distillation.distillation_loss.loss_mode") == "full_reverse_kl", "Expected Full-KL loss")
    require(outer * epochs == expected_updates, "Configured update budget does not match expected_updates")

    updates, completed, evaluations = {}, {}, {}
    for row in rows:
        step = row.get("step/outer")
        if "optim/actor/update_applied" in row:
            update = row.get("step/update")
            require(
                type(update) in (int, float) and math.isfinite(update) and update == int(update),
                "Missing or invalid update coordinate",
            )
            signature = (
                step,
                row.get("step/optimizer"),
                row["optim/actor/update_applied"],
                row.get("optim/actor/update_skipped"),
                row.get("distill/full_kl"),
            )
            require(update not in updates or updates[update] == signature, f"Conflicting update {update}")
            updates[update] = signature
        if "perf/timing/outer_step_seconds" in row:
            signature = (row.get("step/update"), row.get("step/optimizer"))
            require(step not in completed or completed[step] == signature, f"Conflicting outer {step}")
            completed[step] = signature
        if any(k.startswith("eval/") and k.endswith("/acc/mean") for k in row):
            signature = (row.get("step/update"), row.get("step/optimizer"))
            require(step not in evaluations or evaluations[step] == signature, f"Conflicting validation {step}")
            evaluations[step] = signature

    require(set(updates) == set(range(1, expected_updates + 1)), "Missing or extra per-update records")
    for update, (step, optimizer, applied, skipped, kl) in updates.items():
        require(step == (update - 1) // epochs + 1, f"Wrong outer coordinate at update {update}")
        require(optimizer == update and applied == 1 and skipped == 0, f"Skipped/unconfirmed update {update}")
        require(isinstance(kl, (int, float)) and math.isfinite(kl), f"Missing/nonfinite KL at update {update}")
    require(set(completed) == set(range(1, outer + 1)), "Missing or extra completed outer records")
    for step, coordinates in completed.items():
        require(coordinates == (step * epochs, step * epochs), f"Wrong final counters at outer {step}")
    expected_eval = set(range(test_freq, outer + 1, test_freq)) | {outer}
    if get("trainer.val_before_train", False):
        expected_eval.add(0)
    require(set(evaluations) == expected_eval, "Missing or extra validation checkpoints")
    for step, coordinates in evaluations.items():
        require(coordinates == (step * epochs, step * epochs), f"Wrong validation counters at outer {step}")
    return {
        "status": "pass",
        "outer_steps": outer,
        "ppo_epochs": epochs,
        "successful_updates": expected_updates,
        "skipped_updates": 0,
        "configured_prompt_records": outer * batch,
        "validation_optimizer_steps": [step * epochs for step in sorted(evaluations)],
        "scope": "Update/validation budget only; not a data-uniqueness, checkpoint-file or model-quality audit.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="resolved_config.yaml or exported config.json")
    parser.add_argument("--history", type=Path, required=True, help="Exported history.jsonl or history.jsonl.gz")
    parser.add_argument("--expected-updates", type=int, default=4096)
    args = parser.parse_args()
    opener = gzip.open if args.history.suffix == ".gz" else open
    with opener(args.history, "rt") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    try:
        result = verify(OmegaConf.load(args.config), rows, args.expected_updates)
    except ValueError as exc:
        raise SystemExit(f"Budget verification failed: {exc}") from exc
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
