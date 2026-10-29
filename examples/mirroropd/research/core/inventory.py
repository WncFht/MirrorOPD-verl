#!/usr/bin/env python3
# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Build the current experiment register directly from decoded logs and a cloud index.

Only formal Math/Coding/IF runs enter the register. Binary decoding is delegated
to wandb-logs before reading its cache; this does not contact W&B, alter report
snapshots, or infer an evaluation at the last update.
"""

import json
import re

from examples.mirroropd.research.core.io import read


def select(config, key):
    for part in key.split("."):
        if not isinstance(config, dict):
            return None
        config = config.get(part)
    return config


def is_full_kl(config):
    mode = select(config, "distillation.distillation_loss.loss_mode")
    return mode in ("full_reverse_kl", "full_forward_kl") or (
        mode == "mirror"
        and select(config, "distillation.distillation_loss.mirror.objective") == "mdpo"
        and select(config, "distillation.distillation_loss.mirror.estimator") == "full_vocab"
        and select(config, "distillation.distillation_loss.mirror.eta") == 1
    )


def method_label(config):
    if select(config, "distillation.distillation_loss.loss_mode") == "full_forward_kl":
        return "Full-Forward-KL"
    if is_full_kl(config):
        return "Full-KL"
    if select(config, "distillation.distillation_loss.loss_mode") == "mirror":
        m = select(config, "distillation.distillation_loss.mirror") or {}
        return f"{str(m.get('objective')).upper()}/{m.get('estimator')} (κ={m.get('kappa')}, η={m.get('eta')})"
    return "/".join(
        str(select(config, "distillation.distillation_loss." + key)) for key in ("advantage_source", "policy_loss_mode")
    )


def family(name):
    if re.match(r"math-opd-E(48|49)-", name):
        return "polaris-transfer"
    if re.match(r"math-opd-E(44|45|46|47)-", name):
        return "mirror-comparison"
    if name.startswith("math-opd-custom-") or re.match(
        r"math-opd-E\d+-(mdpo|bpo|sampled|full_kl|full_forward_kl)-", name
    ):
        return "method-overrides"
    match = re.match(r"math-opd-E(\d+)-", name)
    if match:
        number = int(match[1])
        return (
            "sampled-ablation"
            if number <= 12
            else "full-kl-comparison"
            if number <= 17
            else "equal-update-budget"
            if number <= 20
            else "failure-reproduction"
        )
    if name.startswith(("math-full-", "code-full-", "if-full-")):
        return "other-runs"
    return None


CONFIG_FIELDS = {
    "student": "actor_rollout_ref.model.path",
    "teacher": "distillation.teacher_models.teacher_model.model_path",
    "train_files": "data.train_files",
    "validation_files": "data.val_files",
    "seed": "data.seed",
    "loss_mode": "distillation.distillation_loss.loss_mode",
    "advantage_source": "distillation.distillation_loss.advantage_source",
    "policy_loss_mode": "distillation.distillation_loss.policy_loss_mode",
    "backend": "distillation.distillation_loss.full_vocab.backend",
    "lr": "actor_rollout_ref.actor.optim.lr",
    "reuse_epochs": "actor_rollout_ref.actor.ppo_epochs",
    "batch_size": "data.train_batch_size",
    "mini_batch_size": "actor_rollout_ref.actor.ppo_mini_batch_size",
    "rollout_n": "actor_rollout_ref.rollout.n",
    "planned_outer": "trainer.total_training_steps",
    "prompt_cap": "data.max_prompt_length",
    "response_cap": "data.max_response_length",
    "eval_n": "actor_rollout_ref.rollout.val_kwargs.n",
    "eval_response_cap": "actor_rollout_ref.rollout.val_kwargs.max_tokens",
    "eval_temperature": "actor_rollout_ref.rollout.val_kwargs.temperature",
    "eval_top_p": "actor_rollout_ref.rollout.val_kwargs.top_p",
    "eval_freq": "trainer.test_freq",
    "eval_before_train": "trainer.val_before_train",
    "metrics_format": "trainer.metrics_format",
    "execution_mode": "distillation.execution_mode",
    "gpus": "trainer.n_gpus_per_node",
}
DATASETS = {"math500": "MATH-500", "amc23": "AMC23", "aime25": "AIME25", "livecodebench": "livecodebench"}


def inspect(directory, cloud):
    meta = read(directory / "run.json")
    config = read(directory / "config.json")
    identity = meta["run"]
    run_id = identity["run_id"]
    name = identity.get("display_name", cloud.get("name", ""))
    result = {
        "run_id": run_id,
        "name": name,
        "family": family(name),
        "method": method_label(config),
        "cloud_state": cloud.get("state"),
        "local_status": meta["status"],
        "exit_code": meta.get("exit_code"),
        "start_time_utc": identity.get("start_time"),
        "history_rows": meta["history_rows"],
        "source_directory": directory.name,
        "recorded_git": identity.get("git"),
        "max_outer": None,
        "completed_outer": None,
        "optimizer_updates": None,
        "applied_update_records": None,
        "skipped_update_records": None,
        "last_eval_outer": None,
        "last_eval_optimizer": None,
        **{key: select(config, value) for key, value in CONFIG_FIELDS.items()},
        **{f"{key}_mean_pct": None for key in DATASETS},
    }
    updates = {}
    outer = optimizer = None
    with (directory / "history.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            outer = row.get("step/outer", row.get("training/global_step", outer))
            optimizer = row.get("step/optimizer", row.get("training/actor_optimizer_step", optimizer))
            if outer is not None:
                result["max_outer"] = max(result["max_outer"] or 0, outer)
            if optimizer is not None:
                result["optimizer_updates"] = max(result["optimizer_updates"] or 0, optimizer)
            if "perf/timing/outer_step_seconds" in row or "timing_s/step" in row:
                result["completed_outer"] = outer
            if "optim/actor/update_applied" in row:
                update = row["step/update"]
                signature = (row["optim/actor/update_applied"], row.get("optim/actor/update_skipped"))
                if update in updates and updates[update] != signature:
                    raise ValueError(f"Conflicting update {run_id}/{update}")
                updates[update] = signature
            metrics = {}
            for slug, old in DATASETS.items():
                keys = [f"eval/{slug}/acc/mean", f"eval/{slug}/acc_mean", f"val-core/{old}/acc/mean@16"]
                for key in keys:
                    if key in row:
                        metrics[slug] = {"key": key, "value": row[key]}
                        result[f"{slug}_mean_pct"] = row[key] * 100
                        break
            if metrics:
                result["last_eval_outer"] = outer
                result["last_eval_optimizer"] = optimizer
    if updates:
        result["applied_update_records"] = sum(v[0] for v in updates.values())
        result["skipped_update_records"] = sum(v[1] for v in updates.values() if v[1] is not None)
    return result
