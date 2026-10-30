# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Check resolved study settings against E20 and the authorized forty-outer changes."""

import argparse
import json
from pathlib import Path

import yaml
from omegaconf import OmegaConf

from verl.utils import omega_conf_to_dataclass
from verl.workers.config import RolloutConfig
from verl.workers.config.distillation_spec import resolve_distillation_spec


def check(config, reference, pair, arm, outers=40, *, smoke=False, scientific=False):
    errors = []
    try:
        omega_conf_to_dataclass(OmegaConf.create(config["actor_rollout_ref"]["rollout"]), dataclass_type=RolloutConfig)
    except (ValueError, TypeError, AssertionError) as exc:
        errors.append(f"RolloutConfig construction failed: {exc}")

    def get(mapping, path):
        for key in path.split("."):
            mapping = mapping.get(key) if isinstance(mapping, dict) else None
        return mapping

    paths = [
        "data.train_batch_size",
        "data.max_prompt_length",
        "data.max_response_length",
        "data.shuffle",
        "data.seed",
        "actor_rollout_ref.actor.ppo_epochs",
        "actor_rollout_ref.actor.ppo_mini_batch_size",
        "actor_rollout_ref.actor.loss_agg_mode",
        "actor_rollout_ref.actor.shuffle",
        "actor_rollout_ref.actor.optim.optimizer",
        "actor_rollout_ref.actor.optim.lr",
        "actor_rollout_ref.actor.optim.betas",
        "actor_rollout_ref.actor.optim.weight_decay",
        "actor_rollout_ref.actor.optim.clip_grad",
        "actor_rollout_ref.actor.optim.lr_scheduler_type",
        "actor_rollout_ref.actor.optim.lr_warmup_steps",
        "actor_rollout_ref.actor.optim.lr_warmup_steps_ratio",
        "actor_rollout_ref.actor.fsdp_config.model_dtype",
        "actor_rollout_ref.actor.fsdp_config.dtype",
        "actor_rollout_ref.actor.fsdp_config.forward_precision",
        "actor_rollout_ref.rollout.n",
        "actor_rollout_ref.rollout.temperature",
        "actor_rollout_ref.rollout.top_p",
        "actor_rollout_ref.rollout.top_k",
        "distillation.execution_mode",
        "distillation.distillation_loss.use_task_rewards",
        "distillation.distillation_loss.use_policy_gradient",
        "distillation.distillation_loss.loss_max_clamp",
        "distillation.distillation_loss.log_prob_min_clamp",
        "algorithm.filter_groups.enable",
        "algorithm.use_kl_in_reward",
    ]
    smoke_budget = {
        "data.train_batch_size": 2,
        "data.max_prompt_length": 1024,
        "data.max_response_length": 32,
        "actor_rollout_ref.actor.ppo_mini_batch_size": 2,
    }
    for path in paths:
        expected_value = smoke_budget[path] if smoke and path in smoke_budget else get(reference, path)
        if get(config, path) != expected_value:
            errors.append(f"{path}: {get(config, path)!r} != expected {expected_value!r}")
    if smoke and outers != 2:
        errors.append("Wiring smoke must use two outers and is not formal acceptance")
    required = {
        "trainer.total_training_steps": outers,
        "trainer.val_before_train": False,
        "trainer.test_freq": -1,
        "trainer.save_freq": -1,
        "trainer.resume_mode": "disable",
        "actor_rollout_ref.rollout.max_model_len": 1056 if smoke else 11264,
        "actor_rollout_ref.rollout.stop_token_ids": [151643],
        "actor_rollout_ref.rollout.seed": 42,
        "actor_rollout_ref.actor.fsdp_config.seed": 42,
        "distillation.distillation_loss.full_vocab.termination_token_ids": [],
        "trainer.v1.sampler.sync_refill_failed_groups": False,
    }
    if outers == 40 and not smoke:
        required["actor_rollout_ref.rollout.engine_kwargs.vllm.attention_config.backend"] = "TRITON_ATTN"
    if scientific:
        from examples.mirroropd.research.core.artifacts import file_hash
        from examples.mirroropd.research.core.io import read
        from examples.mirroropd.research.studies.prefix_fusion.protocol import protocol_identity

        settings = config["trainer"].get("scientific_evaluation", {})
        root = Path(settings.get("root", ""))
        prepared = root / "prepared-03"
        required.update(
            {
                "trainer.save_freq": 1 if smoke else 20,
                "trainer.max_actor_ckpt_to_keep": 2 if smoke else 1,
                "trainer.readout_callback_class": (
                    "examples.mirroropd.research.studies.prefix_fusion.callback.ScientificReadout"
                ),
                "trainer.scientific_evaluation.pair": pair,
                "trainer.scientific_evaluation.smoke": smoke,
                "trainer.n_gpus_per_node": 1,
                "trainer.nnodes": 1,
                "trainer.v1.trainer_mode": "sync",
                "actor_rollout_ref.actor.fsdp_config.param_offload": True,
                "actor_rollout_ref.actor.fsdp_config.optimizer_offload": True,
                "data.val_files": [],
            }
        )
        if str(settings.get("physical_gpu")) not in {"6", "7"}:
            errors.append("Scientific evaluation must use allocated GPU 6 or 7")
        if [Path(p).resolve() for p in config["data"]["train_files"]] != [(prepared / "train.parquet").resolve()]:
            errors.append("Scientific training does not use the frozen disjoint pool")
        try:
            receipt = read(prepared / "complete.json")
            if receipt["inputs"]["protocol"] != protocol_identity():
                errors.append("Prepared protocol changed")
            if file_hash(prepared / "train.parquet") != receipt["artifacts"]["train.parquet"]:
                errors.append("Prepared training pool changed")
        except (OSError, KeyError, ValueError) as exc:
            errors.append(f"Missing or invalid scientific data: {exc}")
    for path, value in required.items():
        if get(config, path) != value:
            errors.append(f"{path}: expected {value!r}, got {get(config, path)!r}")
    spec = resolve_distillation_spec(config["distillation"]["distillation_loss"])
    if not spec.teacher_only_kl or spec.anchor_payload is not None:
        errors.append("Objective is not teacher-only full-vocabulary reverse KL")
    expected = {
        "S0": ("single", "student", "initial", 0.0),
        "S": ("trajectory_mixture", "student", "current", 0.0),
        "LM": ("token_mixture", "student", "current", 0.5),
        "PM": ("token_mixture", "student", "current", 0.5),
        "TM": ("trajectory_mixture", "student", "current", 0.5),
        "T": ("single", "teacher", "current", 1.0),
    }[arm]
    policy = config["rollout_policy"]
    actual = tuple(policy[key] for key in ["mode", "source", "student", "teacher_weight"])
    if policy.get("fusion_space", "logit") != ("probability" if arm == "PM" else "logit"):
        errors.append("Wrong token-fusion space")
    if actual != expected:
        errors.append(f"Wrong prefix policy: {actual!r}")
    student = Path(config["actor_rollout_ref"]["model"]["path"]).name
    teachers = config["distillation"]["teacher_models"]
    teacher = Path(next(iter(teachers.values()))["model_path"]).name
    if (
        student != "DeepSeek-R1-Distill-Qwen-1.5B"
        or teacher != {"J": "JustRL-DeepSeek-1.5B", "P": "Polaris-7B-Preview"}[pair]
        or len(teachers) != 1
    ):
        errors.append("Wrong learner/teacher checkpoint binding")
    if config["trainer"]["n_gpus_per_node"] not in {1, 2} or config["trainer"]["nnodes"] != 1:
        errors.append("Study must use one or two local GPUs")
    return {
        "passed": not errors,
        "errors": errors,
        "pair": pair,
        "arm": arm,
        "outers": outers,
        "expected_updates": outers * 16,
        "scope": "short wiring only" if smoke else "E20-derived learning settings",
        "full_vocab_execution": config["distillation"]["distillation_loss"]["full_vocab"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--pair", choices=["J", "P"], required=True)
    parser.add_argument("--arm", choices=["S0", "S", "LM", "PM", "TM", "T"], required=True)
    parser.add_argument("--outers", type=int, default=40)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = check(
        yaml.safe_load(args.config.read_text()),
        json.loads(args.reference.read_text()),
        args.pair,
        args.arm,
        args.outers,
    )
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
