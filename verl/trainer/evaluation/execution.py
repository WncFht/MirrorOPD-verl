# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""One thin boundary to the native trainer; no independent workers or model pool."""

from omegaconf import OmegaConf
from transformers import AutoConfig

from verl.trainer.distillation.distribution import DistributionSpec


def hidden_dtype_bytes(config):
    # FSDP forward precision, not checkpoint storage or rollout KV precision.
    if OmegaConf.select(config, "distillation.shared_teacher.forward_precision") == "bf16_fp32":
        return 4
    dtype = OmegaConf.select(config, "distillation.shared_teacher.mixed_precision.param_dtype", default="bf16")
    sizes = {"bf16": 2, "bfloat16": 2, "fp16": 2, "float16": 2, "fp32": 4, "float32": 4, "fp64": 8, "float64": 8}
    if dtype not in sizes:
        raise ValueError(f"Unknown teacher hidden precision: {dtype}")
    return sizes[dtype]


def numerical_profile(config):
    """Forward precision changes the scored model function; tiling stays execution metadata."""
    aliases = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32", "fp64": "float64"}
    roles = {}
    for role, prefix in (
        ("student", "actor_rollout_ref.actor.fsdp_config"),
        ("teacher", "distillation.shared_teacher"),
    ):
        dtype = OmegaConf.select(config, prefix + ".mixed_precision.param_dtype", default="bf16")
        roles[role] = {
            "param_dtype": aliases.get(dtype, dtype),
            "forward_precision": OmegaConf.select(config, prefix + ".forward_precision"),
        }
    return {"schema": 1, "roles": roles, "reduction": "stable-rkl-v1", "sample_sum_dtype": "float64"}


class NativeEvaluationExecution:
    def __init__(self, trainer):
        self.trainer = trainer

    def balance(self, meta):
        return self.trainer._balance_batch(meta, {}, logging_prefix="evaluation_balance", inference_only=True)

    def score_teacher(self, meta, timing):
        return self.trainer._compute_shared_teacher_log_probs(meta, timing)

    def distribution(self, *, objective=False):
        config = self.trainer.config
        full = config.distillation.distillation_loss.full_vocab
        trust_remote_code = config.actor_rollout_ref.model.get("trust_remote_code", False)
        student = AutoConfig.from_pretrained(
            config.actor_rollout_ref.model.path, local_files_only=True, trust_remote_code=trust_remote_code
        )
        teacher = AutoConfig.from_pretrained(
            next(iter(config.distillation.teacher_models.values())).model_path,
            local_files_only=True,
            trust_remote_code=trust_remote_code,
        )
        return DistributionSpec.parse(full.get("distribution", {})).compile(
            student.vocab_size, teacher.vocab_size, full.get("termination_token_ids", ()) if objective else ()
        )
