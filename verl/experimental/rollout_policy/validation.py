# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Reject unsupported combinations before allocating workers or GPU memory."""

from omegaconf import OmegaConf, open_dict

from .models import load_model_metadata, requires_token_generator
from .plan import RolloutPolicyConfig


def validate_policy(config, *, policy=None, bindings=None):
    raw = OmegaConf.to_container(config.rollout_policy, resolve=True) if config.get("rollout_policy") else None
    policy = policy or RolloutPolicyConfig.parse(raw)
    if OmegaConf.select(config, "actor_rollout_ref.rollout.name") == "hf_fusion":
        raise ValueError(
            "Keep rollout.name=vllm and select rollout_policy.mode=token_mixture; hf_fusion is an internal target"
        )
    if policy.default_path:
        return policy
    required = {
        "trainer.v1.trainer_mode": "sync",
        "trainer.nnodes": 1,
        "actor_rollout_ref.hybrid_engine": True,
        "actor_rollout_ref.actor.strategy": "fsdp2",
        "actor_rollout_ref.rollout.name": "vllm",
        "actor_rollout_ref.rollout.tensor_model_parallel_size": 1,
        "actor_rollout_ref.rollout.data_parallel_size": 1,
        "actor_rollout_ref.rollout.pipeline_model_parallel_size": 1,
        "actor_rollout_ref.rollout.agent.default_agent_loop": "single_turn_agent",
        "distillation.execution_mode": "shared_sync",
    }
    for key, expected in required.items():
        if OmegaConf.select(config, key) != expected:
            raise ValueError(f"rollout_policy requires {key}={expected}")
    disabled = [
        "algorithm.filter_groups.enable",
        "trainer.v1.sampler.sync_refill_failed_groups",
        "actor_rollout_ref.rollout.multi_turn.enable",
        "actor_rollout_ref.rollout.topk_log_probs",
        "actor_rollout_ref.rollout.ignore_eos",
        "actor_rollout_ref.rollout.disaggregation.enabled",
        "actor_rollout_ref.rollout.mtp.enable",
        "actor_rollout_ref.model.mtp.enable",
        "actor_rollout_ref.rollout.agent.agent_loop_manager_class",
        "actor_rollout_ref.rollout.agent.agent_loop_config_path",
        "trainer.v1.sampler.custom_sampler.path",
        "actor_rollout_ref.rollout.agent.custom_async_server.path",
        "skip.rollout_tq.enable",
    ]
    for key in disabled:
        if OmegaConf.select(config, key, default=False):
            raise ValueError(f"rollout_policy does not support {key}")
    if (
        OmegaConf.select(
            config,
            "actor_rollout_ref.model.lora.rank",
            default=OmegaConf.select(config, "actor_rollout_ref.model.lora_rank", default=0),
        )
        > 0
    ):
        raise ValueError("rollout_policy supports full-parameter dense learners only")
    from verl.workers.config.distillation_spec import resolve_distillation_spec

    from .sampling import SamplingConfig

    if (
        not config.distillation.enabled
        or resolve_distillation_spec(config.distillation.distillation_loss).payload_kind != "hidden"
    ):
        raise ValueError("Local rollout policies currently require full-vocabulary distillation")
    rollout = config.actor_rollout_ref.rollout
    SamplingConfig(
        temperature=rollout.temperature,
        top_p=rollout.top_p,
        top_k=rollout.top_k,
        max_tokens=rollout.response_length,
        fusion_space=policy.fusion_space,
    )
    if rollout.top_k == 0:
        raise ValueError("Use top_k=-1 for the common vLLM/HF no-truncation contract")
    if rollout.get("repetition_penalty", 1.0) != 1.0:
        raise ValueError("Local rollout policies currently require repetition_penalty=1")
    logprob_mode = OmegaConf.select(rollout, "engine_kwargs.vllm.logprobs_mode", default=rollout.logprobs_mode)
    if rollout.calculate_log_probs and logprob_mode != "processed_logprobs":
        raise ValueError("Local rollout behavior probabilities require logprobs_mode=processed_logprobs")
    if policy.mode == "token_mixture" and rollout.dtype != "bfloat16":
        raise ValueError("Production token fusion currently requires BF16 inference weights")
    validate_resume_clock(config, policy.right_weight)

    from .identity import bind_policy
    from .sampling import validate_tokenizers

    student_path = config.actor_rollout_ref.model.path
    if bindings is None:
        bindings, _ = bind_policy(config, policy)
    fusion_needed = requires_token_generator(policy, bindings)
    if fusion_needed and policy.execution.backend == "vllm_fusion":
        validate_native_options(config)
    metadata = load_model_metadata([student_path, *(b.path for b in bindings.values())])
    model_configs = [item[0] for item in metadata.values()]
    learner_config, learner_tokenizer = metadata[student_path]
    if getattr(learner_config, "num_experts", 0) or getattr(learner_config, "vision_config", None):
        raise ValueError("rollout_policy supports dense text learners only")
    for model_config, tokenizer in metadata.values():
        validate_tokenizers(learner_tokenizer, tokenizer, learner_config.vocab_size, model_config.vocab_size)
    components = []
    if fusion_needed:
        first, second = bindings["left"], bindings["right"]
        if second.kind == "current":
            first, second = second, first
        components = [metadata[b.path][0] for b in (first, second)]
    model = config.actor_rollout_ref.model
    if model.get("override_config") or model.get("trust_remote_code", False):
        raise ValueError("Local rollout policies require native checkpoint configs without overrides or remote code")
    for key in ("tokenizer_path", "hf_config_path"):
        if model.get(key) not in (None, student_path):
            raise ValueError(f"Local rollout policies do not support a separate {key}")
    if fusion_needed and policy.execution.backend == "hf_fusion":
        import importlib.util

        import torch
        import transformers
        from packaging.version import Version

        if Version(transformers.__version__) < Version("5.12.0"):
            raise ValueError(
                "Production token fusion requires Transformers >=5.12 with the validated packed-expert API"
            )
        if type(OmegaConf.select(rollout, "engine_kwargs.hf_fusion.reference", default=False)) is not bool:
            raise ValueError("hf_fusion.reference must be a boolean diagnostic switch")
        if components[0].model_type != "qwen3" or any(
            model.model_type not in {"qwen3", "qwen3_moe"} for model in components[1:]
        ):
            raise ValueError(
                "Production token fusion currently supports a Qwen3 dense learner and Qwen3 dense/MoE teacher"
            )
        if any(getattr(model, "quantization_config", None) for model in model_configs):
            raise ValueError("Production token fusion requires unquantized checkpoints")
        if any(importlib.util.find_spec(module) is None for module in ("flash_attn", "triton")):
            raise ValueError("Production token fusion requires FlashAttention 2 and Triton")
        if not hasattr(torch.backends.cuda.matmul, "allow_bf16_reduced_precision_reduction_split_k"):
            raise ValueError("Production token fusion requires Torch >=2.10 with independent split-K control")
    if fusion_needed and policy.execution.backend == "vllm_fusion":
        from importlib.metadata import version

        from packaging.version import Version

        if Version(version("vllm")).base_version != "0.25.1":
            raise ValueError("vllm_fusion currently requires the pinned vLLM 0.25.1/V1 runtime")
        from .topology import FusionTopology

        FusionTopology(config.trainer.n_gpus_per_node, policy.execution.tensor_parallel_size)
        from verl.workers.rollout.vllm_fusion.compat import validate_component_families

        validate_component_families(*components)
        if any(getattr(model, "quantization_config", None) for model in model_configs):
            raise ValueError("vllm_fusion requires unquantized checkpoints")
    if policy.mode == "token_mixture" and config.actor_rollout_ref.rollout.checkpoint_engine.backend != "naive":
        raise ValueError("Token fusion currently requires naive same-topology weight transport")
    import os

    if os.environ.get("VLLM_USE_V2_MODEL_RUNNER", "").lower() in {"1", "true"}:
        raise ValueError("Local rollout policies require vLLM V1 for the common tokenizer-support processor")
    return policy


def policy_engine_config(rollout):
    """Build an independent engine config; validation never mutates caller state."""
    import copy

    from .vllm_support import TokenizerSupport

    rollout = copy.deepcopy(rollout)
    processor = TokenizerSupport.__module__ + ":" + TokenizerSupport.__name__
    with open_dict(rollout.engine_kwargs):
        if rollout.engine_kwargs.get("vllm") is None:
            rollout.engine_kwargs.vllm = {}
    with open_dict(rollout.engine_kwargs.vllm):
        processors = list(rollout.engine_kwargs.vllm.get("logits_processors") or [])
        if processor not in processors:
            rollout.engine_kwargs.vllm.logits_processors = processors + [processor]
    return rollout


def validate_resume_clock(config, rule):
    if rule.kind != "linear" or rule.clock != "optimizer_step":
        return
    import json
    from pathlib import Path

    trainer = config.trainer
    if trainer.resume_mode == "disable":
        return
    if trainer.resume_mode == "resume_path":
        checkpoint = Path(trainer.resume_from_path)
    elif trainer.resume_mode == "auto":
        root = Path(trainer.default_local_dir)
        marker = root / "latest_checkpointed_iteration.txt"
        if not marker.exists():
            return
        checkpoint = root / f"global_step_{int(marker.read_text().strip())}"
    else:
        raise ValueError("Unknown checkpoint resume mode")
    state = checkpoint / "trainer_state.json"
    count = json.loads(state.read_text()).get("actor_optimizer_step") if state.exists() else None
    if type(count) is not int or count < 0:
        raise ValueError("optimizer_step schedule requires a checkpoint with known applied optimizer updates")


def validate_native_options(config):
    for key in (
        "actor_rollout_ref.rollout.quantization",
        "actor_rollout_ref.rollout.quantization_config_file",
        "actor_rollout_ref.rollout.engine_kwargs.vllm.quantization",
        "actor_rollout_ref.rollout.engine_kwargs.vllm.speculative_config",
        "actor_rollout_ref.rollout.engine_kwargs.vllm.enable_eplb",
    ):
        if OmegaConf.select(config, key):
            raise ValueError(f"Native fusion has not accepted {key}")
    cache_dtype = OmegaConf.select(
        config, "actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_dtype", default="auto"
    )
    if cache_dtype not in {None, "auto", "bfloat16"}:
        raise ValueError("Native fusion requires BF16 KV, without KV quantization")
