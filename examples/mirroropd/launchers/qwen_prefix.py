# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Numbered Qwen prefix arms using native distributed evaluation and shared banks."""

import json
import math
from pathlib import Path

BENCHMARKS = (("MATH-500", 500), ("AMC23", 40), ("AIME24", 30), ("AIME25", 30), ("AIME26", 30))


def hydra_value(value):
    """Serialize a nested default as one Hydra override, without shell evaluation."""
    if isinstance(value, dict):
        return "{" + ",".join(f"{key}:{hydra_value(child)}" for key, child in value.items()) + "}"
    if isinstance(value, list | tuple):
        return "[" + ",".join(map(hydra_value, value)) + "]"
    return json.dumps(value)


def evaluation(root, output, world):
    """A new native evaluation protocol; never relabel the legacy HF-bank results."""
    protocols = {}
    for name, cap, prompt, temperature, top_p, seed in [
        ("source", 7168, 4096, 1.0, 1.0, 52000000),
        ("public", 31744, 1024, 0.7, 0.9, 72000000),
    ]:
        protocols[name] = dict(
            temperature=temperature,
            top_p=top_p,
            top_k=-1,
            max_new_tokens=cap,
            prompt_limit=prompt,
            template_kwargs={"enable_thinking": False},
            stop_token_ids=[151645, 151643],
            seed=seed,
            answer_channel="whole_response",
        )
    return dict(
        _target_="verl.trainer.evaluation.coordinator.EvaluationCoordinator",
        output=str(output / "evaluation"),
        source_file=str(root / "questions/source.parquet"),
        bootstrap=str(root / "bootstrap"),
        training_domain="math",
        source_count=64,
        history_size=64,
        seed=42,
        generation_concurrency=world * 16,
        generation_chunk_size=32,
        verifier_workers=16,
        protocols=protocols,
        datasets=[
            dict(
                name=name,
                domain="math",
                path=str(root / f"questions/{name}.parquet"),
                verifier="math",
                n=4,
                protocol="public",
                panel_ids=[f"{name}:{i}" for i in range(min(8, count))],
            )
            for name, count in BENCHMARKS
        ],
        score=dict(payload_bytes=8 * 2**30, host_budget_bytes=128 * 2**30, copy_reserve_factor=2.5, token_chunk=1024),
        schedules={"source": {"every": 5}, "history": {"every": 5}, "banks": {"every": 5}, "bench": {"at": [20, 40]}},
        final_export=True,
        verify_export=True,
    )


def options(env, arm, root):
    """Explicit defaults before method/user overrides; no GPU reservation side effects."""

    def default(key, value):
        if not env.get(key):
            env[key] = str(value)

    default("STUDENT_GPUS", 8)
    world = int(env["STUDENT_GPUS"])
    batch = int(env.get("TRAIN_BATCH_SIZE") or 64)
    if world not in (1, 2, 4, 8) or batch % world:
        raise ValueError("Qwen prefix study requires 1/2/4/8 local GPUs and a divisible global batch")
    default("QWEN_EVAL_ROOT", root / "runs/qwen-prefix-native")
    study = Path(env["QWEN_EVAL_ROOT"]).expanduser().resolve()
    default("TRAIN_FILE", study / "questions/train.parquet")
    for key, value in dict(
        TEACHER_GPUS=0,
        DISTILLATION_EXECUTION_MODE="shared_sync",
        FULL_KL_BACKEND="precompute_fused",
        FULL_KL_TOKEN_CHUNK=1024,
        FULL_KL_TERMINATION_TOKEN_IDS="[]",
        FULL_KL_ASYNC_WRITE="True",
        FULL_KL_PREFETCH="True",
        MIRROR_DIAGNOSTICS="False",
        ROLLOUT_MAX_NUM_SEQS=16,
        ROLLOUT_GPU_MEMORY_UTILIZATION="0.8",
        ROLLOUT_TOKEN_BUDGET=32768,
        PPO_MAX_TOKEN_LEN_PER_GPU=32768 if world > 1 else 16384,
        TEACHER_FSDP_TOKEN_BUDGET=32768 if world > 1 else 16384,
        PARAM_OFFLOAD="True",
        OPTIMIZER_OFFLOAD="True",
        GRADIENT_CHECKPOINTING="True",
        MAX_VAL_RESPONSE_LENGTH=31744,
        VAL_N=4,
        PREFIX_RAY_CPUS=128 if world == 8 else 32,
    ).items():
        default(key, value)
    visible = env.get("CUDA_VISIBLE_DEVICES") or ",".join(map(str, range(world)))
    devices = visible.split(",")
    if len(devices) != world or len(set(devices)) != world or any(d not in set(map(str, range(8))) for d in devices):
        raise ValueError("Set CUDA_VISIBLE_DEVICES to exactly STUDENT_GPUS distinct physical indices")
    context = int(env.get("MAX_PROMPT_LENGTH") or 4096) + int(env.get("MAX_RESPONSE_LENGTH") or 7168)
    default("FUSION_TP", 2 if world > 1 else 1)
    tp = int(env["FUSION_TP"])
    if tp not in (1, 2) or world % tp:
        raise ValueError("Fusion TP must be 1 or 2 and divide the allocated GPUs")
    default("FUSION_MAX_NUM_SEQS", min(32, math.ceil(batch / (world // tp))))
    default(
        "FUSION_KV_BYTES",
        math.ceil(context * int(env["FUSION_MAX_NUM_SEQS"]) * 256 * 1024 / (tp * 8 * 2**30)) * 8 * 2**30,
    )
    # Shared current engine also serves long native accuracy requests. Private
    # fixed rollout generators compute their own training-only KV budgets.
    default("PREFIX_SINGLE_KV_BYTES", 64 * 2**30)
    policy = {
        "S": ["rollout_policy.mode=single", "rollout_policy.model={kind:current}"],
        "S0": ["rollout_policy.mode=single", "rollout_policy.model={kind:initial}"],
        "T": ["rollout_policy.mode=single", "rollout_policy.model={kind:teacher}"],
        "LM": ["rollout_policy.mode=token_mixture", "rollout_policy.fusion_space=logit"],
        "PM": ["rollout_policy.mode=token_mixture", "rollout_policy.fusion_space=probability"],
        "TM": ["rollout_policy.mode=trajectory_mixture"],
    }[arm]
    if arm in ("LM", "PM", "TM"):
        policy += [
            "rollout_policy.left={kind:current}",
            "rollout_policy.right={kind:teacher}",
            "rollout_policy.right_weight=0.5",
        ]
    output = Path(env["RUN_DIR"])
    spec = evaluation(study, output, world)
    return [
        *policy,
        "trainer.val_before_train=True",
        "data.val_files=[]",
        "trainer.resume_mode=disable",
        "actor_rollout_ref.actor.fsdp_config.forward_precision=null",
        "actor_rollout_ref.actor.fsdp_config.model_dtype=fp32",
        "actor_rollout_ref.actor.fsdp_config.dtype=bfloat16",
        "actor_rollout_ref.rollout.temperature=1.0",
        "actor_rollout_ref.rollout.top_p=1.0",
        "actor_rollout_ref.rollout.top_k=-1",
        "actor_rollout_ref.rollout.n=1",
        "actor_rollout_ref.rollout.logprobs_mode=processed_logprobs",
        "actor_rollout_ref.rollout.stop_token_ids=[151645,151643]",
        "++actor_rollout_ref.rollout.engine_kwargs.vllm.attention_config.backend=TRITON_ATTN",
        f"++actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_memory_bytes={env['PREFIX_SINGLE_KV_BYTES']}",
        "rollout_policy.execution.backend=vllm_fusion",
        f"rollout_policy.execution.tensor_parallel_size={tp}",
        f"rollout_policy.execution.max_num_seqs={env['FUSION_MAX_NUM_SEQS']}",
        f"rollout_policy.execution.kv_cache_memory_bytes={env['FUSION_KV_BYTES']}",
        "rollout_policy.execution.enforce_eager=False",
        "actor_rollout_ref.rollout.agent.num_workers=16",
        "reward.num_workers=8",
        "transfer_queue.backend.SimpleStorage.num_data_storage_units=8",
        "actor_rollout_ref.actor.checkpoint.save_contents=[model,hf_model]",
        "actor_rollout_ref.actor.checkpoint.load_contents=[model]",
        "++actor_rollout_ref.actor.checkpoint.preserve_hf_master_weights=True",
        "++trainer.evaluation=" + hydra_value(spec),
        f"++ray_kwargs.ray_init.num_gpus={world}",
        f"++ray_kwargs.ray_init.num_cpus={env['PREFIX_RAY_CPUS']}",
        "++ray_kwargs.ray_init.include_dashboard=False",
    ]


def preflight(environment):
    """Fail before allocating Ray workers when the shared native banks are missing."""
    if environment.get("CONFIG_ONLY") == "1" or environment.get("DRY_RUN") == "1":
        return
    path = Path(environment["QWEN_EVAL_ROOT"]) / "bootstrap/manifest.json"
    if not path.is_file() or json.loads(path.read_text()).get("status") != "complete":
        raise ValueError(
            "E63-E68 need completed native banks under QWEN_EVAL_ROOT. "
            "Prepare questions, resolve a CONFIG_ONLY YAML, then run "
            "verl.trainer.evaluation.bootstrap with initial, teacher and finalize."
        )
