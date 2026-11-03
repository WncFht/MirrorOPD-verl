# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""MiMo SFT, direct-RL and SFT+RL teachers on the native evaluation pipeline."""

import hashlib
import json
import os
from pathlib import Path

from examples.mirroropd.launchers.qwen_prefix import hydra_value

# The paper uses 32K generation for reasoning domains and 8K elsewhere. Reserve
# the existing 8K prompt budget inside the checkpoint's native 32K total context.
PUBLIC_LENGTHS = {"math": (8192, 24576), "code": (8192, 24576), "science": (8192, 24576), "if": (8192, 8192)}


def questions(path, *, required, setting="MIMO_QUESTIONS"):
    """Read the frozen selection, including real question IDs and sampling counts."""
    manifest = Path(path) / "manifest.json"
    if not manifest.is_file():
        raise ValueError(f"Set {setting} to a complete native prepared question artifact")
    document = json.loads(manifest.read_text())
    if document.get("status") != "complete" or document.get("kind") != "question_sets":
        raise ValueError(f"{setting} must contain a complete question_sets manifest")
    metadata = document["metadata"]
    if metadata["split_reports"]["math"]["training"] < required:
        raise ValueError("Stage-chain training pool cannot cover the budget without repeating questions")
    datasets = [{**entry, "protocol": entry["domain"]} for entry in metadata["datasets"]]
    if {entry["domain"] for entry in datasets} != set(PUBLIC_LENGTHS):
        raise ValueError("Stage-chain comparison requires all four public evaluation domains")
    for domain in PUBLIC_LENGTHS:
        panel = [qid for entry in datasets if entry["domain"] == domain for qid in entry["panel_ids"]]
        if len(panel) != 128 or len(set(panel)) != 128:
            raise ValueError(f"Stage-chain {domain} requires 128 distinct fixed panel questions")
    return manifest, metadata, datasets


def options(
    env,
    name,
    root,
    *,
    model_family="qwen25",
    stop_token_ids=(151643, 151645),
    trust_remote_code=True,
    questions_key="MIMO_QUESTIONS",
    bootstrap_key="MIMO_BOOTSTRAP",
    run_namespace="mimo-stage-native",
    default_outer=50,
):
    """Compose one E arm without allocating workers or generating evaluation banks."""

    def default(key, value):
        if not env.get(key):
            env[key] = str(value)

    default(questions_key, root / "runs/unified-evaluation-20261031/questions-v2")
    default(bootstrap_key, root / "runs" / run_namespace / name / "bootstrap")
    world = int(env["STUDENT_GPUS"])
    batch = int(env["TRAIN_BATCH_SIZE"])
    if world < 1 or world > 8 or batch % world:
        raise ValueError("Stage-chain profiles require 1–8 shared GPUs and a divisible global batch")
    outer = int(env.get("TOTAL_TRAINING_STEPS") or default_outer)
    manifest, metadata, datasets = questions(env[questions_key], required=outer * batch, setting=questions_key)
    for key, value in dict(
        TRAIN_FILE=metadata["paths"]["math"]["train"],
        FULL_KL_ASYNC_WRITE="True",
        FULL_KL_PREFETCH="True",
        ROLLOUT_MAX_NUM_SEQS=16,
        ROLLOUT_TOKEN_BUDGET=16384,
        PPO_MAX_TOKEN_LEN_PER_GPU=32768,
        TEACHER_FSDP_TOKEN_BUDGET=32768,
        PARAM_OFFLOAD="True",
        OPTIMIZER_OFFLOAD="True",
        GRADIENT_CHECKPOINTING="True",
        VAL_REPETITION_DETECTION="false",
        SEED=42,
    ).items():
        default(key, value)
    output = Path(env["RUN_DIR"])
    hf_home = Path(env.get("HF_HOME") or Path(env.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "huggingface")
    module_cache = str(Path(env.get("HF_MODULES_CACHE") or hf_home / "modules"))
    # Ray must import the pinned custom config before deserializing HFModelConfig.
    # HF normally adds this path only inside the driver that loaded the config.
    env["PYTHONPATH"] = os.pathsep.join(
        dict.fromkeys([str(root), module_cache, *filter(None, env.get("PYTHONPATH", "").split(os.pathsep))])
    )
    ray_tmp = env.get("SMOKE_RAY_TMPDIR") or (
        "/tmp/mirroropd-mimo-" + hashlib.sha256(str(output).encode()).hexdigest()[:12]
    )
    spec = dict(
        _target_="verl.trainer.evaluation.coordinator.EvaluationCoordinator",
        output=str(output / "evaluation"),
        bootstrap=str(Path(env[bootstrap_key]).expanduser().resolve()),
        source_enabled=False,
        history_enabled=False,
        training_domain="math",
        seed=int(env["SEED"]),
        generation_concurrency=world * 16,
        generation_chunk_size=64,
        datasets=datasets,
        protocols={
            domain: dict(
                model_family=model_family,
                temperature=1.0,
                top_p=0.95,
                top_k=-1,
                prompt_limit=prompt,
                max_new_tokens=response,
                stop_token_ids=list(stop_token_ids),
                seed=int(env["SEED"]),
                answer_channel="after_thinking",
            )
            for domain, (prompt, response) in PUBLIC_LENGTHS.items()
        },
        score=dict(payload_bytes=16 * 2**30, host_budget_bytes=64 * 2**30, token_chunk=256),
        schedules=dict(
            source=dict(at=[], every=None),
            history=dict(at=[], every=None),
            banks=dict(at=[0, 1, 2], every=5),
            bench=dict(at=[step for step in (0, 25, 50) if step < outer] + [outer]),
        ),
        final_export=True,
        verify_export=True,
    )
    return [
        "rollout_policy.mode=single",
        "rollout_policy.model={kind:current}",
        f"actor_rollout_ref.model.trust_remote_code={trust_remote_code}",
        "actor_rollout_ref.model.mtp.enable=False",
        "data.val_files=[]",
        "data.shuffle=False",
        "data.filter_overlong_prompts=False",
        "data.truncation=error",
        "trainer.resume_mode=disable",
        "trainer.validation_data_dir=null",
        "trainer.logger=[console,file,wandb]",
        f"++trainer.prepared_questions_manifest={hydra_value(str(manifest.resolve()))}",
        f"++trainer.prepared_questions_sha256={hashlib.sha256(manifest.read_bytes()).hexdigest()}",
        "actor_rollout_ref.actor.fsdp_config.forward_precision=null",
        "actor_rollout_ref.actor.fsdp_config.model_dtype=fp32",
        "actor_rollout_ref.actor.fsdp_config.dtype=bfloat16",
        "actor_rollout_ref.rollout.max_model_len=32768",
        # Legacy validation is disabled; each native protocol supplies its own
        # prompt/response budget instead of combining training prompts with 3K.
        "actor_rollout_ref.rollout.val_kwargs.max_tokens=null",
        "actor_rollout_ref.rollout.logprobs_mode=processed_logprobs",
        "actor_rollout_ref.rollout.stop_token_ids=" + hydra_value(list(stop_token_ids)),
        "distillation.teacher_models.teacher_model.inference.max_model_len=32768",
        "++actor_rollout_ref.rollout.engine_kwargs.vllm.attention_config.backend=TRITON_ATTN",
        "actor_rollout_ref.rollout.agent.num_workers=16",
        "reward.num_workers=8",
        "transfer_queue.backend.SimpleStorage.num_data_storage_units=8",
        "actor_rollout_ref.actor.checkpoint.save_contents=[model,hf_model]",
        "actor_rollout_ref.actor.checkpoint.load_contents=[model]",
        "++actor_rollout_ref.actor.checkpoint.preserve_hf_master_weights=True",
        "++trainer.evaluation=" + hydra_value(spec),
        "++ray_kwargs.ray_init.address=local",
        f"++ray_kwargs.ray_init.num_gpus={world}",
        f"++ray_kwargs.ray_init.num_cpus={min(64, world * 16)}",
        f"++ray_kwargs.ray_init.object_store_memory={8 * 2**30}",
        f"++ray_kwargs.ray_init._temp_dir={hydra_value(ray_tmp)}",
        "++ray_kwargs.ray_init.runtime_env.env_vars.VERL_FILE_LOGGER_PATH="
        + hydra_value(str(output / "metrics.jsonl")),
        "++ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH=" + hydra_value(env["PYTHONPATH"]),
        "++ray_kwargs.ray_init.include_dashboard=False",
    ]


def preflight(environment, *, bootstrap_key="MIMO_BOOTSTRAP"):
    """Require the matching native bootstrap before entering the training launcher."""
    if environment.get("CONFIG_ONLY") == "1" or environment.get("DRY_RUN") == "1":
        return
    manifest = Path(environment[bootstrap_key]) / "manifest.json"
    if not manifest.is_file() or json.loads(manifest.read_text()).get("status") != "complete":
        raise ValueError(
            f"Stage-chain presets need a completed teacher-specific native bootstrap under {bootstrap_key}. "
            "Resolve a CONFIG_ONLY YAML, then run verl.trainer.evaluation.bootstrap "
            "with initial, teacher and finalize. Source/history producers are disabled."
        )
