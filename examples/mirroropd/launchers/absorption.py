# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""One thirty-pair protocol: native current OPD and full-response fixed-bank RKL."""

import json
import math
from pathlib import Path

from .qwen_prefix import hydra_value
from .repetition import validation_repetition

# IDs are append-only: public launcher, queue and report all consume this table.
PAIRS = (
    (94, "DeepSeek-R1-Distill-Qwen-1.5B", "deepscaler", "deepseek", 128, False),
    (95, "DeepSeek-R1-Distill-Qwen-1.5B", "JustRL-DeepSeek-1.5B", "deepseek", 128, False),
    (96, "DeepSeek-R1-Distill-Qwen-1.5B", "minimath", "deepseek", 128, False),
    (97, "DeepSeek-R1-Distill-Qwen-1.5B", "nemotron", "deepseek", 128, False),
    (98, "DeepSeek-R1-Distill-Qwen-1.5B", "brorl", "deepseek", 128, False),
    (99, "Qwen3-1.7B", "Qwen3-1.7B-GRPO-KL-math-reasoning", "qwen3", 64, True),
    (100, "Qwen3-1.7B", "e3-1.7B", "qwen3", 64, True),
    (101, "Qwen3-1.7B", "Qwen3-1.7B-Math", "qwen3", 64, True),
    (102, "Qwen3-1.7B", "Qwen3-4B-Instruct-2507", "qwen3", 64, False),
    (103, "Llama-3.2-3B-Instruct", "Co-rewarding-I-Llama-3.2-3B-Instruct-MATH", "llama", 32, False),
    (104, "Llama-3.2-3B-Instruct", "Llama-3.2-3B-Instruct-arithmetic-full", "llama", 32, False),
    (105, "MiMo-7B-Base", "MiMo-7B-SFT", "qwen25", 32, False),
    (106, "MiMo-7B-Base", "MiMo-7B-RL-Zero", "qwen25", 32, False),
    (107, "MiMo-7B-SFT", "MiMo-7B-RL", "qwen25", 32, False),
    (108, "DeepSeek-R1-Distill-Qwen-1.5B", "DeepSeek-R1-Distill-Qwen-7B", "deepseek", 128, False),
    (109, "DeepSeek-R1-Distill-Qwen-1.5B", "Polaris-7B-Preview", "deepseek", 128, False),
    (110, "DeepSeek-R1-Distill-Qwen-7B", "Polaris-7B-Preview", "deepseek", 32, False),
    (111, "DeepSeek-R1-Distill-Qwen-7B", "Light-R1-7B-DS", "deepseek", 32, False),
    (112, "Qwen3-0.6B", "Qwen3-4B-Instruct-2507", "qwen3", 32, False),
    (113, "Qwen3-4B", "Qwen3-4B-Instruct-2507", "qwen3", 32, False),
    (114, "DeepSeek-R1-Distill-Qwen-1.5B", "Light-R1-7B-DS", "deepseek", 128, False),
    (115, "MiMo-7B-Base", "MiMo-7B-RL", "qwen25", 32, False),
    (116, "MiMo-7B-SFT", "MiMo-7B-RL-Zero", "qwen25", 32, False),
    (117, "Qwen3-Base-AbsorptionProtocol", "Qwen3-1.7B-SFT", "qwen3", 32, False),
    (118, "Qwen3-Base-AbsorptionProtocol", "GT-Qwen3-1.7B-Base-MATH", "qwen3", 32, False),
    (
        119,
        "DeepSeek-R1-Distill-Qwen-1.5B",
        "agentica-org/DeepCoder-1.5B-Preview@103033da54f104b0d0a3570547fb3721fba99962",
        "deepseek",
        128,
        False,
    ),
    (120, "DeepSeek-R1-Distill-Qwen-1.5B", "UltraData-IF-1.5B", "deepseek", 128, False),
    (121, "Llama-3.1-Tulu-3-8B-SFT", "Llama-3.1-Tulu-3-8B-DPO", "default", 32, False),
    (122, "Llama-3.1-Tulu-3-8B-DPO", "Llama-3.1-Tulu-3-8B", "default", 32, False),
    (123, "Llama-3.1-Tulu-3-8B-SFT", "Llama-3.1-Tulu-3-8B", "default", 32, False),
    (124, "Qwen3-1.7B", "Qwen3-4B-Instruct-2507", "qwen3", 64, False),
    (125, "Qwen3-1.7B", "Qwen3-4B-Instruct-2507", "qwen3", 64, False),
    (126, "Qwen3-1.7B", "Qwen3-4B-Instruct-2507", "qwen3", 64, False),
    (127, "Qwen3-1.7B", "Qwen3-4B-Instruct-2507", "qwen3", 64, False),
    (128, "Qwen3-1.7B", "Qwen3-4B-Instruct-2507", "qwen3", 64, False),
    (129, "Qwen3-1.7B", "Qwen3-4B-Instruct-2507", "qwen3", 64, False),
)
# E124/E125 are explicit full-budget SGD LR probes of the E102 pair.
SGD_LR_PRESETS = {124: 0.01, 125: 0.03}
MUON_LR_PRESETS = {126: 3e-7, 127: 1e-6, 128: 3e-6, 129: 1e-5}

# Matched WD=0 optimizer probes; explicit IDs also drive the public launcher.
OPTIMIZER_PROBES = {
    130: (102, "adamw", 1e-6),
    131: (102, "adamw", 3e-6),
    132: (102, "adamw", 3e-7),
    133: (102, "muon", 3e-5),
}
for _pair, _start in ((95, 134), (109, 140), (117, 146)):
    for _offset, (_kind, _lr) in enumerate(
        (("muon", 3e-6), ("adamw", 1e-6), ("muon", 1e-5), ("adamw", 3e-6), ("muon", 1e-6), ("adamw", 3e-7))
    ):
        OPTIMIZER_PROBES[_start + _offset] = (_pair, _kind, _lr)
_PAIR_BY_ID = {row[0]: row for row in PAIRS}
# Fresh attempts after the E134 local-Ray startup failure; original IDs stay immutable.
OPTIMIZER_PROBES.update({number + 22: probe for number, probe in list(OPTIMIZER_PROBES.items())})
PAIRS += tuple((number, *_PAIR_BY_ID[pair][1:]) for number, (pair, _, _) in OPTIMIZER_PROBES.items())

# Completed E94–E104 retain their protocol. Future MiMo/Base runs declare
# both terminators without coarsening the full-vocabulary KL distribution.
# Frozen attempts and their banks remain identified by their original contracts.
STOP_TOKEN_IDS = {number: (151643, 151645) for number in (105, 106, 107, 115, 116, 117, 118)}
SNAPSHOTS = (1, 2, 3, 4, 8, 12, 20, 30, 40)


def optimizer_matrix():
    """One display row per optimizer arm; retry IDs retain their original protocol identity."""
    rows = []
    for number, lr in SGD_LR_PRESETS.items():
        rows.append(dict(number=number, pair=102, optimizer="SGD", lr=lr, aux_lr=None, wd=0, retry_of=None))
    for number, lr in MUON_LR_PRESETS.items():
        rows.append(dict(number=number, pair=102, optimizer="Muon", lr=lr, aux_lr=1e-6, wd=0, retry_of=None))
    for number, (pair, kind, lr) in OPTIMIZER_PROBES.items():
        rows.append(
            dict(
                number=number,
                pair=pair,
                optimizer="Muon" if kind == "muon" else "AdamW",
                lr=lr,
                aux_lr=1e-6 if kind == "muon" else None,
                wd=0,
                retry_of=number - 22 if 152 <= number <= 173 else None,
            )
        )
    return rows


def optimizer_options(env):
    """Explicit optimizer branches; historical AdamW defaults stay untouched."""
    kind = env.get("ABSORPTION_OPTIMIZER", "adamw").lower()
    if kind == "adamw":
        value = env.get("ABSORPTION_OPTIMIZER_LR")
        if value is None:
            return []
        if not math.isfinite(float(value)) or float(value) <= 0:
            raise ValueError("Set a positive ABSORPTION_OPTIMIZER_LR")
        return [
            "actor_rollout_ref.actor.optim.lr=" + value,
            "actor_rollout_ref.actor.optim.weight_decay=0",
        ]
    if kind not in ("sgd", "muon"):
        raise ValueError("ABSORPTION_OPTIMIZER must be adamw, sgd or muon")
    value = env.get("ABSORPTION_OPTIMIZER_LR")
    if value is None or not math.isfinite(float(value)) or float(value) <= 0:
        raise ValueError("Set a calibrated positive ABSORPTION_OPTIMIZER_LR")
    env["LEARNING_RATE"] = value
    result = [
        "actor_rollout_ref.actor.optim.lr=" + value,
        "actor_rollout_ref.actor.optim.weight_decay=0",
        "actor_rollout_ref.actor.optim.lr_scheduler_type=constant",
        "actor_rollout_ref.actor.optim.lr_warmup_steps=0",
        "actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0",
        "actor_rollout_ref.actor.checkpoint.save_contents=[model,optimizer,extra,hf_model]",
        "actor_rollout_ref.actor.checkpoint.load_contents=[model,optimizer,extra]",
        "trainer.save_freq=5",
        "trainer.max_actor_ckpt_to_keep=2",
    ]
    if kind == "sgd":
        return result + [
            "actor_rollout_ref.actor.optim.optimizer=SGD",
            "actor_rollout_ref.actor.optim.optimizer_impl=torch.optim",
            "actor_rollout_ref.actor.optim.override_optimizer_config={momentum:0.0}",
        ]
    aux = env.get("ABSORPTION_AUX_LR")
    if aux is None or not math.isfinite(float(aux)) or float(aux) <= 0:
        raise ValueError("Muon requires a calibrated positive ABSORPTION_AUX_LR")
    distributed_mode = env.get("ABSORPTION_MUON_DISTRIBUTED_MODE", "owner")
    if distributed_mode not in ("owner", "replicated"):
        raise ValueError("ABSORPTION_MUON_DISTRIBUTED_MODE must be owner or replicated")
    return result + [
        "actor_rollout_ref.actor.optim.optimizer=MuonAdamW",
        "actor_rollout_ref.actor.optim.optimizer_impl=verl.utils.fsdp_optimizers",
        "actor_rollout_ref.actor.optim.override_optimizer_config="
        + hydra_value(
            dict(
                aux_lr=float(aux),
                momentum=0.95,
                nesterov=True,
                ns_steps=5,
                adjust_lr_fn="match_rms_adamw",
                distributed_mode=distributed_mode,
            )
        ),
    ]


def template_kwargs(pair):
    """One input-template contract for training plans and CPU admission."""
    if pair[3] == "qwen3":
        return {"enable_thinking": pair[5]}
    if pair[3] == "llama":
        return {"date_string": "12 Nov 2026"}
    return {}


def options(env, name, root):
    """Compose ordinary launcher overrides; bank scoring never generates current answers."""

    def default(key, value):
        if not env.get(key):
            env[key] = str(value)

    number = int(name.removeprefix("E"))
    if number in OPTIMIZER_PROBES:
        _, kind, lr = OPTIMIZER_PROBES[number]
        default("ABSORPTION_OPTIMIZER", kind)
        default("ABSORPTION_OPTIMIZER_LR", lr)
        if kind == "muon":
            default("ABSORPTION_AUX_LR", 1e-6)
    if number in SGD_LR_PRESETS:
        default("ABSORPTION_OPTIMIZER", "sgd")
        default("ABSORPTION_OPTIMIZER_LR", SGD_LR_PRESETS[number])
    if number in MUON_LR_PRESETS:
        default("ABSORPTION_OPTIMIZER", "muon")
        default("ABSORPTION_OPTIMIZER_LR", MUON_LR_PRESETS[number])
        default("ABSORPTION_AUX_LR", 1e-6)
    default("ABSORPTION_ROOT", root / "runs/absorption-long-20261112")
    study = Path(env["ABSORPTION_ROOT"]).resolve()
    dataset_path = study / "questions/datasets.json"
    if not dataset_path.is_file():
        raise ValueError("Prepare the absorption question tables before resolving absorption presets")
    datasets = json.loads(dataset_path.read_text())
    branch = env.get("ABSORPTION_OPTIMIZER", "adamw").lower()
    branch_options = optimizer_options(env)
    if branch != "adamw":
        if env.get("RUN_DIR") and Path(env["RUN_DIR"]).resolve() == (study / "training" / name).resolve():
            raise ValueError("Optimizer ablations must not overwrite the historical training directory")
        default("EXPERIMENT_NAME", f"math-opd-{name}-{branch}")
    default("RUN_DIR", study / ("training" if branch == "adamw" else f"training-{branch}") / name)
    default("CHECKPOINT_DIR", Path(env["RUN_DIR"]) / "runtime")
    default("TRAIN_FILE", study / "questions/train.parquet")
    for key, value in {
        "STUDENT_GPUS": 4,
        "TEACHER_GPUS": 0,
        "DISTILLATION_EXECUTION_MODE": "shared_sync",
        "FULL_KL_BACKEND": "precompute_fused",
        "FULL_KL_TOKEN_CHUNK": 256,
        "FULL_KL_TERMINATION_TOKEN_IDS": "[]",
        "PARAM_OFFLOAD": "True",
        "OPTIMIZER_OFFLOAD": "True",
        "GRADIENT_CHECKPOINTING": "True",
        "ROLLOUT_TOKEN_BUDGET": 16384,
        "PPO_MAX_TOKEN_LEN_PER_GPU": 16384,
        "TEACHER_FSDP_TOKEN_BUDGET": 16384,
        "ROLLOUT_GPU_MEMORY_UTILIZATION": 0.65,
        "ROLLOUT_TP": 1,
    }.items():
        default(key, value)
    pair = next(row for row in PAIRS if name == f"E{row[0]}")
    world = int(env["STUDENT_GPUS"])
    if world not in (4, 8):
        raise ValueError("Absorption profiles require four or eight GPUs")
    seqs = int(env.get("ROLLOUT_MAX_NUM_SEQS") or pair[4])
    default("ROLLOUT_MAX_NUM_SEQS", seqs)
    template = template_kwargs(pair)
    spec = dict(
        _target_="verl.trainer.evaluation.coordinator.EvaluationCoordinator",
        output=str(Path(env["RUN_DIR"]) / "evaluation"),
        bootstrap=str(study / "bootstrap" / name),
        accuracy_enabled=False,
        source_enabled=False,
        history_enabled=False,
        datasets=datasets,
        protocols={
            "public": dict(
                model_family="default" if pair[3] == "llama" else pair[3],
                prompt_limit=4096,
                max_new_tokens=7168,
                temperature=1.0,
                top_p=1.0,
                top_k=-1,
                template_kwargs=template,
                seed=42,
                stop_token_ids=STOP_TOKEN_IDS.get(pair[0], ()),
                repetition_detection=validation_repetition(env),
            )
        },
        generation_concurrency=world * seqs,
        generation_chunk_size=32,
        schedules={"banks": {"every": 1}, "bench": {"at": [], "final": False}},
        score=dict(payload_bytes=4 * 2**30, host_budget_bytes=64 * 2**30, token_chunk=256),
        export_outers=list(SNAPSHOTS),
        final_export=True,
        verify_export=True,
    )
    return [
        *(
            ["actor_rollout_ref.rollout.stop_token_ids=" + hydra_value(list(STOP_TOKEN_IDS[pair[0]]))]
            if pair[0] in STOP_TOKEN_IDS
            else []
        ),
        *(['+data.apply_chat_template_kwargs.date_string="12 Nov 2026"'] if pair[3] == "llama" else []),
        "data.continuous_token.model_family=" + ("default" if pair[3] == "llama" else pair[3]),
        "rollout_policy.mode=single",
        "rollout_policy.model={kind:current}",
        "data.val_files=[]",
        "data.shuffle=False",
        "data.filter_overlong_prompts=False",
        "data.truncation=error",
        "trainer.val_before_train=True",
        "trainer.resume_mode=disable",
        "trainer.validation_data_dir=null",
        "trainer.logger=[console,file,wandb]",
        "actor_rollout_ref.model.trust_remote_code=" + str(pair[3] == "qwen25"),
        "actor_rollout_ref.model.mtp.enable=False",
        "actor_rollout_ref.actor.fsdp_config.forward_precision=null",
        "actor_rollout_ref.actor.fsdp_config.model_dtype=fp32",
        "actor_rollout_ref.actor.fsdp_config.dtype=bfloat16",
        "actor_rollout_ref.actor.optim.weight_decay=0.01",
        "actor_rollout_ref.rollout.max_model_len=11264",
        "actor_rollout_ref.rollout.val_kwargs.max_tokens=null",
        "actor_rollout_ref.rollout.logprobs_mode=processed_logprobs",
        "actor_rollout_ref.actor.checkpoint.save_contents=[model,hf_model]",
        "actor_rollout_ref.actor.checkpoint.load_contents=[model]",
        "++actor_rollout_ref.actor.checkpoint.preserve_hf_master_weights=True",
        "++actor_rollout_ref.rollout.engine_kwargs.vllm.attention_config.backend=TRITON_ATTN",
        "actor_rollout_ref.rollout.agent.num_workers=16",
        "reward.num_workers=8",
        "transfer_queue.backend.SimpleStorage.num_data_storage_units=8",
        *branch_options,
        "++trainer.evaluation=" + hydra_value(spec),
        f"++ray_kwargs.ray_init.num_gpus={world}",
        "++ray_kwargs.ray_init.num_cpus=64",
        "++ray_kwargs.ray_init.address=local",
        "++ray_kwargs.ray_init.include_dashboard=False",
        "++ray_kwargs.ray_init.object_store_memory=" + str(8 * 2**30),
        "++ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH=" + hydra_value(env.get("PYTHONPATH", str(root))),
        "++ray_kwargs.ray_init.runtime_env.env_vars.VERL_FILE_LOGGER_PATH="
        + hydra_value(str(Path(env["RUN_DIR"]) / "metrics.jsonl")),
    ]


def preflight(env):
    if env.get("CONFIG_ONLY") == "1" or env.get("DRY_RUN") == "1":
        return
    path = Path(env["ABSORPTION_ROOT"]) / "bootstrap" / env["ABSORPTION_EXPERIMENT"] / "manifest.json"
    if not path.is_file() or json.loads(path.read_text()).get("status") != "complete":
        raise ValueError("Generate and finalize this experiment's native bank before training")
