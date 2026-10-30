# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Compose the exact shell overrides without model loading or Ray startup."""

import os
import shlex
import shutil
import subprocess
import sys
from configparser import ConfigParser
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.errors import InstantiationException
from omegaconf import OmegaConf

from examples.mirroropd.assets.paths import registered_model
from verl.utils.config import omega_conf_to_dataclass, validate_config
from verl.workers.config.distillation_spec import resolve_distillation_spec

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "examples/mirroropd/run_math_opd_ablation_fsdp2.sh"


def _model_suffix(alias):
    return str(registered_model(alias, root="/")).lstrip("/")


def _assert_full_kl(loss):
    assert loss.loss_mode == "mirror"
    spec = resolve_distillation_spec(loss)
    assert spec.teacher_only_kl and spec.anchor_payload is None
    assert spec.eta == 1 and not loss.use_policy_gradient


def test_polaris_e48_resolves_full_e20_protocol_and_unchanged_teacher_budget(tmp_path):
    cfg = _config(tmp_path, command=["bash", str(SCRIPT), "E48"], STUDENT_GPUS="4")
    OmegaConf.resolve(cfg)
    validate_config(cfg, use_reference_policy=False, use_critic=False)
    _assert_full_kl(omega_conf_to_dataclass(cfg.distillation).distillation_loss)
    actor, rollout = cfg.actor_rollout_ref.actor, cfg.actor_rollout_ref.rollout
    assert cfg.distillation.teacher_models.teacher_model.model_path.endswith("/" + _model_suffix("Polaris-7B-Preview"))
    assert cfg.distillation.shared_teacher.infer_max_token_len_per_gpu == 32768
    assert cfg.data.train_batch_size == actor.ppo_mini_batch_size == 64
    assert actor.ppo_epochs == 16 and rollout.n == 1
    assert cfg.trainer.total_training_steps == 256
    assert cfg.trainer.n_gpus_per_node == 4 and cfg.trainer.val_before_train
    assert (cfg.trainer.test_freq, cfg.trainer.save_freq) == (32, 64)
    assert actor.optim.lr == 1e-6
    assert (cfg.data.max_prompt_length, cfg.data.max_response_length) == (4096, 7168)
    assert rollout.val_kwargs.max_tokens == 31744 and rollout.val_kwargs.n == 16
    assert cfg.distillation.distillation_loss.full_vocab.backend == "precompute"
    assert cfg.distillation.distillation_loss.full_vocab.token_chunk_size == 512


@pytest.mark.parametrize("domain", ["math", "code", "if"])
def test_full_kl_method(tmp_path, domain):
    cfg = _config(
        tmp_path,
        command=["bash", str(SCRIPT), "custom", "--method", "full_kl"],
        DOMAIN=domain,
        FULL_KL_BACKEND="streaming",
        FULL_KL_ASYNC_WRITE="True",
        FULL_KL_PREFETCH="True",
    )
    OmegaConf.resolve(cfg)
    validate_config(cfg, use_reference_policy=False, use_critic=False)
    loss = omega_conf_to_dataclass(cfg.distillation).distillation_loss
    _assert_full_kl(loss)
    assert loss.full_vocab.backend == "streaming"
    assert loss.full_vocab.async_write and loss.full_vocab.prefetch
    assert not cfg.actor_rollout_ref.rollout.calculate_log_probs


def test_full_reverse_kl_smoke_configuration(tmp_path):
    cfg = _config(tmp_path, script=SCRIPT.parent / "smoke/smoke_full_reverse_kl.sh")
    OmegaConf.resolve(cfg)
    validate_config(cfg, use_reference_policy=False, use_critic=False)
    assert cfg.trainer.total_training_steps == 2
    assert cfg.trainer.n_gpus_per_node == cfg.ray_kwargs.ray_init.num_gpus == 4
    assert cfg.distillation.distillation_loss.full_vocab.backend == "precompute"
    assert cfg.actor_rollout_ref.rollout.engine_kwargs.vllm.attention_config.backend == "TRITON_ATTN"
    assert cfg.distillation.distillation_loss.full_vocab.async_write
    assert cfg.distillation.distillation_loss.full_vocab.prefetch


def _launch(tmp_path, script=SCRIPT, command=None, **overrides):
    env = os.environ.copy()
    for name in (
        "CUDA_VISIBLE_DEVICES",
        "STUDENT_GPUS",
        "TEACHER_GPUS",
        "ROLLOUT_TP",
        "TEACHER_TP",
        "DOMAIN",
        "DISTILLATION_EXECUTION_MODE",
        "TEACHER_FSDP_TOKEN_BUDGET",
        "TEACHER_FSDP_DTYPE",
        "ROLLOUT_ENFORCE_EAGER",
        "ROLLOUT_MAX_NUM_SEQS",
        "ROLLOUT_GPU_MEMORY_UTILIZATION",
        "ROLLOUT_TOKEN_BUDGET",
        "VAL_BATCH_SIZE",
        "VAL_N",
        "CONTINUOUS_TOKEN_MODEL_FAMILY",
        "MAX_PROMPT_LENGTH",
        "MAX_RESPONSE_LENGTH",
        "MAX_VAL_RESPONSE_LENGTH",
        "VAL_REPETITION_DETECTION",
        "VAL_REPETITION_MIN_PATTERN",
        "VAL_REPETITION_MAX_PATTERN",
        "VAL_REPETITION_MIN_COUNT",
        "PPO_EPOCHS",
        "TOTAL_TRAINING_STEPS",
        "TOTAL_EPOCHS",
        "STUDENT_MODEL",
        "TEACHER_MODEL",
        "TEST_FREQ",
        "SAVE_FREQ",
        "LEARNING_RATE",
        "SEED",
        "EXPERIMENT_METHOD",
        "EXPERIMENT_PRESET",
        "MIRROR_OBJECTIVE",
        "MIRROR_ESTIMATOR",
        "MIRROR_KAPPA",
        "MIRROR_ETA",
        "MIRROR_DIAGNOSTICS",
        "OPD_ADVANTAGE_SOURCE",
        "OPD_POLICY_LOSS_MODE",
        "FULL_KL_BACKEND",
        "FULL_KL_TOKEN_CHUNK",
        "FULL_KL_VOCAB_CHUNK",
        "FULL_KL_ASYNC_WRITE",
        "FULL_KL_PREFETCH",
        "FULL_KL_TERMINATION_TOKEN_IDS",
    ):
        env.pop(name, None)
    env.update(
        DRY_RUN="1",
        VERL_USE_UV="0",
        PYTHON_BIN=sys.executable,
        MODEL_ROOT=str(tmp_path / "models with spaces"),
        DATA_ROOT=str(tmp_path / "data"),
        RUN_DIR=str(tmp_path / "run"),
        EXPERIMENT_NAME="test",
        TRAIN_BATCH_SIZE="64",
    )
    env.update(overrides)
    default_args = ["custom"] if script.name == SCRIPT.name else []
    return subprocess.run(command or ["bash", str(script), *default_args], env=env, capture_output=True, text=True)


def _config(tmp_path, **overrides):
    result = _launch(tmp_path, **overrides)
    assert result.returncode == 0, result.stderr
    command = shlex.split(result.stdout)
    with initialize_config_dir(config_dir=str(ROOT / "verl/trainer/config"), version_base=None):
        return compose(config_name="ppo_trainer", overrides=command[command.index("verl.trainer.main_ppo") + 1 :])


@pytest.mark.parametrize("domain,enabled", [("math", True), ("code", False), ("if", False)])
def test_validation_repetition_default_scope(tmp_path, domain, enabled):
    cfg = _config(tmp_path, DOMAIN=domain)
    OmegaConf.resolve(cfg)
    validate_config(cfg, use_reference_policy=False, use_critic=False)
    params = cfg.actor_rollout_ref.rollout.val_kwargs.repetition_detection
    assert (params is not None) == enabled
    if enabled:
        assert dict(params) == {"min_pattern_size": 16, "max_pattern_size": 128, "min_count": 8}


def test_validation_repetition_explicit_enable_uses_selected_parameters(tmp_path):
    cfg = _config(tmp_path, VAL_REPETITION_DETECTION="true")
    assert dict(cfg.actor_rollout_ref.rollout.val_kwargs.repetition_detection) == {
        "min_pattern_size": 16,
        "max_pattern_size": 128,
        "min_count": 8,
    }


def test_validation_repetition_can_be_tuned_and_disabled(tmp_path):
    cfg = _config(
        tmp_path,
        VAL_REPETITION_DETECTION="true",
        VAL_REPETITION_MIN_PATTERN="8",
        VAL_REPETITION_MAX_PATTERN="128",
        VAL_REPETITION_MIN_COUNT="6",
    )
    assert dict(cfg.actor_rollout_ref.rollout.val_kwargs.repetition_detection) == {
        "min_pattern_size": 8,
        "max_pattern_size": 128,
        "min_count": 6,
    }
    cfg = _config(tmp_path, VAL_REPETITION_DETECTION="false")
    assert cfg.actor_rollout_ref.rollout.val_kwargs.repetition_detection is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"VAL_REPETITION_MIN_COUNT": "1"},
        {"VAL_REPETITION_MIN_PATTERN": "512"},
        {"VAL_REPETITION_DETECTION": "maybe"},
        {"VAL_REPETITION_MAX_PATTERN": "oops"},
    ],
)
def test_validation_repetition_invalid_environment_fails_before_launch(tmp_path, overrides):
    assert _launch(tmp_path, **({"VAL_REPETITION_DETECTION": "true"} | overrides)).returncode == 2


@pytest.mark.parametrize("preset,kappa,eta", [("E55", 0, 0.1), ("E56", 0.5, 0.1), ("E57", 1, 0.1), ("E58", 0, 1)])
def test_moe_long_presets_resolve_native_full_vocab_without_checkpoint_saving(tmp_path, preset, kappa, eta):
    cfg = _config(tmp_path, command=["bash", str(SCRIPT), preset])
    OmegaConf.resolve(cfg)
    validate_config(cfg, use_reference_policy=False, use_critic=False)
    loss = omega_conf_to_dataclass(cfg.distillation).distillation_loss
    assert loss.loss_settings.kappa == kappa and loss.loss_settings.eta == eta
    assert loss.loss_settings.teacher_only_kl == (preset == "E58")
    assert not loss.use_policy_gradient and not loss.use_task_rewards
    assert loss.full_vocab.termination_token_ids == [151643, 151645]
    assert loss.full_vocab.backend == "precompute" and loss.full_vocab.token_chunk_size == 512
    assert cfg.actor_rollout_ref.model.path.endswith("/" + _model_suffix("Qwen3-1.7B-Base"))
    assert cfg.distillation.teacher_models.teacher_model.model_path.endswith(
        "/" + _model_suffix("Qwen3-30B-A3B-Instruct-2507")
    )
    assert cfg.distillation.execution_mode == "shared_sync" and cfg.trainer.n_gpus_per_node == 8
    assert cfg.data.apply_chat_template_kwargs.enable_thinking is None
    assert (cfg.data.max_prompt_length, cfg.data.max_response_length) == (1024, 16384)
    actor, rollout = cfg.actor_rollout_ref.actor, cfg.actor_rollout_ref.rollout
    assert cfg.data.train_batch_size == actor.ppo_mini_batch_size == 64
    assert actor.ppo_epochs == 16 and actor.optim.lr == 1e-6 and actor.loss_agg_mode == "token-mean"
    assert rollout.n == 1 and rollout.temperature == rollout.top_p == 1
    assert rollout.max_num_seqs == 64 and rollout.gpu_memory_utilization == 0.8
    assert cfg.trainer.total_training_steps == 100
    assert cfg.trainer.save_freq == -1 and cfg.trainer.test_freq == 10
    assert dict(rollout.val_kwargs.repetition_detection) == {
        "min_pattern_size": 16,
        "max_pattern_size": 128,
        "min_count": 8,
    }
    assert "repetition_detection" not in rollout
    assert not cfg.trainer.val_before_train and cfg.trainer.resume_mode == "disable"
    assert rollout.val_kwargs.n == 16 and rollout.val_kwargs.max_tokens == 31744
    assert (rollout.val_kwargs.temperature, rollout.val_kwargs.top_p) == (0.7, 0.9)
    assert [Path(p).parent.name for p in cfg.data.val_files] == ["MATH-500", "AMC23", "AIME24", "AIME25", "AIME26"]
    assert not (tmp_path / "run").exists()


FAILURE_PAIRS = [
    ("E15", "E13", "DeepSeek-R1-Distill-Qwen-1.5B", "JustRL-DeepSeek-1.5B", 7168, 4096, 128, 64),
    ("E23", "E24", "DeepSeek-R1-Distill-Qwen-1.5B", "JustRL-DeepSeek-1.5B", 15360, 4096, 128, 64),
    ("E25", "E26", "DeepSeek-R1-Distill-Qwen-1.5B", "DeepSeek-R1-Distill-Qwen-7B", 7168, 4096, 128, 32),
    ("E27", "E28", "Qwen3-1.7B-Base", "Qwen3-8B", 7168, 1024, 64, 64),
    ("E29", "E30", "Qwen3-1.7B-Base", "Qwen3-4B", 7168, 1024, 64, 64),
    ("E31", "E32", "Qwen3-1.7B", "Qwen3-4B-Instruct-2507", 4096, 4096, 64, 64),
]


@pytest.mark.parametrize(
    "sampled_preset,full_preset,student,teacher,response,prompt,student_seqs,teacher_seqs", FAILURE_PAIRS
)
@pytest.mark.parametrize("mode", ["sampled", "full"])
def test_failure_training_presets_preserve_pair_protocol_and_update_budget(
    tmp_path, sampled_preset, full_preset, student, teacher, response, prompt, student_seqs, teacher_seqs, mode
):
    cfg = _config(
        tmp_path,
        command=[
            "bash",
            str(SCRIPT.parent / "run_math_opd_ablation_fsdp2.sh"),
            full_preset if mode == "full" else sampled_preset,
        ],
    )
    OmegaConf.resolve(cfg)
    validate_config(cfg, use_reference_policy=False, use_critic=False)
    actor, rollout = cfg.actor_rollout_ref.actor, cfg.actor_rollout_ref.rollout
    assert cfg.actor_rollout_ref.model.path.endswith("/" + _model_suffix(student))
    assert cfg.distillation.teacher_models.teacher_model.model_path.endswith("/" + _model_suffix(teacher))
    assert (cfg.data.max_prompt_length, cfg.data.max_response_length) == (prompt, response)
    assert rollout.max_model_len == prompt + 31744
    assert rollout.max_num_seqs == student_seqs
    assert rollout.gpu_memory_utilization == (0.8 if student == "Qwen3-1.7B-Base" else 0.65)
    assert cfg.data.train_batch_size == actor.ppo_mini_batch_size == 64
    assert actor.ppo_epochs == rollout.n == 1
    assert cfg.trainer.total_training_steps == 250 and cfg.trainer.total_epochs == 1
    assert cfg.trainer.test_freq == 20 and cfg.trainer.save_freq == 100 and cfg.trainer.val_before_train
    assert cfg.trainer.max_actor_ckpt_to_keep == 4 and cfg.trainer.resume_mode == "disable"
    assert actor.optim.lr == 1e-6 and actor.optim.lr_warmup_steps == 0
    assert actor.fsdp_config.seed == actor.data_loader_seed == rollout.seed == cfg.data.seed == 42
    assert rollout.val_kwargs.n == 16 and rollout.val_kwargs.max_tokens == 31744
    assert cfg.data.val_batch_size is None and len(cfg.data.val_files) == 5
    loss = cfg.distillation.distillation_loss
    if mode == "full":
        _assert_full_kl(loss)
    else:
        assert loss.loss_mode == "k1"
    assert loss.use_policy_gradient == (mode == "sampled")
    assert loss.advantage_source == "current" and loss.policy_loss_mode == "vanilla"
    assert not loss.use_task_rewards and not actor.use_kl_loss and actor.entropy_coeff == 0
    if mode == "full":
        assert loss.full_vocab.backend == "precompute" and loss.full_vocab.async_write and loss.full_vocab.prefetch
        assert not rollout.calculate_log_probs and not actor.calculate_entropy
    if student.startswith("Qwen3"):
        assert cfg.data.continuous_token.model_family == "qwen3"
        assert cfg.data.apply_chat_template_kwargs.enable_thinking is False
    else:
        assert cfg.data.continuous_token.model_family == "deepseek"
    assert list(rollout.stop_token_ids or []) == ([151645] if student.endswith("Base") else [])


@pytest.mark.parametrize(
    "sampled_preset,full_preset,student,teacher,response,prompt,student_seqs,teacher_seqs", FAILURE_PAIRS
)
@pytest.mark.parametrize("target", ["student", "teacher"])
def test_failure_validation_selects_generator_resources_without_training(
    tmp_path, sampled_preset, full_preset, student, teacher, response, prompt, student_seqs, teacher_seqs, target
):
    cfg = _config(
        tmp_path,
        command=[
            "bash",
            str(SCRIPT.parent / "run_math_opd_ablation_fsdp2.sh"),
            f"{full_preset if target == 'teacher' else sampled_preset}-valid-{target}",
        ],
    )
    OmegaConf.resolve(cfg)
    validate_config(cfg, use_reference_policy=False, use_critic=False)
    model = teacher if target == "teacher" else student
    assert cfg.actor_rollout_ref.model.path.endswith("/" + _model_suffix(model))
    assert cfg.trainer.val_only and cfg.trainer.val_before_train and cfg.trainer.save_freq == -1
    assert not cfg.distillation.enabled and cfg.distillation.n_gpus_per_node == 0
    rollout = cfg.actor_rollout_ref.rollout
    assert rollout.max_num_seqs == (teacher_seqs if target == "teacher" else student_seqs)
    assert rollout.gpu_memory_utilization == (0.8 if target == "student" and student == "Qwen3-1.7B-Base" else 0.65)
    assert cfg.data.max_prompt_length == (4096 if target == "teacher" else prompt)
    assert rollout.max_model_len == cfg.data.max_prompt_length + 31744
    assert list(rollout.stop_token_ids or []) == ([151645] if model.endswith("Base") else [])
    assert not rollout.calculate_log_probs
    if model.startswith("Qwen3"):
        assert cfg.data.apply_chat_template_kwargs.enable_thinking is False


def test_numbered_failure_preset_and_explicit_overrides(tmp_path):
    cfg = _config(
        tmp_path,
        command=[
            "bash",
            str(SCRIPT.parent / "run_math_opd_ablation_fsdp2.sh"),
            "E27",
            "trainer.test_freq=1",
            "actor_rollout_ref.rollout.stop_token_ids=[]",
            "actor_rollout_ref.rollout.gpu_memory_utilization=0.6",
        ],
        STUDENT_MODEL="/custom student",
        TEACHER_MODEL="/custom teacher",
        MAX_RESPONSE_LENGTH="1024",
        ROLLOUT_MAX_NUM_SEQS="8",
        ROLLOUT_GPU_MEMORY_UTILIZATION="0.55",
        SEED="43",
        TOTAL_TRAINING_STEPS="2",
        TEST_FREQ="10",
    )
    assert cfg.actor_rollout_ref.model.path == "/custom student"
    assert cfg.distillation.teacher_models.teacher_model.model_path == "/custom teacher"
    assert cfg.distillation.distillation_loss.loss_mode == "k1"
    assert cfg.trainer.total_training_steps == 2 and cfg.trainer.test_freq == 1
    assert cfg.actor_rollout_ref.rollout.max_num_seqs == 8
    assert cfg.actor_rollout_ref.rollout.gpu_memory_utilization == 0.6
    assert cfg.data.max_response_length == 1024 and cfg.data.seed == 43
    assert list(cfg.actor_rollout_ref.rollout.stop_token_ids or []) == []


@pytest.mark.parametrize("setting", ["E25", "E27"])
def test_failure_smoke_reuses_formal_pair_and_stop_protocol(tmp_path, setting):
    cfg = _config(
        tmp_path,
        command=["bash", str(SCRIPT.parent / "smoke/smoke_failure_sampled.sh"), setting],
    )
    assert cfg.trainer.total_training_steps == 2 and not cfg.trainer.val_before_train
    assert cfg.actor_rollout_ref.rollout.max_num_seqs == 8 and cfg.data.max_response_length == 512
    assert list(cfg.actor_rollout_ref.rollout.stop_token_ids or []) == ([151645] if setting == "E27" else [])


@pytest.mark.parametrize("preset", ["E21-unknown", "E99", "E43-unknown", "A", "A-full"])
def test_unknown_failure_preset_fails_without_launching(tmp_path, preset):
    result = _launch(tmp_path, command=["bash", str(SCRIPT.parent / "run_math_opd_ablation_fsdp2.sh"), preset])
    assert result.returncode == 2


@pytest.mark.parametrize("canonical,alias", [("E13", "E22"), ("E15", "E21")])
@pytest.mark.parametrize("suffix", ["", "-valid-student", "-valid-teacher"])
def test_failure_baseline_aliases_resolve_to_identical_configs(tmp_path, canonical, alias, suffix):
    configs = [
        _config(tmp_path, command=["bash", str(SCRIPT.parent / "run_math_opd_ablation_fsdp2.sh"), preset + suffix])
        for preset in (canonical, alias)
    ]
    assert OmegaConf.to_container(configs[0], resolve=True) == OmegaConf.to_container(configs[1], resolve=True)


REUSE_PRESETS = [
    ("E15", "E33"),
    ("E13", "E17"),
    ("E23", "E34"),
    ("E24", "E35"),
    ("E25", "E36"),
    ("E26", "E37"),
    ("E27", "E38"),
    ("E28", "E39"),
    ("E29", "E40"),
    ("E30", "E41"),
    ("E31", "E42"),
    ("E32", "E43"),
]


@pytest.mark.parametrize("base,reuse", REUSE_PRESETS)
def test_failure_reuse16_preserves_budget_and_declares_e41_termination_override(tmp_path, base, reuse):
    configs = [
        _config(tmp_path, command=["bash", str(SCRIPT.parent / "run_math_opd_ablation_fsdp2.sh"), preset])
        for preset in (base, reuse)
    ]
    first, repeated = configs
    assert first.actor_rollout_ref.actor.ppo_epochs == 1
    assert repeated.actor_rollout_ref.actor.ppo_epochs == 16
    shortened = reuse in ("E41", "E43")
    assert repeated.trainer.total_training_steps == (100 if shortened else 250)
    assert repeated.trainer.total_training_steps * repeated.actor_rollout_ref.actor.ppo_epochs == (
        1600 if shortened else 4000
    )
    assert repeated.trainer.save_freq == (50 if shortened else 100)
    assert repeated.trainer.test_freq == (10 if shortened else 20)
    assert repeated.trainer.val_before_train == (reuse != "E41")
    if shortened:
        assert repeated.data.max_response_length == 7168
        assert repeated.actor_rollout_ref.rollout.max_num_seqs == 64
        assert repeated.actor_rollout_ref.rollout.gpu_memory_utilization == 0.8
        for field in ("total_training_steps", "save_freq", "test_freq", "val_before_train"):
            OmegaConf.update(repeated, f"trainer.{field}", first.trainer[field])
        OmegaConf.update(repeated, "data.max_response_length", first.data.max_response_length)
        OmegaConf.update(
            repeated,
            "actor_rollout_ref.rollout.gpu_memory_utilization",
            first.actor_rollout_ref.rollout.gpu_memory_utilization,
        )
        OmegaConf.update(
            repeated,
            "distillation.teacher_models.teacher_model.inference.max_model_len",
            first.distillation.teacher_models.teacher_model.inference.max_model_len,
        )
    OmegaConf.update(repeated, "actor_rollout_ref.actor.ppo_epochs", 1)
    if reuse == "E41":
        assert list(repeated.distillation.distillation_loss.full_vocab.termination_token_ids) == [151643, 151645]
        OmegaConf.update(repeated, "distillation.distillation_loss.full_vocab.termination_token_ids", [])
    assert OmegaConf.to_container(first, resolve=True) == OmegaConf.to_container(repeated, resolve=True)


def test_e41_termination_configuration_and_override_precedence(tmp_path):
    command = ["bash", str(SCRIPT.parent / "run_math_opd_ablation_fsdp2.sh"), "E41"]
    cfg = _config(tmp_path, command=command)
    OmegaConf.resolve(cfg)
    validate_config(cfg, use_reference_policy=False, use_critic=False)
    assert list(cfg.distillation.distillation_loss.full_vocab.termination_token_ids) == [151643, 151645]
    disabled = _config(tmp_path, command=command, FULL_KL_TERMINATION_TOKEN_IDS="[]")
    assert not disabled.distillation.distillation_loss.full_vocab.termination_token_ids
    disabled = _config(
        tmp_path, command=command + ["distillation.distillation_loss.full_vocab.termination_token_ids=[]"]
    )
    assert not disabled.distillation.distillation_loss.full_vocab.termination_token_ids
    for override in [
        "actor_rollout_ref.rollout.ignore_eos=True",
        "distillation.distillation_loss.full_vocab.backend=dense",
    ]:
        invalid = _config(tmp_path, command=command + [override])
        with pytest.raises(ValueError):
            validate_config(invalid, use_reference_policy=False, use_critic=False)
    validation = _config(
        tmp_path,
        command=["bash", str(SCRIPT.parent / "run_math_opd_ablation_fsdp2.sh"), "E41-valid-student"],
        FULL_KL_TERMINATION_TOKEN_IDS="[151643,151645]",
    )
    assert not validation.distillation.enabled
    assert not validation.distillation.distillation_loss.full_vocab.termination_token_ids
    invalid = _config(
        tmp_path, command=command + ["--method", "sampled"], FULL_KL_TERMINATION_TOKEN_IDS="[151643,151645]"
    )
    with pytest.raises(InstantiationException, match="termination equivalence"):
        validate_config(invalid, use_reference_policy=False, use_critic=False)


@pytest.mark.parametrize("preset", ["E33", "E43"])
def test_failure_reuse_presets_allow_explicit_epoch_and_budget_overrides(tmp_path, preset):
    cfg = _config(
        tmp_path,
        command=["bash", str(SCRIPT.parent / "run_math_opd_ablation_fsdp2.sh"), preset],
        PPO_EPOCHS="2",
        TOTAL_TRAINING_STEPS="3",
        SAVE_FREQ="-1",
    )
    assert cfg.actor_rollout_ref.actor.ppo_epochs == 2 and cfg.trainer.total_training_steps == 3
    assert cfg.trainer.save_freq == -1


def test_launcher_composes_native_single_teacher_baseline(tmp_path):
    cfg = _config(tmp_path, DISTILLATION_EXECUTION_MODE="standalone")
    OmegaConf.resolve(cfg)
    validate_config(cfg, use_reference_policy=False, use_critic=False)
    assert cfg.trainer.use_v1 and cfg.trainer.v1.trainer_mode == "sync"
    assert "opd_metrics" not in cfg.trainer.v1
    assert cfg.trainer.n_gpus_per_node == 4 and cfg.distillation.n_gpus_per_node == 4
    assert cfg.actor_rollout_ref.actor.strategy == "fsdp2"
    assert cfg.data.train_batch_size == cfg.actor_rollout_ref.actor.ppo_mini_batch_size == 64
    assert cfg.data.continuous_token.model_family == "deepseek"
    assert cfg.actor_rollout_ref.actor.ppo_epochs == 1
    assert cfg.actor_rollout_ref.rollout.response_length == 7168
    assert cfg.actor_rollout_ref.rollout.val_kwargs.max_tokens == 31744
    assert cfg.actor_rollout_ref.rollout.val_kwargs.n == 16
    assert cfg.actor_rollout_ref.rollout.max_model_len == 35840
    assert cfg.distillation.teacher_models.teacher_model.inference.max_model_len == 11265
    assert cfg.distillation.distillation_loss.loss_mode == "k1"
    assert not cfg.distillation.distillation_loss.use_task_rewards
    assert cfg.distillation.distillation_loss.loss_max_clamp is None
    assert [Path(p).parent.name for p in cfg.data.val_files] == ["MATH-500", "AMC23", "AIME24", "AIME25", "AIME26"]
    assert cfg.actor_rollout_ref.model.path == str(
        registered_model("DeepSeek-R1-Distill-Qwen-1.5B", root=tmp_path / "models with spaces")
    )
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("gpus", [4, 8])
@pytest.mark.parametrize("entry", ["run_math_opd_ablation_fsdp2.sh", "launchers/_run_single_teacher_fsdp2.sh"])
def test_default_graph_and_validation_window_keep_training_budget(tmp_path, gpus, entry):
    cfg = _config(tmp_path, script=SCRIPT.parent / entry, STUDENT_GPUS=str(gpus))
    assert not cfg.actor_rollout_ref.rollout.enforce_eager
    assert list(cfg.actor_rollout_ref.rollout.cudagraph_capture_sizes) == [1, 2, 4, 8, 16, 32, 64, 128]
    assert cfg.actor_rollout_ref.rollout.max_num_seqs == 128
    assert cfg.actor_rollout_ref.rollout.gpu_memory_utilization == 0.65
    assert cfg.actor_rollout_ref.rollout.max_num_batched_tokens == 8192
    assert cfg.data.val_batch_size is None
    assert cfg.trainer.n_gpus_per_node == gpus
    assert cfg.distillation.n_gpus_per_node == 0
    assert cfg.actor_rollout_ref.rollout.val_kwargs.n == 16
    assert cfg.data.train_batch_size == cfg.actor_rollout_ref.actor.ppo_mini_batch_size == 64


def test_validation_environment_and_final_hydra_overrides(tmp_path):
    cfg = _config(tmp_path, VAL_BATCH_SIZE="null", ROLLOUT_MAX_NUM_SEQS="96", ROLLOUT_GPU_MEMORY_UTILIZATION="0.5")
    assert cfg.data.val_batch_size is None
    assert cfg.actor_rollout_ref.rollout.max_num_seqs == 96
    assert cfg.actor_rollout_ref.rollout.gpu_memory_utilization == 0.5
    assert cfg.actor_rollout_ref.rollout.cudagraph_capture_sizes[-1] == 96
    cfg = _config(
        tmp_path,
        command=[
            "bash",
            str(SCRIPT),
            "custom",
            "data.val_batch_size=64",
            "actor_rollout_ref.rollout.gpu_memory_utilization=0.6",
        ],
        VAL_BATCH_SIZE="null",
        ROLLOUT_GPU_MEMORY_UTILIZATION="0.5",
    )
    assert cfg.data.val_batch_size == 64
    assert cfg.actor_rollout_ref.rollout.gpu_memory_utilization == 0.6


@pytest.mark.parametrize("window", ["0", "-1", "1.5", "None", "false"])
def test_invalid_validation_window_is_rejected_before_startup(tmp_path, window):
    result = _launch(tmp_path, VAL_BATCH_SIZE=window)
    assert result.returncode == 2
    assert "VAL_BATCH_SIZE must be null or a positive integer" in result.stderr


def test_graph_capture_covers_concurrency_override_and_eager_fallback(tmp_path):
    cfg = _config(tmp_path, ROLLOUT_MAX_NUM_SEQS="64", ROLLOUT_ENFORCE_EAGER="True", VAL_BATCH_SIZE="16")
    assert cfg.actor_rollout_ref.rollout.enforce_eager
    assert list(cfg.actor_rollout_ref.rollout.cudagraph_capture_sizes) == [1, 2, 4, 8, 16, 32, 64]
    assert cfg.data.val_batch_size == 16


def test_zero_rollout_concurrency_is_rejected_before_capture_derivation(tmp_path):
    result = _launch(tmp_path, ROLLOUT_MAX_NUM_SEQS="0")
    assert result.returncode == 2
    assert "ROLLOUT_MAX_NUM_SEQS must be a positive integer" in result.stderr


@pytest.mark.parametrize(
    "domain,teacher,train,validation",
    [
        ("code", "DeepCoder-1.5B-Preview", "code/TACO/train.parquet", ["LCBv6"]),
        ("if", "UltraData-IF-1.5B", "if/MultiIF/train.parquet", []),
    ],
)
def test_domain_launchers_select_teacher_and_dataset_protocol(tmp_path, domain, teacher, train, validation):
    script = ROOT / f"examples/mirroropd/run_{domain}_deepseek_r1_1_5b_fsdp2.sh"
    cfg = _config(tmp_path, script=script)
    OmegaConf.resolve(cfg)
    validate_config(cfg, use_reference_policy=False, use_critic=False)
    assert cfg.distillation.teacher_models.teacher_model.model_path.endswith(_model_suffix(teacher))
    assert cfg.data.train_files == [str(tmp_path / "data" / train)]
    assert [Path(p).parent.name for p in cfg.data.val_files] == validation
    assert cfg.data.continuous_token.model_family == "deepseek"
    assert cfg.trainer.n_gpus_per_node == 8 and cfg.distillation.n_gpus_per_node == 0
    assert cfg.data.val_batch_size == 32
    assert cfg.actor_rollout_ref.rollout.max_num_seqs == 32
    assert cfg.actor_rollout_ref.rollout.gpu_memory_utilization == 0.5
    if domain == "if":
        assert cfg.trainer.test_freq == -1
        assert not cfg.trainer.val_before_train
        assert cfg.actor_rollout_ref.rollout.max_model_len == 11264


def test_performance_controls_reach_student_and_teacher(tmp_path):
    cfg = _config(
        tmp_path,
        DISTILLATION_EXECUTION_MODE="standalone",
        ROLLOUT_ENFORCE_EAGER="False",
        TEACHER_ENFORCE_EAGER="False",
        PARAM_OFFLOAD="False",
        OPTIMIZER_OFFLOAD="False",
        GRADIENT_CHECKPOINTING="False",
        ROLLOUT_TOKEN_BUDGET="16384",
        TEACHER_TOKEN_BUDGET="16384",
    )
    assert not cfg.actor_rollout_ref.rollout.enforce_eager
    assert not cfg.distillation.teacher_models.teacher_model.inference.enforce_eager
    assert not cfg.actor_rollout_ref.actor.fsdp_config.param_offload
    assert not cfg.actor_rollout_ref.actor.fsdp_config.optimizer_offload
    assert not cfg.actor_rollout_ref.model.enable_gradient_checkpointing
    assert cfg.actor_rollout_ref.rollout.max_num_batched_tokens == 16384
    assert cfg.distillation.teacher_models.teacher_model.inference.max_num_batched_tokens == 16384


def test_if_checker_environment_reaches_ray_workers(tmp_path):
    cfg = _config(
        tmp_path,
        script=ROOT / "examples/mirroropd/run_if_deepseek_r1_1_5b_fsdp2.sh",
        OPENCOMPASS_ROOT=str(tmp_path / "opencompass source"),
        NLTK_DATA=str(tmp_path / "nltk_data"),
    )
    variables = cfg.ray_kwargs.ray_init.runtime_env.env_vars
    assert variables.OPENCOMPASS_ROOT == str(tmp_path / "opencompass source")
    assert variables.NLTK_DATA == str(tmp_path / "nltk_data")


def test_if_rejects_single_turn_validation(tmp_path):
    result = _launch(tmp_path, script=ROOT / "examples/mirroropd/run_if_deepseek_r1_1_5b_fsdp2.sh", TEST_FREQ="20")
    assert result.returncode == 2
    assert "three-turn" in result.stderr


def test_if_checks_checker_source_before_starting_training(tmp_path):
    train = tmp_path / "data/if/MultiIF/train.parquet"
    train.parent.mkdir(parents=True)
    train.touch()
    result = _launch(
        tmp_path,
        script=ROOT / "examples/mirroropd/run_if_deepseek_r1_1_5b_fsdp2.sh",
        DRY_RUN="0",
        CONFIG_ONLY="0",
        CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7",
        OPENCOMPASS_ROOT=str(tmp_path / "missing"),
    )
    assert result.returncode == 2
    assert "OPENCOMPASS_ROOT" in result.stderr
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("teacher_gpus,teacher_tp,replicas", [(4, 1, 4), (4, 2, 2), (1, 1, 1)])
def test_native_teacher_config_uses_entire_requested_pool(tmp_path, teacher_gpus, teacher_tp, replicas):
    cfg = _config(
        tmp_path,
        DISTILLATION_EXECUTION_MODE="standalone",
        TEACHER_GPUS=str(teacher_gpus),
        TEACHER_TP=str(teacher_tp),
        CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7",
    )
    distillation = omega_conf_to_dataclass(cfg.distillation)
    assert list(distillation.teacher_models) == ["default"]
    teacher = distillation.teacher_models["default"]
    assert teacher.num_replicas == replicas
    assert teacher.per_replica_world_size == teacher_tp
    assert teacher.world_size == teacher_gpus


@pytest.mark.parametrize(
    "overrides",
    [
        {"STUDENT_GPUS": "3"},
        {"DISTILLATION_EXECUTION_MODE": "standalone", "TEACHER_TP": "3"},
        {"MAX_VAL_RESPONSE_LENGTH": "0"},
        {"CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6"},
    ],
)
def test_launcher_rejects_invalid_resource_or_budget_settings(tmp_path, overrides):
    assert _launch(tmp_path, **overrides).returncode == 2


def test_validation_context_checked_before_worker_startup(tmp_path):
    cfg = _config(tmp_path)
    cfg.actor_rollout_ref.rollout.max_model_len = 11265
    with pytest.raises(ValueError, match="prompt_length"):
        validate_config(cfg, use_reference_policy=False, use_critic=False)


def test_qwen3_base_32k_validation_window_fits_exactly(tmp_path):
    cfg = _config(tmp_path, MAX_PROMPT_LENGTH="1024", MAX_VAL_RESPONSE_LENGTH="31744")
    assert cfg.actor_rollout_ref.rollout.max_model_len == 32768
    assert cfg.distillation.teacher_models.teacher_model.inference.max_model_len == 8193


def test_validation_budget_rejects_default_multiturn_agent(tmp_path):
    cfg = _config(tmp_path)
    cfg.actor_rollout_ref.rollout.agent.default_agent_loop = "tool_agent"
    with pytest.raises(ValueError, match="single_turn_agent"):
        validate_config(cfg, use_reference_policy=False, use_critic=False)


def test_validation_budget_rejects_backend_that_overwrites_request_limit(tmp_path):
    # Isolate the response-budget guard from the validation repetition backend guard.
    cfg = _config(tmp_path, DISTILLATION_EXECUTION_MODE="standalone", VAL_REPETITION_DETECTION="false")
    cfg.actor_rollout_ref.rollout.name = "trtllm"
    with pytest.raises(ValueError, match="vllm or sglang"):
        validate_config(cfg, use_reference_policy=False, use_critic=False)


def test_uv_launch_includes_math_dependencies_for_driver_and_workers(tmp_path):
    result = _launch(tmp_path, VERL_USE_UV="1", DEVICE="gpu")
    assert result.returncode == 0
    command = shlex.split(result.stdout)
    assert command[:2] == ["uv", "run"]
    assert "math" in command[: command.index("-m")]
    worker_python = next(arg for arg in command if arg.startswith("ray_kwargs.ray_init.runtime_env.py_executable="))
    assert "--extra math" in worker_python


def test_default_launch_uses_absolute_native_interpreter():
    """Default launch selection must not depend on activation or the ambient Python PATH."""
    env = os.environ.copy()
    for name in (
        "PYTHON_BIN",
        "VERL_USE_UV",
        "CUDA_VISIBLE_DEVICES",
        "MODEL_ROOT",
        "DATA_ROOT",
        "STUDENT_MODEL",
        "TEACHER_MODEL",
        "TRAIN_FILE",
        "MATH_FILE",
        "AMC_FILE",
        "AIME_FILE",
    ):
        env.pop(name, None)
    env["DRY_RUN"] = "1"
    result = subprocess.run(["bash", str(SCRIPT), "custom"], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    command = shlex.split(result.stdout)
    expected = str(ROOT / ".venv-native/bin/python")
    assert command[0] == expected
    assert f"ray_kwargs.ray_init.runtime_env.py_executable='{expected}'" in command
    model = registered_model("DeepSeek-R1-Distill-Qwen-1.5B", root=ROOT.parent / "model")
    assert f"actor_rollout_ref.model.path='{model}'" in command
    assert any(str(ROOT.parent / "data/math/DAPO/train.parquet") in argument for argument in command)


def test_explicit_native_interpreter_reaches_ray_workers(tmp_path):
    """The Ray runtime override retains the requested absolute environment path."""
    interpreter = str(tmp_path / "custom-env/bin/python")
    cfg = _config(tmp_path, PYTHON_BIN=interpreter)
    assert cfg.ray_kwargs.ray_init.runtime_env.py_executable == interpreter


def test_launcher_fixes_offline_wandb_location_for_ray(tmp_path):
    """The training driver and Ray task runner share a checkout-relative offline log root."""
    cfg = _config(tmp_path, WANDB_DIR=str(tmp_path / "unrelated"), WANDB_MODE="offline")
    assert cfg.trainer.logger == ["console", "wandb"]
    assert cfg.ray_kwargs.ray_init.runtime_env.env_vars.WANDB_DIR == str(ROOT)
    assert cfg.ray_kwargs.ray_init.runtime_env.env_vars.WANDB_MODE == "offline"


def test_real_launch_requires_explicit_gpu_allocation(tmp_path):
    result = _launch(tmp_path, DRY_RUN="0", CONFIG_ONLY="0")
    assert result.returncode == 2
    assert "allocated to this run" in result.stderr
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("exit_code", [0, 23])
@pytest.mark.parametrize("domain", ["math", "code", "if"])
@pytest.mark.parametrize("append_source", [False, True])
def test_source_archive_launch_saves_both_log_streams_and_training_exit_status(
    tmp_path, exit_code, domain, append_source
):
    """An unpacked tree logs stdout/stderr while retaining the trainer's success or failure."""
    checkout = tmp_path / "source checkout"
    entry = SCRIPT if domain == "math" else SCRIPT.with_name(f"run_{domain}_deepseek_r1_1_5b_fsdp2.sh")
    script = checkout / "examples/mirroropd" / entry.name
    script.parent.mkdir(parents=True)
    shutil.copy2(entry, script)
    shutil.copytree(SCRIPT.parent / "launchers", script.parent / "launchers")
    shutil.copytree(SCRIPT.parent / "assets", script.parent / "assets")
    datasets = {
        "math": [
            "math/DAPO/train",
            "math/MATH-500/test",
            "math/AMC23/test",
            "math/AIME24/test",
            "math/AIME25/test",
            "math/AIME26/test",
        ],
        "code": ["code/TACO/train", "code/LCBv6/test"],
        "if": ["if/MultiIF/train"],
    }
    for name in datasets[domain]:
        path = tmp_path / "data" / f"{name}.parquet"
        path.parent.mkdir(parents=True)
        path.touch()
    checker = tmp_path / "checker source"
    (checker / "opencompass/datasets/MultiIF").mkdir(parents=True)
    interpreter = tmp_path / "mock-python"
    interpreter.write_text(
        '#!/bin/sh\ncase " $* " in *" -c "*) exit 0 ;; *" --cfg "*) printf "mock: true\\n"; exit 0 ;; esac\n'
        'printf "mock training stdout\\n"\nprintf "mock training stderr\\n" >&2\n'
        # A running shell may resume reading the file after its child exits.
        # Updating the launcher must not execute newly appended source text.
        'if [ "$MUTATE_LAUNCHER" = 1 ]; then\n'
        '  printf \'\\nprintf "unexpected" > "$RUN_DIR/unexpected"\\n\' >> "$LAUNCHER_TO_CHANGE"\n'
        "fi\n"
        f"exit {exit_code}\n"
    )
    interpreter.chmod(0o755)
    result = _launch(
        tmp_path,
        script=script,
        DRY_RUN="0",
        CONFIG_ONLY="0",
        PYTHON_BIN=str(interpreter),
        CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7",
        OPENCOMPASS_ROOT=str(checker),
        MUTATE_LAUNCHER="1" if append_source else "0",
        LAUNCHER_TO_CHANGE=str(script.parent / "launchers/_run_single_teacher_fsdp2.sh"),
    )
    assert result.returncode == exit_code, result.stderr
    assert not (tmp_path / "run/unexpected").exists()
    assert "source archive without Git metadata" in (tmp_path / "run/source_commit.txt").read_text()
    assert (tmp_path / "run/source_changes.patch").read_bytes() == b""
    assert "verl.trainer.main_ppo" in (tmp_path / "run/command.sh").read_text()
    log = (tmp_path / "run/train.log").read_text()
    for message in ("mock training stdout", "mock training stderr"):
        assert message in log
        assert message in result.stdout


def _local_hope_command(worker_command=None):
    if worker_command is None:
        config = ConfigParser(interpolation=None)
        config.read(ROOT / "hope/mirroropd/mirroropd.hope")
        worker_command = config.get("roles", "worker.script")
    remote_root = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/MirrorOPD-verl"
    assert remote_root in worker_command
    return shlex.split(worker_command.replace(remote_root, str(ROOT)))


@pytest.mark.parametrize("devices,expected_code", [(None, 0), ("GPU-a,GPU-b", 2)])
def test_direct_hope_command_preserves_allocation_or_uses_job_local_fallback(tmp_path, devices, expected_code):
    overrides = {} if devices is None else {"CUDA_VISIBLE_DEVICES": devices}
    result = _launch(tmp_path, command=_local_hope_command(), **overrides)
    assert result.returncode == expected_code, result.stderr
    if expected_code:
        assert "CUDA_VISIBLE_DEVICES does not cover" in result.stderr


@pytest.mark.parametrize("domain", ["math", "code", "if"])
@pytest.mark.parametrize("mode", ["shared_sync", "standalone"])
def test_hope_submission_and_worker_preview_match_eight_gpu_allocation(tmp_path, domain, mode):
    """Exercise the real Make submission and worker config while stubbing only the Hope CLI."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    hope = fake_bin / "hope"
    record = tmp_path / "submission.txt"
    hope.write_text('#!/bin/sh\nprintf "%s\\n" "$PWD" "$@" > "$HOPE_SUBMIT_RECORD"\n')
    hope.chmod(0o755)
    env = os.environ.copy()
    env.update(PATH=str(fake_bin) + os.pathsep + env["PATH"], HOPE_SUBMIT_RECORD=str(record))
    submitted = subprocess.run(["make", "opd-submit"], cwd=ROOT, env=env, capture_output=True, text=True)
    assert submitted.returncode == 0, submitted.stderr
    directory, action, filename = record.read_text().splitlines()
    assert action == "run"
    config = ConfigParser(interpolation=None)
    config.read(Path(directory) / filename)
    assert config.getint("roles", "workers") == 1
    allocated_gpus = config.getint("roles", "worker.gcores80g")

    # Execute the configured worker command itself, translating only the remote mount.
    worker_command = config.get("roles", "worker.script")
    assert "hope/mirroropd/run.sh" not in worker_command
    assert "STUDENT_GPUS=" not in worker_command and "TEACHER_GPUS=" not in worker_command
    preview = _launch(
        tmp_path,
        command=_local_hope_command(worker_command),
        PYTHON_BIN="",
        DISTILLATION_EXECUTION_MODE=mode,
        DOMAIN=domain,
        CUDA_VISIBLE_DEVICES="GPU-0,GPU-1,GPU-2,GPU-3,GPU-4,GPU-5,GPU-6,GPU-7",
        VERL_USE_UV="1",
    )
    assert preview.returncode == 0, preview.stderr
    command = shlex.split(preview.stdout)
    native_python = str(ROOT / ".venv-native/bin/python")
    assert command[0] == native_python
    with initialize_config_dir(config_dir=str(ROOT / "verl/trainer/config"), version_base=None):
        cfg = compose(config_name="ppo_trainer", overrides=command[command.index("verl.trainer.main_ppo") + 1 :])
    assert cfg.trainer.n_gpus_per_node + cfg.distillation.n_gpus_per_node == allocated_gpus == 8
    assert cfg.distillation.execution_mode == mode
    assert cfg.trainer.n_gpus_per_node == (8 if mode == "shared_sync" else 4)
    assert cfg.distillation.n_gpus_per_node == (0 if mode == "shared_sync" else 4)
    assert cfg.ray_kwargs.ray_init.runtime_env.py_executable == native_python
    assert cfg.ray_kwargs.ray_init.runtime_env.env_vars.WANDB_MODE == "offline"
    teachers = {"math": "JustRL-DeepSeek-1.5B", "code": "DeepCoder-1.5B-Preview", "if": "UltraData-IF-1.5B"}
    assert cfg.distillation.teacher_models.teacher_model.model_path.endswith(_model_suffix(teachers[domain]))
    assert not (tmp_path / "run").exists()


def test_default_shared_pipeline_and_teacher_forward_controls(tmp_path):
    cfg = _config(tmp_path, TEACHER_FSDP_TOKEN_BUDGET="16384", TEACHER_FSDP_DTYPE="bfloat16")
    OmegaConf.resolve(cfg)
    validate_config(cfg, use_reference_policy=False, use_critic=False)
    distillation = omega_conf_to_dataclass(cfg.distillation)
    assert distillation.execution_mode == "shared_sync"
    assert cfg.trainer.n_gpus_per_node == 8
    assert distillation.n_gpus_per_node == distillation.nnodes == 0
    assert distillation.shared_teacher.infer_max_token_len_per_gpu == 16384
    assert distillation.shared_teacher.model_dtype == "bfloat16"
    assert distillation.shared_teacher.mixed_precision["param_dtype"] == "bfloat16"
    assert distillation.teacher_models["default"].num_replicas == 0
    # Shared configuration does not rewrite the vLLM prompt/response budget.
    assert distillation.teacher_models["default"].inference.response_length == 7168


@pytest.mark.parametrize(
    "overrides",
    [
        {"DISTILLATION_EXECUTION_MODE": "invalid"},
        {"TEACHER_GPUS": "4"},
        {"ROLLOUT_TP": "2"},
        {"PARAM_OFFLOAD": "False"},
        {"OPTIMIZER_OFFLOAD": "False"},
        {"TEACHER_FSDP_TOKEN_BUDGET": "8192"},
    ],
)
def test_shared_launcher_rejects_incompatible_settings(tmp_path, overrides):
    assert _launch(tmp_path, **overrides).returncode == 2


@pytest.mark.parametrize(
    "group,source,mode",
    [
        ("E1", "rollout", "token_ppo"),
        ("E2", "rollout", "token_is"),
        ("E3", "current", "token_ppo"),
        ("E4", "current", "token_is"),
        ("E5", "current", "reinforce"),
        ("E6", "rollout", "reinforce"),
        ("E7", "rollout", "token_ppo"),
        ("E8", "rollout", "token_is"),
        ("E9", "current", "token_ppo"),
        ("E10", "current", "token_is"),
        ("E11", "current", "reinforce"),
        ("E12", "rollout", "reinforce"),
    ],
)
@pytest.mark.parametrize("execution_mode", ["shared_sync", "standalone"])
def test_ablation_launcher_composes_supported_modes(tmp_path, group, source, mode, execution_mode):
    script = ROOT / "examples/mirroropd/run_math_opd_ablation_fsdp2.sh"
    result = _launch(tmp_path, command=["bash", str(script), group], DISTILLATION_EXECUTION_MODE=execution_mode)
    assert result.returncode == 0, result.stderr
    command = shlex.split(result.stdout)
    with initialize_config_dir(config_dir=str(ROOT / "verl/trainer/config"), version_base=None):
        cfg = compose(config_name="ppo_trainer", overrides=command[command.index("verl.trainer.main_ppo") + 1 :])
    OmegaConf.resolve(cfg)
    validate_config(cfg, use_reference_policy=False, use_critic=False)
    assert cfg.distillation.distillation_loss.advantage_source == source
    assert cfg.distillation.distillation_loss.policy_loss_mode == mode
    if int(group[1:]) <= 6:
        assert cfg.actor_rollout_ref.actor.ppo_epochs == 4
        assert cfg.trainer.total_training_steps == cfg.trainer.save_freq == cfg.trainer.test_freq == 32
    else:
        assert cfg.actor_rollout_ref.actor.ppo_epochs == 16
        assert cfg.trainer.total_training_steps == 200
        assert cfg.trainer.test_freq == 50
        assert cfg.trainer.save_freq == 100
    assert not cfg.trainer.val_before_train
    assert cfg.trainer.v1.trainer_mode == "sync"
    assert cfg.trainer.v1.sync.get("parameter_sync_step", 1) == 1
    assert cfg.actor_rollout_ref.actor.optim.lr_warmup_steps == 0
    assert not cfg.actor_rollout_ref.actor.checkpoint.async_save


@pytest.mark.parametrize("group", ["E1", "E10"])
def test_ablation_launcher_accepts_budget_clip_epochs_and_final_overrides(tmp_path, group):
    script = ROOT / "examples/mirroropd/run_math_opd_ablation_fsdp2.sh"
    result = _launch(
        tmp_path,
        command=["bash", str(script), group, "trainer.total_training_steps=7"],
        TOTAL_TRAINING_STEPS="200",
        PPO_EPOCHS="2",
        TEST_FREQ="200",
        SAVE_FREQ="200",
        OPD_CLIP_EPS="0.1",
    )
    command = shlex.split(result.stdout)
    with initialize_config_dir(config_dir=str(ROOT / "verl/trainer/config"), version_base=None):
        cfg = compose(config_name="ppo_trainer", overrides=command[command.index("verl.trainer.main_ppo") + 1 :])
    assert cfg.trainer.total_training_steps == 7
    assert cfg.trainer.save_freq == cfg.trainer.test_freq == 200
    assert cfg.actor_rollout_ref.actor.ppo_epochs == 2
    loss = cfg.distillation.distillation_loss
    assert loss.clip_ratio == loss.clip_ratio_low == loss.clip_ratio_high == 0.1


@pytest.mark.parametrize(
    "group,lr,epochs",
    [("E13", 1e-6, 1), ("E14", 5e-7, 1), ("E15", 1e-6, 1), ("E16", 1e-6, 4), ("E17", 1e-6, 16)],
)
def test_full_kl_and_vanilla_presets(tmp_path, group, lr, epochs):
    script = SCRIPT.parent / "run_math_opd_ablation_fsdp2.sh"
    cfg = _config(tmp_path, command=["bash", str(script), group], DOMAIN="code")
    OmegaConf.resolve(cfg)
    validate_config(cfg, use_reference_policy=False, use_critic=False)
    distillation = omega_conf_to_dataclass(cfg.distillation)
    loss = distillation.distillation_loss
    assert cfg.trainer.n_gpus_per_node == 8 and distillation.execution_mode == "shared_sync"
    assert cfg.data.train_batch_size == cfg.actor_rollout_ref.actor.ppo_mini_batch_size == 64
    assert cfg.actor_rollout_ref.actor.ppo_epochs == epochs
    assert cfg.trainer.total_training_steps == 250
    assert cfg.trainer.test_freq == 20 and cfg.trainer.save_freq == 100
    assert cfg.trainer.max_actor_ckpt_to_keep == 4 and cfg.trainer.val_before_train
    assert cfg.actor_rollout_ref.actor.optim.lr == pytest.approx(lr)
    assert cfg.actor_rollout_ref.actor.optim.lr_scheduler_type == "constant"
    assert cfg.actor_rollout_ref.actor.optim.lr_warmup_steps == 0
    assert cfg.data.seed == cfg.actor_rollout_ref.rollout.seed == cfg.actor_rollout_ref.actor.fsdp_config.seed == 42
    assert cfg.actor_rollout_ref.actor.data_loader_seed == 42
    assert cfg.data.max_response_length == 7168 and cfg.actor_rollout_ref.rollout.val_kwargs.n == 16
    assert cfg.distillation.teacher_models.teacher_model.model_path.endswith(_model_suffix("JustRL-DeepSeek-1.5B"))
    assert loss.advantage_source == "current" and loss.policy_loss_mode == "vanilla"
    assert not loss.use_task_rewards
    assert cfg.algorithm.rollout_correction.rollout_is is None
    assert not cfg.algorithm.rollout_correction.bypass_mode
    if group == "E15":
        assert loss.loss_mode == "k1" and loss.use_policy_gradient
        assert cfg.actor_rollout_ref.rollout.calculate_log_probs
    else:
        _assert_full_kl(loss)
        assert loss.full_vocab.backend == "precompute"
        assert loss.full_vocab.async_write and loss.full_vocab.prefetch
        assert loss.full_vocab.token_chunk_size == 512 and loss.full_vocab.vocab_chunk_size == 8192
        assert not cfg.actor_rollout_ref.rollout.calculate_log_probs


@pytest.mark.parametrize("group,field", [("E14", "optim.lr"), ("E16", "ppo_epochs"), ("E17", "ppo_epochs")])
def test_full_kl_pairs_only_change_the_ablation_factor(tmp_path, group, field):
    script = SCRIPT.parent / "run_math_opd_ablation_fsdp2.sh"
    first = _config(tmp_path, command=["bash", str(script), "E13"])
    second = _config(tmp_path, command=["bash", str(script), group])
    key = "actor_rollout_ref.actor." + field
    OmegaConf.update(second, key, OmegaConf.select(first, key))
    assert OmegaConf.to_container(first, resolve=True) == OmegaConf.to_container(second, resolve=True)


@pytest.mark.parametrize("group", ["E13", "E14", "E15", "E16", "E17"])
def test_full_kl_and_vanilla_presets_preserve_explicit_overrides(tmp_path, group):
    script = SCRIPT.parent / "run_math_opd_ablation_fsdp2.sh"
    cfg = _config(
        tmp_path,
        command=["bash", str(script), group, "actor_rollout_ref.actor.optim.lr=8e-7", "trainer.test_freq=7"],
        LEARNING_RATE="3e-7",
        PPO_EPOCHS="2",
        TOTAL_TRAINING_STEPS="12",
        TEST_FREQ="6",
        SAVE_FREQ="12",
        SEED="43",
        FULL_KL_BACKEND="streaming",
        FULL_KL_ASYNC_WRITE="False",
        FULL_KL_PREFETCH="False",
    )
    assert cfg.actor_rollout_ref.actor.optim.lr == pytest.approx(8e-7)
    assert cfg.actor_rollout_ref.actor.ppo_epochs == 2 and cfg.trainer.total_training_steps == 12
    assert cfg.trainer.test_freq == 7 and cfg.trainer.save_freq == 12
    assert cfg.data.seed == cfg.actor_rollout_ref.rollout.seed == cfg.actor_rollout_ref.actor.fsdp_config.seed == 43
    if group != "E15":
        assert cfg.distillation.distillation_loss.full_vocab.backend == "streaming"
        assert not cfg.distillation.distillation_loss.full_vocab.async_write
        assert not cfg.distillation.distillation_loss.full_vocab.prefetch


@pytest.mark.parametrize("group", ["E13", "E14", "E16", "E17"])
def test_full_kl_presets_require_shared_teacher(tmp_path, group):
    result = _launch(
        tmp_path,
        command=["bash", str(SCRIPT.parent / "run_math_opd_ablation_fsdp2.sh"), group],
        DISTILLATION_EXECUTION_MODE="standalone",
    )
    assert result.returncode == 2 and "shared_sync" in result.stderr


def test_shared_launcher_remains_vanilla_and_validates_epoch_input(tmp_path):
    cfg = _config(tmp_path)
    assert cfg.distillation.distillation_loss.policy_loss_mode == "vanilla"
    assert cfg.distillation.distillation_loss.advantage_source == "current"
    assert cfg.actor_rollout_ref.actor.ppo_epochs == 1
    result = _launch(tmp_path, PPO_EPOCHS="0")
    assert result.returncode == 2 and "PPO_EPOCHS" in result.stderr


def test_code_launcher_forwards_lcb_policy_and_audit_directory(tmp_path):
    script = ROOT / "examples/mirroropd/run_code_deepseek_r1_1_5b_fsdp2.sh"
    cfg = _config(tmp_path, script=script, MIRROROPD_LCB_MEMORY_MIB="2048", MIRROROPD_LCB_MAX_CONCURRENT="2")
    env = cfg.ray_kwargs.ray_init.runtime_env.env_vars
    assert env.MIRROROPD_LCB_MEMORY_MIB == "2048"
    assert env.MIRROROPD_LCB_MAX_CONCURRENT == "2"
    assert env.MIRROROPD_LCB_RUN_ID == "test"
    assert env.MIRROROPD_LCB_AUDIT_DIR == str(tmp_path / "run" / "lcb-audit")


@pytest.mark.parametrize(
    "group,epochs,outer,test,save",
    [
        ("E18", 64, 64, 8, 16),
        ("E19", 32, 128, 16, 32),
        ("E20", 16, 256, 32, 64),
    ],
)
def test_equal_update_presets_use_one_original_data_pass(tmp_path, group, epochs, outer, test, save):
    cfg = _config(tmp_path, command=["bash", str(SCRIPT.parent / "run_math_opd_ablation_fsdp2.sh"), group])
    OmegaConf.resolve(cfg)
    validate_config(cfg, use_reference_policy=False, use_critic=False)
    loss = omega_conf_to_dataclass(cfg.distillation).distillation_loss
    assert cfg.data.train_files[0].endswith("/math/DAPO/train.parquet")
    assert cfg.data.train_batch_size == cfg.actor_rollout_ref.actor.ppo_mini_batch_size == 64
    assert cfg.data.train_max_samples == -1 and cfg.data.shuffle and cfg.data.seed == 42
    assert cfg.trainer.total_epochs == 1 and cfg.trainer.total_training_steps == outer
    assert cfg.actor_rollout_ref.actor.ppo_epochs == epochs and outer * epochs == 4096
    assert cfg.trainer.test_freq == test and test * epochs == 512
    assert cfg.trainer.save_freq == save and save * epochs == 1024
    assert cfg.trainer.max_actor_ckpt_to_keep == 4
    assert not cfg.trainer.val_before_train and cfg.trainer.resume_mode == "disable"
    assert not cfg.algorithm.filter_groups.enable and not cfg.trainer.v1.sampler.sync_refill_failed_groups
    assert outer * 64 <= 17917
    _assert_full_kl(loss)
    assert not loss.use_task_rewards
    assert loss.full_vocab.backend == "precompute" and loss.full_vocab.async_write and loss.full_vocab.prefetch
    assert cfg.actor_rollout_ref.actor.optim.lr == 1e-6


@pytest.mark.parametrize(
    "target,model",
    [
        ("student", "DeepSeek-R1-Distill-Qwen-1.5B"),
        ("teacher", "JustRL-DeepSeek-1.5B"),
    ],
)
def test_standalone_validation_uses_selected_generator_without_distillation(tmp_path, target, model):
    cfg = _config(tmp_path, command=["bash", str(SCRIPT.parent / "run_math_validation_fsdp2.sh"), target])
    OmegaConf.resolve(cfg)
    validate_config(cfg, use_reference_policy=False, use_critic=False)
    assert cfg.actor_rollout_ref.model.path.endswith("/" + _model_suffix(model))
    assert not cfg.distillation.enabled and cfg.distillation.n_gpus_per_node == 0
    assert cfg.trainer.val_only and cfg.trainer.val_before_train
    assert cfg.trainer.resume_mode == "disable" and cfg.trainer.save_freq == -1
    assert not cfg.actor_rollout_ref.rollout.calculate_log_probs
    assert cfg.actor_rollout_ref.rollout.seed == cfg.data.seed == 42
    assert cfg.actor_rollout_ref.rollout.val_kwargs.n == 16
    assert cfg.actor_rollout_ref.rollout.val_kwargs.max_tokens == 31744
    assert cfg.data.val_batch_size is None
    assert cfg.actor_rollout_ref.rollout.max_num_seqs == 128
    assert cfg.actor_rollout_ref.rollout.gpu_memory_utilization == 0.65
    assert not cfg.actor_rollout_ref.rollout.enforce_eager


def test_teacher_validation_selects_teacher_override_and_retains_hydra_precedence(tmp_path):
    script = SCRIPT.parent / "run_math_validation_fsdp2.sh"
    cfg = _config(
        tmp_path,
        command=["bash", str(script), "teacher"],
        STUDENT_MODEL="/unused/student",
        TEACHER_MODEL="/teacher with spaces",
        STUDENT_GPUS="4",
    )
    assert cfg.actor_rollout_ref.model.path == "/teacher with spaces"
    assert cfg.trainer.n_gpus_per_node == 4
    cfg = _config(
        tmp_path,
        command=[
            "bash",
            str(script),
            "student",
            "actor_rollout_ref.model.path=/checkpoint",
            "actor_rollout_ref.rollout.val_kwargs.n=2",
        ],
        SEED="43",
    )
    assert cfg.actor_rollout_ref.model.path == "/checkpoint"
    assert cfg.actor_rollout_ref.rollout.val_kwargs.n == 2 and cfg.actor_rollout_ref.rollout.seed == 43
    result = _launch(tmp_path, command=["bash", str(script), "unknown"])
    assert result.returncode == 2


@pytest.mark.parametrize("group", ["E18", "E19", "E20"])
def test_equal_update_presets_allow_explicit_smoke_overrides(tmp_path, group):
    cfg = _config(
        tmp_path,
        command=["bash", str(SCRIPT.parent / "run_math_opd_ablation_fsdp2.sh"), group, "trainer.test_freq=1"],
        PPO_EPOCHS="2",
        TOTAL_TRAINING_STEPS="2",
        TRAIN_BATCH_SIZE="8",
        TEST_FREQ="2",
        SAVE_FREQ="-1",
    )
    assert cfg.actor_rollout_ref.actor.ppo_epochs == cfg.trainer.total_training_steps == 2
    assert cfg.data.train_batch_size == cfg.actor_rollout_ref.actor.ppo_mini_batch_size == 8
    assert cfg.trainer.test_freq == 1 and cfg.trainer.save_freq == -1


def test_e49_forward_preserves_e48_protocol_and_explicit_support(tmp_path):
    cfg = _config(tmp_path, command=["bash", str(SCRIPT), "E49"], STUDENT_GPUS="4")
    OmegaConf.resolve(cfg)
    validate_config(cfg, use_reference_policy=False, use_critic=False)
    loss = omega_conf_to_dataclass(cfg.distillation).distillation_loss
    assert loss.loss_mode == "full_forward_kl"
    assert loss.loss_settings.anchor_payload is None
    assert loss.loss_settings.teacher_only_forward_kl
    assert loss.full_vocab.forward_teacher_support == "student"
    assert loss.full_vocab.backend == "precompute_fused"
    assert loss.full_vocab.token_chunk_size == 512
    assert loss.full_vocab.async_write and loss.full_vocab.prefetch
    assert cfg.distillation.teacher_models.teacher_model.model_path.endswith("/" + _model_suffix("Polaris-7B-Preview"))
    assert cfg.data.train_batch_size == cfg.actor_rollout_ref.actor.ppo_mini_batch_size == 64
    assert cfg.actor_rollout_ref.actor.ppo_epochs == 16
    assert cfg.trainer.total_training_steps == 256 and cfg.trainer.val_before_train
    assert (cfg.trainer.test_freq, cfg.trainer.save_freq) == (32, 64)
    assert cfg.actor_rollout_ref.actor.optim.lr == 1e-6
    assert cfg.data.max_response_length == 7168
    assert cfg.actor_rollout_ref.rollout.val_kwargs.max_tokens == 31744
    assert cfg.actor_rollout_ref.rollout.val_kwargs.n == 16
    assert not cfg.actor_rollout_ref.rollout.calculate_log_probs


def test_forward_smoke_is_small_and_records_file_metrics(tmp_path):
    cfg = _config(
        tmp_path,
        script=SCRIPT.parent / "smoke/smoke_full_forward_kl.sh",
        TRAIN_BATCH_SIZE="8",
        PPO_EPOCHS="2",
        STUDENT_GPUS="4",
    )
    OmegaConf.resolve(cfg)
    validate_config(cfg, use_reference_policy=False, use_critic=False)
    assert cfg.distillation.distillation_loss.loss_mode == "full_forward_kl"
    assert cfg.distillation.distillation_loss.full_vocab.forward_teacher_support == "student"
    assert cfg.trainer.total_training_steps == 2 and not cfg.trainer.val_before_train
    assert list(cfg.trainer.logger) == ["console", "file"]
    assert cfg.data.train_batch_size == 8 and cfg.actor_rollout_ref.actor.ppo_epochs == 2
    assert cfg.distillation.shared_teacher.infer_max_token_len_per_gpu == 32768


@pytest.mark.parametrize(
    "preset,profile,eta", [("E50", "fp16", 1), ("E51", "fp16", 0.3), ("E52", "bf16_fp32", 1), ("E53", "bf16_fp32", 0.3)]
)
def test_precision_presets_preserve_protocol_and_teacher(tmp_path, preset, profile, eta):
    cfg = _config(tmp_path, command=["bash", str(SCRIPT), preset], STUDENT_GPUS="4")
    OmegaConf.resolve(cfg)
    engine = omega_conf_to_dataclass(cfg.actor_rollout_ref.actor.fsdp_config)
    assert engine.forward_precision == profile
    assert engine.mixed_precision["param_dtype"] == ("float16" if profile == "fp16" else "bfloat16")
    assert cfg.distillation.shared_teacher.model_dtype == "bfloat16"
    assert cfg.actor_rollout_ref.rollout.dtype == "bfloat16"
    assert cfg.distillation.distillation_loss.mirror.eta == eta
    assert cfg.trainer.total_training_steps == 256
    assert cfg.actor_rollout_ref.actor.ppo_epochs == 16
    assert cfg.trainer.save_freq == 128 and cfg.actor_rollout_ref.rollout.val_kwargs.n == 16


def test_precision_option_and_hydra_override_precedence(tmp_path):
    cfg = _config(
        tmp_path,
        command=[
            "bash",
            str(SCRIPT),
            "E52",
            "--precision",
            "fp16",
            "actor_rollout_ref.actor.fsdp_config.forward_precision=bf16",
        ],
        TRAIN_PRECISION="fp32",
    )
    assert cfg.actor_rollout_ref.actor.fsdp_config.forward_precision == "bf16"


def test_checkpoint_store_is_canonical_and_validation_keeps_run_directory(tmp_path):
    from examples.mirroropd.assets.paths import production_checkpoints

    cfg = _config(tmp_path)
    expected = production_checkpoints(tmp_path / "run", root=tmp_path / "models with spaces", repository=ROOT)
    assert cfg.trainer.default_local_dir == str(expected)
    assert cfg.trainer.validation_data_dir == str(tmp_path / "run/validation")
    assert not expected.exists() and not (tmp_path / "run").exists()
    cfg = _config(tmp_path, CHECKPOINT_DIR=str(tmp_path / "explicit weights"))
    assert cfg.trainer.default_local_dir == str(tmp_path / "explicit weights")


def test_existing_checkpoint_data_is_rejected_before_training(tmp_path):
    checkpoint = tmp_path / "weights"
    checkpoint.mkdir()
    (checkpoint / "state.pt").write_bytes(b"existing")
    result = _launch(
        tmp_path, DRY_RUN="0", CUDA_VISIBLE_DEVICES="0,1", STUDENT_GPUS="2", CHECKPOINT_DIR=str(checkpoint)
    )
    assert result.returncode == 2
    assert "already contains checkpoint data" in result.stderr
    assert (checkpoint / "state.pt").read_bytes() == b"existing"
    assert not (tmp_path / "run").exists()
