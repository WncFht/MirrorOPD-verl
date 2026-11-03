# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""New chains retain the shared scientific budget and explicit tokenizer views."""

import pytest
from omegaconf import OmegaConf
from test_mimo_launcher_on_cpu import prepared as prepared_fixture
from test_mirroropd_math_launcher_on_cpu import SCRIPT, _config

prepared = prepared_fixture


@pytest.mark.parametrize(
    "number,student,teacher,family,stop",
    [
        (74, "Olmo3-Base-BayesProtocol", "Olmo3-C3PO-BayesProtocol", "default", 100257),
        (75, "Olmo3-Base-BayesProtocol", "Olmo3-SFT-BayesProtocol", "default", 100257),
        (76, "Qwen2.5-14B-R1Protocol", "Light-R1-14B-R1Protocol", "deepseek", 151643),
        (77, "Qwen2.5-14B-R1Protocol", "R1-Distill-14B-R1Protocol", "deepseek", 151643),
    ],
)
def test_native_stage_chain_config(prepared, number, student, teacher, family, stop):
    cfg = _config(
        prepared,
        command=["bash", str(SCRIPT), f"E{number}"],
        STAGE_CHAIN_QUESTIONS=str(prepared),
        TRAIN_BATCH_SIZE="128",
    )
    OmegaConf.resolve(cfg)
    assert ("MirrorOPD--" + student) in cfg.actor_rollout_ref.model.path
    assert ("MirrorOPD--" + teacher) in cfg.distillation.teacher_models.teacher_model.model_path
    assert not cfg.actor_rollout_ref.model.trust_remote_code
    assert cfg.data.continuous_token.model_family == family
    assert list(cfg.actor_rollout_ref.rollout.stop_token_ids) == [stop]
    assert cfg.data.train_batch_size == cfg.actor_rollout_ref.actor.ppo_mini_batch_size == 128
    assert cfg.actor_rollout_ref.actor.ppo_epochs == 32 and cfg.trainer.total_training_steps == 50
    assert (cfg.data.max_prompt_length, cfg.data.max_response_length) == (2048, 7168)
    assert cfg.trainer.total_training_steps == 50 and cfg.actor_rollout_ref.actor.ppo_epochs == 32
    assert cfg.actor_rollout_ref.rollout.max_model_len == 32768
    assert cfg.actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu >= 32768
    assert cfg.distillation.shared_teacher.infer_max_token_len_per_gpu >= 32768
    assert not cfg.trainer.evaluation.source_enabled and not cfg.trainer.evaluation.history_enabled
    assert cfg.trainer.evaluation.protocols.math.max_new_tokens == 24576
    assert cfg.trainer.evaluation.protocols["if"].max_new_tokens == 8192


@pytest.mark.parametrize("world,cpu_offload", [(2, True), (4, False)])
def test_light_resource_plan_preserves_fp32_adam_with_fewer_gpus(prepared, world, cpu_offload):
    cfg = _config(
        prepared,
        command=["bash", str(SCRIPT), "E76"],
        STAGE_CHAIN_QUESTIONS=str(prepared),
        TRAIN_BATCH_SIZE="128",
        STUDENT_GPUS=str(world),
    )
    OmegaConf.resolve(cfg)
    assert cfg.actor_rollout_ref.actor.fsdp_config.offload_policy is cpu_offload
    assert cfg.actor_rollout_ref.actor.fsdp_config.model_dtype == "fp32"
    assert cfg.actor_rollout_ref.actor.optim.optimizer == "AdamW"
    assert cfg.actor_rollout_ref.actor.ppo_epochs == 32 and cfg.trainer.total_training_steps == 50
