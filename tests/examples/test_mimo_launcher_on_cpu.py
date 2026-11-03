# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Verify MiMo stages retain common protocols with the explicitly revised budgets."""

import json

import pytest
from omegaconf import OmegaConf
from test_math_experiment_plan_on_cpu import plan
from test_mirroropd_math_launcher_on_cpu import SCRIPT, _assert_full_kl, _config

from examples.mirroropd.assets.paths import registered_model
from examples.mirroropd.launchers.mimo import preflight, questions
from verl.trainer.evaluation.config import EvaluationConfig
from verl.trainer.ppo.readout_callback import readout_definition
from verl.utils.config import omega_conf_to_dataclass, validate_config


@pytest.fixture
def prepared(tmp_path):
    datasets = [
        dict(
            name=name,
            domain=domain,
            verifier=verifier,
            n=n,
            path=str(tmp_path / f"{name}.parquet"),
            protocol="public",
            panel_ids=[
                f"{name}:{i}" for i in range(128 if name in {"MATH-500", "LCBv5", "GPQA-Diamond", "IFEval"} else 0)
            ],
        )
        for name, domain, verifier, n in (
            ("MATH-500", "math", "math", 1),
            ("AMC23", "math", "math", 16),
            ("AIME24", "math", "math", 16),
            ("AIME25", "math", "math", 16),
            ("AIME26", "math", "math", 16),
            ("LCBv5", "code", "lcb", 10),
            ("GPQA-Diamond", "science", "gpqa", 4),
            ("IFEval", "if", "ifeval", 1),
        )
    ]
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            dict(
                kind="question_sets",
                status="complete",
                metadata=dict(
                    datasets=datasets,
                    split_reports={"math": {"training": 6400}},
                    paths={"math": {"train": str(tmp_path / "train.parquet")}},
                ),
            )
        )
    )
    return tmp_path


@pytest.mark.parametrize("number,teacher", [(71, "MiMo-7B-RL"), (72, "MiMo-7B-RL-Zero"), (73, "MiMo-7B-SFT")])
def test_native_mimo_config(prepared, number, teacher):
    cfg = _config(
        prepared,
        command=["bash", str(SCRIPT), f"E{number}"],
        MIMO_QUESTIONS=str(prepared),
        TRAIN_BATCH_SIZE="128",
    )
    OmegaConf.resolve(cfg)
    validate_config(cfg, use_reference_policy=False, use_critic=False)
    _assert_full_kl(omega_conf_to_dataclass(cfg.distillation).distillation_loss)
    assert cfg.actor_rollout_ref.model.path.endswith(str(registered_model("MiMo-7B-Base", root="/")))
    assert cfg.distillation.teacher_models.teacher_model.model_path.endswith(str(registered_model(teacher, root="/")))
    assert cfg.actor_rollout_ref.model.trust_remote_code
    assert not cfg.actor_rollout_ref.model.mtp.enable and not cfg.actor_rollout_ref.rollout.mtp.enable
    assert cfg.data.train_batch_size == cfg.actor_rollout_ref.actor.ppo_mini_batch_size == 128
    reuse, outer = (32, 25) if number == 71 else (16, 20)
    assert cfg.actor_rollout_ref.actor.ppo_epochs == reuse and cfg.trainer.total_training_steps == outer
    assert cfg.data.continuous_token.model_family == "qwen25"
    assert not cfg.data.shuffle and not cfg.data.filter_overlong_prompts
    assert (cfg.data.max_prompt_length, cfg.data.max_response_length) == (2048, 7168)
    assert cfg.actor_rollout_ref.rollout.max_model_len == 32768
    assert cfg.actor_rollout_ref.rollout.val_kwargs.max_tokens is None
    assert list(cfg.actor_rollout_ref.rollout.stop_token_ids) == [151643, 151645]
    assert cfg.trainer.test_freq == cfg.trainer.save_freq == -1
    assert cfg.trainer.val_before_train and cfg.data.val_files == []
    _, declaration = readout_definition(cfg)
    evaluation = EvaluationConfig.parse(declaration)
    assert not evaluation.source_enabled and not evaluation.history_enabled
    assert "source" not in evaluation.protocols and "source" not in evaluation.bootstrap_children
    assert all(p.prompt_limit + p.max_new_tokens <= 32768 for p in evaluation.protocols.values())
    assert evaluation.protocols["math"].max_new_tokens == 24576
    assert evaluation.protocols["if"].max_new_tokens == 8192
    assert [d.n for d in evaluation.datasets] == [1, 16, 16, 16, 16, 10, 4, 1]
    assert sum(len(d.panel_ids) for d in evaluation.datasets) == 512
    assert [i for i in range(outer + 1) if evaluation.schedules["banks"].due(i, outer)] == [
        0,
        1,
        2,
        *range(5, outer + 1, 5),
    ]
    assert [i for i in range(outer + 1) if evaluation.schedules["bench"].due(i, outer)] == [0, outer]
    assert evaluation.final_export and evaluation.verify_export


def test_mimo_arms_share_the_initialization_and_protocol(prepared):
    env = {"MIMO_QUESTIONS": str(prepared)}
    sft, rl = (plan([name], env) for name in ("E73", "E71"))
    # Budgets and the terminal accuracy boundary differ; all other arguments agree.
    common = lambda values: tuple(
        value for value in values if not value.startswith(("trainer.total_training_steps=", "++trainer.evaluation="))
    )
    assert common(sft.arguments) == common(rl.arguments)
    changed = {key for key in sft.environment if sft.environment[key] != rl.environment[key]}
    assert changed == {"TEACHER_MODEL", "MIMO_BOOTSTRAP", "PPO_EPOCHS"}
    assert sft.environment["PPO_EPOCHS"] == "16" and rl.environment["PPO_EPOCHS"] == "32"
    assert sft.environment["TRAIN_BATCH_SIZE"] == "128"
    for role in ("student", "teacher"):
        with pytest.raises(ValueError, match="native bootstrap"):
            plan([f"E71-valid-{role}"], env)
    for name in ["E69", "E70"]:
        with pytest.raises(ValueError, match="withdrawn"):
            plan([name], env)


def test_mimo_rejects_incomplete_panels_and_repeated_question_budget(prepared):
    with pytest.raises(ValueError, match="without repeating"):
        questions(prepared, required=6401)
    path = prepared / "manifest.json"
    data = json.loads(path.read_text())
    data["metadata"]["datasets"][-1]["panel_ids"].pop()
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="128 distinct"):
        questions(prepared, required=6400)


def test_mimo_bootstrap_preflight_has_no_gpu_side_effects(tmp_path):
    preflight({"CONFIG_ONLY": "1"})
    preflight({"DRY_RUN": "1"})
    with pytest.raises(ValueError, match="teacher-specific"):
        preflight({"MIMO_BOOTSTRAP": str(tmp_path)})
    (tmp_path / "manifest.json").write_text(json.dumps({"status": "complete"}))
    preflight({"MIMO_BOOTSTRAP": str(tmp_path)})
