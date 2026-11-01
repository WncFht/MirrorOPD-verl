# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Plan tests do not need Torch, Hydra, model assets or a running shell launcher."""

import importlib.util
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "math_experiment_plan", ROOT / "examples/mirroropd/launchers/math_experiment.py"
)
planner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = planner
SPEC.loader.exec_module(planner)


def plan(argv, env=None):
    args, hydra = planner.parser().parse_known_args(argv)
    return planner.build_plan(args, hydra, {"EXPERIMENT_NAME": "test", **(env or {})})


def test_catalog_covers_every_number_without_alias_duplication():
    catalog = set(planner.PRESETS) | set(planner.ALIASES)
    highest = max(int(name[1:]) for name in catalog)
    assert catalog == {f"E{i}" for i in range(1, highest + 1)}
    assert set(planner.PRESETS).isdisjoint(planner.ALIASES)
    assert len(planner.MODEL_MATRIX) == 6
    with pytest.raises(FrozenInstanceError):
        planner.PRESETS["E13"].budget.steps = 1


@pytest.mark.parametrize(
    "selector,canonical,role",
    [
        ("E21", "E15", None),
        ("E22-valid-teacher", "E13", "teacher"),
        ("E27-valid-student", "E27", "student"),
        ("custom", "custom", None),
    ],
)
def test_aliases_and_roles(selector, canonical, role):
    name, _, actual = planner.resolve_selector(selector)
    assert (name, actual) == (canonical, role)


def test_planning_does_not_mutate_inputs_or_accumulate_previous_defaults():
    env = {"EXPERIMENT_NAME": "fixed", "MIRROR_ETA": ".2", "PPO_EPOCHS": "2"}
    original = env.copy()
    args, hydra = planner.parser().parse_known_args(["E17", "--method", "bpo", "--eta", ".4", "trainer.test_freq=-1"])
    result = planner.build_plan(args, hydra, env)
    assert env == original
    assert result.environment["MIRROR_ETA"] == ".4" and result.environment["PPO_EPOCHS"] == "2"
    assert result.arguments[-1] == "trainer.test_freq=-1"
    assert planner.PRESETS["E17"].budget.reuse == 16
    assert plan(["custom"]).environment.get("PPO_EPOCHS") is None


def test_full_defaults_and_validation_do_not_leak_training_budget():
    result = plan(["E28-valid-teacher"])
    assert result.entry.name == "run_math_validation_fsdp2.sh"
    assert result.environment["MAX_PROMPT_LENGTH"] == "4096"
    assert result.environment["ROLLOUT_MAX_NUM_SEQS"] == "64"
    assert result.environment["ROLLOUT_GPU_MEMORY_UTILIZATION"] == "0.65"
    assert not any(x.startswith("trainer.total_training_steps=") for x in result.arguments)
    assert "PPO_EPOCHS" not in result.environment
    full = plan(["E18"])
    assert full.environment["FULL_KL_PREFETCH"] == "True"
    assert "trainer.total_training_steps=64" in full.arguments
    assert full.environment["PPO_EPOCHS"] == "64"


def test_base_memory_default_preserves_explicit_environment_override():
    assert plan(["E27"]).environment["ROLLOUT_GPU_MEMORY_UTILIZATION"] == "0.8"
    result = plan(["E27-valid-student"], {"ROLLOUT_GPU_MEMORY_UTILIZATION": "0.55"})
    assert result.environment["ROLLOUT_GPU_MEMORY_UTILIZATION"] == "0.55"


def test_e41_termination_alignment_is_scoped_and_can_be_disabled():
    assert plan(["E41"]).environment["FULL_KL_TERMINATION_TOKEN_IDS"] == "[151643,151645]"
    assert plan(["E41"], {"FULL_KL_TERMINATION_TOKEN_IDS": "[]"}).environment["FULL_KL_TERMINATION_TOKEN_IDS"] == "[]"
    for args in (["E30"], ["E39"], ["E44"], ["E41-valid-student"], ["E41", "--method", "sampled"]):
        assert "FULL_KL_TERMINATION_TOKEN_IDS" not in plan(args).environment


def test_empty_environment_matches_shell_default_semantics():
    result = plan(["E13"], {"PPO_EPOCHS": "", "TOTAL_TRAINING_STEPS": "", "TEST_FREQ": ""})
    assert result.environment["PPO_EPOCHS"] == "1" and result.environment["TEST_FREQ"] == "20"
    assert "trainer.total_training_steps=250" in result.arguments
    assert (
        plan(["custom"], {"OPD_ADVANTAGE_SOURCE": "", "OPD_POLICY_LOSS_MODE": ""}).environment["EXPERIMENT_METHOD"]
        == "sampled"
    )


def test_polaris_e48_retains_e20_budget_and_supports_independent_validation():
    from dataclasses import replace

    e48 = planner.PRESETS["E48"]
    assert e48.budget == replace(planner.PRESETS["E20"].budget, initial_validation=True)
    training = plan(["E48"])
    from examples.mirroropd.assets.paths import registered_model

    assert training.environment["TEACHER_MODEL"] == str(registered_model("Polaris-7B-Preview"))
    assert training.environment["STUDENT_MODEL"] == str(registered_model("DeepSeek-R1-Distill-Qwen-1.5B"))
    assert training.environment["PPO_EPOCHS"] == "16"
    assert training.environment["LEARNING_RATE"] == "1e-6"
    assert training.environment["FULL_KL_BACKEND"] == "precompute"
    assert "TEACHER_FSDP_TOKEN_BUDGET" not in training.environment
    assert "trainer.total_training_steps=256" in training.arguments
    assert "trainer.val_before_train=True" in training.arguments
    assert "algorithm.filter_groups.enable=False" in training.arguments
    teacher = plan(["E48-valid-teacher"])
    assert teacher.entry.name == "run_math_validation_fsdp2.sh"
    assert teacher.environment["ROLLOUT_MAX_NUM_SEQS"] == "32"
    assert "PPO_EPOCHS" not in teacher.environment


def test_exec_preserves_literal_arguments_and_separator(monkeypatch, tmp_path):
    captured = []
    monkeypatch.setattr(planner.os, "execvpe", lambda program, args, env: captured.append((program, args, env)))
    marker = tmp_path / "must-not-exist"
    literal = f"actor_rollout_ref.model.path='$(touch {marker}) with spaces'"
    planner.main(["E13", "--", literal, "--eta=untouched"])
    _, args, _ = captured[0]
    assert args[-2:] == [literal, "--eta=untouched"]
    assert not marker.exists()


def test_listing_is_derived_from_catalog_and_never_launches(monkeypatch, capsys):
    monkeypatch.setattr(planner.os, "execvpe", lambda *args: pytest.fail("listing must not execute training"))
    planner.main(["--list-presets"])
    output = capsys.readouterr().out
    assert "E43" in output and "E21=E15" in output and "Qwen3-8B" in output


@pytest.mark.parametrize(
    "args",
    [
        ["E1-valid-student"],
        ["E99"],
        ["E13", "--method", "bpo", "--estimator", "sampled"],
        ["E13-valid-teacher", "--eta", ".3"],
        ["E13", "--method", "sampled", "--eta", ".3"],
    ],
)
def test_invalid_plans(args):
    with pytest.raises(ValueError):
        plan(args)


def test_entry_comment_overview_matches_catalog():
    entry = (ROOT / "examples/mirroropd/run_math_opd_ablation_fsdp2.sh").read_text()
    block = entry.split("# BEGIN GENERATED EXPERIMENT OVERVIEW\n", 1)[1]
    block = block.split("\n# END GENERATED EXPERIMENT OVERVIEW", 1)[0]
    assert all(line.startswith("#") for line in block.splitlines())
    actual = "\n".join(line[2:] if line.startswith("# ") else "" for line in block.splitlines())
    assert actual == planner.preset_overview()


def test_overview_is_read_only_and_budget_derived(monkeypatch, capsys):
    from dataclasses import replace

    catalog = planner.PRESETS.copy()
    catalog["E20"] = replace(catalog["E20"], budget=replace(catalog["E20"].budget, steps=257))
    monkeypatch.setattr(planner, "PRESETS", catalog)
    monkeypatch.setattr(planner.os, "execvpe", lambda *args: pytest.fail("overview must not launch"))
    planner.main(["--overview"])
    text = capsys.readouterr().out
    line = next(line for line in text.splitlines() if line.startswith("E20 "))
    assert line.split()[1:4] == ["16", "257", "4112"]
    assert "Sampled-r16" in text and "E43" in text and "E21=E15" in text


@pytest.mark.parametrize("preset,method,kappa", [("E45", "mdpo", "0"), ("E46", "bpo", "0"), ("E47", "mdpo", "0.5")])
def test_registered_mirror_method_and_target_defaults(preset, method, kappa):
    result = plan([preset])
    assert result.environment["EXPERIMENT_METHOD"] == method
    assert result.environment["MIRROR_ESTIMATOR"] == "full_vocab"
    assert result.environment["MIRROR_KAPPA"] == kappa
    assert result.environment["MIRROR_ETA"] == "0.3"
    assert result.environment["FULL_KL_BACKEND"] == "precompute_fused"
    assert result.environment["FULL_KL_TOKEN_CHUNK"] == "128"


def test_registered_defaults_preserve_override_precedence_and_endpoint_switch():
    result = plan(
        ["E45", "--eta", "0.5", "distillation.distillation_loss.mirror.eta=0.6"],
        {"MIRROR_ETA": "0.4", "MIRROR_KAPPA": "0.25", "FULL_KL_TOKEN_CHUNK": "256"},
    )
    assert result.environment["MIRROR_ETA"] == "0.5"
    assert result.environment["MIRROR_KAPPA"] == "0.25"
    assert result.environment["FULL_KL_TOKEN_CHUNK"] == "256"
    assert result.arguments[-1] == "distillation.distillation_loss.mirror.eta=0.6"
    endpoint = plan(["E45", "--method", "full_kl"])
    assert endpoint.environment["EXPERIMENT_METHOD"] == "full_kl"
    assert "MIRROR_KAPPA" not in endpoint.environment and "MIRROR_ETA" not in endpoint.environment


def test_e54_moe_teacher_keeps_e41_training_protocol():
    from dataclasses import replace

    assert planner.PRESETS["E54"] == replace(
        planner.PRESETS["E41"],
        pair=replace(planner.PRESETS["E41"].pair, teacher="Qwen3-30B-A3B-Instruct-2507"),
    )
    result = plan(["E54"])
    from examples.mirroropd.assets.paths import registered_model

    assert result.environment["TEACHER_MODEL"] == str(registered_model("Qwen3-30B-A3B-Instruct-2507"))
    assert result.environment["FULL_KL_TERMINATION_TOKEN_IDS"] == "[151643,151645]"
    assert not planner.PRESETS["E54"].budget.initial_validation
    assert "trainer.total_training_steps=100" in result.arguments


@pytest.mark.parametrize("name", ["E55", "E56", "E57", "E58"])
def test_moe_long_native_template_is_shared_with_independent_validation(name):
    training = plan([name])
    assert training.environment["SAVE_FREQ"] == "-1"
    assert training.environment["TEST_FREQ"] == "10"
    assert "trainer.max_actor_ckpt_to_keep=4" not in training.arguments
    for selector in (name, name + "-valid-student", name + "-valid-teacher"):
        assert "+data.apply_chat_template_kwargs.enable_thinking=null" in plan([selector]).arguments
    for old in ("E41", "E43", "E54"):
        assert "+data.apply_chat_template_kwargs.enable_thinking=False" in plan([old]).arguments


@pytest.mark.parametrize("name,reference", [("E59", "E55"), ("E60", "E56"), ("E61", "E57"), ("E62", "E58")])
def test_fused_eos_presets_preserve_scientific_protocol(name, reference):
    old, new = planner.PRESETS[reference], planner.PRESETS[name]
    assert new.pair == old.pair and new.budget == old.budget
    assert (new.method, new.kappa, new.eta) == (old.method, old.kappa, old.eta)
    previous, current = dict(old.environment), dict(new.environment)
    assert previous["FULL_KL_BACKEND"] == "precompute"
    assert current.pop("TRAIN_PRECISION") == "bf16"
    assert current.pop("FULL_KL_BACKEND") == "precompute_fused"
    previous.pop("FULL_KL_BACKEND")
    assert current == previous
    actual = plan([name])
    assert actual.environment["MIRROR_DIAGNOSTICS"] == "True"
    assert actual.environment["FULL_KL_TERMINATION_TOKEN_IDS"] == "[151643,151645]"


@pytest.mark.parametrize("number,arm", [(63, "S"), (64, "S0"), (65, "T"), (66, "LM"), (67, "PM"), (68, "TM")])
def test_qwen_prefix_eight_gpu_defaults(number, arm):
    result = plan([f"E{number}"])
    assert planner.PRESETS[f"E{number}"].rollout_arm == arm
    assert result.environment["STUDENT_GPUS"] == "8"
    assert result.environment["ROLLOUT_MAX_NUM_SEQS"] == "16"
    assert result.environment["FUSION_KV_BYTES"] == str(24 * 2**30)
    assert result.environment["PREFIX_SINGLE_KV_BYTES"] == str(64 * 2**30)
    assert result.environment["PPO_MAX_TOKEN_LEN_PER_GPU"] == "32768"
    assert result.environment["PARAM_OFFLOAD"] == result.environment["OPTIMIZER_OFFLOAD"] == "True"
    assert result.environment["FULL_KL_TERMINATION_TOKEN_IDS"] == "[]"
    assert "trainer.total_training_steps=40" in result.arguments
    assert result.environment["PPO_EPOCHS"] == "16"
    assert "rollout_policy.execution.tensor_parallel_size=2" in result.arguments
    assert "actor_rollout_ref.rollout.stop_token_ids=[151645,151643]" in result.arguments
    assert "+data.apply_chat_template_kwargs.enable_thinking=False" in result.arguments


def test_qwen_resource_overrides_and_method_precedence():
    result = plan(
        ["E66", "--precision", "bf16", "rollout_policy.execution.max_num_seqs=3"],
        {"STUDENT_GPUS": "2", "CUDA_VISIBLE_DEVICES": "2,3", "FUSION_KV_BYTES": "4294967296"},
    )
    assert result.environment["FUSION_KV_BYTES"] == "4294967296"
    assert result.arguments[-1] == "rollout_policy.execution.max_num_seqs=3"
    precisions = [v for v in result.arguments if v.startswith("actor_rollout_ref.actor.fsdp_config.forward_precision=")]
    assert precisions[-1].endswith("=bf16")
    with pytest.raises(ValueError, match="distinct physical"):
        plan(["E66"], {"STUDENT_GPUS": "8", "CUDA_VISIBLE_DEVICES": "6,7"})


def test_qwen_native_banks_are_required_only_for_actual_launch(tmp_path):
    import json

    from examples.mirroropd.launchers.qwen_prefix import preflight

    root = tmp_path / "native"
    preflight({"CONFIG_ONLY": "1"})
    preflight({"DRY_RUN": "1"})
    with pytest.raises(ValueError, match="completed native banks"):
        preflight({"QWEN_EVAL_ROOT": str(root)})
    (root / "bootstrap").mkdir(parents=True)
    (root / "bootstrap/manifest.json").write_text(json.dumps({"status": "complete"}))
    preflight({"QWEN_EVAL_ROOT": str(root)})


def test_qwen_bootstrap_resources_keep_capture_shapes_within_replica_limit(tmp_path):
    from omegaconf import OmegaConf

    from examples.mirroropd.research.studies.prefix_fusion.prepare_native import configs

    source = tmp_path / "input.yaml"
    OmegaConf.save(
        OmegaConf.create(
            {
                "actor_rollout_ref": {
                    "rollout": {
                        "max_num_seqs": 16,
                        "cudagraph_capture_sizes": [1, 2, 4, 8, 16],
                        "engine_kwargs": {"vllm": {"kv_cache_memory_bytes": 0}},
                    }
                }
            }
        ),
        source,
    )
    configs(source, tmp_path / "configs")
    teacher = OmegaConf.load(tmp_path / "configs/teacher.yaml").actor_rollout_ref.rollout
    assert teacher.max_num_seqs == max(teacher.cudagraph_capture_sizes) == 8
    assert teacher.engine_kwargs.vllm.kv_cache_memory_bytes == 40 * 2**30


def test_native_qwen_readout_uses_the_canonical_factory(tmp_path):
    from omegaconf import OmegaConf

    from examples.mirroropd.launchers.qwen_prefix import evaluation
    from verl.trainer.evaluation.config import EvaluationConfig
    from verl.trainer.evaluation.coordinator import EvaluationCoordinator
    from verl.trainer.ppo.readout_callback import readout_definition

    declaration = evaluation(tmp_path, tmp_path / "run", 2)
    callback, options = readout_definition(OmegaConf.create({"trainer": {"evaluation": declaration}}))
    assert callback is EvaluationCoordinator
    assert EvaluationConfig.parse(options).source_count == 64
