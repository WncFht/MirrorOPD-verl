# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Standard-library experiment planner. No model imports, shell evaluation or training.

The public shell entry delegates here. Model/budget presets are data; planning
merges preset < environment < method options < forwarded Hydra overrides.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class ModelPair:
    student: str = "DeepSeek-R1-Distill-Qwen-1.5B"
    teacher: str = "JustRL-DeepSeek-1.5B"
    family: str = "deepseek"
    response: int = 7168
    prompt: int = 4096
    student_sequences: int = 128
    teacher_sequences: int = 64
    stop_token: int | None = None
    student_memory_utilization: float = 0.65
    teacher_memory_utilization: float = 0.65
    enable_thinking: bool | None = False


@dataclass(frozen=True)
class Budget:
    steps: int
    reuse: int
    test: int | None
    save: int | None
    initial_validation: bool = False
    keep: int | None = None
    align_seeds: bool = False
    original_data: bool = False
    lr: str | None = None


@dataclass(frozen=True)
class Preset:
    budget: Budget
    method: str = "sampled"
    advantage: str = "current"
    policy: str = "vanilla"
    pair: ModelPair | None = None
    kappa: str | None = None
    eta: str | None = None
    environment: tuple[tuple[str, str], ...] = ()
    rollout_arm: str | None = None
    evaluation_protocol: str | None = None
    overrides: tuple[str, ...] = ()


SHORT = Budget(32, 4, None, None)
LONG = Budget(200, 16, 50, 100)
MODEL = Budget(250, 1, 20, 100, initial_validation=True, keep=4, align_seeds=True, lr="1e-6")
DEFAULT_PAIR = ModelPair()
QWEN_BASE = ModelPair(
    student="Qwen3-1.7B-Base",
    teacher="Qwen3-8B",
    family="qwen3",
    prompt=1024,
    student_sequences=64,
    student_memory_utilization=0.8,
    teacher_sequences=64,
    stop_token=151645,
)
QWEN_CHAT = ModelPair(
    student="Qwen3-1.7B",
    teacher="Qwen3-4B-Instruct-2507",
    family="qwen3",
    response=4096,
    student_sequences=64,
    teacher_sequences=64,
)
# sampled / full / sampled-reuse16 / full-reuse16; one row owns each model protocol.
MODEL_MATRIX = (
    (15, 13, 33, 17, DEFAULT_PAIR),
    (23, 24, 34, 35, replace(DEFAULT_PAIR, response=15360)),
    (25, 26, 36, 37, replace(DEFAULT_PAIR, teacher="DeepSeek-R1-Distill-Qwen-7B", teacher_sequences=32)),
    (27, 28, 38, 39, QWEN_BASE),
    (29, 30, 40, 41, replace(QWEN_BASE, teacher="Qwen3-4B")),
    (31, 32, 42, 43, QWEN_CHAT),
)
ALIASES = {"E21": "E15", "E22": "E13"}
RETIRED_PRESETS = {"E69", "E70"}

# One E20-based protocol, four registered arms. Runtime/environment overrides
# retain the same precedence as older presets; these are defaults, not locks.
MIRROR_COMPARISON_ENV = (
    ("VAL_N", "16"),
    ("FULL_KL_BACKEND", "precompute_fused"),
    ("FULL_KL_TOKEN_CHUNK", "128"),
    ("FULL_KL_ASYNC_WRITE", "True"),
    ("FULL_KL_PREFETCH", "True"),
    ("MIRROR_DIAGNOSTICS", "True"),
)
MIRROR_COMPARISON_METHODS = (
    (44, "full_kl", None, None),
    (45, "mdpo", "0", "0.3"),
    (46, "bpo", "0", "0.3"),
    (47, "mdpo", "0.5", "0.3"),
)

QWEN_MOE_LONG_METHODS = (
    (55, "mdpo", "0", "0.1"),
    (56, "mdpo", "0.5", "0.1"),
    (57, "mdpo", "1", "0.1"),
    (58, "full_kl", None, None),
)
QWEN_MOE_LONG_ENV = (
    ("STUDENT_GPUS", "8"),
    ("TEACHER_GPUS", "0"),
    ("DISTILLATION_EXECUTION_MODE", "shared_sync"),
    ("VAL_N", "16"),
    ("FULL_KL_BACKEND", "precompute"),
    ("FULL_KL_TOKEN_CHUNK", "512"),
    ("FULL_KL_ASYNC_WRITE", "True"),
    ("FULL_KL_PREFETCH", "True"),
    ("FULL_KL_TERMINATION_TOKEN_IDS", "[151643,151645]"),
    ("MIRROR_DIAGNOSTICS", "True"),
)


PRECISION_COMPARISONS = (
    (50, 51, "fp16", "FP16 + GradScaler"),
    (52, 53, "bf16_fp32", "BF16 GEMM / FP32 intermediates"),
)


SAMPLED_SIGNALS = (
    ("rollout", "token_ppo"),
    ("rollout", "token_is"),
    ("current", "token_ppo"),
    ("current", "token_is"),
    ("current", "reinforce"),
    ("rollout", "reinforce"),
)


def preset_catalog() -> dict[str, Preset]:
    presets = {}
    for i, (source, policy) in enumerate(SAMPLED_SIGNALS, 1):
        for offset, budget in ((0, SHORT), (6, LONG)):
            presets[f"E{i + offset}"] = Preset(budget, advantage=source, policy=policy)
    for sampled, full, sampled16, full16, pair in MODEL_MATRIX:
        for number, method, reuse in (
            (sampled, "sampled", 1),
            (full, "full_kl", 1),
            (sampled16, "sampled", 16),
            (full16, "full_kl", 16),
        ):
            presets[f"E{number}"] = Preset(replace(MODEL, reuse=reuse), method=method, pair=pair)
    # Follow-up Qwen runs use the same 100-step validation/rollout protocol.
    # Keep the longer budgets and chat-model cap of the other arms intact.
    for name in ("E41", "E43"):
        presets[name] = replace(
            presets[name],
            budget=replace(MODEL, reuse=16, steps=100, test=10, save=50, initial_validation=name == "E43"),
            pair=replace(presets[name].pair, response=7168, student_sequences=64, student_memory_utilization=0.8),
        )
    presets["E14"] = replace(presets["E13"], budget=replace(MODEL, lr="5e-7"))
    presets["E16"] = replace(presets["E13"], budget=replace(MODEL, reuse=4))
    for number, reuse, steps, test, save in ((18, 64, 64, 8, 16), (19, 32, 128, 16, 32), (20, 16, 256, 32, 64)):
        presets[f"E{number}"] = Preset(
            Budget(steps, reuse, test, save, keep=4, align_seeds=True, original_data=True, lr="1e-6"), method="full_kl"
        )
    for number, method, kappa, eta in MIRROR_COMPARISON_METHODS:
        presets[f"E{number}"] = Preset(
            replace(presets["E20"].budget, save=128),
            method=method,
            pair=DEFAULT_PAIR,
            kappa=kappa,
            eta=eta,
            environment=MIRROR_COMPARISON_ENV,
        )
    # Polaris cross-origin teacher, retaining E20's optimizer and rollout budget.
    presets["E48"] = replace(
        presets["E20"],
        budget=replace(presets["E20"].budget, initial_validation=True),
        pair=replace(DEFAULT_PAIR, teacher="Polaris-7B-Preview", teacher_sequences=32),
    )
    presets["E49"] = replace(
        presets["E48"],
        method="full_forward_kl",
        environment=(("FULL_KL_BACKEND", "precompute_fused"), ("FULL_KL_FORWARD_TEACHER_SUPPORT", "student")),
    )
    for opd, mdpo, precision, _ in PRECISION_COMPARISONS:
        for number, base in ((opd, "E44"), (mdpo, "E45")):
            presets[f"E{number}"] = replace(
                presets[base], environment=(*presets[base].environment, ("TRAIN_PRECISION", precision))
            )
    presets["E54"] = replace(presets["E41"], pair=replace(presets["E41"].pair, teacher="Qwen3-30B-A3B-Instruct-2507"))
    for number, method, kappa, eta in QWEN_MOE_LONG_METHODS:
        presets[f"E{number}"] = Preset(
            replace(presets["E54"].budget, save=-1, keep=None),
            method=method,
            pair=replace(presets["E54"].pair, response=16384, enable_thinking=None),
            kappa=kappa,
            eta=eta,
            environment=QWEN_MOE_LONG_ENV,
        )
    # New IDs preserve the recorded E55-E58 execution settings. The fused
    # comparison changes only the numerical backend and makes BF16 explicit.
    for number, reference in enumerate(("E55", "E56", "E57", "E58"), start=59):
        environment = dict(presets[reference].environment)
        environment.update(FULL_KL_BACKEND="precompute_fused", TRAIN_PRECISION="bf16")
        presets[f"E{number}"] = replace(presets[reference], environment=tuple(environment.items()))
    for number, arm in ((63, "S"), (64, "S0"), (65, "T"), (66, "LM"), (67, "PM"), (68, "TM")):
        presets[f"E{number}"] = Preset(
            Budget(40, 16, -1, -1, initial_validation=True, align_seeds=True, original_data=True, lr="1e-6"),
            method="full_kl",
            pair=replace(QWEN_CHAT, response=7168, student_sequences=8),
            rollout_arm=arm,
        )
    for number, teacher in ((71, "MiMo-7B-RL"), (72, "MiMo-7B-RL-Zero"), (73, "MiMo-7B-SFT")):
        presets[f"E{number}"] = Preset(
            Budget(50, 32, -1, -1, initial_validation=True, align_seeds=True, original_data=True, lr="1e-6"),
            method="full_kl",
            pair=ModelPair(
                student="MiMo-7B-Base",
                teacher=teacher,
                family="qwen25",
                prompt=2048,
                response=7168,
                student_sequences=16,
                teacher_sequences=16,
            ),
            environment=(
                ("STUDENT_GPUS", "4"),
                ("TEACHER_GPUS", "0"),
                ("DISTILLATION_EXECUTION_MODE", "shared_sync"),
                ("TRAIN_BATCH_SIZE", "128"),
                ("MAX_VAL_RESPONSE_LENGTH", "24576"),
                ("FULL_KL_BACKEND", "precompute_fused"),
                ("FULL_KL_TOKEN_CHUNK", "256"),
                ("FULL_KL_TERMINATION_TOKEN_IDS", "[]"),
                ("MIRROR_DIAGNOSTICS", "False"),
            ),
            evaluation_protocol="mimo",
        )
    for number, student, teacher, chain, family in (
        (74, "Olmo3-Base-BayesProtocol", "Olmo3-C3PO-BayesProtocol", "bayesrl", "default"),
        (75, "Olmo3-Base-BayesProtocol", "Olmo3-SFT-BayesProtocol", "bayesrl", "default"),
        (76, "Qwen2.5-14B-R1Protocol", "Light-R1-14B-R1Protocol", "light_r1", "deepseek"),
        (77, "Qwen2.5-14B-R1Protocol", "R1-Distill-14B-R1Protocol", "light_r1", "deepseek"),
    ):
        presets[f"E{number}"] = replace(
            presets["E71"],
            pair=ModelPair(
                student=student,
                teacher=teacher,
                family=family,
                prompt=2048,
                response=7168,
                student_sequences=4 if chain == "light_r1" else 8,
                teacher_sequences=4 if chain == "light_r1" else 8,
            ),
            evaluation_protocol=chain,
        )
    # MiMo budget amendment; keep the independently planned Olmo/Light-R1 budgets above.
    for number, steps, reuse in ((71, 25, 32), (72, 20, 16), (73, 20, 16)):
        preset = presets[f"E{number}"]
        presets[f"E{number}"] = replace(preset, budget=replace(preset.budget, steps=steps, reuse=reuse))
    # E91 tests whether an already-SFT MiMo student absorbs the existing
    # direct-RL teacher. E72 already covers the same teacher with Base student.
    presets["E91"] = replace(
        presets["E72"],
        pair=replace(presets["E72"].pair, student="MiMo-7B-SFT"),
    )
    for number, arm in ((78, "S"), (79, "S0"), (80, "T")):
        presets[f"E{number}"] = Preset(
            Budget(100, 16, -1, -1, initial_validation=True, align_seeds=True, original_data=True, lr="1e-6"),
            method="full_kl",
            pair=replace(QWEN_CHAT, response=7168, student_sequences=16),
            rollout_arm=arm,
            environment=(("TRAIN_BATCH_SIZE", "128"), ("QWEN_PREFIX_PROTOCOL", "source_b128_v1")),
        )
    # A/B retain the E57 target/reuse and use the accepted E61 fused path.
    # Stopping follows current defaults; historical E57 used natural stopping.
    for number, steps in ((81, 25), (82, 100)):
        environment = dict(presets["E61"].environment)
        environment.update(
            STUDENT_GPUS="4",
            TRAIN_BATCH_SIZE="256",
            ROLLOUT_TP="1",
            VAL_BATCH_SIZE="null",
            TOTAL_EPOCHS="2" if number == 82 else "1",
        )
        presets[f"E{number}"] = replace(
            presets["E61"],
            budget=replace(presets["E61"].budget, steps=steps, test=5, initial_validation=False),
            pair=replace(presets["E61"].pair, student_sequences=64),
            environment=tuple(environment.items()),
        )
    for number, future, aggregation in ((83, "none", "seq-mean-token-mean"), (84, "mean", "token-mean")):
        environment = dict(presets["E82"].environment)
        if number == 84:
            environment["STUDENT_GPUS"] = "8"
            environment["VAL_N_PROFILE"] = "math4_rest16"
        presets[f"E{number}"] = replace(
            presets["E82"],
            budget=replace(presets["E82"].budget, test=20 if number == 84 else 5, save=25, keep=1),
            environment=tuple(environment.items()),
            overrides=(
                f"distillation.distillation_loss.full_vocab.future.mode={future}",
                "distillation.distillation_loss.full_vocab.future.gamma=1.0",
                f"actor_rollout_ref.actor.loss_agg_mode={aggregation}",
                "actor_rollout_ref.actor.checkpoint.save_contents=[model,optimizer,extra,hf_model]",
                "actor_rollout_ref.actor.checkpoint.load_contents=[model,optimizer,extra]",
                "+actor_rollout_ref.actor.checkpoint.preserve_hf_master_weights=True",
                "trainer.checkpoint_callback_class=examples.mirroropd.research.studies.weighted_targets.checkpoint.LatestTrainingState",
            ),
        )
    # Eight-GPU follow-ups deliberately run without durable model or optimizer
    # checkpoints.  Keep E82's budget and MDPO target fixed while isolating the
    # remaining aggregation and future-credit choices.
    eight_gpu_ablation = (
        (85, "none", "token-sum", "aggregation: token-sum"),
        (86, "none", "seq-mean-token-sum", "aggregation: seq-mean-token-sum"),
        (87, "none", "seq-mean-token-sum-norm", "aggregation: seq-mean-token-sum-norm"),
        (88, "sum", "token-mean", "future: sum"),
        (89, "discounted_sum", "token-mean", "future: discounted-sum, gamma=.99"),
        (90, "discounted_mean", "token-mean", "future: discounted-mean, gamma=.99"),
    )
    for number, future, aggregation, _label in eight_gpu_ablation:
        environment = dict(presets["E82"].environment)
        environment["STUDENT_GPUS"] = "8"
        if number >= 87:
            environment["VAL_N_PROFILE"] = "math4_rest16"
        presets[f"E{number}"] = replace(
            presets["E82"],
            budget=replace(presets["E82"].budget, test=20, save=-1, keep=None),
            environment=tuple(environment.items()),
            overrides=(
                f"distillation.distillation_loss.full_vocab.future.mode={future}",
                "distillation.distillation_loss.full_vocab.future.gamma=0.99"
                if future.startswith("discounted_")
                else "distillation.distillation_loss.full_vocab.future.gamma=1.0",
                f"actor_rollout_ref.actor.loss_agg_mode={aggregation}",
            ),
        )
    # Keep E89/E90 at gamma=.99; new IDs isolate the shorter discount horizon.
    for number, reference in ((92, "E89"), (93, "E90")):
        preset = presets[reference]
        presets[f"E{number}"] = replace(
            preset,
            overrides=tuple(
                "distillation.distillation_loss.full_vocab.future.gamma=0.95"
                if value.startswith("distillation.distillation_loss.full_vocab.future.gamma=")
                else value
                for value in preset.overrides
            ),
        )
    return presets


PRESETS = preset_catalog()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        prog="run_math_opd_ablation_fsdp2.sh",
        description="One math experiment entry: model/budget preset + independent method.",
        epilog=", ".join(f"{alias}={name}" for alias, name in ALIASES.items())
        + ". Model presets accept -valid-student/-valid-teacher.\n"
        "Precedence: preset < environment < method options < final Hydra overrides.\n"
        "custom uses shared defaults without an E budget; DRY_RUN=1 prints the final command.\n"
        "Use --list-presets for the catalog; see examples/mirroropd/docs/settings/训练设置.md for protocols.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    result.add_argument("preset", nargs="?", help=f"E1–E{max(int(k[1:]) for k in PRESETS)} or custom")
    result.add_argument("--list-presets", action="store_true", help="list every preset without launching")
    result.add_argument("--overview", action="store_true", help="show grouped comparison tables without launching")
    result.add_argument(
        "--precision",
        choices=("bf16", "fp16", "fp32", "bf16_fp32"),
        help="actor/anchor execution profile; teacher and rollout keep their own precision",
    )
    result.add_argument("--method", choices=("sampled", "full_kl", "full", "full_forward_kl", "mdpo", "bpo"))
    result.add_argument("--estimator", choices=("sampled", "full_vocab"))
    result.add_argument("--kappa", help="MDPO/BPO target shape (preset default, otherwise 0.5)")
    result.add_argument("--eta", help="MDPO/BPO strength (preset default, otherwise 0.3); full_kl fixes eta=1")
    result.add_argument("--advantage", choices=("current", "rollout"), help="legacy sampled only")
    result.add_argument(
        "--policy-loss", choices=("vanilla", "token_is", "token_ppo", "reinforce"), help="legacy sampled only"
    )
    return result


def defaults(env: dict[str, str], **values) -> None:
    """Match Bash ${NAME:-default}, including explicitly empty environment values."""
    for key, value in values.items():
        if value is not None and not env.get(key):
            env[key] = str(value)


def positive_steps(value: str) -> str:
    if not re.fullmatch(r"[1-9][0-9]*", value):
        raise ValueError("TOTAL_TRAINING_STEPS must be a positive integer")
    return value


def resolve_selector(selector: str) -> tuple[str, Preset | None, str | None]:
    if selector == "custom":
        return selector, None, None
    match = re.fullmatch(r"(E[0-9]+)(?:-valid-(student|teacher))?", selector)
    if not match:
        raise ValueError(f"Unknown preset: {selector}")
    if match[1] in RETIRED_PRESETS:
        raise ValueError("E69/E70 DeepSeekMath presets were withdrawn; use MiMo E71/E72/E73")
    name = ALIASES.get(match[1], match[1])
    preset = PRESETS.get(name)
    if preset is None or (match[2] and preset.pair is None):
        raise ValueError(f"Unsupported preset/validation selector: {selector}")
    return name, preset, match[2]


def model_defaults(env: dict[str, str], pair: ModelPair, role: str | None, root: Path) -> list[str]:
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from examples.mirroropd.assets.paths import available_model

    defaults(env, MODEL_ROOT=root.parent / "model")
    for key, alias in (("STUDENT_MODEL", pair.student), ("TEACHER_MODEL", pair.teacher)):
        if not env.get(key):
            env[key] = str(available_model(alias, root=env["MODEL_ROOT"]))
    defaults(
        env,
        CONTINUOUS_TOKEN_MODEL_FAMILY=pair.family,
        MAX_PROMPT_LENGTH=4096 if role == "teacher" else pair.prompt,
        MAX_RESPONSE_LENGTH=pair.response,
        MAX_VAL_RESPONSE_LENGTH=31744,
        VAL_N=16,
        VAL_BATCH_SIZE="null",
        ROLLOUT_MAX_NUM_SEQS=pair.teacher_sequences if role == "teacher" else pair.student_sequences,
        ROLLOUT_GPU_MEMORY_UTILIZATION=(
            pair.teacher_memory_utilization if role == "teacher" else pair.student_memory_utilization
        ),
        SEED=42,
    )
    overrides = []
    if pair.family == "qwen3":
        thinking = "null" if pair.enable_thinking is None else str(pair.enable_thinking)
        overrides.append(f"+data.apply_chat_template_kwargs.enable_thinking={thinking}")
    if pair.stop_token is not None and role != "teacher":
        overrides.append(f"actor_rollout_ref.rollout.stop_token_ids=[{pair.stop_token}]")
    return overrides


def seed_overrides(env: dict[str, str]) -> list[str]:
    defaults(env, SEED=42)
    return [
        f"actor_rollout_ref.{field}={env['SEED']}"
        for field in ("actor.fsdp_config.seed", "actor.data_loader_seed", "rollout.seed")
    ]


def budget_overrides(env: dict[str, str], budget: Budget) -> list[str]:
    steps = positive_steps(env.get("TOTAL_TRAINING_STEPS") or str(budget.steps))
    defaults(
        env,
        TRAIN_BATCH_SIZE=64,
        PPO_EPOCHS=budget.reuse,
        TOTAL_EPOCHS=1,
        LEARNING_RATE=budget.lr,
        TEST_FREQ=steps if budget.test is None else budget.test,
        SAVE_FREQ=steps if budget.save is None else budget.save,
    )
    result = [
        f"trainer.total_training_steps={steps}",
        "actor_rollout_ref.actor.optim.lr_scheduler_type=constant",
        "actor_rollout_ref.actor.optim.lr_warmup_steps=0",
        "actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0",
        "actor_rollout_ref.actor.checkpoint.async_save=False",
    ]
    if budget.initial_validation:
        result.append("trainer.val_before_train=True")
    if budget.keep is not None:
        result.append(f"trainer.max_actor_ckpt_to_keep={budget.keep}")
    if budget.align_seeds:
        result.extend(seed_overrides(env))
    if budget.original_data:
        result.extend(
            (
                "data.train_max_samples=-1",
                "algorithm.filter_groups.enable=False",
                "trainer.v1.sampler.sync_refill_failed_groups=False",
            )
        )
    return result


def method_options(args, env: dict[str, str], default_method: str) -> tuple[str, str, str]:
    method = args.method or env.get("EXPERIMENT_METHOD") or default_method
    method = "full_kl" if method == "full" else method
    if method not in ("sampled", "full_kl", "full_forward_kl", "mdpo", "bpo"):
        raise ValueError(f"Unknown method: {method}")
    mirror = method in ("mdpo", "bpo")
    if not mirror and (args.kappa is not None or args.eta is not None):
        raise ValueError("--kappa/--eta require --method mdpo or bpo; full_kl fixes eta=1")
    if method != "sampled" and (args.advantage or args.policy_loss):
        raise ValueError("--advantage/--policy-loss apply only to legacy sampled")
    for field, key in (
        ("kappa", "MIRROR_KAPPA"),
        ("eta", "MIRROR_ETA"),
        ("advantage", "OPD_ADVANTAGE_SOURCE"),
        ("policy_loss", "OPD_POLICY_LOSS_MODE"),
    ):
        value = getattr(args, field)
        if value is not None:
            if not value:
                raise ValueError(f"Empty value for --{field.replace('_', '-')}")
            env[key] = value
    default_estimator = "sampled" if method == "sampled" else "full_vocab"
    if mirror:
        default_estimator = env.get("MIRROR_ESTIMATOR") or default_estimator
    estimator = args.estimator or default_estimator
    allowed = {
        "sampled": ("sampled",),
        "full_kl": ("full_vocab",),
        "full_forward_kl": ("full_vocab",),
        "mdpo": ("sampled", "full_vocab"),
        "bpo": ("full_vocab",),
    }
    if estimator not in allowed[method]:
        raise ValueError(f"Unsupported method/estimator: {method}/{estimator}")
    if method == "sampled":
        if (env.get("OPD_ADVANTAGE_SOURCE") or "current") not in ("current", "rollout"):
            raise ValueError("Invalid sampled advantage")
        if (env.get("OPD_POLICY_LOSS_MODE") or "vanilla") not in ("vanilla", "token_is", "token_ppo", "reinforce"):
            raise ValueError("Invalid sampled policy loss")
    env["EXPERIMENT_METHOD"] = method
    if mirror:
        env["MIRROR_ESTIMATOR"] = estimator
        label = f"-{method}-{estimator}-k{env.get('MIRROR_KAPPA') or '0.5'}-eta{env.get('MIRROR_ETA') or '0.3'}"
    else:
        label = "" if method == default_method else f"-{method}"
    return method, estimator, label


@dataclass(frozen=True)
class LaunchPlan:
    entry: Path
    arguments: tuple[str, ...]
    environment: dict[str, str]

    @property
    def command(self) -> list[str]:
        return ["bash", str(self.entry), *self.arguments]


def build_plan(args, hydra: list[str], environment: dict[str, str], root: Path = ROOT) -> LaunchPlan:
    name, preset, role = resolve_selector(args.preset)
    env = environment.copy()
    if preset:
        defaults(env, **dict(preset.environment))
    entry_dir = root / "examples/mirroropd"
    options = ["++rollout_model_resolver=examples.mirroropd.assets.paths.rollout_model_path"]
    prefix_options = []
    if preset and preset.rollout_arm:
        if role:
            raise ValueError(
                "Qwen source presets use their scientific readouts; independent model validation remains under E43"
            )
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from examples.mirroropd.launchers.qwen_prefix import options as prefix_defaults

        stamp = datetime.now().strftime("%Y%m%d-%H%M%S") + f"-{os.getpid()}"
        defaults(env, EXPERIMENT_NAME=f"math-opd-{name}-{stamp}")
        defaults(env, RUN_DIR=root / "outputs" / env["EXPERIMENT_NAME"])
        defaults(env, CHECKPOINT_DIR=Path(env["RUN_DIR"]) / "runtime")
        prefix_options = prefix_defaults(env, preset.rollout_arm, root)
    if preset and preset.evaluation_protocol in {"mimo", "bayesrl", "light_r1"}:
        if role:
            raise ValueError(
                "Stage-chain presets evaluate original models through their native bootstrap, not -valid selectors"
            )
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S") + f"-{os.getpid()}"
        defaults(env, EXPERIMENT_NAME=f"math-opd-{name}-{stamp}")
        namespace = "mimo-stage-native" if preset.evaluation_protocol == "mimo" else "stage-chain-native"
        defaults(env, RUN_DIR=root / "runs" / namespace / env["EXPERIMENT_NAME"])
        defaults(env, CHECKPOINT_DIR=Path(env["RUN_DIR"]) / "checkpoints")
        if preset.evaluation_protocol == "mimo":
            from examples.mirroropd.launchers.mimo import options as mimo_defaults

            prefix_options = mimo_defaults(env, name, root, default_outer=preset.budget.steps)
        else:
            from examples.mirroropd.launchers.stage_chains import options as stage_defaults

            prefix_options = stage_defaults(env, name, root, chain=preset.evaluation_protocol)
    protocol = model_defaults(env, preset.pair, role, root) if preset and preset.pair else []
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S") + f"-{os.getpid()}"
    if role:
        if any(
            getattr(args, k) is not None
            for k in ("method", "estimator", "kappa", "eta", "advantage", "policy_loss", "precision")
        ) or env.get("EXPERIMENT_METHOD"):
            raise ValueError("Independent validation does not accept method options")
        defaults(env, EXPERIMENT_NAME=f"math-opd-{name}-valid-{role}-{stamp}")
        return LaunchPlan(
            entry_dir / "run_math_validation_fsdp2.sh", tuple([role, *seed_overrides(env), *protocol, *hydra]), env
        )
    if preset:
        env["DOMAIN"] = "math"
        defaults(env, OPD_ADVANTAGE_SOURCE=preset.advantage, OPD_POLICY_LOSS_MODE=preset.policy)
        options.extend(budget_overrides(env, preset.budget))
        options.extend(preset.overrides)
    elif env.get("TOTAL_TRAINING_STEPS"):
        options.append(f"trainer.total_training_steps={positive_steps(env['TOTAL_TRAINING_STEPS'])}")
    default_method = preset.method if preset else "sampled"
    if preset and (args.method or env.get("EXPERIMENT_METHOD") or default_method) in ("mdpo", "bpo"):
        defaults(env, MIRROR_KAPPA=preset.kappa, MIRROR_ETA=preset.eta)
    method, estimator, label = method_options(args, env, default_method)
    precision = args.precision or env.get("TRAIN_PRECISION")
    if precision:
        if precision not in ("bf16", "fp16", "fp32", "bf16_fp32"):
            raise ValueError(f"Unknown TRAIN_PRECISION: {precision}")
        options.append(f"actor_rollout_ref.actor.fsdp_config.forward_precision={precision}")
        env["TRAIN_PRECISION"] = precision
        label += f"-{precision}"

    if name in ("E41", "E54") and (method == "full_kl" or (method == "mdpo" and estimator == "full_vocab")):
        # Both IDs already terminate this single-turn Base rollout. Coarsen
        # their loss identity only for the Base student protocol used in E41 and E54.
        defaults(env, FULL_KL_TERMINATION_TOKEN_IDS="[151643,151645]")
    # Preserve preset IO defaults, including inactive full-vocab fields in sampled MDPO.
    if preset and (preset.method == "full_kl" or estimator == "full_vocab"):
        defaults(env, FULL_KL_BACKEND="precompute", FULL_KL_ASYNC_WRITE="True", FULL_KL_PREFETCH="True")
    defaults(env, EXPERIMENT_NAME=f"{env.get('DOMAIN') or 'math'}-opd-{name}{label}-{stamp}")
    return LaunchPlan(
        entry_dir / "launchers/_run_single_teacher_fsdp2.sh", tuple([*prefix_options, *options, *protocol, *hydra]), env
    )


def _aligned_table(headers, rows):
    table = [tuple(map(str, headers)), *(tuple(map(str, row)) for row in rows)]
    widths = [max(len(row[i]) for row in table) for i in range(len(headers))]
    return "\n".join(
        "  ".join(value.ljust(width) for value, width in zip(row, widths, strict=True)).rstrip() for row in table
    )


def _short_model(name):
    return (
        name.replace("DeepSeek-R1-Distill-Qwen-", "R1-").replace("JustRL-DeepSeek-", "JustRL-").replace("Qwen3-", "Q3-")
    )


def _preset_ranges(numbers):
    groups = []
    for number in sorted(numbers):
        if groups and number == groups[-1][-1] + 1:
            groups[-1].append(number)
        else:
            groups.append([number])
    return ",".join(f"E{g[0]}" if len(g) == 1 else f"E{g[0]}-E{g[-1]}" for g in groups)


def preset_overview() -> str:
    """Compact human reference, shared by --overview and the public entry comment."""
    pairs = []
    for sampled, full, sampled16, full16, pair in MODEL_MATRIX:
        names = [f"E{i}" for i in (sampled, full, sampled16, full16)]
        # A selected arm may override the family's rollout cap.
        for cap in dict.fromkeys(PRESETS[name].pair.response for name in names):
            pairs.append(
                tuple(name if PRESETS[name].pair.response == cap else "-" for name in names)
                + (_short_model(pair.student), _short_model(pair.teacher), cap)
            )
    signals = [
        (f"E{i}", f"E{i + len(SAMPLED_SIGNALS)}", PRESETS[f"E{i}"].advantage, PRESETS[f"E{i}"].policy)
        for i in range(1, len(SAMPLED_SIGNALS) + 1)
    ]
    comparison = [
        (
            name,
            p.method,
            "full_vocab",
            p.kappa if p.kappa is not None else "-",
            p.eta or "1",
            _short_model(p.pair.student),
            _short_model(p.pair.teacher),
            dict(p.environment)["VAL_N"],
            dict(p.environment)["FULL_KL_BACKEND"],
            dict(p.environment)["FULL_KL_TOKEN_CHUNK"],
        )
        for number, *_ in MIRROR_COMPARISON_METHODS
        for name in (f"E{number}",)
        for p in (PRESETS[name],)
    ]
    grouped = {}
    for name, preset in sorted(PRESETS.items(), key=lambda item: int(item[0][1:])):
        grouped.setdefault(preset.budget, []).append(int(name[1:]))
    budgets = [
        (
            _preset_ranges(ids),
            b.reuse,
            b.steps,
            b.steps * b.reuse,
            b.lr or "shared",
            f"{b.steps if b.test is None else b.test}/{b.steps if b.save is None else b.save}",
            "yes" if b.initial_validation else "no",
        )
        for b, ids in grouped.items()
    ]
    # Read frequently compared settings from the catalog so the comment stays
    # aligned when an arm changes GPU count, future mode or validation budget.
    ablations = []
    for number in (*range(81, 91), 92, 93):
        name = f"E{number}"
        preset = PRESETS[name]
        env = dict(preset.environment)
        overrides = dict(value.split("=", 1) for value in preset.overrides)
        future_prefix = "distillation.distillation_loss.full_vocab.future."
        ablations.append(
            (
                name,
                env["STUDENT_GPUS"],
                overrides.get(future_prefix + "mode", "none"),
                overrides.get(future_prefix + "gamma", "1.0"),
                overrides.get("actor_rollout_ref.actor.loss_agg_mode", "token-mean"),
                preset.budget.test,
                "4/16" if env.get("VAL_N_PROFILE") == "math4_rest16" else "16/16",
                preset.budget.save,
            )
        )
    return "\n\n".join(
        [
            "用法与配置优先级\n"
            "  bash examples/mirroropd/run_math_opd_ablation_fsdp2.sh --help\n"
            "  bash examples/mirroropd/run_math_opd_ablation_fsdp2.sh --overview\n"
            "  bash examples/mirroropd/run_math_opd_ablation_fsdp2.sh --list-presets\n"
            "  DRY_RUN=1 bash examples/mirroropd/run_math_opd_ablation_fsdp2.sh E84\n"
            "  bash examples/mirroropd/run_math_opd_ablation_fsdp2.sh E84 -- trainer.test_freq=40\n"
            "优先级：编号预设 < 环境变量 < --method/--kappa/--eta 等选项 < 最后的 Hydra 覆盖。\n"
            "DRY_RUN=1 仅打印最终命令；-- 后的参数原样传给 Hydra。custom 使用公共默认值，不加载编号预算。\n"
            "本注释与 --overview 共用 launchers/math_experiment.py 的 preset_overview()；修改源后同步生成区。\n"
            "以下均为当前预设默认值；已启动实验以其 resolved config 为准。",
            "一、E81–E90 / E92–E93：Qwen Base 的聚合与 future reward 对照\n"
            "学生 Qwen3-1.7B-Base ← 教师 Qwen3-30B-A3B-Instruct-2507。\n"
            "共同训练：full-vocab MDPO kappa=1/eta=0.1，LR=1e-6；shared_sync。\n"
            "每 outer 抽取 256 题，rollout n=1，生成 256 条新回答；mini-batch=256，同批 reuse=16。\n"
            "E81 为 25 outer/400 次更新；E82–E90、E92/E93 为 100 outer/1600 次更新。\n"
            "执行：precompute_fused/chunk512、bf16；TP1，每引擎并发 64、显存利用率 0.8。\n"
            "长度：训练 prompt/response=1024/16384，验证 response cap=31744。\n"
            + _aligned_table(
                ("Preset", "GPUs", "Future", "Gamma", "Aggregation", "Valid every", "Val n M/rest", "Save every"),
                ablations,
            )
            + "\n表中 Valid/Save 均以 outer 为单位；Save=-1 表示不保存 checkpoint。\n"
            "验证：MATH-500、AMC23、AIME24/25/26；M/rest 表示 MATH-500/其余四集的每题回答数。\n"
            "16/16 每次 10080 条回答，4/16 每次 4080 条；各组均无初始验证。\n"
            "每 20 outer 验证对应 20/40/60/80/100，共 5 次；每 5 outer 的组也包含终点验证。\n"
            "训练与验证均默认开启 repetition stop：16–128 token 模式重复至少 8 次时提前停止。\n"
            "E83/E84 每 25 outer 保存完整训练状态，仅保留最新一份，另保留最终 HF 模型。\n"
            "E81/E82、E85–E90、E92/E93 不保存 checkpoint、最终模型或 optimizer。\n"
            "E85–E90、E92/E93 resume_mode=disable。\n"
            "本组未配置固定 bank RKL；训练 teacher RKL 读取当前训练前缀。\n"
            "实验解释见 docs/experiments/learning-signal/weighted-targets.md。",
            "二、E1–E53：算法、模型、长度与精度对照",
            "模型/长度对照：Sampled=legacy current+vanilla；Full=Full-KL（MDPO eta=1）。\n"
            + _aligned_table(("Sampled", "Full", "Sampled-r16", "Full-r16", "Student", "Teacher", "Train cap"), pairs),
            "模型缩写：R1=DeepSeek-R1-Distill-Qwen；Q3=Qwen3；JustRL=JustRL-DeepSeek。\n"
            + (
                f"E14 与 E13 同模型，LR={PRESETS['E14'].budget.lr}；"
                f"E16 与 E13 同模型，reuse={PRESETS['E16'].budget.reuse}。"
            ),
            "Sampled 算法对照：Short/Long 的预算见下表。\n"
            + _aligned_table(("Short", "Long", "Advantage", "Policy loss"), signals),
            "Mirror 正式对照：直接调用编号即可；prompt/response=4096/7168，验证 cap=31744。\n"
            + _aligned_table(
                ("Preset", "Method", "Estimator", "Kappa", "Eta", "Student", "Teacher", "Val n", "Backend", "Chunk"),
                comparison,
            )
            + "\nE44–E47：batch=64，seed=42，IO async/prefetch 开启；MDPO/BPO 开启诊断，Full-KL 沿用 endpoint 指标。",
            "E48：R1-1.5B student / Polaris-7B-Preview teacher；沿用 E20 Full-KL、reuse16 与预算，开启初始验证。\n"
            "Teacher FSDP token budget 沿用公共默认 32768；支持 E48-valid-student / E48-valid-teacher。\n"
            "E49：沿用 E48 模型、预算和验证；改为 full-vocab Forward KL，precompute_fused，"
            "teacher 支持显式条件化到 student 输出行。",
            "精度对照：复用 E44/E45 的模型、256 outer / reuse16 / save128，教师与 rollout 保持原配置。\n"
            + _aligned_table(
                ("Full-KL", "MDPO", "Actor/anchor precision"),
                [
                    ("E44", "E45", "BF16 baseline"),
                    *((f"E{opd}", f"E{mdpo}", label) for opd, mdpo, _, label in PRECISION_COMPARISONS),
                ],
            )
            + "\n--precision bf16|fp16|bf16_fp32 可覆盖预设；fp32 是较慢的 packed math attention 参考。",
            "三、E54–E62：Qwen Base 的 TOP-D 与执行后端对照",
            "E54：Qwen3-1.7B-Base ← Qwen3-30B-A3B-Instruct-2507；沿用 E41 的 100 outer/reuse16，无初始验证。",
            "E55–E58：同一 Base ← 30B-A3B-Instruct-2507 长程对照，从初始权重开始。\n"
            + _aligned_table(
                ("Preset", "Method", "Kappa", "Eta"),
                [(f"E{n}", method, kappa or "-", eta or "1") for n, method, kappa, eta in QWEN_MOE_LONG_METHODS],
            )
            + "\n统一 8 卡 shared_sync、batch64、reuse16、100 outer、LR=1e-6、train cap=16384；"
            "每 10 outer 验证五个数学 bench，val n=16、cap=31744，无初始验证。\n"
            "原生共同前缀 enable_thinking=null；EOS 合并 [151643,151645]；precompute/chunk512。\n"
            "不保存任何 checkpoint（save_freq=-1，包括最后一步）；E54 保留原 7k/旧前缀协议。",
            "E59–E62：分别沿用 E55–E58 的模型、目标和预算，改用已验收的 EOS precompute_fused/chunk512；"
            "actor/anchor 显式 bf16，完整诊断保留。旧 E55–E58 默认不变。",
            "四、E63–E68 / E78–E80：rollout 来源对照",
            "E63–E68：Qwen3-1.7B non-thinking ← Qwen3-4B-Instruct-2507，六种前缀。\n"
            + _aligned_table(
                ("Preset", "Rollout"),
                [
                    (f"E{n}", a)
                    for n, a in (
                        (63, "current"),
                        (64, "initial"),
                        (65, "teacher"),
                        (66, "logit fusion"),
                        (67, "probability fusion"),
                        (68, "batch fusion"),
                    )
                ],
            )
            + "\n默认每实验 8 卡；single 用 8 个 TP1，fusion 用 4 个 TP2；全局 batch64、40 outer/reuse16。"
            "fusion 每副本并发 16/每卡 KV24 GiB，current+ 评估并发 16/KV64 GiB；"
            "计算 token budget32768；shared_sync 保持已支持的参数/优化器卸载。\n"
            "采用 main 统一原生评估与连续调度；QWEN_EVAL_ROOT 指定新银行。旧 HF 评估不可混接；八卡性能须实测。",
            "E78–E80：Qwen3-1.7B non-thinking ← Qwen3-4B-Instruct-2507；current/initial/teacher。\n"
            "batch=mini-batch128，100 outer/reuse16，1600 次更新；默认 2 卡 TP1，可显式配置 1/2/4/8 卡。\n"
            "原生 bench0/50/100，MATH500 n4、AMC23 n16、AIME24/25/26 n32；source128 和双银行各 128，每 10 轮 RKL。",
            "五、E71–E77 / E91：教师训练阶段与学生初始化对照",
            "E69/E70 DeepSeekMath 预设已撤销；E71–E73 固定 MiMo-7B-Base student。\n"
            "E71=MiMo-7B-RL（SFT+RL）；E72=MiMo-7B-RL-Zero（直接 RL）；E73=MiMo-7B-SFT。\n"
            "默认 4 卡 shared_sync、batch128、LR=1e-6；训练题按顺序取用，无重复题目。\n"
            "E71：25 outer/reuse32/800 次更新/3200 题；E72/E73：20 outer/reuse16/320 次更新/2560 题。\n"
            "原生 32K：训练 prompt/response=2048/7168；公共 prompt8192，Math/Code/Science response24576，IF8192。\n"
            "source/history RKL 关闭；四域固定 bank 在 0/1/2 及每 5 outer 评分。\n"
            "完整 bench 在 0 与终点评测；预算超过 25 outer 时增加第 25 轮评测。\n"
            "MIMO_QUESTIONS 指定 prepared 题目，MIMO_BOOTSTRAP 指定对应 teacher 的新银行；"
            "关闭 MTP 推测解码与辅助训练；最终只导出 HF 模型并重载核对。\n"
            "完整设置见 docs/experiments/teacher-transfer/post-training-stages.md。",
            "E91：MiMo-7B-SFT student ← MiMo-7B-RL-Zero direct-RL teacher。\n"
            "复用 E72/E73 的 20 outer、reuse16、320 updates、batch128 设置，仅改变 student 初始化。\n"
            "E72 已覆盖同一 teacher 的 RL→Base。",
            "E74/E75：Olmo3 Base 学习 C3PO（SFT+RL）/IVON-SFT；E76/E77：Qwen2.5-14B Base 学习 Light-R1/R1-Distill。\n"
            "E76/E77 阶段不匹配：Light-R1 在 R1-Distill 之后额外做 3K SFT 再 GRPO，缺 RL 直接起点的 SFT-only 权重。\n"
            "四组使用明确登记的共同 SFT tokenizer 协议视图，原始参数不改；Light-R1 记录 7 个特殊 ID 的标签变化。\n"
            "14B 少于 4 卡时自动使用 FSDP2 逐层 CPU offload，保留 FP32 主权重和 AdamW；默认 4 卡使用阶段搬运。\n"
            "E74–E77 使用 batch128/reuse32/50 outer、训练 2048+7168、公共推理回答 24576/IF8192，关闭 source/history。\n"
            "STAGE_CHAIN_QUESTIONS 指定 prepared 题目，STAGE_CHAIN_BOOTSTRAP 指定独立银行；详见阶段链选型及接入验收。",
            "六、全部编号的预算与覆盖规则",
            "训练预算：Eval/Save 均按 outer step；Updates*=默认 batch==mini-batch 时的更新尝试数。\n"
            + _aligned_table(("Presets", "Reuse", "Outer", "Updates*", "LR", "Eval/Save", "Initial val"), budgets)
            + "\nEval=-1 关闭常规验证；原生 evaluation 协议的 bank/bench 仍按各组独立日程执行。\n"
            "Save=-1 关闭周期 checkpoint；原生协议的最终 HF 导出见各组说明。",
            "shared LR 继承公共 launcher；其余模型、资源、验证长度以最终配置为准。\n"
            + "编号别名："
            + "; ".join(f"{a}={b}" for a, b in ALIASES.items())
            + "。\n"
            + "E44–E47 沿用 E20 预算、每 128 outer 保存；"
            "其他探索可用 --method/--kappa/--eta 覆盖；custom 不加载编号预算。\n"
            + "model 预设可加 -valid-student/-valid-teacher，独立验证原始模型。",
            "E41/E54 Full-KL 与 full-vocab MDPO 默认对齐终止动作 [151643,151645]；"
            "FULL_KL_TERMINATION_TOKEN_IDS=[] 可恢复原始词表目标。",
        ]
    )


def list_presets() -> str:
    rows = ["Preset  Method    Reuse  Outer  Pair / signal"]
    for name, p in sorted(PRESETS.items(), key=lambda kv: int(kv[0][1:])):
        detail = (
            f"{p.pair.student} -> {p.pair.teacher}; cap={p.pair.response}" if p.pair else f"{p.advantage}/{p.policy}"
        )
        if p.rollout_arm:
            world = 2 if dict(p.environment).get("QWEN_PREFIX_PROTOCOL") == "source_b128_v1" else 8
            detail += f"; prefix={p.rollout_arm}; default={world} GPUs; scientific readouts"
        if p.kappa is not None:
            detail += f"; full_vocab kappa={p.kappa} eta={p.eta}"
        if p.environment:
            detail += "; " + " ".join(f"{key}={value}" for key, value in p.environment)
        rows.append(f"{name:<7} {p.method:<9} {p.budget.reuse:<6} {p.budget.steps:<6} {detail}")
    rows.append("Aliases: " + "; ".join(f"{a}={b}" for a, b in ALIASES.items()) + ". custom has no numbered budget.")
    return "\n".join(rows)


def main(argv=None):
    cli = parser()
    argv = list(sys.argv[1:] if argv is None else argv)
    # A literal -- stops our parsing; keep every subsequent Hydra argument intact.
    end = argv.index("--") if "--" in argv else len(argv)
    args, hydra = cli.parse_known_args(argv[:end])
    hydra += argv[end + 1 :] if end < len(argv) else []
    if args.overview:
        print(preset_overview())
        return
    if args.list_presets:
        print(list_presets())
        return
    if args.preset is None:
        if argv:
            cli.error("a preset or custom is required")
        cli.print_help()
        return
    try:
        plan = build_plan(args, hydra, dict(os.environ))
        if PRESETS.get(args.preset) and PRESETS[args.preset].rollout_arm:
            from examples.mirroropd.launchers.qwen_prefix import preflight

            preflight(plan.environment)
        if PRESETS.get(args.preset) and PRESETS[args.preset].evaluation_protocol == "mimo":
            from examples.mirroropd.launchers.mimo import preflight

            preflight(plan.environment)
        elif PRESETS.get(args.preset) and PRESETS[args.preset].evaluation_protocol in {"bayesrl", "light_r1"}:
            from examples.mirroropd.launchers.stage_chains import preflight

            preflight(plan.environment)
    except ValueError as error:
        cli.error(str(error))
    os.execvpe(plan.command[0], plan.command, plan.environment)


if __name__ == "__main__":
    main()
