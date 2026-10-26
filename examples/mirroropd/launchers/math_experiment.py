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


SHORT = Budget(32, 4, None, None)
LONG = Budget(200, 16, 50, 100)
MODEL = Budget(250, 1, 20, 100, initial_validation=True, keep=4, align_seeds=True, lr="1e-6")
DEFAULT_PAIR = ModelPair()
QWEN_BASE = ModelPair(
    student="Qwen3-1.7B-Base",
    teacher="Qwen3-8B",
    family="qwen3",
    prompt=1024,
    student_sequences=24,
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
        "Use --list-presets for the catalog; see examples/mirroropd/docs/训练配方.md for protocols.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    result.add_argument("preset", nargs="?", help=f"E1–E{max(int(k[1:]) for k in PRESETS)} or custom")
    result.add_argument("--list-presets", action="store_true", help="list every preset without launching")
    result.add_argument("--overview", action="store_true", help="show grouped comparison tables without launching")
    result.add_argument("--method", choices=("sampled", "full_kl", "full", "mdpo", "bpo"))
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
    name = ALIASES.get(match[1], match[1])
    preset = PRESETS.get(name)
    if preset is None or (match[2] and preset.pair is None):
        raise ValueError(f"Unsupported preset/validation selector: {selector}")
    return name, preset, match[2]


def model_defaults(env: dict[str, str], pair: ModelPair, role: str | None, root: Path) -> list[str]:
    defaults(env, MODEL_ROOT=root.parent / "model")
    defaults(
        env,
        STUDENT_MODEL=f"{env['MODEL_ROOT']}/{pair.student}",
        TEACHER_MODEL=f"{env['MODEL_ROOT']}/{pair.teacher}",
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
        overrides.append("+data.apply_chat_template_kwargs.enable_thinking=False")
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
    if method not in ("sampled", "full_kl", "mdpo", "bpo"):
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
    options = []
    protocol = model_defaults(env, preset.pair, role, root) if preset and preset.pair else []
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S") + f"-{os.getpid()}"
    if role:
        if any(
            getattr(args, k) is not None for k in ("method", "estimator", "kappa", "eta", "advantage", "policy_loss")
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
    elif env.get("TOTAL_TRAINING_STEPS"):
        options.append(f"trainer.total_training_steps={positive_steps(env['TOTAL_TRAINING_STEPS'])}")
    default_method = preset.method if preset else "sampled"
    if preset and (args.method or env.get("EXPERIMENT_METHOD") or default_method) in ("mdpo", "bpo"):
        defaults(env, MIRROR_KAPPA=preset.kappa, MIRROR_ETA=preset.eta)
    method, estimator, label = method_options(args, env, default_method)
    if name == "E41" and method == "full_kl":
        # Both IDs already terminate this single-turn Base rollout. Coarsen
        # their loss identity only for the model/objective tested in E41.
        defaults(env, FULL_KL_TERMINATION_TOKEN_IDS="[151643,151645]")
    # Preserve preset IO defaults, including inactive full-vocab fields in sampled MDPO.
    if preset and (preset.method == "full_kl" or estimator == "full_vocab"):
        defaults(env, FULL_KL_BACKEND="precompute", FULL_KL_ASYNC_WRITE="True", FULL_KL_PREFETCH="True")
    defaults(env, EXPERIMENT_NAME=f"{env.get('DOMAIN') or 'math'}-opd-{name}{label}-{stamp}")
    return LaunchPlan(entry_dir / "launchers/_run_single_teacher_fsdp2.sh", tuple([*options, *protocol, *hydra]), env)


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
    pairs = [
        (f"E{s}", f"E{f}", f"E{s16}", f"E{f16}", _short_model(pair.student), _short_model(pair.teacher), pair.response)
        for s, f, s16, f16, pair in MODEL_MATRIX
    ]
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
    return "\n\n".join(
        [
            "E 编号速查：以下为默认设置，环境、方法选项和 Hydra 可以覆盖。",
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
            "Teacher FSDP token budget 沿用公共默认 32768；支持 E48-valid-student / E48-valid-teacher。",
            "训练预算：Eval/Save 均按 outer step；Updates*=默认 batch==mini-batch 时的更新尝试数。\n"
            + _aligned_table(("Presets", "Reuse", "Outer", "Updates*", "LR", "Eval/Save", "Initial val"), budgets),
            "shared LR 继承公共 launcher；其余模型、资源、验证长度以最终配置为准。\n"
            + "编号别名："
            + "; ".join(f"{a}={b}" for a, b in ALIASES.items())
            + "。\n"
            + "E44–E47 沿用 E20 预算、每 128 outer 保存；"
            "其他探索可用 --method/--kappa/--eta 覆盖；custom 不加载编号预算。\n"
            + "model 预设可加 -valid-student/-valid-teacher，独立验证原始模型。",
            "E41 Full-KL 默认对齐终止动作 [151643,151645]；FULL_KL_TERMINATION_TOKEN_IDS=[] 可恢复原始词表 KL。",
        ]
    )


def list_presets() -> str:
    rows = ["Preset  Method    Reuse  Outer  Pair / signal"]
    for name, p in sorted(PRESETS.items(), key=lambda kv: int(kv[0][1:])):
        detail = (
            f"{p.pair.student} -> {p.pair.teacher}; cap={p.pair.response}" if p.pair else f"{p.advantage}/{p.policy}"
        )
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
    except ValueError as error:
        cli.error(str(error))
    os.execvpe(plan.command[0], plan.command, plan.environment)


if __name__ == "__main__":
    main()
