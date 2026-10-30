# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Regenerate the lineage inventory and fixed-panel risks without conflating modes or supports."""

import json
from pathlib import Path

from examples.mirroropd.research.core.artifacts import file_hash
from examples.mirroropd.research.core.io import dump, write_csv, write_text
from examples.mirroropd.research.core.paths import REPO, workspace
from examples.mirroropd.research.studies.boundary import common as old
from examples.mirroropd.research.studies.boundary import round2_common as r2
from examples.mirroropd.research.studies.fixed_transfer.display_policy import MAX_DISPLAY_UPDATES, in_main_comparison


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--update-report", action="store_true", help="Refresh only marked numerical report blocks")
    args = parser.parse_args()
    w = workspace()
    root = w / "wandb-logs/MirrorOPD-verl/analysis"
    plans = REPO / "examples/mirroropd/research/studies/fixed_transfer/plans"
    out = REPO / "examples/mirroropd/docs/experiments/teacher-transfer/transfer-mechanism/data/overview-20261029"
    out.mkdir(parents=True, exist_ok=True)
    sets = {
        "R1": {k: dict(repo=v[0], revision=v[1], path=str(r2.model_path(k))) for k, v in r2.MODELS.items()},
        "Qwen3": json.loads((plans / "qwen3-mode-protocol.json").read_text())["models"],
        "Llama": json.loads((plans / "llama-lineage-protocol.json").read_text())["models"],
    }
    for key, model in json.loads((plans / "stage-lineage-models.json").read_text())["models"].items():
        sets["Qwen3" if key.startswith("Q") else "Llama"][key] = model
    inventory = []
    for family, models in sets.items():
        for key, m in models.items():
            p = Path(m.get("path", str(w / "model/teacher-study" / m["repo"].replace("/", "--") / m["revision"])))
            if not p.is_absolute():
                p = w / p
            present = bool(list(p.glob("*.safetensors"))) and (p / "config.json").exists()
            if family != "R1":
                present = present and (p / "complete.json").exists()
            index = p / "model.safetensors.index.json"
            if index.exists():
                present = present and all(
                    (p / f).exists() for f in set(json.loads(index.read_text())["weight_map"].values())
                )
            inventory.append(
                dict(
                    family=family,
                    key=key,
                    repo=m["repo"],
                    revision=m["revision"],
                    subfolder=m.get("subfolder"),
                    weights_present=present,
                    path=str(p),
                )
            )
    records, pairs = [], []

    def add(family, arm, student, teacher, initial_file, final_file, steps, support, recent_file, window):
        if steps > MAX_DISPLAY_UPDATES:
            steps = MAX_DISPLAY_UPDATES
            final_file = final_file.parent / f"fixed-{steps:03d}.json"
            recent_file = final_file.parent / f"fixed-{steps - window:03d}.json"
        initial = json.loads(initial_file.read_text())
        final = json.loads(final_file.read_text())
        recent = json.loads(recent_file.read_text()) if recent_file.exists() else None
        pairs.append(dict(family=family, arm=arm, student=student, teacher=teacher, steps=steps, support=support))
        for panel, groups in [
            ("old_train_ad", list("ABCD")),
            ("train_seen", [chr(65 + i) for i in range(min(20, steps // 16))]),
            ("held_math", ["M01", "M02", "M03", "M04"]),
            ("held_code", ["Code01", "Code02"]),
            ("held_if", ["IF01", "IF02"]),
        ]:
            a = [r for r in initial if r["group"] in groups]
            b = [r for r in final if r["group"] in groups]
            assert a and [r["id"] for r in a] == [r["id"] for r in b]
            for metric in ["rkl", "js"]:

                def mean(rows, metric=metric):
                    return sum(r["stats"]["all"][metric] for r in rows) / (len(rows) * 1024)

                av, bv = mean(a), mean(b)
                rv = mean([r for r in recent if r["group"] in groups]) if recent else None
                records.append(
                    dict(
                        family=family,
                        arm=arm,
                        student=student,
                        teacher=teacher,
                        steps=steps,
                        support=support,
                        panel=panel,
                        metric=metric,
                        prompts=len(a),
                        actual_tokens=sum(r["stats"]["all"]["tokens"] for r in b),
                        budget_positions=len(b) * 1024,
                        initial=av,
                        final=bv,
                        drop_percent=100 * (1 - bv / av),
                        recent_window=window if recent else None,
                        recent_initial=rv,
                        recent_drop_percent=100 * (1 - bv / rv) if rv and rv > 0 else None,
                        initial_source=str(initial_file),
                        final_source=str(final_file),
                    )
                )

    cont = {v["old_arm"]: k for k, v in r2.ARMS.items() if v["old_arm"]}
    for arm, (student, teacher) in old.ARMS.items():
        d = old.OUT / "arms" / arm
        assert (d / "complete.json").exists()
        steps, finalroot = 64, d
        support = "native"
        if arm in cont:
            finalroot = r2.OUT / "arms" / cont[arm]
            assert (finalroot / "complete.json").exists()
            steps = 128
            extension = root / "r1-samesize-stage320-20261030"
            candidate = extension / "arms" / cont[arm]
            if verified_extension(extension, cont[arm]):
                d = finalroot = candidate
                steps = 320
            replica = root / "r1-samesize-deterministic320-20261030"
            if verified_extension(replica, cont[arm]):
                d = finalroot = replica / "arms" / cont[arm]
                steps, support = 320, "native_deterministic_fa2_from_initial"
        window = 16
        add(
            "R1",
            arm,
            student,
            teacher,
            d / "fixed-000.json",
            finalroot / f"fixed-{steps:03d}.json",
            steps,
            support,
            finalroot / f"fixed-{steps - window:03d}.json",
            window,
        )
    for arm, c in r2.ARMS.items():
        if c["old_arm"]:
            continue
        d = r2.OUT / "arms" / arm
        assert (d / "complete.json").exists()
        steps = 64
        support = "native"
        extension = root / "r1-samesize-stage320-20261030"
        candidate = extension / "arms" / arm
        if verified_extension(extension, arm):
            d, steps = candidate, 320
        replica = root / "r1-samesize-deterministic320-20261030"
        if verified_extension(replica, arm):
            d, steps, support = replica / "arms" / arm, 320, "native_deterministic_fa2_from_initial"
        add(
            "R1",
            arm,
            "r1",
            c["teacher"],
            d / "fixed-000.json",
            d / f"fixed-{steps:03d}.json",
            steps,
            support,
            d / f"fixed-{steps - 16:03d}.json",
            16,
        )
    for arm, teacher in [("TB3-R7", "r1"), ("TB3-J7", "justrl")]:
        d = root / "teacher-boundary-small-to-large-20261029/arms" / arm
        assert (d / "complete.json").exists()
        steps, window, support = 64, 16, "conditional_C151936"
        for budget in [320, 640]:
            stage = root / f"teacher-boundary-small-to-large-deterministic{budget}-20261030"
            candidate = stage / "arms" / arm
            if (candidate / "complete.json").exists() and (stage / "analysis" / f"verified-{arm}.json").exists():
                d = candidate
                steps, window, support = budget, 64, "conditional_C151936_deterministic_fa2"
        add(
            "R1",
            arm,
            "r7",
            teacher,
            d / "fixed-000.json",
            d / f"fixed-{steps:03d}.json",
            steps,
            support,
            d / f"fixed-{steps - window:03d}.json",
            window,
        )
    thinking = {}
    for plan_name in [
        "qwen3-lineage",
        "qwen3-thinking-wave1",
        "qwen3-thinking-stage640",
        "qwen3-thinking-rlvr-control320",
        "qwen3-thinking-e3-stage1280",
        "qwen-stage-lineage-thinking320",
        "qwen-stage-lineage-thinking640",
        "qwen-stage-grpo-thinking640",
    ]:
        plan = json.loads((plans / f"{plan_name}.json").read_text())
        for arm in plan["arms"]:
            directory = plan_directory(plan, w) / "arms" / arm["name"]
            marker = directory / "complete.json"
            if not marker.exists():
                if plan_name == "qwen3-lineage":
                    raise FileNotFoundError(f"Historical reference receipt missing: {marker}")
                continue
            steps = json.loads(marker.read_text())["updates"]
            if steps >= thinking.get(arm["teacher"], (0,))[0]:
                support = "native_old_thinking" if plan_name == "qwen3-lineage" else "native_thinking"
                thinking[arm["teacher"]] = (steps, directory, arm["name"], support)
    for teacher, (steps, directory, arm, support) in thinking.items():
        window = 64 if steps >= 320 else 16
        add(
            "Qwen3",
            arm,
            "q3-1.7",
            teacher,
            directory / "fixed-000.json",
            directory / f"fixed-{steps:03d}.json",
            steps,
            support,
            directory / f"fixed-{steps - window:03d}.json",
            window,
        )
    # Completed stages replace earlier endpoints of the same mode-specific arm.
    latest = {}
    for stage in [
        "stage320",
        "stage640",
        "wave2",
        "wave3",
        "crosssize320",
        "origin-controls320",
        "origin-controls640",
        "origin-controls1280",
    ]:
        plan = json.loads((plans / f"qwen3-nonthinking-{stage}.json").read_text())
        for arm in plan["arms"]:
            directory = plan_directory(plan, w) / "arms" / arm["name"]
            receipt = directory / "complete.json"
            if receipt.exists():
                steps = json.loads(receipt.read_text())["updates"]
                if steps > latest.get(arm["name"], (0,))[0]:
                    latest[arm["name"]] = (steps, directory, arm["teacher"])
    for arm, (steps, directory, teacher) in latest.items():
        add(
            "Qwen3",
            arm,
            "q3-1.7-nonthinking",
            teacher,
            directory / "fixed-000.json",
            directory / f"fixed-{steps:03d}.json",
            steps,
            "native_nonthinking",
            directory / f"fixed-{steps - 64:03d}.json",
            64,
        )
    # Separate input conditions share collection logic but never share plot rankings.
    lines = [
        (
            "Qwen3",
            "q3-1.7-native-thinking2507",
            "native_template_thinking2507",
            ["qwen3-2507-thinking-native-template"],
        ),
        (
            "Qwen3",
            "q3-1.7-native-instruct2507",
            "native_template_instruct2507",
            ["qwen3-2507-instruct-native-template", "qwen3-2507-instruct-native-template640"],
        ),
        (
            "Llama",
            "llama3-3b-off",
            "native_common_off",
            [
                "llama-reasoning-off-stage320",
                "llama-reasoning-off-stage640",
                "llama-reasoning-off-stage1280",
                "llama-stage-lineage320",
            ],
        ),
    ]
    for family, student, support, plan_names in lines:
        endpoints = {}
        for name in plan_names:
            plan = json.loads((plans / f"{name}.json").read_text())
            for arm in plan["arms"]:
                directory = plan_directory(plan, w) / "arms" / arm["name"]
                marker = directory / "complete.json"
                if not marker.exists():
                    continue
                steps = json.loads(marker.read_text())["updates"]
                if steps >= endpoints.get(arm["name"], (0,))[0]:
                    endpoints[arm["name"]] = (steps, directory, arm["teacher"])
        for arm, (steps, directory, teacher) in endpoints.items():
            add(
                family,
                arm,
                student,
                teacher,
                directory / "fixed-000.json",
                directory / f"fixed-{steps:03d}.json",
                steps,
                support,
                directory / f"fixed-{steps - 64:03d}.json",
                64,
            )
    appendix_records = [r for r in records if not in_main_comparison(r["teacher"])]
    appendix_pairs = [r for r in pairs if not in_main_comparison(r["teacher"])]
    appendix_inventory = [r for r in inventory if not in_main_comparison(r["key"])]
    dump(out / "appendix-inventory.json", dict(inventory=appendix_inventory, pairs=appendix_pairs))
    if appendix_records:
        write_csv(out / "appendix-risks.csv", appendix_records)
    records = [r for r in records if in_main_comparison(r["teacher"])]
    pairs = [r for r in pairs if in_main_comparison(r["teacher"])]
    inventory = [r for r in inventory if in_main_comparison(r["key"])]
    summary = dict(
        scope="main comparison within 0-320 updates; later budgets are historical only",
        maximum_display_updates=MAX_DISPLAY_UPDATES,
        weight_versions=len({(x["repo"], x["revision"], x["subfolder"]) for x in inventory}),
        local_weight_versions=sum(x["weights_present"] for x in inventory),
        completed_unique_pairs=len({(student_weight_key(x["student"]), x["teacher"]) for x in pairs}),
        completed_conditions=len(pairs),
        completed_updates=sum(x["steps"] for x in pairs),
        families={
            f: dict(
                weight_versions=sum(x["family"] == f for x in inventory),
                local_weights=sum(x["family"] == f and x["weights_present"] for x in inventory),
                completed_pairs=len(
                    {(student_weight_key(x["student"]), x["teacher"]) for x in pairs if x["family"] == f}
                ),
            )
            for f in sets
        },
        inventory=inventory,
        pairs=pairs,
        normalization="sum KL / (prompts * 1024); support and mode remain separate",
    )
    dump(out / "inventory.json", summary)
    write_csv(out / "risks.csv", records)
    if args.update_report:
        update_report(summary, records, out.parent.parent / "2026-10-29-教师谱系实验总览与RKL进展.md")
    print({k: v for k, v in summary.items() if k not in ["inventory", "pairs"]})
    if args.plot:
        plot(records, out)


def plan_directory(plan, work):
    path = Path(plan["output"])
    return path if path.is_absolute() else work / path


def verified_extension(directory, arm):
    proof = directory / "analysis" / f"verified-{arm}.json"
    if not proof.exists() or not (directory / "arms" / arm / "complete.json").exists():
        return False
    value = json.loads(proof.read_text())
    if not value.get("complete") or not value.get("evidence"):
        raise ValueError(f"Incomplete extension verification: {proof}")
    for item in value["evidence"]:
        if file_hash(item["path"]) != item["sha256"]:
            raise ValueError(f"Extension evidence changed after verification: {item['path']}")
    return True


def replace_block(text, key, value):
    opening, closing = f"<!-- generated:{key}:start -->", f"<!-- generated:{key}:end -->"
    if text.count(opening) != 1 or text.count(closing) != 1:
        raise ValueError(f"Expected exactly one report block: {key}")
    start, end = text.index(opening) + len(opening), text.index(closing)
    if end < start:
        raise ValueError(f"Reversed report block: {key}")
    return text[:start] + "\n\n" + value.strip() + "\n\n" + text[end:]


def update_report(summary, records, path):
    text = path.read_text()
    text = replace_block(
        text,
        "counts",
        (
            f"状态：本轮 R1、Qwen3、Llama 教师谱系主比较集合共有 {summary['weight_versions']} 个固定权重版本，"
            f"其中 {summary['local_weight_versions']} 个已落地。按当前完成回执，已完成 "
            f"{summary['completed_unique_pairs']} 个不同师生权重配对、{summary['completed_conditions']} 个输入条件，"
            f"展示范围内的终点预算合计 {summary['completed_updates']} 更新。每个条件仅展示 0–320 内最新有效读出，"
            "不重复累计中间阶段或重放预算；不统计其他研究线，也不把自教师、teacher-copy、重复评分和数值验证算作训练分支。"
            "准确率不作为本总览主表或下一阶段开训门槛。"
        ),
    )
    table = ["| 谱系 | 计划中的权重版本 | 权重已到本地 | 已完成训练的不同师生配对 |", "| --- | ---: | ---: | ---: |"]
    for family, counts in summary["families"].items():
        table.append(
            f"| {family} | {counts['weight_versions']} | {counts['local_weights']} | {counts['completed_pairs']} |"
        )
    table.append(
        f"| 合计 | **{summary['weight_versions']}** | **{summary['local_weight_versions']}** | "
        f"**{summary['completed_unique_pairs']}** |"
    )
    text = replace_block(text, "inventory", "\n".join(table))
    ranking = heldout_summary(records)
    index = {(r["student"], r["arm"], r["panel"]): r for r in records if r["metric"] == "rkl"}
    for student in ["r1", "r7", "q3-1.7", "q3-1.7-nonthinking", "q3-1.7-native-instruct2507", "llama3-3b-off"]:
        table = [
            "| Teacher（条件见图） | 更新数 | 整体初始 RKL | 整体最终 RKL | "
            "整体下降 | Math 下降 | Code 下降 | IF 下降 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        selected = [r for r in ranking if r["student"] == student]
        if not selected:
            raise ValueError(f"Refusing to replace report with an empty panel: {student}")
        for row in selected:
            label = row["model_label"] + (" †" if row["support"].startswith("conditional") else "")
            domains = " | ".join(
                f"{index[(student, row['arm'], p)]['drop_percent']:.2f}%" for p in ["held_math", "held_code", "held_if"]
            )
            table.append(
                f"| {label} | {row['steps']} | {row['initial']:.6f} | {row['final']:.6f} | "
                f"{row['drop_percent']:.2f}% | {domains} |"
            )
        text = replace_block(text, student, "\n".join(table))
    write_text(path, text)


def student_weight_key(key):
    """Plot lines describe input conditions, not additional student weight versions."""
    return "q3-1.7" if key.startswith("q3-1.7") else key


MODEL_LABELS = {
    "Q1": ("Qwen3-4B Math-CoT SFT step80", 4, "Base-derived SFT; thinking input"),
    "Q2": ("Qwen3-4B Math-CoT SFT step640", 4, "Base-derived SFT; thinking input"),
    "L1": ("Llama-3.1-Tulu-3-8B-SFT", 8, "SFT; shared student template"),
    "L2": ("Llama-3.1-Tulu-3-8B-DPO", 8, "SFT + DPO; shared student template"),
    "L3": ("Llama-3.1-Tulu-3-8B", 8, "SFT + DPO + RLVR; shared student template"),
    "L4": ("Llama-3.1-8B-SimpleRL-Zoo", 8, "Base + RL; shared-prefix mismatch control"),
    "r1": ("DeepSeek-R1-Distill-Qwen-1.5B", 1.5, "R1 reasoning prefix"),
    "r7": ("DeepSeek-R1-Distill-Qwen-7B", 7, "R1 reasoning prefix"),
    "justrl": ("JustRL-DeepSeek-1.5B", 1.5, "RL; R1 reasoning prefix"),
    "nemotron": ("Nemotron-Research-Reasoning-Qwen-1.5B v1", 1.5, "RL; R1 reasoning prefix"),
    "polaris": ("Polaris-7B-Preview", 7, "RL; R1 reasoning prefix"),
    "skywork": ("Skywork-OR1-Math-7B", 7, "RL; R1 reasoning prefix"),
    "light": ("Light-R1-7B-DS", 7, "R1 reasoning prefix"),
    "minimath": ("MiniMath-R1-1.5B", 1.5, "R1 reasoning prefix"),
    "deepscaler": ("DeepScaleR-1.5B-Preview", 1.5, "RL; R1 reasoning prefix"),
    "brorl": ("Nemotron-Research-Reasoning-Qwen-1.5B / BroRL", 1.5, "RL; R1 reasoning prefix"),
    "deepcoder": ("DeepCoder-1.5B-Preview", 1.5, "R1 reasoning prefix"),
    "q3-1.7": ("Qwen3-1.7B", 1.7, "post-trained dual-mode; thinking input"),
    "q3-1.7-nonthinking": ("Qwen3-1.7B", 1.7, "post-trained dual-mode; non-thinking input"),
    "q3-0.6": ("Qwen3-0.6B", 0.6, "post-trained dual-mode"),
    "q3-1.7-native-thinking2507": ("Qwen3-1.7B", 1.7, "Thinking-2507 native prefix"),
    "q3-1.7-native-instruct2507": ("Qwen3-1.7B", 1.7, "Instruct-2507 native prefix; S0 thinking"),
    "q3-4-instruct2507": ("Qwen3-4B-Instruct-2507", 4, "instruct-only; native template"),
    "base-gt-math": ("GT-Qwen3-1.7B-Base-MATH", 1.7, "RL; training mode unreported"),
    "coreward-i": ("Co-rewarding-I-Qwen3-1.7B-Base-MATH", 1.7, "RL; training mode unreported"),
    "coreward-ii": ("Co-rewarding-II-Qwen3-1.7B-Base-MATH", 1.7, "RL; training mode unreported"),
    "base-grpo-4": ("Qwen3-4B-Base-GRPO / Thinking-Space", 4, "early-fork RL; training mode unreported"),
    "rlvr-gsm8k": ("tocsa/qwen3-1.7b-rlvr-gsm8k", 1.7, "RLVR/LoRA; thinking input; train mode unreported"),
    "llama3-3b-off": ("Llama-3.2-3B-Instruct / Unsloth", 3, "shared reasoning-off system"),
    "gt": ("GT-Llama-3.2-3B-Instruct-MATH", 3, "RL; shared off system"),
    "coreward-i-llama": ("Co-rewarding-I-Llama-3.2-3B-Instruct-MATH", 3, "RL; shared off system"),
    "arithmetic-full": ("Llama-3.2-3B-Instruct-arithmetic-full", 3, "author-declared full SFT; shared off system"),
    "llama-1b": ("Llama-3.2-1B-Instruct / Unsloth", 1, "shared off system"),
    "llama-8b": ("Llama-3.1-8B-Instruct / Unsloth", 8, "shared off system"),
    "nemotron-8b": ("Llama-3.1-Nemotron-Nano-8B-v1", 8, "SFT + RL + merge; shared off system"),
    "q3-base": ("Qwen3-1.7B-Base", 1.7, "Base (not Instruct); thinking input"),
    "e3": ("e3-1.7B", 1.7, "RL; thinking input"),
    "grpo-kl": ("Qwen3-1.7B-GRPO-KL-Math", 1.7, "RL; thinking input"),
    "sft-gsm8k": ("Qwen3-1.7B-Math / Samsung SFT", 1.7, "thinking input; native mode unverified"),
    "sft-deepmath": ("Qwen3-1.7B-SFT / Thinking-Space", 1.7, "thinking input; non-thinking trained"),
    "base-opd": ("Qwen3-1.7B-Base-OPD / Thinking-Space", 1.7, "thinking input; non-thinking trained"),
    "polaris-4": ("Polaris-4B-Preview", 4, "RL; thinking input; YaRN"),
    "q3-4": ("Qwen3-4B", 4, "post-trained dual-mode; thinking input"),
    "q3-8": ("Qwen3-8B", 8, "post-trained dual-mode; thinking input"),
    "q3-4-thinking2507": ("Qwen3-4B-Thinking-2507", 4, "thinking-only; shared-prefix mismatch"),
}


def heldout_summary(records):
    """Pool per-question risks before taking a ratio, rather than average domain percentages."""
    groups = {}
    for row in records:
        if row["metric"] == "rkl" and row["panel"] in {"held_math", "held_code", "held_if"}:
            key = (row["family"], row["student"], row["arm"])
            if row["panel"] in groups.setdefault(key, {}):
                raise ValueError("Duplicate held-out domain")
            groups[key][row["panel"]] = row
    result = []
    for panels in groups.values():
        if set(panels) != {"held_math", "held_code", "held_if"}:
            raise ValueError("All three held-out domains are required for ranking")
        count = sum(int(r["prompts"]) for r in panels.values())
        initial = sum(int(r["prompts"]) * float(r["initial"]) for r in panels.values()) / count
        final = sum(int(r["prompts"]) * float(r["final"]) for r in panels.values()) / count
        if initial <= 0:
            raise ValueError("Cannot rank a zero initial-risk denominator")
        row = dict(
            panels["held_math"],
            panel="held_all",
            prompts=count,
            initial=initial,
            final=final,
            drop_percent=100 * (1 - final / initial),
        )
        if len({(r["steps"], r["support"], r["teacher"]) for r in panels.values()}) != 1:
            raise ValueError("Cannot pool different endpoint conditions")
        row["actual_tokens"] = sum(int(r.get("actual_tokens", 0)) for r in panels.values())
        row["budget_positions"] = sum(int(r.get("budget_positions", int(r["prompts"]) * 1024)) for r in panels.values())
        windows = {r.get("recent_window") for r in panels.values()}
        if (
            len(windows) == 1
            and None not in windows
            and all(r.get("recent_initial") is not None for r in panels.values())
        ):
            recent = sum(int(r["prompts"]) * float(r["recent_initial"]) for r in panels.values()) / count
            row.update(recent_initial=recent, recent_drop_percent=100 * (1 - final / recent) if recent > 0 else None)
        else:
            row.update(recent_initial=None, recent_drop_percent=None, recent_window=None)
        label_key = (
            "coreward-i-llama" if row["family"] == "Llama" and row["teacher"] == "coreward-i" else row["teacher"]
        )
        name, size, mode = MODEL_LABELS[label_key]
        if row["support"] == "native_nonthinking":
            mode = "non-thinking input"
            if row["teacher"] in {
                "sft-gsm8k",
                "base-gt-math",
                "coreward-i",
                "coreward-ii",
                "base-grpo-4",
                "rlvr-gsm8k",
            }:
                mode += "; training mode unreported"
            if row["teacher"] in {"q3-0.6", "q3-4", "q3-8"}:
                mode = "post-trained dual-mode; non-thinking input"
        if row["support"] == "native_thinking" and row["teacher"] in {"q3-0.6", "q3-4", "q3-8"}:
            mode = "post-trained dual-mode; thinking input"
        if row["support"].startswith("native_template_"):
            mode = "specialist native template; isolated control"
        if "deterministic" in row["support"]:
            mode += "; deterministic FA2 backward"
        row.update(
            model_label=name, input_mode=mode, teacher_size_b=size, same_size=size == MODEL_LABELS[row["student"]][1]
        )
        result.append(row)
    return sorted(result, key=lambda r: (r["student"], -r["drop_percent"], r["arm"]))


def plot(records, output):
    """Ledger: training seen/A-D and held-out pooled/Math/Code/IF share rows and scales; hatch encodes size.

    One row is one latest endpoint. Pooled risk weights each prompt equally (64/32/32),
    then computes reduction. Mode labels describe actual inputs, not inferred capabilities.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Patch

    from examples.mirroropd.research.core.plotting import load_style

    records = [r for r in records if in_main_comparison(r["teacher"])]
    pooled = heldout_summary(records)
    write_csv(output / "heldout-ranking.csv", pooled)
    fv = load_style()
    fv.apply_rc("paper", fontsize=10)
    audits, orders = {}, {}
    index = {(r["arm"], r["panel"], r["metric"]): r for r in records}
    for student in sorted({r["student"] for r in pooled}):
        ranked = [r for r in pooled if r["student"] == student]
        n = len(ranked)
        orders[student] = [
            dict(
                teacher=r["model_label"],
                size_b=r["teacher_size_b"],
                mode=r["input_mode"],
                pooled_drop=r["drop_percent"],
                same_size=r["same_size"],
            )
            for r in ranked
        ]
        fig, axes = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(19, max(6.8, n + 3)))
        fig.subplots_adjust(left=0.34, right=0.98, top=0.87, bottom=0.24 if n <= 6 else 0.17, wspace=0.12)
        panels = [
            (
                "Training data",
                [
                    ("train_seen", "All seen training prompts", fv.HUES["vblue"], 1.0),
                    ("old_train_ad", "Old training A-D (64)", fv.NEUTRALS["vmist"], 1.0),
                ],
            ),
            (
                "Held-out data",
                [
                    ("held_all", "All held-out (128)", fv.HUES["vgray"], 1.0),
                    ("held_math", "Math (64)", fv.HUES["vteal"], 1.0),
                    ("held_code", "Code (32)", fv.HUES["vorange"], 1.0),
                    ("held_if", "IF (32)", fv.HUES["vviolet"], 1.0),
                ],
            ),
        ]

        def values_for(panel, ranked=ranked):
            return [
                r["drop_percent"] if panel == "held_all" else float(index[r["arm"], panel, "rkl"]["drop_percent"])
                for r in ranked
            ]

        all_values = [v for _, series in panels for panel, _, _, _ in series for v in values_for(panel)]
        span = max(100, max(all_values) - min(all_values))
        xmin = min(all_values) - 0.12 * span if min(all_values) < 0 else 0
        xmax = max(113, max(all_values) + 0.12 * span)
        for ax, (title, series) in zip(axes, panels, strict=True):
            if xmin < 0:
                ax.axvline(0, color=fv.NEUTRALS["baseline"], linewidth=0.7)
            offsets = [-0.14, 0.14] if len(series) == 2 else [-0.27, -0.09, 0.09, 0.27]
            for offset, (panel, label, color, alpha) in zip(offsets, series, strict=True):
                values = values_for(panel)
                bars = ax.barh(
                    np.arange(n) + offset,
                    values,
                    height=0.22 if len(series) == 2 else 0.145,
                    color=color,
                    alpha=alpha,
                    edgecolor=fv.NEUTRALS["vink"],
                    linewidth=0.35,
                    label=label,
                )
                for pos, value, bar, row in zip(np.arange(n) + offset, values, bars, ranked, strict=True):
                    if not row["same_size"]:
                        bar.set_hatch("///")
                    ax.text(
                        value + (0.01 * span if value >= 0 else -0.01 * span),
                        pos,
                        f"{value:.1f}",
                        ha="left" if value >= 0 else "right",
                        va="center",
                        fontsize=9,
                        color=fv.NEUTRALS["vink"],
                    )
            ax.set(xlim=(xmin, xmax), xlabel="RKL reduction (%)", title=title)
            ax.spines[["top", "right"]].set_visible(False)
        labels = []
        for r in ranked:
            support = "; conditional KL" if r["support"].startswith("conditional") else ""
            if "deterministic_fa2" in r["support"]:
                support += "; deterministic FA2"
            count = index[r["arm"], "train_seen", "rkl"]["prompts"]
            labels.append(
                f"{r['model_label']}\n{r['input_mode']}; {r['steps']} updates{support}\nSeen training prompts: {count}"
            )
        axes[0].set_yticks(np.arange(n), labels)
        axes[0].invert_yaxis()
        axes[1].tick_params(axis="y", left=False)
        fig.suptitle(f"Student: {MODEL_LABELS[student][0]} | {MODEL_LABELS[student][2]}", y=0.965)
        handles = [Patch(facecolor=c, alpha=a, label=label) for _, series in panels for _, label, c, a in series]
        handles += [
            Patch(facecolor=fv.HUES["vgray"], edgecolor=fv.NEUTRALS["vink"], label="Same size"),
            Patch(facecolor=fv.HUES["vgray"], edgecolor=fv.NEUTRALS["vink"], hatch="///", label="Different size"),
        ]
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.64, 0.065), ncol=4, fontsize=9)
        fig.text(
            0.5,
            0.014,
            "Rows sorted by pooled held-out RKL reduction (Math64 + Code32 + IF32). Shared scale across panels.\n"
            "Training = prompts actually used for updates; A-D is a subset, not an extra dataset. Frozen S0; H1024.",
            ha="center",
            fontsize=9,
            color=fv.NEUTRALS["baseline"],
        )
        fig.canvas.draw()
        # The right panel shares the teacher labels displayed on the left.
        problems = fv.audit(fig, print_slot=fig.get_figwidth(), scoped_ignore=[("missing-axis-label", axes[1])])
        audits[student] = [str(p) for p in problems]
        if any(fv.problem_severity(p) == "error" for p in problems):
            dump(output / "figure-audit.json", audits)
            raise RuntimeError(audits[student])
        for ext in ["png", "pdf"]:
            fig.savefig(output / f"{student}-final-rkl.{ext}", dpi=180)
        plt.close(fig)
    dump(output / "figure-audit.json", audits)
    dump(output / "figure-order.json", orders)
    print(audits)


if __name__ == "__main__":
    main()
