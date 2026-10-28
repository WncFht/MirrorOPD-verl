#!/usr/bin/env python3
# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Generate execution and evaluation views from one local evidence inventory.

Research labels are derived from run identities. Config/history
remain in wandb-logs; per-dataset evaluation coordinates are kept separate from
training progress. No cloud access or training is performed.
"""

import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from examples.mirroropd.research.analysis.build_inventory import inspect, read, select, write

DATASETS = {
    "math500": "MATH-500",
    "amc23": "AMC23",
    "aime24": "AIME24",
    "aime25": "AIME25",
    "aime26": "AIME26",
    "livecodebench": "livecodebench",
}


RUN_OVERRIDES = {
    "f0e44ec7": {"experiment": "E39", "primary_question": "teacher-transfer", "variant": "200 outer，终止对齐"},
    "53dd553d": {
        "experiment": "E41",
        "primary_question": "teacher-transfer",
        "variant": "100 outer，终止对齐，无初始验证",
    },
    "eccc09a1": {"experiment": "E43", "primary_question": "teacher-transfer", "variant": "100 outer，五集验证"},
}
PRESET_FAMILIES = {
    "E1": "learning-signal",
    "E2": "learning-signal",
    "E3": "learning-signal",
    "E4": "learning-signal",
    "E5": "learning-signal",
    "E6": "learning-signal",
    "E7": "learning-signal",
    "E8": "learning-signal",
    "E9": "learning-signal",
    "E10": "learning-signal",
    "E11": "learning-signal",
    "E12": "learning-signal",
    "E13": "learning-signal",
    "E14": "learning-signal",
    "E15": "learning-signal",
    "E16": "reuse-budget",
    "E17": "reuse-budget",
    "E18": "reuse-budget",
    "E19": "reuse-budget",
    "E20": "reuse-budget",
    "E21": "teacher-transfer",
    "E22": "teacher-transfer",
    "E23": "teacher-transfer",
    "E24": "teacher-transfer",
    "E25": "teacher-transfer",
    "E26": "teacher-transfer",
    "E27": "teacher-transfer",
    "E28": "teacher-transfer",
    "E29": "teacher-transfer",
    "E30": "teacher-transfer",
    "E31": "teacher-transfer",
    "E32": "teacher-transfer",
    "E33": "teacher-transfer",
    "E34": "teacher-transfer",
    "E35": "reuse-budget",
    "E36": "teacher-transfer",
    "E37": "teacher-transfer",
    "E38": "teacher-transfer",
    "E39": "teacher-transfer",
    "E40": "teacher-transfer",
    "E41": "teacher-transfer",
    "E42": "teacher-transfer",
    "E43": "teacher-transfer",
    "E44": "teacher-transfer",
    "E45": "teacher-transfer",
    "E46": "teacher-transfer",
    "E47": "teacher-transfer",
    "E48": "teacher-transfer",
    "E49": "teacher-transfer",
    "E50": "teacher-transfer",
    "E51": "teacher-transfer",
    "E52": "teacher-transfer",
    "E53": "teacher-transfer",
    "E54": "infra",
}
RESEARCH_LINES = [
    {"id": "learning-signal", "title": "训练信号与目标"},
    {"id": "reuse-budget", "title": "Reuse 与训练预算"},
    {"id": "rollout-policy", "title": "Rollout 来源与混合效果"},
    {"id": "teacher-transfer", "title": "师生迁移与优化机制"},
    {"id": "infra", "title": "工程诊断与验收"},
]


def classify(identity):
    rid, name = identity["run_id"], identity.get("display_name", "")
    if rid in RUN_OVERRIDES:
        return RUN_OVERRIDES[rid]
    match = re.match(r"math-opd-(E\d+)-", name)
    if match:
        preset = match[1]
        if preset not in PRESET_FAMILIES:
            raise ValueError(f"Register the research question for {preset} before publishing its views")
        return {"experiment": preset, "primary_question": PRESET_FAMILIES[preset], "variant": ""}
    if name.startswith(("math-full-", "code-full-", "if-full-")):
        return {"experiment": "historical", "primary_question": "historical", "variant": "旧协议"}
    return None


def last_evaluations(history):
    """Keep each dataset's last observed metrics, never combine different events."""
    result = {}
    outer = optimizer = None
    for row in history:
        outer = row.get("step/outer", row.get("training/global_step", outer))
        optimizer = row.get("step/optimizer", row.get("training/actor_optimizer_step", optimizer))
        for slug, old_name in DATASETS.items():
            candidates = (f"eval/{slug}/acc/mean", f"eval/{slug}/acc_mean", f"val-core/{old_name}/acc/mean@16")
            key = next((k for k in candidates if k in row), None)
            if key is None:
                continue
            prefix = f"eval/{slug}/acc/"
            result[slug] = {
                "dataset": slug,
                "outer": outer,
                "optimizer": optimizer,
                "history_step": row.get("_step"),
                "mean": row[key],
                "mean_key": key,
                "pass16": row.get(prefix + "pass@16"),
                "majority16": row.get(prefix + "majority@16"),
                "ci_low": row.get(prefix + "mean_ci_low"),
                "ci_high": row.get(prefix + "mean_ci_high"),
                "legacy_estimators": {
                    k: v
                    for k, v in row.items()
                    if (slug in k or old_name in k) and any(t in k for t in ("best@16", "maj@16"))
                },
                "prompt_count": row.get(f"eval/{slug}/prompt_count"),
                "samples_min": row.get(f"eval/{slug}/samples_per_prompt_min"),
                "samples_max": row.get(f"eval/{slug}/samples_per_prompt_max"),
            }
    return result


def collect(logs):
    runs, seen = [], set()
    for path in sorted((logs / "derived/MirrorOPD-verl").glob("*/run.json")):
        meta = read(path)
        if meta.get("run", {}).get("project") != "MirrorOPD-verl":
            continue
        membership = classify(meta["run"])
        if membership is None:
            continue
        rid = meta["run"]["run_id"]
        if rid in seen:
            raise ValueError(f"Duplicate local run ID: {rid}; select a snapshot explicitly")
        seen.add(rid)
        if meta["status"] not in ("finished", "snapshot", "partial_snapshot"):
            raise ValueError(f"Unusable local decode: {rid}: {meta['status']}")
        key = meta["cache_key"]
        raw = Path(key["source"])
        stat = raw.stat()
        if (stat.st_size, stat.st_mtime_ns) != (key["bytes"], key["mtime_ns"]):
            raise ValueError(f"Stale local decode: {rid}; run the decoder first")
        config = read(path.parent / "config.json")
        row = inspect(path.parent, {})
        history = [json.loads(line) for line in (path.parent / "history.jsonl").read_text().splitlines()]
        row.update(membership)
        row["evaluations"] = last_evaluations(history)
        row["precision"] = select(config, "actor_rollout_ref.actor.fsdp_config.forward_precision")
        row["termination_ids"] = select(config, "distillation.distillation_loss.full_vocab.termination_token_ids")
        row["source_sha256"] = hashlib.sha256(raw.read_bytes()).hexdigest()
        row["raw_source"] = str(raw.relative_to(logs))
        row["decode_status"] = meta["status"]
        row["execution_status"] = (
            "completed"
            if meta["status"] == "finished" and meta.get("exit_code") == 0
            else "failed_exit"
            if meta.get("exit_code") not in (None, 0)
            else "partial_evidence"
        )
        runs.append(row)
    return runs


def fmt(value):
    return "—" if value is None else str(value)


def pct(value):
    return "—" if value is None else f"{100 * value:.2f}"


def model_name(path):
    return Path(path).name if path else "—"


def header(title, inventory):
    return [
        "# " + title,
        "",
        "<!-- Generated by research.analysis.refresh_tables; do not edit data rows. -->",
        "",
        f"本地证据读取时间：{inventory['observed_at']}。共 {len(inventory['runs'])} 个已识别运行；这不是预设完成数量。",
        "",
        "研究动机与解释见[问题索引](README.md)，登记规则见[复现与维护](维护说明.md)。"
        "完整配置、逐数据集坐标和原始文件摘要保存在日志仓库的 `derived/MirrorOPD-verl/inventory/inventory.json`。",
        "",
    ]


def render_runs(inventory):
    lines = header("实际运行记录", inventory)
    lines += [
        "`completed` 要求本地退出码为0；`partial_evidence` 只描述证据不足，不能判断进程当前是否运行。"
        "训练进度与最后验证位置分开；E编号表示预设，实际预算、精度和终止配置来自当前run。",
        "",
    ]
    for question in [*RESEARCH_LINES, {"id": "historical", "title": "历史协议运行"}]:
        chosen = [r for r in inventory["runs"] if r["primary_question"] == question["id"]]
        if not chosen:
            continue
        lines += [
            f"## {question['id']} {question['title']}",
            "",
            "| 实验 / run | Student ← Teacher | 方法 / 实际配置 | "
            "训练outer / 成功更新 | 最后验证outer（各集） | 证据状态 |",
            "| --- | --- | --- | ---: | --- | --- |",
        ]
        for r in chosen:
            coord = "; ".join(f"{d}:{fmt(e['outer'])}" for d, e in r["evaluations"].items()) or "—"
            setting = f"{r['method']}；reuse{fmt(r['reuse_epochs'])}；LR{fmt(r['lr'])}；cap{fmt(r['response_cap'])}"
            setting += f"；precision={r['precision'] or '原mixed配置'}；stop={fmt(r['termination_ids'])}"
            if r["variant"]:
                setting += "；" + r["variant"]
            lines.append(
                f"| {r['experiment']} · `{r['run_id']}` | "
                f"{model_name(r['student'])} ← {model_name(r['teacher'])} | {setting} | "
                f"{fmt(r['completed_outer'])} / {fmt(r['optimizer_updates'])} | {coord} | "
                f"{r['execution_status']}；exit={fmt(r['exit_code'])} |"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_evaluations(inventory):
    lines = header("验证结果汇总", inventory)
    lines += [
        "每行是该run、该数据集最后一次实际记录的评测，百分数为mean/pass@16/majority@16。"
        "未记录的项留空，不把旧best/maj估计器改名，也不把其他集合的更新坐标移植过来。"
        "题目置信区间不是训练seed方差。不同模型、预算和协议不得按本表作统一排名。",
        "",
    ]
    for question in [*RESEARCH_LINES, {"id": "historical", "title": "历史协议运行"}]:
        chosen = [r for r in inventory["runs"] if r["primary_question"] == question["id"] and r["evaluations"]]
        if not chosen:
            continue
        lines += [
            f"## {question['id']} {question['title']}",
            "",
            "| 实验 / run | 数据集 | 验证outer / 成功更新 | n / cap / T / top-p | "
            "Mean % [CI] | Pass@16 % | Majority@16 % | 字段口径 |",
            "| --- | --- | ---: | --- | ---: | ---: | ---: | --- |",
        ]
        for r in chosen:
            for d, e in r["evaluations"].items():
                ci = (
                    "" if e["ci_low"] is None or e["ci_high"] is None else f" [{pct(e['ci_low'])}, {pct(e['ci_high'])}]"
                )
                budget = " / ".join(
                    fmt(r.get(k)) for k in ("eval_n", "eval_response_cap", "eval_temperature", "eval_top_p")
                )
                lines.append(
                    f"| {r['experiment']} · `{r['run_id']}` | {DATASETS[d]} | "
                    f"{fmt(e['outer'])} / {fmt(e['optimizer'])} | {budget} | {pct(e['mean'])}{ci} | "
                    f"{pct(e['pass16'])} | {pct(e['majority16'])} | `{e['mean_key']}` |"
                )
        lines.append("")
    lines += [
        "旧best/maj字段保留在生成JSON的 `legacy_estimators` 中；"
        "跨模型协议、初始验证和原始teacher参考由对应研究报告解释。",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs-repo", required=True, type=Path)
    parser.add_argument("--docs-dir", type=Path, default=Path(__file__).resolve().parents[2] / "docs/experiments")
    parser.add_argument(
        "--skip-decode", action="store_true", help="Use existing decoded inputs, rejecting stale sources"
    )
    args = parser.parse_args()
    logs = args.logs_repo.resolve()
    if not args.skip_decode:
        subprocess.run(
            [
                str(logs / ".venv/bin/python"),
                str(logs / "tools/wandb_local.py"),
                "decode",
                "--project",
                "MirrorOPD-verl",
            ],
            check=True,
        )
    runs = collect(logs)
    if not runs:
        raise ValueError("No matching local evidence; refusing to replace views")
    commit = subprocess.check_output(["git", "-C", str(logs), "rev-parse", "HEAD"], text=True).strip()
    inventory = {"observed_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "logs_head": commit, "runs": runs}
    run_text = render_runs(inventory)
    eval_text = render_evaluations(inventory)
    # Prepare all outputs before replacing either human-readable view.
    write(logs / "derived/MirrorOPD-verl/inventory/inventory.json", inventory)
    root = args.docs_dir.resolve()
    for name, text in (("运行记录.md", run_text), ("验证结果汇总.md", eval_text)):
        temp = root / (name + ".tmp")
        temp.write_text(text)
        temp.replace(root / name)
    print(f"Generated both views from {len(runs)} runs; {sum(len(r['evaluations']) for r in runs)} dataset evaluations")


if __name__ == "__main__":
    main()
