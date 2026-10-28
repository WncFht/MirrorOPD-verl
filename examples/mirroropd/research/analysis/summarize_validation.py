#!/usr/bin/env python3
# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Maintain Math ablation validation results from local decoded history.

Last evaluations stay separate from terminal training counters. Legacy bootstrap
best/maj fields are never relabelled as modern pass/majority. Full checkpoint
records are derived in wandb-logs, while only the readable summary enters source.
"""

import argparse
import csv
import json
import re
import subprocess
from datetime import datetime, timedelta, timezone
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

from examples.mirroropd.research.analysis.build_inventory import family, method_label, select

DATASETS = {"math500": "MATH-500", "amc23": "AMC23", "aime25": "AIME25"}
PATTERN = re.compile(r"math-opd-(E\d+|custom)(?:-(valid-student|valid-teacher))?(?:-(.+?))?-\d{8}-")
FIELDS = [
    "mean",
    "ci_low",
    "ci_high",
    "pass16",
    "majority16",
    "legacy_best16",
    "legacy_maj16",
    "length_mean",
    "length_count",
    "prediction_missing",
    "response_count",
    "failed_count",
]


def percent(value):
    if value is None:
        return "—"
    return str((Decimal(str(value)) * 100).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def extract(row, dataset):
    prefix = f"eval/{dataset}/"
    old = f"val-core/{DATASETS[dataset]}/acc/"
    if prefix + "acc/mean" in row:
        mapping = {
            "mean": "acc/mean",
            "ci_low": "acc/mean_ci_low",
            "ci_high": "acc/mean_ci_high",
            "pass16": "acc/pass@16",
            "majority16": "acc/majority@16",
            "length_mean": "generation/response_length/mean",
            "length_count": "generation/finish_reason/length_count",
            "prediction_missing": "acc/prediction_missing_count",
            "response_count": "response_count",
            "failed_count": "failed_count",
        }
        return "policy-v3", {field: row.get(prefix + key) for field, key in mapping.items()}
    if prefix + "acc_mean" in row:
        return "policy-v2", {
            "mean": row[prefix + "acc_mean"],
            "response_count": row.get(prefix + "response_count"),
            "failed_count": row.get(prefix + "failed_count"),
        }
    if old + "mean@16" in row:
        return "legacy", {
            "mean": row[old + "mean@16"],
            "legacy_best16": row.get(old + "best@16/mean"),
            "legacy_maj16": row.get(old + "maj@16/mean"),
        }
    return None


def read_run(path):
    meta = json.loads((path / "run.json").read_text())
    match = PATTERN.match(meta["run"].get("display_name", ""))
    if not match:
        return None
    if meta["status"] not in ("finished", "snapshot", "partial_snapshot"):
        raise ValueError(f"Unusable decode: {path}: {meta['status']}")
    config = json.loads((path / "config.json").read_text())
    points = {}
    max_outer = max_update = 0
    for line in (path / "history.jsonl").open():
        row = json.loads(line)
        outer = row.get("step/outer", row.get("training/global_step"))
        update = row.get("step/optimizer", row.get("training/actor_optimizer_step"))
        if outer is not None:
            max_outer = max(max_outer, outer)
        if update is not None:
            max_update = max(max_update, update)
        for dataset in DATASETS:
            value = extract(row, dataset)
            if value is None:
                continue
            if outer is None or update is None:
                raise ValueError(f"Evaluation lacks explicit outer/optimizer coordinates: {path}")
            point = points.setdefault((outer, update), {"outer": outer, "update": update, "datasets": {}})
            version, metrics = value
            data = {"version": version, **metrics}
            previous = point["datasets"].get(dataset)
            if previous is not None and previous != data:
                raise ValueError(f"Conflicting evaluation: {path} / {outer} / {dataset}")
            point["datasets"][dataset] = data
    evaluations = sorted(points.values(), key=lambda x: (x["update"], x["outer"]))
    # Keep incomplete latest evaluations explicit rather than borrowing older dataset scores.
    method = method_label(config)
    return {
        "preset": match[1],
        "role": match[2],
        "variant": match[3],
        "id": meta["run"]["run_id"],
        "name": meta["run"]["display_name"],
        "source_directory": path.name,
        "state": meta["status"],
        "exit_code": meta["exit_code"],
        "max_outer": max_outer,
        "max_update": max_update,
        "evaluations": evaluations,
        "method": method,
        "reuse": select(config, "actor_rollout_ref.actor.ppo_epochs"),
        "lr": select(config, "actor_rollout_ref.actor.optim.lr"),
        "protocol": {
            k: select(config, v)
            for k, v in {
                "student": "actor_rollout_ref.model.path",
                "teacher": "distillation.teacher_models.teacher_model.model_path",
                "n": "actor_rollout_ref.rollout.val_kwargs.n",
                "temperature": "actor_rollout_ref.rollout.val_kwargs.temperature",
                "top_p": "actor_rollout_ref.rollout.val_kwargs.top_p",
                "cap": "actor_rollout_ref.rollout.val_kwargs.max_tokens",
                "seed": "data.seed",
            }.items()
        },
    }


def label(run):
    variant = " / " + run["variant"] if run.get("variant") else ""
    return run["preset"] + variant + (" / " + run["role"] if run["role"] else "") + " · `" + run["id"] + "`"


def latest(run):
    return run["evaluations"][-1] if run["evaluations"] else {"outer": "—", "update": "—", "datasets": {}}


def triple(point, key):
    return " / ".join(percent(point["datasets"].get(ds, {}).get(key)) for ds in DATASETS)


def render(runs, logs_commit, raw_commit, references):
    date = datetime.now(timezone(timedelta(hours=8))).date().isoformat()
    seen = {
        r["preset"]
        for r in runs
        if not r["role"] and (not r.get("variant") or family(r["name"]) == "mirror-comparison")
    }
    missing = "、".join(f"E{i}" for i in range(1, 48) if i not in [21, 22] and f"E{i}" not in seen) or "无"
    lines = [
        "# 数学消融实验验证结果汇总",
        "",
        f"更新：{date}（北京时间）。"
        f"覆盖 [统一消融脚本](../../run_math_opd_ablation_fsdp2.sh)在本地日志中已有记录的 **{len(runs)} 个 run**。"
        f"读取日志仓库提交 `{logs_commit[:9]}`，"
        f"这些原始 run 最近的更新提交为 `{raw_commit[:9]}`。",
        "",
        "本页维护 **最后一次实际验证** 的 mean accuracy、pass@16、majority@16 和可用的 95% CI。"
        "所有分数以百分比表示；`—` 表示未记录。主表不挑选最佳 checkpoint，"
        "不把不同 checkpoint 的三个集合最高分拼成一行。",
        "",
        "## 1. 评测设置与口径",
        "",
        "验证集合为 MATH-500、AMC23、AIME25，完整协议对应 500/40/30 题。"
        "以下模型和生成参数来自每个 run 的实际 config；相同配置合并展示。",
        "",
        "| 对应 run | Student | Teacher | 每题回答数 | Temperature / top-p | 验证 response cap | Seed |",
        "| --- | --- | --- | ---: | --- | ---: | --- |",
    ]
    groups = {}
    for run in runs:
        signature = json.dumps(run["protocol"], sort_keys=True)
        groups.setdefault(signature, []).append(run)
    for signature, group in groups.items():
        p = json.loads(signature)
        name = "全部已列 run" if len(groups) == 1 else "、".join(sorted({r["preset"] for r in group}))
        lines.append(
            f"| {name} | {Path(p['student']).name} | "
            f"{Path(p['teacher']).name} | {p['n']} | "
            f"{p['temperature']} / {p['top_p']} | {p['cap']} | {p['seed']} |"
        )
    lines += [
        "",
        "Mean accuracy 是题目等权的回答正确率均值；pass@16 是 16 条候选中至少一次正确的概率；"
        "majority@16 是答案投票结果。旧日志的 bootstrap `best@16/maj@16` 另列，"
        "不能改名为新版 pass/majority。旧 `std` 也不当作 mean 的 95% CI。"
        "定义见[指标字典](metrics/指标字典.md)。",
        "",
        "各实验训练预算、硬件与代码阶段不同，下表用于盘点，不能直接按末点排名宣称算法优劣。"
        "E13–E17 当前脚本默认已改为 250 outer 并做初始验证；"
        "历史主 run 的实际预算为 200 outer、无初始验证。",
        "",
        "## 2. 末次验证 mean accuracy",
        "",
        "| 实验 / run | 实际方法 × 复用；LR | 验证 outer / update | MATH-500 | AMC23 | AIME25 | 本地日志状态 |",
        "| --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for r in runs:
        point = latest(r)
        scores = [percent(point["datasets"].get(d, {}).get("mean")) for d in DATASETS]
        state = (
            "完整，exit=0"
            if r["state"] == "finished" and r["exit_code"] == 0
            else ("部分快照，无退出记录" if r["exit_code"] is None else f"{r['state']}，exit={r['exit_code']}")
        )
        lines.append(
            f"| {label(r)}[^run-{r['id']}] | {r['method']} ×{r['reuse']}；"
            f"{r['lr']:g} | {point['outer']} / {point['update']} | " + " | ".join(scores) + f" | {state} |"
        )
    lines += ["", "### 训练进度与末次验证的区别", ""]
    for r in runs:
        p = latest(r)
        if p["update"] != r["max_update"] or r["exit_code"] != 0:
            lines.append(
                f"- {label(r)}：训练记录到 outer {r['max_outer']} / update {r['max_update']}；"
                f"最后验证在 outer {p['outer']} / update {p['update']}。末次分数属于该验证点，"
                f"不能当成训练末点结果。"
            )
    lines += [
        "",
        "同一编号的重跑分开登记；例如 E15 的完整 run 与部分 run 不合并。"
        "`partial_snapshot` 仅说明本地日志末尾不完整，不能单独判断远端训练是否失败。",
        "",
        "## 3. 新口径的 pass@16、majority@16 与 CI",
        "",
        "以下只列实际发出 policy-v3 字段的 run，每行与主表使用同一个末次验证点。"
        "CI 为日志记录的题目 bootstrap 区间，不是跨训练 seed 方差，也没有跨 run 配对检验。",
    ]
    modern = [r for r in runs if any(v.get("version") == "policy-v3" for v in latest(r)["datasets"].values())]
    for dataset, name in DATASETS.items():
        lines += [
            "",
            f"### {name}",
            "",
            "| 实验 / run | 验证 update | Mean [95% CI] | Pass@16 | Majority@16 |",
            "| --- | ---: | --- | ---: | ---: |",
        ]
        for r in modern:
            p = latest(r)
            m = p["datasets"].get(dataset, {})
            score = f"{percent(m.get('mean'))} [{percent(m.get('ci_low'))}, {percent(m.get('ci_high'))}]"
            lines.append(
                f"| {label(r)} | {p['update']} | {score} | "
                f"{percent(m.get('pass16'))} | {percent(m.get('majority16'))} |"
            )
    lines += [
        "",
        "E8–E11 的 policy-v2 日志只有对应 `acc_mean`、回答数和失败计数，没有新版 pass/majority；"
        "这些值不能从 mean accuracy 推回。完整记录仍在主表中。",
        "",
        "## 4. 早期 bootstrap 指标单独保留",
        "",
        "下面每格按 **MATH / AMC / AIME** 顺序列出。"
        "旧 best/maj 的估计器与新版不同，只用于解释早期实验，不与第 3 节作同口径横向排名。",
        "",
        "| 实验 / run | 验证 update | 旧 best@16/mean | 旧 maj@16/mean |",
        "| --- | ---: | --- | --- |",
    ]
    for r in runs:
        p = latest(r)
        if any(v.get("version") == "legacy" for v in p["datasets"].values()):
            lines.append(f"| {label(r)} | {p['update']} | {triple(p, 'legacy_best16')} | {triple(p, 'legacy_maj16')} |")
    lines += [
        "",
        "## 5. E18–E20 的共同更新预算",
        "",
        "末次验证表中 E18/E19 已到 4096，E20 只到 2560。下面固定 update=2560，"
        "以免把不同训练进度放在一起比较。每格均按 MATH / AMC / AIME 排列。",
        "",
        "| 实验 / run | Outer | Mean | Pass@16 | Majority@16 |",
        "| --- | ---: | --- | --- | --- |",
    ]
    for r in runs:
        if r.get("variant") or r["preset"] not in ["E18", "E19", "E20"]:
            continue
        p = next((p for p in r["evaluations"] if p["update"] == 2560), None)
        if p:
            lines.append(
                f"| {label(r)} | {p['outer']} | {triple(p, 'mean')} | "
                f"{triple(p, 'pass16')} | {triple(p, 'majority16')} |"
            )
    lines += [
        "",
        "其他直接可比的已记录终点包括 E2–E5 的 128 updates、E13/E14/E15 的 200 updates、E8–E11 与 E17 的 3200 updates，"
        "以及 E18/E19 的 4096 updates；预算相同仍需核对指标版本、硬件与实际配置。"
        "详细解释见[等更新预算分析](equal-update-budget/README.md)及[Full-KL 对照](full-kl-comparison/README.md)。",
        "",
        "## 6. 原始师生参考",
        "",
        "这两条是历史 One-Shot-OPD val-only 参考，采样预算相近，但输入／执行管线不同，"
        "不作为上述每个 run 的真实 step 0。只列可解释的 mean accuracy。",
        "",
        "| 参考 / run | MATH-500 | AMC23 | AIME25 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for r in references:
        lines.append(f"| {r['kind']} · `{r['id']}` | " + " | ".join(percent(r["means"][d]) for d in DATASETS) + " |")
    lines += [
        "",
        "来源与限制见[师生参考分析](full-kl-comparison/2026-10-24-Full-KL收敛与师生参考评测.md)。"
        "新的统一脚本 `-valid-student/-valid-teacher` 正式评测应独立登记，不能用历史参考填补。",
        "",
        "## 7. 尚无正式验证结果的配置",
        "",
        f"当前归档未发现正式训练记录的标准编号：{missing}。短测和实现验收不计入本页。"
        "E21/E22 是脚本对 E15/E13 的别名，不重复登记。"
        "更多模型／长度计划见[失败复现矩阵](failure-reproduction/README.md)。",
        "",
        "## 8. 更新方法与来源字段",
        "",
        "本页由项目脚本生成，"
        "完整逐 checkpoint 明细写在日志仓库 `derived/MirrorOPD-verl/validation-summary/`，"
        "不向源码复制 config/history。CSV 中准确率等概率值保留 0–1，正文显示百分比。"
        "每次刷新先更新原始日志；脚本会调用通用解码器，再从 history 提取验证值和坐标。",
        "",
        "```bash",
        "python3 examples/mirroropd/research/analysis/summarize_validation.py \\",
        "  --logs-repo ../wandb-logs \\",
        "  --output examples/mirroropd/docs/experiments/验证结果汇总.md",
        "```",
        "",
        "| 语义 | 新字段 | 早期字段 |",
        "| --- | --- | --- |",
        "| Mean accuracy | `eval/{dataset}/acc/mean` | `eval/{dataset}/acc_mean` 或 `val-core/{dataset}/acc/mean@16` |",
        "| Mean 的 95% CI | `acc/mean_ci_low/high` | 未记录时留空，旧 std 不替代 |",
        "| Pass@16 | `acc/pass@16` | 旧 `best@16/mean` 独立列示 |",
        "| Majority@16 | `acc/majority@16` | 旧 `maj@16/mean` 独立列示 |",
        "",
        "更新时保留同一验证点的三个集合；数据缺失就留空，不从更早 checkpoint 借分数。"
        "需要最佳 checkpoint 时另表注明选择规则和坐标，不覆盖末次结果。"
        "新增模型对或评测协议时核对配置分组，并同步更新缺口说明。",
        "",
        "### 参考文献",
        "",
    ]
    lines += [
        f"[^run-{r['id']}]: W&B. {r['name']}. [运行页面](https://forge.coreweave.com/wandb/wncfht/MirrorOPD-verl/runs/{r['id']})。"
        f"数值来自本地原始日志。"
        for r in runs
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logs = args.logs_repo.resolve()
    result = subprocess.run(
        [str(logs / ".venv/bin/python"), str(logs / "tools/wandb_local.py"), "decode", "--project", "MirrorOPD-verl"],
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise SystemExit(result.stdout + result.stderr)
    runs = [r for p in (logs / "derived/MirrorOPD-verl").glob("*/run.json") if (r := read_run(p.parent))]
    runs.sort(key=lambda r: (int(r["preset"][1:]) if r["preset"].startswith("E") else 999, r["name"]))
    if len({r["id"] for r in runs}) != len(runs):
        raise ValueError("Duplicate local directories for a run ID")
    references = []
    for kind, rid in [("Student", "6uorrk2k"), ("Teacher", "u87buult")]:
        p = logs / f"MirrorOPD-verl/analysis/oneshot-{kind.lower()}-{rid}-api-20261020/history.jsonl"
        row = json.loads(p.read_text().splitlines()[0])
        references.append({"kind": kind, "id": rid, "means": {ds: extract(row, ds)[1]["mean"] for ds in DATASETS}})
    logs_commit = subprocess.check_output(["git", "-C", str(logs), "rev-parse", "HEAD"], text=True).strip()
    paths = [f"MirrorOPD-verl/wandb/{r['source_directory']}/run-{r['id']}.wandb" for r in runs]
    raw_commit = subprocess.check_output(
        ["git", "-C", str(logs), "log", "-1", "--format=%H", "--", *paths], text=True
    ).strip()
    output = logs / "derived/MirrorOPD-verl/validation-summary"
    output.mkdir(exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(
            {"logs_commit": logs_commit, "raw_commit": raw_commit, "runs": runs, "references": references},
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    records = []
    for r in runs:
        for p in r["evaluations"]:
            for ds, m in p["datasets"].items():
                records.append(
                    {
                        "preset": r["preset"],
                        "run_id": r["id"],
                        "outer": p["outer"],
                        "optimizer": p["update"],
                        "dataset": ds,
                        "version": m["version"],
                        **{k: m.get(k) for k in FIELDS},
                    }
                )
    with (output / "checkpoints.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    args.output.write_text(render(runs, logs_commit, raw_commit, references))
    print(f"Wrote {args.output}: {len(runs)} runs, {len(records)} dataset/checkpoint records")


if __name__ == "__main__":
    main()
