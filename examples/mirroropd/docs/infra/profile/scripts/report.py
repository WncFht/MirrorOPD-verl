# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Generate a factual report from completed jobs; failed/absent runs stay explicit."""

import argparse
import json
import shutil
import statistics
from datetime import datetime
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--publish-dir", type=Path)
    args = p.parse_args()
    state = json.loads((args.root / "status.json").read_text())
    text = [
        "# Full-KL 分层 profile 实验记录",
        "",
        f"生成时间：{datetime.now().isoformat(timespec='seconds')}。",
        f"源码提交：`{state['source_commit']}`。原始输出：`{args.root}`。",
        "",
        "这份文档由测量文件自动生成。算子测试使用合成 BF16 hidden；固定输入管线使用真实"
        "模型和历史数学回答重新分词，lr=0 保持对照权重一致；完整"
        "训练另走 Ray/TQ/vLLM。候选融合算子仍在实验目录，未替换生产默认。",
        "",
        "## 执行状态",
        "",
        "| 任务 | 退出码 | 墙钟秒数 |",
        "| --- | ---: | ---: |",
    ]
    for name, row in state["jobs"].items():
        text.append(f"| {name} | {row['returncode']} | {row['wall_seconds']:.1f} |")
    text += [
        "",
        "## 算子对照",
        "",
        "| backend | 有效 token | chunk | 前后向中位数 ms | 峰值 allocated GiB |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for file in sorted(args.root.glob("kernel*/benchmark.json")):
        result = json.loads(file.read_text())
        for row in result["rows"]:
            if "wall_ms_median" in row:
                text.append(
                    f"| {row['backend']} | {row['tokens']} | {row['chunk']} | "
                    f"{row['wall_ms_median']:.2f} | {row['peak_allocated_gib']:.3f} |"
                )
    text += [
        "",
        "## 固定输入的真实模型管线",
        "",
        "排除首步预热和详细 profile 步；阶段包含对应模型上下卡。未经过 TQ，不能据此推断异步 IO 收益。",
        "",
        "| 配置 | 稳态样本数 | teacher 秒 | actor 秒 | 合计秒 | 最后 loss |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    pipes = {}
    for file in sorted(args.root.glob("pipeline-*/pipeline.json")):
        result = json.loads(file.read_text())
        rows = [r for r in result["records"] if not r["warmup"] and not r["profiled"]]
        if not rows:
            continue
        med = lambda k, rows=rows: statistics.median(r[k] for r in rows)
        name = file.parent.name
        pipes[name] = result
        text.append(
            f"| {name} | {len(rows)} | {med('teacher_s'):.3f} | {med('actor_s'):.3f} | "
            f"{med('wall_s'):.3f} | {rows[-1]['loss']:.7f} |"
        )
    if "pipeline-baseline" in pipes and "pipeline-fused" in pipes:
        a, b = [pipes[k] for k in ("pipeline-baseline", "pipeline-fused")]
        delta = abs(a["records"][-1]["loss"] - b["records"][-1]["loss"])
        text += ["", f"固定权重末次 loss 绝对差：`{delta:.9g}`。这项检查不等价于训练质量评估。"]
    text += [
        "",
        "## 完整训练与 trace",
        "",
        "完整训练日志保存在各 `e2e-io*/stdout.log`，profile 仅采第 3"
        " 步。两个运行不保证生成 token 完全一致，吞吐比较必须同时核对输入和有效 token"
        " 数。",
        "",
        "`trace-summary.json` 记录 CUDA kernel 区间并集与计算/通"
        "信重叠；kernel duration 累加值表示工作量，不"
        "等于整步墙钟时间。NCCL 未与计算重叠的时间不自动等于关键路径等待。图中的 80 ms "
        "窗口是局部时间线，完整交互分析使用原始 trace。",
        "",
        "## 复现和证据",
        "",
        "每个任务目录包含 `launch.json`、`stdout.log`、`exit.jso"
        "n`。profile 采集与非采集计时分开；候选必须通过归一"
        "化和梯度对照后才进入多卡实验。部分任务失败时，请读取其日志，不能将未完成的任务记为性能改进。",
        "",
    ]
    report = "\n".join(text)
    (args.root / "report.md").write_text(report)
    if args.publish_dir:
        args.publish_dir.mkdir(parents=True, exist_ok=True)
        (args.publish_dir / "2026-10-23-Full-KL分层测量记录.md").write_text(report)
        destination = args.publish_dir / "data" / args.root.name
        destination.mkdir(parents=True, exist_ok=True)
        for pattern in (
            "kernel*/benchmark.json",
            "pipeline-*/pipeline.json",
            "trace-summary.json",
            "figure-audit.json",
            "status.json",
        ):
            for file in args.root.glob(pattern):
                target = destination / file.relative_to(args.root)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file, target)
        if (args.root / "figures").exists():
            shutil.copytree(args.root / "figures", args.publish_dir / "figures" / args.root.name, dirs_exist_ok=True)


if __name__ == "__main__":
    main()
