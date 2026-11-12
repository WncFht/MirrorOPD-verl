# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Publish paired sigma0 RKL curves and self-contained data outside run directories.

Ledger: each point is one completed checkpoint with all 512 fixed trajectories.
Columns distinguish the two prefix banks; rows distinguish mean/Math/Code/IF.
Color encodes original versus restored weights consistently; horizontal position
is optimizer updates, vertical position is question RKL or paired RKL difference.
"""

import argparse
import gzip
import json
import time
from pathlib import Path

from examples.mirroropd.research.core.plotting import save_figure
from verl.trainer.evaluation.artifacts import file_sha256

from ..spectral.update_dynamics import atomic_json
from .geometry_plotting import grid, legend, setup


def collect(sources, destination):
    result = dict(schema="sigma0-bank-readout-v1", captured_at=time.time(), pairs={}, sources={})
    for source in sources:
        protocol = json.loads((source / "protocol.json").read_text())
        if protocol["smoke"]:
            raise ValueError("Smoke rows cannot be published as full-bank scientific results")
        for name in protocol["pairs"]:
            if name in result["pairs"]:
                raise ValueError("Pair appears in multiple score sources")
            result["pairs"][name] = dict(expected_stages=9, stages=[])
            for path in sorted((source / name).glob("step-*/complete.json")):
                stage = json.loads(path.read_text())
                if stage["rows"] != 512 or not stage["no_edit_gate"]["passed"]:
                    raise ValueError("Incomplete full-bank condition or no-edit gate")
                result["pairs"][name]["stages"].append(stage)
                relative = f"{name}/{path.parent.name}"
                result["sources"][relative] = dict(sha256=file_sha256(path))
                # Preserve original and edited per-question, per-position statistics.
                for filename in ("original.json", "sigma0.json", "edits.json", "no-edit-gate.json"):
                    raw = (path.parent / filename).read_bytes()
                    target = destination / "rows" / name / path.parent.name / (filename + ".gz")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if not target.exists():
                        temporary = target.with_suffix(".tmp")
                        temporary.write_bytes(gzip.compress(raw, mtime=0))
                        temporary.replace(target)
    destination.mkdir(parents=True, exist_ok=True)
    atomic_json(destination / "curve-data.json", result)
    return result


def publish(data, report):
    plt, fv = setup()
    colors = fv.series_colors("paper", 2)
    figures = report / "figures/sigma0-bank"
    figures.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 恢复初始奇异值后的完整 bank RKL",
        "",
        "对每个 checkpoint 的完整权重执行 $W_t=U_t\\Sigma_tV_t^\\top \\to U_t\\Sigma_0V_t^\\top$。"
        "编辑所有 Transformer 层的 q/k/v/o/gate/up/down；embedding、head、norm 等参数保留该 checkpoint。",
        "",
        "固定使用本轮 initial 和 teacher 两种 bank，每个 bank 为 Math128、Code64、IF64，共 512 条轨迹。"
        "保留原始 token 与 mask，评分所有有效响应位置，上限 7168；每题先按有效 token 平均，三域平均再等权。",
        "",
        "编辑在 FP64 中分解并重构，前向使用原生 BF16 权重、FA2 和主线稳定全词表 RKL 内核。"
        "未编辑与编辑条件均在独立 HF 路径重新评分，差值定义为编辑后减编辑前。"
        "训练时 FSDP 的逐题分数另作执行对照，不能替代这里的成对基线。"
        "每个阶段还对双 bank 三域各一条完整响应重复无编辑前向，要求最大逐题 RKL 差不超过 1e-8。",
        "",
        "[曲线数据](data/sigma0-bank/curve-data.json)与相邻 rows 子目录保存逐题及位置分段结果，绘图不依赖运行目录。",
        "",
        "| Pair | 完整阶段 | 最新更新数 | initial-bank Math：原始→恢复谱 | "
        "teacher-bank Math：原始→恢复谱 | 与训练记录最大逐题偏差 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, pair in data["pairs"].items():
        stages = pair["stages"]
        if not stages:
            lines.append(f"| {name} | 0/9 | — | — | — | — |")
            continue
        last = stages[-1]
        cells = []
        for bank in ("public_initial", "public_teacher"):
            key = f"eval/rkl/{bank}/math/question_mean"
            cells.append(f"{last['original'][key]:.6g} → {last['restored'][key]:.6g}")
        lines.append(
            f"| {name} | {len(stages)}/9 | {last['step']} | {' | '.join(cells)} | "
            f"{last['training_comparison']['max_question_abs_error']:.3g} |"
        )
        steps = [s["step"] for s in stages]
        for kind in ("absolute", "delta"):
            fig, axes = grid(4, 2, width=10, height=12)
            for col, bank in enumerate(("public_initial", "public_teacher")):
                for row, domain in enumerate(("all", "math", "code", "if")):
                    ax = axes[row, col]
                    key = f"eval/rkl/{bank}/{domain}/question_mean"
                    if kind == "absolute":
                        for color, field, label in zip(
                            colors, ("original", "restored"), ("Original", "Restore sigma0"), strict=True
                        ):
                            ax.plot(steps, [s[field][key] for s in stages], color=color, marker="o", label=label)
                    else:
                        ax.axhline(0, color=fv.NEUTRALS["baseline"], linewidth=0.8)
                        ax.plot(
                            steps,
                            [s["delta"][key] for s in stages],
                            color=colors[1],
                            marker="o",
                            label="Restored minus original",
                        )
                    ax.set_title(f"{bank.replace('public_', '')} / {domain}")
                    ax.set_xlabel("Optimizer updates")
                    ax.set_ylabel("Question-mean RKL" if kind == "absolute" else "Paired RKL difference")
            values = [
                s[field][f"eval/rkl/{bank}/{domain}/question_mean"]
                for s in stages
                for bank in ("public_initial", "public_teacher")
                for domain in ("all", "math", "code", "if")
                for field in (("original", "restored") if kind == "absolute" else ("delta",))
            ]
            if kind == "absolute":
                upper = max(max(values) * 1.08, 1e-6)
                for ax in axes.flat:
                    ax.set_ylim(0, upper)
            else:
                bound = max(max(abs(v) for v in values) * 1.1, 1e-8)
                for ax in axes.flat:
                    ax.set_ylim(-bound, bound)
            legend(axes[0, 0], 2)
            fig.suptitle(f"{name}: {'Paired full-bank RKL' if kind == 'absolute' else 'Effect of restoring sigma0'}")
            save_figure(
                fig,
                figures / f"{name}-{kind}",
                f"{len(stages)}/9 checkpoints; 512 complete trajectories each. all = equal-domain mean.",
            )
            plt.close(fig)
    lines += [
        "",
        "## 图与指标",
        "",
        "| 图 | 纵轴 | 用途 |",
        "| --- | --- | --- |",
        "| absolute | 未编辑与恢复谱后的绝对 RKL | 判断已学分布匹配是否保留 |",
        "| delta | 恢复谱后 RKL − 未编辑 RKL | 判断训练阶段对谱变化的依赖；正值表示恢复谱损伤匹配 |",
        "",
    ]
    for name, pair in data["pairs"].items():
        if pair["stages"]:
            lines += [
                f"## {name}",
                "",
                f"![{name} 的绝对 RKL](figures/sigma0-bank/{name}-absolute.png)",
                "",
                f"![{name} 的换谱效应](figures/sigma0-bank/{name}-delta.png)",
                "",
            ]
    temporary = report / "谱恢复与RKL.tmp"
    temporary.write_text("\n".join(lines) + "\n")
    temporary.replace(report / "谱恢复与RKL.md")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=int, default=300)
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("Positive publication interval required")
    while True:
        data = collect(args.source, args.report / "data/sigma0-bank")
        publish(data, args.report)
        if not args.watch:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
