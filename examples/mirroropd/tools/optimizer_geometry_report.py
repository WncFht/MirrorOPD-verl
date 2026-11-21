# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Publish matched optimizer geometry and held-out readouts without touching training."""

import argparse
import time
from pathlib import Path

import numpy as np

from examples.mirroropd.research.core.io import dump, read
from examples.mirroropd.research.core.plotting import save_figure
from examples.mirroropd.research.studies.absorption_dynamics.geometry_plotting import band, grid, setup
from examples.mirroropd.research.studies.absorption_dynamics.geometry_probes import relation
from examples.mirroropd.research.studies.absorption_dynamics.geometry_readouts import collect, publish


def statistic(matrix, key, index=-1):
    stage = matrix["stages"][index]
    if key == "relative_norm":
        return np.sqrt(stage["energy"] / stage["weight"]["base_energy"])
    if key == "weight_overlap":
        return next(r["adjusted_overlap"] for r in stage["weight"]["frames"] if r["requested_rank"] == 64)
    if key == "future_overlap":
        return relation(matrix, stage["step"], 640, 64)["adjusted_overlap"]
    if key == "delta_sigma":
        return stage["weight"][key]
    return stage[key]


def summarize(data, plan):
    """Align coordinates before aggregation; matrix variation is not seed uncertainty."""
    pairs = data["pairs"]
    common = set.intersection(*(set(p["complete_layers"]) for p in pairs.values()))
    tables = {}
    for name, pair in pairs.items():
        matrices = [m for m in pair["matrices"] if m["layer"] in common]
        row = dict(optimizer=plan["pairs"][name]["optimizer"], lr=plan["pairs"][name]["lr"], matrices=len(matrices))
        for key in ("stable_rank", "rank90", "relative_norm", "weight_overlap", "delta_sigma"):
            values = [statistic(m, key) for m in matrices]
            row[key] = (
                dict(zip(("q25", "median", "q75"), np.quantile(values, [0.25, 0.5, 0.75]).tolist(), strict=True))
                if values
                else None
            )
        for first in (16, 480):
            values = [relation(m, first, 640, 64)["adjusted_overlap"] for m in matrices]
            row[f"overlap_{first}_640"] = float(np.median(values)) if values else None
        row["rkl"] = {}
        for bank in ("public_initial", "public_teacher"):
            row["rkl"][bank] = {}
            for domain in ("all", "math", "code", "if"):
                key = f"eval/rkl/{bank}/{domain}/question_mean"
                events = pair["risks"]
                y = [e["metrics"][key] for e in events]
                tail = y[-10:]
                slope = float(np.polyfit([e["outer"] for e in events[-10:]], tail, 1)[0])
                row["rkl"][bank][domain] = dict(
                    initial=y[0],
                    final=y[-1],
                    tail_mean=float(np.mean(tail)),
                    tail_std=float(np.std(tail)),
                    slope_per_outer=slope,
                    fitted_10outer_change_fraction=10 * slope / float(np.mean(tail)),
                )
        tables[name] = row
    return dict(common_layers=sorted(common), complete=all(p["complete"] for p in pairs.values()), experiments=tables)


def render_comparison(data, summary, figures):
    _, fv = setup()
    colors = fv.series_colors("paper", 2)
    styles = {}
    for family, color in zip(("SGD", "Muon"), colors, strict=True):
        names = sorted(
            (n for n, r in summary["experiments"].items() if r["optimizer"] == family),
            key=lambda n: summary["experiments"][n]["lr"],
        )
        for name, linestyle in zip(names, ("-", "--", "-.", ":"), strict=False):
            styles[name] = color, linestyle
    fig, axes = grid(2, 2, width=12, height=8)
    for name, pair in data["pairs"].items():
        color, linestyle = styles[name]
        meta = summary["experiments"][name]
        label = f"{name} {meta['optimizer']} {meta['lr']:g}"
        for col, bank in enumerate(("public_initial", "public_teacher")):
            for row, domain in enumerate(("all", "math")):
                ax = axes[row, col]
                events = pair["risks"]
                ax.plot(
                    [e["outer"] for e in events],
                    [e["metrics"][f"eval/rkl/{bank}/{domain}/question_mean"] for e in events],
                    color=color,
                    linestyle=linestyle,
                    label=label,
                )
                ax.set_xlabel("Outer iteration")
                ax.set_ylabel(f"{bank.replace('public_', '')} bank: {domain} RKL")
    fig.legend(
        *axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.065), fontsize=9
    )
    fig.get_layout_engine().set(rect=(0, 0.18, 1, 0.80))
    save_figure(
        fig,
        figures / "optimizer-rkl",
        "Same Qwen pair; E125 has independently generated banks. All = equal domain weights.",
    )
    common = summary["common_layers"]
    if not common:
        return
    fig, axes = grid(2, 3, width=15, height=8)
    fields = [
        ("stable_rank", "Update stable rank"),
        ("rank90", "Update rank90"),
        ("relative_norm", "Update norm / initial weight norm"),
        ("future_overlap", "Update-frame overlap to final, k64"),
        ("weight_overlap", "Weight-frame overlap to initial, k64"),
        ("delta_sigma", "Full-weight spectral distance"),
    ]
    for name, pair in data["pairs"].items():
        color, linestyle = styles[name]
        matrices = [m for m in pair["matrices"] if m["layer"] in common]
        steps = [s["step"] for s in matrices[0]["stages"]]
        for ax, (key, label) in zip(axes.flat, fields, strict=True):
            band(ax, steps, [[statistic(m, key, i) for i in range(len(steps))] for m in matrices], color, name)
            ax.lines[-1].set_linestyle(linestyle)
            ax.set_xlabel("Optimizer updates")
            ax.set_ylabel(label)
    for ax in (axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 2]):
        ax.set_yscale("log")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=6, bbox_to_anchor=(0.5, 0.065))
    fig.get_layout_engine().set(rect=(0, 0.17, 1, 0.81))
    save_figure(
        fig,
        figures / "optimizer-geometry",
        f"Matched {len(common)} layers / {len(common) * 7} matrices per run; "
        "median and IQR, not seed confidence intervals.",
    )


def write_report(data, plan, summary, report, source):
    names = sorted(data["pairs"], key=lambda n: int(n[1:]))
    lines = [
        "# SGD 与 Muon 的九时刻权重动力学",
        "",
        "## 设置与比较范围",
        "",
        (
            "E124–E129 均为 Qwen3-1.7B ← Qwen3-4B-Instr"
            "uct-2507，40 outer × reuse"
            "16，共 640 次优化器更新。E123 只有预备配置，没有训练产物，按确认标记"
            "缺失。这里不包含 AdamW 的新测量，也不能用这"
            "六组比较 AdamW 的优劣。"
        ),
        "",
        (
            "复用 [checkpoint-dynamics](../README.md) 的测量实现：outer"
            " 1/2/3/4/8/12/20/30/40；每层"
            " q/k/v/o/gate/up/down，共 28×7=196 矩阵。读取经 "
            "native reload RKL 验收的 FP3"
            "2 模型导出，在 GPU 4–7 上执行 FP64 精确 SVD（gesvd），"
            "右子空间 rank=16/32/64/128/25"
            "6。测量与训练分离，不需要 optimizer state。"
        ),
        "",
        (
            "累计更新为 ΔWₜ=Wₜ−W₀，区间增量为 Wᵦ−Wₐ；完整权重谱 σ(Wₜ)、"
            "累计更新谱 σ(ΔWₜ) 与谱差 σ(Wₜ)−σ("
            "W₀) 分开报告。Frame 使用主夹角和随机基线校正 overlap，原图同时"
            "报告能量捕获与边界谱隙。这里的 frame 是各配"
            "置内部随时间的比较，没有计算两个优化器的直接跨配置主夹角。"
        ),
        "",
        (
            "E126–E129 复用 E124 的固定 bank；E125 独立生成固定 b"
            "ank。各组题单相同不等于生成的回答相同，E125"
            " 与其余组的绝对 RKL 比较包含 bank 差异。Muon 的矩阵学习率变化，"
            "辅助参数仍用 AdamW LR=1e-6、WD=0"
            "；不能解释为所有参数均只使用 Muon。"
        ),
        "",
        "## 覆盖与数据",
        "",
        f"数据读取时间：{time.strftime('%Y-%m-%d %H:%M:%S %z')}。"
        f"几何状态：{'全部完成' if summary['complete'] else '测量进行中，以下几何为部分层预览'}。"
        f"共同完整层：{summary['common_layers']}。",
        "",
        "| 实验 | Optimizer | LR | 已测矩阵 | 快照/矩阵 |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for name in names:
        p, s = data["pairs"][name], summary["experiments"][name]
        lines.append(f"| {name} | {s['optimizer']} | {s['lr']:g} | {len(p['matrices'])}/{p['expected']} | 9 |")
    lines += [
        "",
        f"原始测量：`{source}`。"
        "运行协议、张量 SHA256、逐矩阵原始奇异值、计时和失败回执均保留；"
        "[覆盖](data/coverage.json)、[统计](data/summa"
        "ry.json)、[精确 readout](data/readout.json."
        "gz)。"
        "每条成功矩阵记录覆盖九时刻；196/196 才是该组全层完成。",
        "",
        "## Held-out RKL 与后段变化",
        "",
        "all 为 Math/Code/IF 三域等权平均。后段均值和斜率使用 oute"
        "r31–40；斜率单位为每 outer RKL。这是对后段趋势的描述，单 see"
        "d 和十个时刻不足以确定渐近上限。",
        "",
        (
            "| 实验 | initial-bank 终点 | teacher-bank 终点"
            " | initial 后10均值 | teache"
            "r 后10均值 | initial 后段斜率 | teacher 后段斜率 |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in names:
        s = summary["experiments"][name]
        a, b = [s["rkl"][bank]["all"] for bank in ("public_initial", "public_teacher")]
        lines.append(
            f"| {name} | {a['final']:.6f} | {b['final']:.6f} | "
            f"{a['tail_mean']:.6f} | {b['tail_mean']:.6f} | "
            f"{a['slope_per_outer']:+.3g} | {b['slope_per_outer']:+.3g} |"
        )
    lines += [
        "",
        "![两套固定 bank 的 RKL](figures/optimizer-rkl.png)",
        "",
        (
            "Muon 的 LR 从 3e-7 提至 3e-6 时，initial-bank "
            "all 终点改善，但 teacher-bank a"
            "ll 终点变差；到 1e-5 时 teacher-bank all 终点超过初始"
            "值。较大的 LR 改变了对两种状态分布的适配取舍。"
            "E124 的 teacher-bank all 终点同样高于初始值，SGD 也没"
            "有消除这种分布差别。不能只按 initial ba"
            "nk 的下降认定平台改善。"
        ),
        "",
        "## 谱与 frame 的共同坐标比较",
        "",
        "以下逐矩阵指标只使用六组都测齐七类投影的共同层，取矩阵中位数。矩阵间 IQR 是"
        "不同层/模块的差异，不是多 seed 置信区间。",
        "",
        (
            "| 实验 | 纳入矩阵 | 最终 stable rank | 最终 rank90"
            " | ‖ΔW‖/‖W₀‖ | ΔW frame 1"
            "6→640 k64 | ΔW frame 480→640 k64 | W frame 0→640 k64 |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in names:
        s = summary["experiments"][name]
        if not s["matrices"]:
            continue
        lines.append(
            f"| {name} | {s['matrices']} | {s['stable_rank']['median']:.3g} | "
            f"{s['rank90']['median']:.3g} | {s['relative_norm']['median']:.4g} | "
            f"{s['overlap_16_640']:.4f} | {s['overlap_480_640']:.4f} | {s['weight_overlap']['median']:.4f} |"
        )
    if summary["common_layers"]:
        lines += ["", "![优化器的共同层几何轨迹](figures/optimizer-geometry.png)", ""]
        sgd = [
            summary["experiments"][n]["stable_rank"]["median"]
            for n in names
            if summary["experiments"][n]["optimizer"] == "SGD"
        ]
        muon = [
            summary["experiments"][n]["stable_rank"]["median"]
            for n in names
            if summary["experiments"][n]["optimizer"] == "Muon"
        ]
        coverage_note = (
            "这是完整 196 矩阵的描述。"
            if summary["complete"]
            else "覆盖尚未完整，数值随层数增加更新，暂不推广到全模型。"
        )
        lines += [
            f"共同层累计更新的最终 stable rank：SGD 为 {min(sgd):.3g}–{max(sgd):.3g}，"
            f"Muon 为 {min(muon):.3g}–{max(muon):.3g}。" + coverage_note,
            "",
        ]
    if summary["common_layers"]:
        records = summary["experiments"]
        selected = [(name, records[name]) for name in names]
        low, high = records["E126"], records["E129"]
        lines += [
            f"Muon 3e-7→1e-5 的最终相对更新范数中位数扩大 "
            f"{high['relative_norm']['median'] / low['relative_norm']['median']:.2f} 倍。"
            f"对应的 k64 早期→最终 overlap 为 {low['overlap_16_640']:.3f}→{high['overlap_16_640']:.3f}，"
            f"末段 480→640 为 {low['overlap_480_640']:.3f}→{high['overlap_480_640']:.3f}。",
            "",
            "Stable rank=‖ΔW‖²_F/‖ΔW‖²₂ 由最大奇异值主导；rank90 统计累积到 90% 能量所需维数。"
            "SGD 即便 stable rank 很低，rank90 仍可能很大，表示存在主导方向及长谱尾，不能直接称为严格低秩更新。",
            "",
        ]
        for name, record in selected:
            row = record["rkl"]["public_teacher"]["all"]
            lines += [
                f"{name}：最终 stable rank={record['stable_rank']['median']:.3g}，"
                f"rank90={record['rank90']['median']:.3g}；teacher-bank all 后段均值={row['tail_mean']:.5f}，"
                f"拟合十个 outer 的变化为均值的 {100 * row['fitted_10outer_change_fraction']:+.2f}%。",
                "",
            ]
    lines += [
        (
            "Muon 每步对动量矩阵做近似正交化，弱化梯度奇异值幅度的差别；SGD 沿原梯度"
            "/动量的谱幅度更新。这使更新谱是否集中成为可检验差"
            "别，但累计 ΔW 是各步矩阵之和，仍会受到方向旋转与相消影响。LR 改变轨迹以后"
            "，累计谱形状与 frame 也会变化，不能把两档 "
            "LR 视为同一个最终方向的纯缩放。"
        ),
        "",
        (
            "较低的初始—最终 frame overlap 表示主更新子空间继续变化，不能单独"
            "解释为优化更好或泛化更好。应结合 late-inc"
            "rement 被早期 frame 捕获的能量、rank 边界谱隙以及两套 ban"
            "k 的后段 RKL 判断。完整权重 frame 保"
            "持稳定而 ΔW frame 变化也不矛盾：在更新相对 W₀ 很小时，两者描述的对象与尺度不同。"
        ),
        "",
        (
            "现有数据能描述优化器如何分配更新能量、方向何时形成，以及这些量是否伴随 held"
            "-out 改善；不能从相关性推出某种谱造成平台。优"
            "先检查同坐标下各 LR 的增量范数是否仍大、480→640 frame 是否稳定"
            "、两套 bank 是否同时下降。如果参数持续变化而"
            " teacher bank 不再改善，才进一步区分目标分布取舍与有效迁移方向不足"
            "。训练批内拟合的因果检验还需要结合 batch-b"
            "efore/after 指标，当前权重测量没有新增训练干预。"
        ),
        "",
        "## 全部原协议图",
        "",
        "每图同时保存 PNG、PDF 与 audit 文件。完整精确奇异值保留在原始矩阵"
        "记录；readout 为绘图用能量 CDF 和角度分位数。",
        "",
    ]
    for name in names:
        images = sorted((report / "figures").glob(f"{name}-*.png"))
        if images:
            lines += [f"### {name}", ""]
        for image in images:
            lines += [f"![{image.stem}](figures/{image.name})", ""]
    (report / "README.md").write_text("\n".join(lines))


def refresh(source, report, *, original=True):
    plan = read(source / "protocol.json")
    data = collect(source, report / "data")
    summary = summarize(data, plan)
    dump(report / "data/summary.json", summary)
    figures = report / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    if original:
        publish(data, figures)
    render_comparison(data, summary, figures)
    write_report(data, plan, summary, report, source)
    dump(
        report / "data/publication.json",
        dict(complete=summary["complete"], captured_at=data["captured_at"], failed_tasks=data["failed_tasks"]),
    )
    from examples.mirroropd.tools.maintenance.pipeline import REPO, format_paths

    if format_paths(REPO, [(report / "README.md").resolve()], write=True):
        raise RuntimeError("Optimizer report formatting failed")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=int, default=600)
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("interval must be positive")
    while True:
        try:
            summary = refresh(args.source, args.report)
        except Exception as error:
            dump(args.report / "publication-error.json", dict(at=time.time(), error=repr(error)))
            raise
        if not args.watch:
            return
        if (args.source / "exit.json").exists():
            receipt = read(args.source / "exit.json")
            if receipt["exit_code"] != 0 or not summary["complete"]:
                raise RuntimeError("Measurement exited without complete coverage; inspect raw receipts")
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
