# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Refresh result tables and figures in a separately maintained research report."""

import os
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np

from .figure_catalog import FAMILIES


def write_report(data, destination, figures):
    """Replace only the generated results block; preserve the authored interpretation."""
    destination.mkdir(parents=True, exist_ok=True)
    page = destination / "README.md"
    text = page.read_text()
    start, end = "<!-- results:start -->", "<!-- results:end -->"
    before, rest = text.split(start)
    _, after = rest.split(end)
    timestamp = datetime.fromtimestamp(data["captured_at"], ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"数据更新于 {timestamp}（北京时间）。矩阵完成后包含全部九个 checkpoint；完整层要求七类投影齐全。",
        "",
        "| Pair | Student → teacher | 已测矩阵 | 完整层 | 全模型几何覆盖 |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for name, p in data["pairs"].items():
        lines.append(
            f"| {name} | {p['student']} → {p['teacher']} | "
            f"{len(p['matrices'])}/{p['expected']} | {len(p['complete_layers'])} | "
            f"{'完成' if p['complete'] else '部分'} |"
        )
    lines += [
        "",
        "所有 pair 的 held-out RKL 均来自完成的 640 更新训练。"
        "initial 和 teacher 两种 bank 分开报告；括号为终点相对初始的剩余百分比。",
        "",
        "| Pair | initial bank：平均 RKL | teacher bank：平均 RKL | initial bank：Math RKL | teacher bank：Math RKL |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, p in data["pairs"].items():
        a, b = p["risks"][0]["metrics"], p["risks"][-1]["metrics"]
        cells = []
        for bank, domain in [
            ("public_initial", "all"),
            ("public_teacher", "all"),
            ("public_initial", "math"),
            ("public_teacher", "math"),
        ]:
            k = f"eval/rkl/{bank}/{domain}/question_mean"
            cells.append(f"{b[k]:.6g}（{100 * b[k] / a[k]:.2f}%）")
        lines.append("| " + name + " | " + " | ".join(cells) + " |")
    lines += [
        "",
        "以下几何统计使用每个 pair 当前完整层的相同矩阵集合，均为逐矩阵中位数。不同 pair 尚未测齐时不据此排序。",
        "",
        "| Pair | 纳入矩阵 | 完整权重：0→640 SD 距离 | 累计更新：16→640 SD 距离 | k64：16→640 调整后 overlap | "
        "OPD/teacher 更新幅度 | 参数残差能量比 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    included = []
    for name, p in data["pairs"].items():
        matrices = [m for m in p["matrices"] if m["layer"] in p["complete_layers"] and "sigma_variation" in m]
        if not matrices:
            continue
        included.append((name, p, matrices))
        vals = []
        for kind, first in [("weight", 0), ("update", 16)]:
            vals.append(
                np.median(
                    [
                        next(
                            x["sd_variation"]
                            for x in m["sigma_variation"][kind]["pairs"]
                            if x["step_a"] == first and x["step_b"] == 640
                        )
                        for m in matrices
                    ]
                )
            )
        overlap = np.median(
            [
                next(
                    x["adjusted_overlap"]
                    for x in m["pairs"]
                    if x["step_a"] == 16 and x["step_b"] == 640 and x["requested_rank"] == 64
                )
                for m in matrices
            ]
        )
        teacher = [m["stages"][-1].get("teacher_comparison") for m in matrices]
        cells = [
            f"{np.median([t[k] for t in teacher]):.4g}" if all(teacher) else "跨尺寸，不比较"
            for k in ["norm_ratio", "residual_energy_ratio"]
        ]
        lines.append(
            f"| {name} | {len(matrices)} | {vals[0]:.4g} | {vals[1]:.4g} | {overlap:.4f} | " + " | ".join(cells) + " |"
        )
    lines += ["", "### 跨 pair 总览", ""]
    for image in sorted(figures.glob("overview-*.png")):
        lines += [f"![{image.stem}]({os.path.relpath(image, destination)})", ""]
    lines += ["完整分类图集见[图集](图集.md)。下列四组图优先展示更新维度、方向和后续学习。", ""]
    gallery = [
        "# Checkpoint 动力学图集",
        "",
        f"数据时间：{timestamp}（北京时间）。",
        "",
        "[返回设置、总表与分析](README.md)。所有图均有同名 PDF 和审计文件；精确数据在相邻 data 目录。",
        "",
        "交互式 HTML 对比页面 `explorer.html`：选择多个 pair 叠加轨迹，筛选 rank、层与模块，或比较共同色标热图。"
        "用法与复算见 [HTML 使用说明](README.md#交互式-html-对比)。",
        "",
        "主角分位曲线使用 33 点插值；ECDF 在每 5 度位置精确统计。部分层预览标明覆盖，跨 pair 总览只用完整组。",
        "",
    ]
    for name, p, matrices in included:
        coverage = f"{len(matrices)}/{p['expected']} 个纳入矩阵，{len(p['complete_layers'])} 个完整层；" + (
            "全模型覆盖。" if p["complete"] else "部分层预览。"
        )
        lines += [f"### {name}：{p['student']} → {p['teacher']}", "", coverage, ""]
        for suffix in ("update-dimension", "frame-rank-time", "frame-increments", "singular-energy-rank"):
            image = figures / f"{name}-{suffix}.png"
            if image.exists():
                lines += [f"![{name}：{suffix}]({os.path.relpath(image, destination)})", ""]
        gallery += [f"## {name}：{p['student']} → {p['teacher']}", "", coverage, ""]
        for category, entries in FAMILIES.items():
            available = [(suffix, label) for suffix, label in entries if (figures / f"{name}-{suffix}.png").exists()]
            if not available:
                continue
            gallery += [f"### {category}", ""]
            for suffix, label in available:
                image = figures / f"{name}-{suffix}.png"
                gallery += [f"![{name}：{label}]({os.path.relpath(image, destination)})", ""]
    (destination / "图集.md").write_text("\n".join(gallery))
    if data["failed_tasks"]:
        lines += ["", f"测量错误记录：{data['failed_tasks']} 个。完成状态还需核对剩余任务。"]
    tmp = page.with_suffix(".tmp")
    tmp.write_text(before + start + "\n\n" + "\n".join(lines) + "\n" + end + after)
    tmp.replace(page)
    # Keep live-generated pages in the same canonical format as authored docs.
    from examples.mirroropd.tools.maintenance.pipeline import REPO, format_paths

    if format_paths(REPO, [page.resolve(), (destination / "图集.md").resolve()], write=True):
        raise RuntimeError("Generated geometry report formatting failed")
