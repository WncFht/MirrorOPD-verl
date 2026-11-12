# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Scientific figure families, shared by publication and the standalone gallery."""

FAMILIES = {
    "输出与 teacher 对齐": [
        ("sigma-teacher", "三域 RKL 与 teacher 谱距离"),
        ("sigma-teacher-distance", "Teacher SD 距离与相对初始剩余距离"),
        ("frame-teacher-final", "更新 frame：teacher 与自身终点对齐"),
        ("parameter-residual", "更新幅度、方向与参数残差"),
        ("residual-layer-modules", "各层各模块的参数残差"),
    ],
    "Frame 的方向与时间结构": [
        ("early-final", "Early→final overlap 与相对最优能量捕获"),
        ("frame-angles", "主角分位曲线与角度 ECDF"),
        ("frame-angle-time", "主角 median、P90、最大值动态"),
        ("frame-pairwise", "Checkpoint 两两 frame 相似度"),
        ("frame-rank-time", "五档 rank × 时间：overlap 与能量覆盖"),
        ("frame-modules", "七类模块的 frame 动态"),
        ("frame-layer-modules", "层 × 模块 × 阶段的 frame"),
        ("weight-update-frames", "完整权重与累计更新的 frame 转角"),
    ],
    "早期空间对后续增量的覆盖": [
        ("frame-increments", "Anchor × 后续区间的相对最优捕获"),
        ("frame-outside-energy", "早期空间内部、外部的绝对能量与外部比例"),
    ],
    "Singular spectrum 与更新维度": [
        ("update-dimension", "绝对/相对更新范数、stable rank、rank90/95、熵有效秩"),
        ("singular-energy-rank", "完整权重、累计更新、区间增量的能量–rank 曲线"),
        ("singular-profiles", "七类模块的实际完整权重奇异值曲线"),
        ("singular-update-profiles", "七类模块的累计更新奇异值曲线"),
        ("singular-rank-time", "奇异值 rank × 时间的有符号变化"),
        ("singular-scale-shape", "整体谱缩放、形状残差与最大谱扰动"),
        ("singular-layer-modules", "更新维度与幅度的层 × 模块差异"),
    ],
    "谱形状变化及其与 frame 的关系": [
        ("weight-spectrum", "完整权重 ISO 三指标"),
        ("sigma-dynamics", "完整权重和累计更新的 SD variation"),
        ("sigma-pairwise", "Checkpoint 两两谱距离"),
        ("sigma-layers", "逐层谱变化速度"),
        ("singular-gap-angles", "谱隙与相邻 frame 变化"),
        ("singular-frame-phase", "谱变化与 frame 变化的联合分布"),
    ],
}
