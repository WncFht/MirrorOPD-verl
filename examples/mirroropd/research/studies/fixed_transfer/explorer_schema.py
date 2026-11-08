# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Shared labels, symbols and view contracts for the offline absorption explorer.

Storage field names remain stable for existing CSV consumers. Symbol labels are
presentation metadata, never a second implementation of the numerical metrics.
"""

METRICS = {
    "rkl": dict(symbol="K", label="RKL 残差", unit="nats / H1024", scale="log", formula="K_D(t)"),
    "log_absorption": dict(
        symbol="A", label="对数吸收量", unit="dex", scale="linear", formula="A_D(t) = log₁₀[K_D(0) / K_D(t)]"
    ),
    "relative": dict(symbol="R", label="相对下降", unit="%", scale="linear", formula="R_D(t) = 100[1 − K_D(t)/K_D(0)]"),
    "absolute": dict(
        symbol="B", label="绝对下降", unit="nats / H1024", scale="linear", formula="B_D(t) = K_D(0) − K_D(t)"
    ),
    "gap": dict(symbol="Δ", label="累计吸收差", unit="dex", scale="linear", formula="Δ_D(t) = A_T(t) − A_D(t)"),
    "efficiency": dict(
        symbol="η", label="累计迁移效率", unit="无量纲", scale="linear", formula="η_D(t) = A_D(t) / A_T(t)"
    ),
}
VIEWS = {
    "phase": dict(label="训练池 → 留出轨迹", group="原始轨迹", diagnostic=False, sampling="all", fixed_metric=None),
    "time": dict(label="随更新变化", group="原始轨迹", diagnostic=False, sampling="all", fixed_metric=None),
    "endpoint": dict(
        label="终点散点（所选指标）", group="原始轨迹", diagnostic=False, sampling="end", fixed_metric=None
    ),
    "absorption": dict(
        label="终点吸收量 · (A_T, A_D)",
        group="量化诊断",
        diagnostic=True,
        sampling="end",
        fixed_metric="log_absorption",
    ),
    "efficiency": dict(
        label="累计迁移效率 · η_D", group="量化诊断", diagnostic=True, sampling="end", fixed_metric="efficiency"
    ),
    "drift": dict(label="累计吸收差 · Δ_D", group="量化诊断", diagnostic=True, sampling="all", fixed_metric="gap"),
    "windows": dict(
        label="早期—末窗迁移效率", group="量化诊断", diagnostic=True, sampling="end", fixed_metric="efficiency"
    ),
}
STATUSES = {
    "insufficient_budget": "检查点不足，窗口尚不可用",
    "train_rebound": "训练吸收增量非正（停滞或反弹）",
    "small_train_progress": "训练吸收增量未超过分母阈值",
    "eligible": "比值可计算",
}
SYMBOLS = {
    "train_initial": "K_T(0)",
    "held_initial": "K_D(0)",
    "train_rkl": "K_T",
    "held_rkl": "K_D",
    "train_absorption": "A_T",
    "held_absorption": "A_D",
    "gap": "Δ_D",
    "efficiency": "η_D",
    "early_train": "ΔA_T[0,64]",
    "early_held": "ΔA_D[0,64]",
    "late_train": "ΔA_T[e−64,e]",
    "late_held": "ΔA_D[e−64,e]",
    "early_efficiency": "η_D[0,64]",
    "late_efficiency": "η_D[e−64,e]",
}
HUES = ["4F8FA5", "EE995B", "C95B5B", "8A74B5"]


def teacher_rgb(index):
    hue = HUES[index % len(HUES)]
    shade = 1 - 0.17 * (index // len(HUES))
    return tuple(int(hue[k : k + 2], 16) / 255 * shade for k in (0, 2, 4))


def presentation(runs):
    students = sorted({r["student"] for r in runs})
    for student in students:
        chosen = sorted([r for r in runs if r["student"] == student], key=lambda r: r["label"])
        for index, run in enumerate(chosen):
            run["display_order"] = index
            run["color"] = "rgb(" + ",".join(f"{v * 255:.5f}" for v in teacher_rgb(index)) + ")"
    return runs
