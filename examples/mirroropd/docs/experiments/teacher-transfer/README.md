# 师生迁移与优化机制

## 背景与 motivation

教师监督有时能被学生持续吸收，有时只降低当前批loss而难以积累。这里比较同源/异源关系，并用受控实验定位优化与迁移瓶颈；当前主要使用acc等既有指标，“完全蒸馏”的严格定义暂作开放问题。

## Setting 与实验设计

| 分支 | 主变量与控制 | 入口 |
| --- | --- | --- |
| 模型与教师配对 | 谱系、规模、后训练和实际功能差异 | [公共模型资料](../../settings/模型与师生配对.md)、[配对结果](teacher-comparison/README.md) |
| 异源教师方向对照 | E48/E49：Polaris下RKL/FKL | [Polaris正式训练](cross-origin/polaris/README.md) |
| 局部到跨state迁移 | 固定前缀、旧批/未见批、来源、覆盖、fresh交叉评分 | [迁移机制](transfer-mechanism/README.md) |
| 实际优化能力 | optimizer历史、步幅、精度、近目标MDPO | [优化器与精度](optimizer-precision/README.md) |

小实验、正式训练和CPU诊断分别保留各自预算、采样及归一化。师生来源是设置维度，不以训练后是否容易学习反推“异源程度”。

## 实验结果

E49改善了部分完成行为，但新批分布拟合仍平台。E51在其253批快照中有252批取得内层净下降，系统性反弹基本消失；任务优势仍需独立判断。Polaris暖状态受控FP32实验当前批KL下降78.81%，held-out仅0.75%，说明局部求解改善未自动解除迁移瓶颈。

## 分析

按“当前批拟合→未见固定state→fresh访问分布→任务表现”区分证据。优化器与精度实验既解释OPD，也解释MDPO；目标正确性由理论/infra保证。Qwen的EOS/模板修复和E54异常诊断归[模型兼容性](../../infra/model-compatibility/README.md)，不作为本线核心问题。
