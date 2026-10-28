# Rollout 来源与混合效果

## 背景与 motivation

本线从2610.11291的复现和直接扩展出发，检验在同一师生配对和learner配方下，不同训练数据生成策略是否改变最终accuracy。当前不承担同源/异源跨batch迁移的微观解释。

## Setting 与实验设计

比较当前student、初始冻结student、teacher，以及轨迹混合和逐token混合。每项实验固定模型配对、learner loss、optimizer、训练与验证协议，明确更新、token和生成成本的匹配方式；实际setting在具体方案中登记。数学上定义的融合中心与实际双模型后端分别见[理论](../../theory/rollout-policy/2026-10-28-Rollout策略的目标设计与求解.md)和[工程实现](../../infra/rollout/fusion/README.md)。

| 内容 | 当前边界 |
| --- | --- |
| 论文设置与比较依据 | [失败复现案例调研](../../literature/opd/2026-10-24-OPD失败复现设置与Full-vocab对照.md)的Semi-OPD部分 |
| 来源、版本和混合实现 | 工程生命周期和短测，不能当成长程学习结果 |
| 新来源/混合的最终acc比较 | 尚未由上述工程验收建立；完成后按独立实际运行报告 |

## 实验结果与分析

现阶段不填入模拟或工程smoke的acc作为正式结果。后续首先比较训练曲线、末次accuracy和成本；不预设混合策略更优，也不以分布中心半径下降代替学习效果。

Polaris的固定来源、跨batch及交叉评分用于迁移假说，完整保留在[迁移机制](../teacher-transfer/transfer-mechanism/README.md)，不计作本线已完成效果实验。
