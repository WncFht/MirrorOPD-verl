# Full-vocab 蒸馏工程

本目录维护 full baseline 与 full-vocab reverse KL 的数据通路、算子、资源预算和实现验收。数学目标与计算图由[理论目录](../theory/README.md)维护，支持集选择、源码审计和实验比较由 [OPD 审计目录](../opd-objective/README.md)维护，符号遵循[统一约定](../符号约定.md)。

| 内容 | 入口 |
| --- | --- |
| 当前 Full-KL 完整调用链、具体源码行号、默认与可选优化 | [全流程源码走读](2026-10-23-Full-KL全流程源码走读.md) |
| η、κ 目标的理论与拟议 Full-vocab 接入 | [形状与插值专题](../mirror-descent/README.md) |
| E13/E14 Full-KL、E15 sampled、E16/E17 Full-KL 批内复用 | [数学对照预设](../训练配方.md#full-kl-与-sampled-对照入口) |
| 2026-10-22 初次接入的数值、多卡与 smoke 验收结果 | [实现与验收](2026-10-22-Full-KL实现与验收.md) |
| Full baseline 后续设计与原始实施阶段 | [实现计划](2026-10-22-Full-baseline与Full-KL实现计划.md) |
| CPU 正确性与 H800 算子实测、复现命令 | [算子小实验](2026-10-22-Full-KL算子小实验.md) |
| 共用流式公式、math 配方容量与历史步时模型 | [词表蒸馏的计算与显存预算](2026-10-22-词表蒸馏的计算与显存预算.md) |
| 用户提供的生产实现参考原文 | [Full-vocab OPD](reference/fullvocab-opd.md) |
| KL 数值、baseline 与直接梯度的区别 | [数值与梯度实现](../theory/2026-10-22-Full-vocab计算图与四种更新.md#graphs) |

## 资料维护

2026-10-22 将原 `opd-objective/reference/fullvocab-opd.md` 原样迁入 `reference/`，将共用工程预算迁入本目录并更新仓库内引用；不保留正文副本。参考原文缺少作者、代码链接和部分图表，性能数字属于原文报告，不能当成本机复现结果。

已有估计器与数学审计脚本保留在 `opd-objective/scripts/`，它们服务多种 OPD 方法。本目录新增实验仅维护实现决策所需的算子证据。

当前流程、数据合同和算子机制只在源码走读中维护；验收文档保留各自日期的原始结果，profile 文档维护测量、图表与性能结论，历史设计保留未实现的 baseline 方案。外部参考原文不改写。

## 历史队列

2026-10-22 的冻结任务、旧 sampled 参考与对照计划见[Math 历史消融队列](../archive/2026-10-23-Math历史消融队列与对照设计.md)。当前 E13–E17 配置从上表的数学预设进入；该旧队列不作为新的默认启动说明。
