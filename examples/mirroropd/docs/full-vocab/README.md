# Full-vocab 蒸馏工程

本目录维护 full baseline 与 full-vocab reverse KL 的数据通路、算子、资源预算和实现验收。数学目标与计算图由[理论目录](../theory/README.md)维护，支持集选择、源码审计和实验比较由 [OPD 审计目录](../opd-objective/README.md)维护，符号遵循[统一约定](../符号约定.md)。

| 内容 | 入口 |
| --- | --- |
| 2026-10-24 Qwen3 tied head、输出支持修正、运行时 EOS 配置与保存的验收结果 | [实现验收与对齐诊断](../failure-reproduction/2026-10-24-OPD实现验收与对齐诊断.md) |
| 2026-10-23 Full-KL 调用链、默认与可选优化；模型和词表边界以上方合并文档为准 | [全流程源码走读](2026-10-23-Full-KL全流程源码走读.md) |
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

流程、数据合同和算子机制在源码走读中维护；2026-10-24 的模型适配与不等长输出支持由上方合并文档补充。验收部分保留各自实验的原始结果，profile 文档维护测量、图表与性能结论，历史设计保留未实现的 baseline 方案。外部参考原文不改写。

## 历史队列

2026-10-22 的冻结任务、旧 sampled 参考与对照计划见[Math 历史消融队列](../archive/2026-10-23-Math历史消融队列与对照设计.md)。当前 E13–E17 配置从上表的数学预设进入；该旧队列不作为新的默认启动说明。

位置诊断：Full-KL 新增 response 相对位置十分桶的 sum/count/mean，复用原始 KL 计算且不改梯度；字段及精度验收见[KL精度与位置分桶](../research/2026-10-24-评测预算语义与KL精度及位置分桶.md)。

诊断接口使用统一 `FullKLResult`，普通／fused precompute共用分块和VJP；精度误差推导、验证与代码走读见[KL精度模型](../research/2026-10-24-KL精度误差模型与诊断接口走读.md)。

当前计算图与公共指标分层：[KL计算与位置指标分层](2026-10-24-KL计算与位置指标分层.md)，含Mermaid、训练／诊断分支及通用token指标接口。
