# 研究与方法

本目录维护相关工作与数值机制研究。正式实验的设计、运行进度和结果分析见[实验目录](../experiments/README.md)。固定前缀下的目标与证明见[理论目录](../theory/README.md)。

| 文档 | 内容 |
| --- | --- |
| [陈旧数据训练的指标与实验对照](2026-10-21-陈旧数据训练的指标与实验对照.md) | p99、KL、ESS、clipping；相关 RL/OPD 工作的设置与论文图表 |
| [KL 精度误差模型与诊断接口](2026-10-24-KL精度误差模型与诊断接口走读.md) | Logit 扰动、误差相关性、初始权重 probe 与诊断接口 |
| [大规模预训练后期的数值精度证据](2026-10-25-大规模预训练后期的数值精度证据.md) | 输出 CE、attention、参数更新和 FP8/FP4 长训练的证据及适用边界 |

跨实验比较与训练动态在[Full-KL 系列](../experiments/full-kl-comparison/README.md)，评测统计语义在[实验指标](../experiments/metrics/README.md)，失败复现论文依据与执行矩阵在[模型／长度系列](../experiments/failure-reproduction/README.md)。
