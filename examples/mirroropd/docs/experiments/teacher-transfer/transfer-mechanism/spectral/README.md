# 谱滤波与师生来源对照

## 背景与 motivation

检验共享梯度方向与 teacher 残差可学习性的关系，避免把 loss-gradient Gram 等同于输出 NTK。综合研究问题见[跨状态迁移与能力获取](../../README.md)。

更新：2026-10-28。当前只完成 CPU 谱诊断和模型元数据核对。用户要求等通知有 GPU 后再测新的 NTK/Jacobian；大型训练先讨论 setting，再接入数学配方入口。

| 内容                                                                            | 状态与入口                                               |
| ------------------------------------------------------------------------------- | -------------------------------------------------------- |
| [谱滤波解释、实测结果和下一阶段设置](2026-10-28-谱滤波诊断与同架构教师对照.md)  | 4 个已有全参数梯度 Gram 的新谱分析；不是新测完整输出 NTK |
| [谱图](figures/gradient-spectrum.png)／[PDF](figures/gradient-spectrum.pdf)     | 梯度方向集中度与独立 Held64 梯度投影                     |
| [子集稳定性](figures/gradient-subsets.png)／[PDF](figures/gradient-subsets.pdf) | 每个大小 256 个训练梯度子集，5–95% 范围不是跨 seed 区间  |
| [分析程序](../../../../../research/transfer/analyze_gradient_spectrum.py)       | CPU，只读取冻结输入，输出独立目录                        |
| [模型核对程序](../../../../../research/transfer/audit_teacher_candidates.py)    | 只下载固定版本的小型配置和 tokenizer 元数据，不下载权重  |

原始输入：`wandb-logs/MirrorOPD-verl/analysis/batch-size-stage-20261027/noise-{justrl,polaris}-{initial,warm128}.json`。

新分析与模型审计：`wandb-logs/MirrorOPD-verl/analysis/spectral-filter-20261028/`。两条路径均相对源码仓库的父目录。`spectral-results.json` 包含原始、单位梯度归一化、中心化和子集结果；`model-audit/model-comparison.json` 包含固定远程 revision、结构字段差异和 token-ID 差异。

返回[实验目录](../../../README.md)；相关工作见[RL/SFT 调研](../../../../literature/transfer-generalization/README.md)。
