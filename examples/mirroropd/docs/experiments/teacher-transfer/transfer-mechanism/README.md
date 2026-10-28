# 跨 batch 与跨 state 迁移机制

## 背景与 motivation

同源 JustRL 与异源 Polaris 的局部拟合收益向其他 batch 传播的程度不同。需要固定输入、复测旧批和未见批，区分保持、迁移与访问分布变化；换批 loss 回升本身不是遗忘率。

## Setting 与证据

- [S/I/B/N 统一设置及结果](2026-10-27-跨源蒸馏泛化研究总览.md)。
- [跨批次及 L 单遍结果](cross-batch/2026-10-28-跨批次迁移与单遍训练结果.md)。
- [待执行双来源 R 方案](rollout-sources/2026-10-28-双来源Rollout四分支实验方案.md)，不与旧结果合计。
- [能力筛选假说](capability-filter/2026-10-28-OPD能力筛选假说的受控实验.md)。
- [谱与同架构诊断](spectral/2026-10-28-谱滤波诊断与同架构教师对照.md)，已有 Gram 不是输出 NTK。

## 结果与分析

这些实验回答已见与未见 state 的收益为何不同，teacher/student rollout 是机制控制变量。它们不替代以最终 acc 为主要目的的来源/混合效果实验；每个系列的固定风险、fresh 风险与任务采样范围分别解释。
