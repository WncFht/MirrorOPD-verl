# 训练信号与目标

## 背景与 motivation

给定同一批前缀，current/fixed信号、IS/NoIS/PPO以及动作积分方式会产生不同更新。需要区分改变目标、改变估计器和改变更新约束，才能解释sampled与full-vocab的效果。

## Setting 与实验设计

主历史配对是R1-Distill-Qwen-1.5B ← JustRL-DeepSeek-1.5B，DAPO、batch64。E2–E5/E8–E11比较信号与修正；E13/E15比较一次复用的full/sample；E13/E14比较LR。不同运行的实际预算、验证和实现差异见各报告。

## 实验结果

长程sampled各基准的最佳末点来自不同分支，没有逐集合统一赢家；E13/E15也没有给出full-vocab全面占优的单run结论。固定前缀梯度身份见[理论](../../theory/README.md)，不能由方法名字代替。

## 分析与证据

- [Sampled信号消融](sampled-controls/2026-10-25-Sampled信号消融.md)。
- [Full/sample基础对照](sampled-vs-full/2026-10-28-基础对照设置与结果索引.md)。
- E13/E16/E17的复用因素另归[预算线](../reuse-budget/README.md)。
- MDPO/BPO的目标定义见[镜像理论](../../theory/mirror-descent/形状与插值目标.md)，实际求解见[精度研究](../teacher-transfer/optimizer-precision/README.md)。

下一项配对应固定同版本实现与预算，再区分目标和估计器收益；本次整理不启动新实验。
