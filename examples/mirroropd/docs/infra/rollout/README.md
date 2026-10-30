# Rollout 来源与执行

[模型来源与组合](fusion/README.md)维护 current、固定 checkpoint、single、按轨迹组成 batch，以及逐 token 融合的统一语义。先阅读其架构与当前能力边界，再按后续专项计划讨论执行优化。

来源策略的效果与最终准确率归[rollout 策略实验](../../experiments/rollout-policy/README.md)，几何目标推导归[理论](../../theory/rollout-policy/2026-10-28-Rollout策略的目标设计与求解.md)。开发与冻结执行目录的归属见[工作区说明](fusion/workspaces.md)。
