# 实验研究

总问题是：OPD 在不同师生条件下能取得怎样的任务效果，哪些因素限制学习，怎样提高学习效率。目前以 accuracy 等既有指标评价；“完全蒸馏教师”的严格定义保留为开放问题。

| 主线                   | 主比较与范围                                                               | 入口                                           |
| ---------------------- | -------------------------------------------------------------------------- | ---------------------------------------------- |
| 训练信号与目标         | 给定数据，比较 sampled/full-vocab、current/fixed、IS/NoIS/PPO 及目标关系   | [learning-signal](learning-signal/README.md)   |
| Reuse 与训练预算       | 一批 rollout 复用多久，固定更新/生成预算及 response cap 怎样影响效果与成本 | [reuse-budget](reuse-budget/README.md)         |
| Rollout 来源与混合效果 | 围绕 2610.11291 的复现和直接扩展，比较生成策略下的最终 acc                 | [rollout-policy](rollout-policy/README.md)     |
| 师生迁移与优化机制     | 同源/异源配对、跨 batch 迁移、局部拟合、optimizer 与精度                   | [teacher-transfer](teacher-transfer/README.md) |

固定 state、teacher/student 来源控制和跨 batch 探针用于解释迁移时，归第四条；固定配对改变生成策略、主要看最终 acc 时，归第三条。两者可以共享工具，结果不能互相代替。

[运行记录](运行记录.md)与[验证汇总](验证结果汇总.md)来自同一次本地读取；[公共设置](../settings/README.md)只提供资料与参数含义，各报告记录实际 setting。小实验按科学或工程目的归属，不设杂项小实验目录。

新增/修改报告和复算方法见[维护说明](维护说明.md)。

## 最新日志支持的分析

日志仓库 `31775eb` 对应的生成表包含 35 个已识别运行，其中 27 个正常完成、7 个证据不完整、1 个非零退出。[E50–E53 对照](teacher-transfer/optimizer-precision/production/2026-10-29-E50至E53完整结果与混合精度对照.md)新增 E51 完整终点、E52 完整训练及 E53 前期结果，分别讨论目标净下降、任务准确率和运行成本。E53 最后训练为 outer103、验证为 outer96，不能作 256 轮终点比较。

训练信号线已有 sampled/full 与 IS/NoIS/PPO 对照；预算线已有 reuse1/4/16、4096 次等更新及长 response cap 对照；迁移线已有 Polaris RKL/FKL、Qwen 配对和固定 state/跨 batch 研究。Rollout 来源与混合线目前仍以方案和工程验收为主，尚未建立长程最终 accuracy 对照。各线沿用自己的实际协议，不将总表作为跨模型统一排名。
