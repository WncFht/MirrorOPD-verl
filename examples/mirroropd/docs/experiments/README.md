# 实验研究

研究 OPD 在不同条件下能学到什么、收益与成本如何变化，以及哪些因素限制教师吸收。五个 RQ 组织阅读路线，各专题维护实际对照、结果和复算材料。

| RQ                     | 核心比较                                                      | 正文入口                                                                                             |
| ---------------------- | ------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| RQ1 学习信号与估计器   | current/rollout、IS/NoIS/PPO、sampled/full 与归一化           | [学习信号](learning-signal/README.md)                                                                |
| RQ2 Reuse 与预算       | 新增轨迹、批内更新、回答长度与计算成本                        | [训练预算](reuse-budget/README.md)                                                                   |
| RQ3 Rollout 来源与组合 | initial/teacher/current、logit/prob/batch fusion              | [Rollout 来源](rollout-policy/README.md)                                                             |
| RQ4 失败复现与吸收机制 | pair/setting 边界、SFT/SFT+RL 阶段、跨 domain 与跨 batch 迁移 | [教师吸收](teacher-transfer/README.md)；[师生配对 RKL 总览](teacher-transfer/fixed-state-results.md) |
| RQ5 TOP-D 复现与 κ/η   | 算法、设置、结果的复现程度与 weighted-target 参数消融         | [TOP-D 对照](learning-signal/weighted-targets.md)                                                    |

## 数据与方法

[运行记录](运行记录.md)和[验证汇总](验证结果汇总.md)展示已识别正式 run 的本地快照；固定状态、机制探针和工程验收从各专题进入。报告注明实际 checkpoint、输入、支持集、分母与预算，分别解释训练拟合、独立状态读出和任务成绩。“完全吸收”的评价标准仍待研究确定。

[公共设置](../settings/README.md)维护参数与指标定义，[理论](../theory/README.md)维护推导。[精度与 optimizer](../infra/numerics/optimization/README.md)、[模型兼容](../infra/model-compatibility/README.md)和[性能测量](../infra/performance/README.md)提供工程依据。复现差异、单 seed 和预算限制在对应比较中说明；新增结果与复算见[维护说明](维护说明.md)。
