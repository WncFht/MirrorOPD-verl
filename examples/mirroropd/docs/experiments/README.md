# 实验研究

[师生配对 RKL 总览：R1、Qwen3、Llama](teacher-transfer/fixed-state-results.md)汇集已有配对的训练与留出结果。

研究 OPD 在不同条件下能学到什么、收益与成本如何变化，以及哪些因素限制教师吸收。五个 RQ 组织阅读路线，各专题连接实际对照、结果和复算材料。

| RQ                     | 研究内容与入口                                                                   | 当前证据边界                                                         |
| ---------------------- | -------------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| RQ1 学习信号与估计器   | [current/rollout、IS/NoIS/PPO、sampled/full与归一化](learning-signal/README.md)  | 历史配方并非全部同版本受控，单 seed 末点不能给出统一算法排名         |
| RQ2 Reuse 与预算       | [批内更新、新轨迹、回答长度及成本](reuse-budget/README.md)                       | 固定生成量、更新量、token 和墙钟是不同问题；旧共同窗口不回填         |
| RQ3 Rollout 来源与组合 | [initial/teacher/current、logit/prob/batch fusion](rollout-policy/README.md)     | 纯 student 任务结果与共同状态拟合分开；不同师生及新旧评估协议分开    |
| RQ4 失败复现与吸收机制 | [pair/setting边界、阶段对照、跨domain与跨batch原因](teacher-transfer/README.md)  | 当前批拟合、旧批保持、未见状态与真实任务分别解释；不预设完全吸收阈值 |
| RQ5 TOP-D 复现与κ/η    | [算法/设置/结果对照与 weighted-target 消融](learning-signal/weighted-targets.md) | target 对应不等于完整复现；精度和 optimizer 实验由 infra 支撑        |

## 数据与方法

[运行记录](运行记录.md)和[验证汇总](验证结果汇总.md)展示已识别正式 run 的本地快照；固定状态、机制探针和工程验收从各专题进入。报告用实际 checkpoint、输入、支持集、分母和预算解释结果。

[公共设置](../settings/README.md)维护参数与指标定义，[理论](../theory/README.md)维护推导。[精度与 optimizer](../infra/numerics/optimization/README.md)、[模型兼容](../infra/model-compatibility/README.md)和[性能测量](../infra/performance/README.md)提供工程依据。新增结果、文档合并与复算见[维护说明](维护说明.md)。
