# Reuse 与训练预算

## 背景与 motivation

多次复用减少新生成开销，但把更新预算更多花在固定前缀上。response cap还影响完整推理覆盖、token工作量和墙钟。这里以预算为主变量，rollout来源尽量固定。

## Setting 与实验设计

| 对照 | 固定预算 | 改变因素 |
| --- | --- | --- |
| E13/E16/E17 | 相同outer200 | reuse1/4/16，更新数随之变化 |
| E18/E19/E20 | 4096次更新 | reuse64/32/16，新回答数按1:2:4变化 |
| E35/E17 | 共同outer200/update3200 | cap15360/7168；硬件/代码存在历史差异 |

具体报告分别记录实际配置，固定更新不等于固定token或算力。

## 实验结果

增加reuse没有带来所有基准一致的末点提升。旧共同2560-update统计中，高reuse E18比E20少生成74.77%的response token，任务排序混合。E35共同outer200的AIME比E17高3.96个百分点，新token增加22.71%、记录训练时间增加45.18%。旧统计不外推为其他预算结果。

## 分析与证据

- [等更新预算冻结分析](equal-update-budget/2026-10-25-等更新预算冻结分析.md)。
- [同批拟合与新批起点](reuse-comparison/2026-10-24-E16E17位置指标与后期收敛.md)。
- [长窗口的结果与成本](response-cap/2026-10-28-E35长窗口与生成预算.md)。

复用次数不是实际策略漂移量。改变生成来源的最终acc对照归[rollout线](../rollout-policy/README.md)；固定state的迁移机制归[师生迁移](../teacher-transfer/README.md)。
