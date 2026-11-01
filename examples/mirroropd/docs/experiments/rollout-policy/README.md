# Rollout 来源与混合效果

本研究线比较同一 learner 配方下，current、initial、teacher、逐 token logit/probability fusion 和 trajectory mixture 对学习与最终 accuracy 的影响。来源不同会改变训练状态分布，不能仅按各自 rollout loss 排序；比较使用共同状态读出与纯 student 任务评测。

## 实验报告与协议边界

| 实验                                       | 正式报告与状态                                                                                   | 比较边界                                                                |
| ------------------------------------------ | ------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------- |
| R1-1.5B ← JustRL-1.5B / Polaris-7B，十二组 | [E20 六种前缀对照](2026-11-01-E20设置下的六种前缀对照.md)：各 40 outer/640 更新与最终 avg@4 完成 | `c52e3eb6`，Math/Code/IF 与固定 H；单训练 seed                          |
| Qwen3-1.7B non-thinking ← Qwen3-4B，旧六组 | [Qwen 六种前缀报告](2026-11-01-Qwen3非思考模式六种前缀对照.md)：报告已纳入 LM/PM 完整结果        | `462e3f3a`，其余四组不据此标为完成                                      |
| Qwen 新八卡 E63–E68                        | [公共训练设置](../../settings/训练设置.md#e63e68qwen3-八卡前缀对照与原生评估)                    | 新 source/history/public 原生评估，单独建立基线；编号存在不代表实验完成 |

R1/JustRL 六种来源最终成绩接近；R1/Polaris 的混合与 teacher 来源低于 current，且更长回答和截断同时出现。完整数值、统计单位与限制归各报告，周报只引用摘要。

## 历史与相关研究

[五种前缀原计划](archive/2026-10-30-E20设置下的五种前缀对照.md)已被六种来源及共同评估协议替代；[S0 无评估诊断](archive/2026-10-30-S0无评估优化诊断.md)只说明旧优化运行，不能推断 accuracy。执行沿革和失败记录保留在十二组报告及其 manifest。

模型来源与执行合同见[Rollout 工程入口](../../infra/rollout/fusion/README.md)，融合中心定义见[理论](../../theory/rollout-policy/2026-10-28-Rollout策略的目标设计与求解.md)。固定来源与跨 batch 传播机制归[师生迁移](../teacher-transfer/transfer-mechanism/README.md)，不混入来源策略最终效果的比较。
