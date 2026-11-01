# Rollout 模型来源、组合与执行

模型来源与组合方式分别配置。Current、initial、teacher 和 checkpoint 引用不改变 learner 和监督 teacher 的身份；来源策略最终效果由[六来源实验](../../experiments/rollout-policy/README.md)维护。

| 任务                                              | 正文                                                                       |
| ------------------------------------------------- | -------------------------------------------------------------------------- |
| 理解模型引用、single、trajectory 与 token mixture | [统一语义](semantics.md)                                                   |
| 启动、资源、保存恢复与旧字段迁移                  | [使用说明](usage.md)                                                       |
| 核对模型、后端和实际验收范围                      | [支持与验收](support.md)                                                   |
| 配置重复检测及提前停止                            | [使用与风险](repetition-detection.md)，离线 token 节省不等于已测 GPU 加速  |
| 查看热点、TP、并发与显存                          | [性能测量](../performance/inference/2026-11-01-Token融合热点与并发测量.md) |

[几何分布定义](../../theory/rollout-policy/objectives.md)负责数学，[历史设计](fusion/archive/README.md)保留分版本方案与失败候选。当前行为随源码维护，冻结运行按原 manifest 与代码解释。
