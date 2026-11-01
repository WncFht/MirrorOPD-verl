# Rollout 模型来源与组合

日常训练从 `examples/mirroropd/run_math_opd_ablation_fsdp2.sh` 启动。模型来源与组合方式独立配置；生成模型不改变 learner 和监督 teacher 的身份。

| 阅读目的                                         | 唯一维护入口                                                                  |
| ------------------------------------------------ | ----------------------------------------------------------------------------- |
| current、initial、teacher、checkpoint 与组合语义 | [模型来源与组合](semantics.md)                                                |
| 内部调用关系、阶段与资源所有权                   | [内部职责](semantics.md#6-内部职责与迁移)                                     |
| 启动、资源参数、保存恢复与旧字段迁移             | [使用说明](usage.md)                                                          |
| 已实现能力、验证范围及剩余限制                   | [支持与验收](support.md)                                                      |
| 热点、并发、TP 与显存实测                        | [性能记录](../../performance/inference/2026-11-01-Token融合热点与并发测量.md) |
| 来源策略的学习效果                               | [Rollout 策略实验](../../../experiments/rollout-policy/README.md)             |
| 开发与冻结执行目录                               | [工作区维护](../../maintenance/workspaces.md)                                 |

## 历史依据

[历史设计与验收索引](archive/README.md)保留旧方案、失败候选、分版本验收和工作区清理回执。历史文档的待办不作为当前开发清单；后续能力变更更新支持表及其证据，配置和数学定义分别更新使用说明与统一语义。
