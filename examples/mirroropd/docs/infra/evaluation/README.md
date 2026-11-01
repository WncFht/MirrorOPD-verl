# 原生评估、判分与产物

本目录说明如何使用原生评估和核对工程证据。公共 suite、source/history 与其他 readout 从同一个构造边界进入；准确字段、组件 options、失败行为与生命周期由[评估包合同](../../../../../verl/trainer/evaluation/README.md)维护，不并行维护另一套配置手册。

## 使用与扩展

| 任务                                              | 主要入口                                                                                                                                                                      |
| ------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 配置 suite、bootstrap、任务频率与 model-only 导出 | [评估包 README](../../../../../verl/trainer/evaluation/README.md)，包括可执行的配置增量与命令                                                                                 |
| 选择科学题集、轨迹、n/cap 和读出预算              | [四域协议](../../experiments/teacher-transfer/protocols/online-four-domain.md)及[阶段对照](../../experiments/teacher-transfer/post-training-stages.md)，以各 run 实际配置为准 |
| 原生/共同支持、EOS 合并及 ID 映射                 | [动作空间](../model-compatibility/action-space.md)，分布与模型合同独立于银行来源                                                                                              |
| 数学、代码、IF 等任务判分                         | [判分器与依赖](判分器与依赖.md)；[LCB 隔离](lcb/README.md)                                                                                                                    |
| 理解训练、评分、权重与资源的衔接                  | [总体架构](../architecture/README.md)、[教师生命周期](../architecture/teacher.md)                                                                                             |
| 检查评分性能和资源分组                            | [RKL/评估实测](../performance/full-vocab/training.md)                                                                                                                         |

JustRL 已完成运行启用 source/history 读出；MiMo／阶段链的验收版本关闭这两项读出，并保留真实训练轨迹。各 run 的长度与读出设置见对应实验协议，当前可用字段与默认值由评估包合同维护。银行身份、冻结生成恢复和训练恢复是不同合同；旧银行不通过补字段自动升级。

评估纯 current learner，即使训练来源为 teacher 或 fusion 也不使用混合模型代替任务成绩。已确认结束、状态审计、TQ 清理及必要恢复完成后才发布事件；字段和完整失败语义以包合同为准。

## 验收范围与历史依据

[分阶段验收记录](../../archive/engineering/evaluation/2026-11-01-分阶段代码走读与验收.md)保留最初阅读基点、短测和后续实现证据；其早期缺口不作为当前缺陷清单。[版本快照](../../archive/engineering/evaluation/2026-11-01-原生评估版本快照.md)保留当时 main 与工作区的差别，不用旧分支标签推断当前部署。

真实长程的科学结果见[JustRL 跨域报告](../../experiments/teacher-transfer/cross-domain-transfer.md)，不同模型的实际接入见[模型兼容性](../model-compatibility/README.md)。配置可解析、短程完整链路、长序列前向和正式训练是不同证据，不能互相替代。

任务抽象、配置演变及存储取舍分别保存在[原接入设计](../../archive/engineering/evaluation/2026-10-31-四域接入设计.md)、[配置草案](../../archive/engineering/evaluation/2026-10-31-配置与调度草案.md)、[执行与存储依据](../../archive/engineering/evaluation/2026-11-01-执行与存储设计依据.md)。后续行为改变更新包合同和相应当前文档，不在历史方案末尾追加新的当前接口。
