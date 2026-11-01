# 实现、运行与工程实验

Infra 维护当前实现、使用方法，以及数值、optimizer、模型接入和性能实验。科学结论由[研究正文](../experiments/README.md)解释，工程验收按实际源码和负载限定范围。

| 使用任务                                     | 入口                                                                        |
| -------------------------------------------- | --------------------------------------------------------------------------- |
| 理解跨模块数据流、生命周期与扩展边界         | [架构](architecture/README.md)、[代码走读](architecture/代码走读.md)        |
| 准备资产、选择方法、启动与恢复               | [训练运行](training/README.md)                                              |
| 使用不同模型来源、融合和重复检测             | [Rollout](rollout/README.md)                                                |
| 使用原生评估、判分、银行与产物               | [评估](evaluation/README.md)                                                |
| 核对模板、token-ID、输出支持、EOS 和模型接入 | [兼容性](model-compatibility/README.md)                                     |
| 核对目标算子、梯度、精度与 optimizer         | [数值实现](numerics/README.md)、[工程对照](numerics/optimization/README.md) |
| 比较算子、生成、训练、评分的时间和容量       | [性能](performance/README.md)                                               |
| 升级旧配置与内部调用者                       | [接口迁移](architecture/runtime-migration.md)                               |
| 维护文档、源码工作区与历史记录               | [维护流程](maintenance/README.md)、[历史依据](maintenance/history.md)       |

CPU oracle、CUDA 误差、分布式短测、长序列前向、正式任务结果分别提供证据，不相互替代。当前支持以配置/源码合同及对应验收为准，历史提案中的待办不自动成为当前开发清单。
