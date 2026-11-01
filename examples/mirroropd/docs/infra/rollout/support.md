# Rollout 支持与验收

核对日期：2026-11-01。已提交基线为 main `00e8af30`；下表区分接口实现和实际验收。冻结实验继续使用各自 manifest 中的版本，不随文档整理切换。模型引用、阶段所有权和采样合同由[统一语义](semantics.md)维护，启动及资源参数由[使用说明](usage.md)维护。

## 能力与证据

| 能力                                                     | 实现状态           | 已有证据与边界                                                              |
| -------------------------------------------------------- | ------------------ | --------------------------------------------------------------------------- |
| current、initial、teacher、checkpoint 的 single          | 已合入 main        | 默认 single 两轮训练、保存恢复后第三轮更新；initial 身份不随恢复改变        |
| current＋固定模型、固定 A＋固定 B                        | 已合入 main        | current 在右侧、固定 A/B 概率融合与 batch mixture 完成 TP1 短生命周期       |
| logit 几何融合、probability 算术融合、trajectory mixture | 已合入 main        | 数学与来源合同分别检查；不是任意模型族均可使用                              |
| 端点、同源和空配额归一化                                 | 已合入 main        | 右侧权重 1→0.5→0 的真实更新及中途恢复；端点只准备活跃模型                   |
| 具名 current refit、固定权重隔离、阶段释放               | 已合入 main        | 重构验收为 9 段生命周期、16 次更新、100 条五来源验证记录                    |
| 公共 single 采样和恢复 schema 2                          | 已合入 main        | 限支持的单节点 FSDP2、colocated vLLM V1、单轮文本；旧字段直接拒绝           |
| Qwen2 dense、Qwen3 dense 与 dense/MoE 生成               | 有实现与分版本证据 | 原生模型与长历史结果见历史验收；后续架构短测未重验所有 MoE 组合             |
| TP2、双 TP1 replica、graph 和正式生成负载                | 有分范围实测       | P/Q 配对独立生成已有测量，不能代替所有配置的 actor refit、训练恢复验收      |
| Qwen LM/PM 长程训练                                      | 冻结科学版本完成   | `462e3f3a`、TP1、C16/KV48，各 40 outer/640 更新；不外推为新 main 或八卡验收 |
| HF 参考后端                                              | 显式保留           | Qwen3 小负载两轮更新与再次生成；不继续以 HF 全性能矩阵阻塞原生路径          |

重构短生命周期使用 R1-1.5B/JustRL-1.5B，HF 参考使用 Qwen3-1.7B-Base/Qwen3-8B，batch2/cap16/reuse1。测试参数、源码及 patch、失败尝试和回执见[统一接口验收记录](fusion/archive/2026-10-31-Rollout统一接口实施计划.md#架构重构交付)与[原生后端历史验收](fusion/archive/2026-10-28-Rollout双融合实现与验收记录.md)。验证来源的小样本 accuracy 只检查接线，不证明学习效果。

完整生成的实际模型、源码、三次测量与未覆盖路径见[性能记录](../performance/inference/2026-11-01-Token融合热点与并发测量.md)。长程训练结果见[Qwen 六种前缀报告](../../experiments/rollout-policy/qwen-sources.md)。

## 未覆盖范围

- 模型引用统一不表示任意架构、词表或不同分词模型都能融合；当前生成能力按后端预检。
- 重构后的 TP1 短测不能覆盖 MoE、TP2、多 replica、graph 和正式长负载的全部训练生命周期。
- 长负载 probability、chunked prefill、preemption 和替换 attention 后端需要各自证据，不能由 logit 自然生成结果代替。
- E63–E68 使用新原生评估协议；旧 R1/Qwen 的共同状态结果不作为它们已经完成的证明。
- 自动失败重试、TRB、量化、多轮等不属于当前交付范围；请求排空、权重失败和批次完整性检查继续保留。

新验收应在对应能力行补充实际代码版本、配置及证据。仅实现、配置解析通过、GPU 小测通过和长程完成分别记录，不将其合并成单一“已完成”。
