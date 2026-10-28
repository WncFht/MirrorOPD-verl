# 架构重构

后续优化：[Rollout 双模型融合：vLLM 重构、优化与验收方案](../../rollout/fusion/2026-10-28-vLLM双模型融合重构与优化方案.md)。对照当前工作区、公开实现和 vLLM 固定版本源码，说明后端解耦、异构模型/KV、student 专项 refit、teacher 休眠恢复、采样语义及分阶段验收；附源码指纹清单。

执行入口：[本地 Rollout 双融合实施交接说明](../../rollout/fusion/2026-10-28-Rollout双融合实施交接说明.md)。交给实现者从这里开始；详细要求见配套实施计划。SKD、AdaSwitch 不进入实施范围。

待实施方案：[本地模型 Rollout：轨迹混合与逐 Token Logit 融合](../../rollout/fusion/2026-10-28-可组合Rollout策略架构与实现方案.md)。所有生成模型从本地 checkpoint 加载；同 batch 的 trajectory mixture 与逐 decode 的 logit fusion 共用 current/initial student 选择，尚未实现。

实施配套：[双融合实现细节、优化与验收计划](../../rollout/fusion/2026-10-28-Rollout双融合实现细节与验收计划.md)。包含 logit-fusion 博客/代码核对、全局采样计划、TQ 汇合、权重接收、双 KV、停止概率、容量估算及分阶段 gate；保留 TRB 的 KL 预算求解与 EOS 对齐设计。

2026-10-25：五项重构已实现，完成 CPU、真实 TQ、分布式 GPU 及 1.5B 短端到端验收，独立 review 最终结论为 approve。

| 文档                                                                            | 内容                                                                              |
| ------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| [主分支四卡验收](../../../archive/engineering/2026-10-25-主分支四卡验收.md)     | 在 GPU 0–3 上重跑四组集成测试及 response cap 512 的真实 sampled/full-vocab 短训练 |
| [实施与验收](../../../archive/engineering/2026-10-25-架构重构实施与验收.md)     | 当前模块边界、已裁决语义、分批提交、测试与实验结果、未覆盖范围                    |
| [原始背景与计划](../../../archive/engineering/2026-10-24-架构背景与重构计划.md) | 固定评审基点、现状／目标图、原计划与 review 修订；保留当时的未实施状态            |
| [验收数据与复现](../acceptance/data/2026-10-25-acceptance/README.md)            | 测试范围、GPU 对照、真实短运行配置和原始指标                                      |

Full-KL 数值目标仍见[计算与位置指标分层](../metrics.md#相对位置统计)。本次工程重构不改变其数学定义或正式实验预算。
