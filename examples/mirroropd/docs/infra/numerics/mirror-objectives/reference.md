# MDPO/BPO 当前实现参考

本页维护配置、数据需求、数值边界及验证入口。目标定义和完整推导见[镜像理论](../../../theory/mirror-descent/objectives.md)，当前调用链见[代码走读](../../architecture/代码走读.md)，初版设计、分阶段实验和未验证条件保留在[历史验收](../../../archive/engineering/2026-11-01-Mirror实现方案与历史验收.md)。

<a id="settings"></a>

## 方法、配置与数据

统一入口见[方法选择](../../training/launch.md)。`loss_mode=mirror`配合`MirrorConfig`选择 objective、estimator、κ和η；范围为κ∈[0,1]、η∈(0,1]，BPO 仅支持 full_vocab。旧运行配置名`full_reverse_kl`已退役，新 Full-RKL 使用 MDPO/full_vocab/η=1；数学函数仍保留。旧配置和调用者按[迁移指南](../../architecture/runtime-migration.md)处理，历史 run 仍使用冻结代码。

| 路径                    | 实际目标与数据需求                                                                                   |
| ----------------------- | ---------------------------------------------------------------------------------------------------- |
| MDPO/full_vocab/η=1     | teacher-only Full-RKL endpoint，无 anchor                                                            |
| MDPO/full_vocab/η<1     | 冻结 teacher 和 anchor hidden/head，当前student每次更新重新forward                                   |
| BPO/full_vocab，包括η=1 | 冻结 q 加权的中心化残差回归，始终需要 anchor hidden/head                                             |
| MDPO/sampled            | 行为 IS 的 sampled surrogate，需要 teacher 及按目标建立的 scalar anchor；不启用 PPO/NoIS/ratio guard |

`resolve_distillation_spec`只在配置构造阶段解析目标与 payload_kind。派生`loss_settings`为只读运行元数据，不接受外部配置输入；mirror 参数不在训练期间切换。完整配置校验由[distillation.py](../../../../../../verl/workers/config/distillation.py)和[目标元数据](../../../../../../verl/workers/config/distillation_spec.py)执行。

`ppo_epochs`控制同一批的重复更新次数，不是独立的镜像步长。q 在一次 actor 调用内冻结，下一次调用重新捕获；η不是 optimizer LR。教师阶段、anchor 准备及异常清理由[生命周期](../../architecture/teacher.md)维护；位置与全局分母见[数据合同](../../architecture/data-and-loss.md)。

## 后端与动作空间

| 目标                   | 可用计算路径                                                                                                 |
| ---------------------- | ------------------------------------------------------------------------------------------------------------ |
| teacher-only Full-RKL  | dense、recompute、cached、streaming、precompute、precompute_fused 及显式 liger 对照，能力由当前配置/模型校验 |
| 一般 full MDPO/BPO     | dense 参考、precompute、precompute_fused                                                                     |
| EOS 合并训练           | 同词表 teacher-only RKL 或 full MDPO，precompute/precompute_fused；BPO 和 sampled 拒绝                       |
| 显式共同支持与 ID 映射 | 当前配置仅支持 teacher-only reverse KL；模型和分词语义由配对合同校验                                         |

`precompute_fused`要求 CUDA FP16/BF16/FP32输入；CPU或FP64参考明确选择precompute/dense，不静默fallback。数值入口和错误条件见[full_vocab.py](../../../../../../verl/trainer/distillation/full_vocab.py)。Dense 数值 oracle 可覆盖一些直接调用，不能据此宣称训练配置支持全部组合。

Full-vocab 接入、支持集和模型 adapter 见[数据流](../../architecture/full-vocab.md)，稳定 target、chunk VJP 与 Triton 见[算子走读](../../architecture/full-kl-kernel-walkthrough.md)。Precompute 在每次当前 student 前向内计算 VJP，不跨 optimizer 更新缓存学生梯度；frozen hidden/head复用不等于缓存完整词表target。

<a id="metrics"></a>

## 训练目标与观测

Engine 通过唯一的`model_output["full_vocab"]`返回[FullVocabOutput](../../../../../../verl/trainer/distillation/output.py)。它使用可被 FSDP 遍历的 NamedTuple，分别承载可微 objective_sum、规定的 divisor 和 detached observations。LossContext 在调用级提供全局 token/sequence/DP 分母，不将运行时总体注入静态 config。

一般 MDPO/BPO 的优化目标应用 1/η；原 teacher KL、target KL、anchor KL 和 target log Z 各自保持定义，不因训练缩放而一起除η。MDPO raw 目标与 target KL 之间有冻结 normalizer 项，不能把两者的前向值混为一谈。η=1 endpoint 沿用自己的指标，不为缺失 anchor 补零。

EOS 合并、条件支持、FKL 附加 RKL 诊断与位置分桶分别命名。完整含义及当前覆盖见[指标系统](../../architecture/metrics.md)与[公共指标字典](../../../settings/指标口径.md)。逐 update 观测来自产生该次梯度的 forward，不等于更新后的额外评分。

## 更新、恢复与扩展

Optimizer 执行事实独立于 metrics；显式 None 表示未知，缺字段报错。Anchor 为调用级临时状态，不进入 checkpoint。计数、scheduler、保存与恢复保证只由[进度合同](../../architecture/progress-checkpoint.md)维护。

新增目标先定义数学对象、payload 与缩放，再复用现有 teacher/anchor、head-loss 和 Observation 出口。新增指标明确概率空间、总体和时间，不按某个旧输出键是否存在分派算法。新增模型先通过[兼容性](../../model-compatibility/README.md)核对输入、head 和动作支持。

<a id="validation"></a>

## 验证证据

[Full-vocab 验证矩阵](../full-vocab/validation.md)索引 CPU oracle、CUDA 数值、FSDP、真实短训练与 EOS 回归；[运行合同重构记录](../../../archive/engineering/2026-11-01-运行合同审计与重构方案.md)保留对应代码基点和检查范围。各结果按自己的源码和配置解释，不将不同日期测试项数相加为当前覆盖率。

[精度与 optimizer 实验](../optimization/README.md)检查内层反弹和实际策略步幅；[性能](../../performance/full-vocab/README.md)检查时间和容量。它们不能由数学 oracle 代替，也不作为 TOP-D 任务分数必然领先的承诺。

<a id="remaining-validation"></a>

## 支持与验证边界

未覆盖的模型、后端、长序列、自然生成负载和恢复边界仍需相应证据。两卡 engine 后续更新一致不能替代完整 V1/Ray 冷启动恢复，不承诺 inner-update 中途恢复；旧验收的具体缺口见历史记录。配置解析通过、短 smoke、长程科研结果分别报告，不由文档迁移扩大支持范围。
