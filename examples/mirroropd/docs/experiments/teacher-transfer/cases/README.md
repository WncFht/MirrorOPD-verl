# Teacher-transfer position-KL

三个正式文档按 student family 维护，图内标明实验与 teacher。

| 文档                              | 实验            | 内容                                                            |
| --------------------------------- | --------------- | --------------------------------------------------------------- |
| [MiMo](mimo-position-kl.md)       | E71 / E72 / E73 | RL、RL-Zero、SFT teacher；位置图、训练 loss 与 Math-bank RKL    |
| [R1-1.5B](r1-1.5b-position-kl.md) | E48 / E49 / E35 | Polaris 的 RKL / 条件 FKL 对照；JustRL E35 单独图               |
| [Qwen](qwen-position-kl.md)       | E39 / E41 / E43 | 三节独立图：first / last heatmap、位置曲线、首尾五个 outer 放大 |

## 新实验设计

| 文档                                                                               | 实验      | 内容                                                                                             |
| ---------------------------------------------------------------------------------- | --------- | ------------------------------------------------------------------------------------------------ |
| [MiMo E91](../post-training-stages.md#e91)                                         | E91       | SFT student 吸收同源纯 direct-RL teacher；训练、validation 与统一评估均开启 repetition detection |
| [Qwen2.5-1.5B SFT/RFT teacher 对照](2026-11-07-E92-Qwen25-SFT与RFT教师吸收对照.md) | E92       | 从共同 Base 生成数学 SFT/RFT teachers，再用匹配 MiMo 的 OPD 预算比较吸收情况                     |
| [MiMo E91](../post-training-stages.md#e91)；[E93](../post-training-stages.md#e93)  | E91 / E93 | 同一 SFT student 对 direct-RL 与 SFT+RL 两条 MiMo teacher 谱系的吸收对照                         |

每份文档保留设置表、原始 RKL 口径、首尾数值表和可复算产物；字段覆盖见[数据清单附录](../appendix/position-kl-inventory.md)，Polaris 完整任务与方向证据见[长训附录](../appendix/polaris-long-run.md)。

[初始与末 outer 的 first-update 位置曲线](first-update-position-profiles.md)汇总以上九组实验、六组 MDPO 和两组终止动作诊断；每个实验独立成图，两端均取 first update，比较前段、中段与末段的 KL 形状。MDPO 随 outer 的热力图与训练线图见[独立专题](../../learning-signal/mdpo-position-kl.md)。
