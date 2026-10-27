# 实验目录

这里维护正式训练、模型评测与方法性能对照的设计、实际 setting、运行记录和结果分析。实验编号表示配置；run ID 表示一次实际执行。同一编号的失败、重跑和后续 seed 分别登记。公共性能测量方法、算子证据、实现验收和重构记录由工程专题维护。

当前盘点冻结于 **2026-10-25 09:25（北京时间）**，本地日志提交 `ff8b9f0`，并核对 W&B 项目清单。共登记 **23 个正式训练 run：17 个 Math E 系列、3 个历史 Math、3 个 Coding**；工程测试等 20 个 run 不进入实验总表。历史师生 val-only 参考来自另一个项目的冻结快照，单独注明来源。

冻结盘点之后完成的运行在下表单独登记，不回写上述历史总数。2026-10-27 已补齐 Qwen3-1.7B-Base ← Qwen3-8B 的200步完整训练、验证和样本分析。

## 实验总表

| 系列 | 要回答的问题 | 执行进度 | 分析进度与缺口 |
| --- | --- | --- | --- |
| [Sampled 消融 E1–E12](sampled-ablation/README.md) | rollout/current 信号与 IS、PPO、NoIS 如何影响复用训练？ | E2–E5 各完成 128 次更新；E8–E11 各完成 3200 次更新；E1/E6/E7/E12 未发现运行记录 | 已整理末次验证；尚缺完整学习曲线、共同预算与跨 seed 分析 |
| [Full-KL 对照 E13–E17](full-kl-comparison/README.md) | 完整词表、学习率与复用次数如何影响结果？ | 五个主 run 均完成 200 outer；E15 另一次执行为 failed | 四篇详细分析使用原冻结快照；E17 已补齐，完整 200-step 末点见系列入口，旧报告的截断边界保留 |
| [等更新预算 E18–E20](equal-update-budget/README.md) | 固定 4096 次更新时，复用和新增轨迹如何取舍？ | E18/E19 各完成 4096 更新；E20 到 3024 更新，云端 failed | 已完成 2235 指标分析、共同 2560-update 对照和八组图；E20 后三轮验证与跨 seed 证据仍缺 |
| [失败复现模型／长度矩阵](failure-reproduction/README.md) | 长窗口、强教师、师生失配下，full-vocab 能否改善 sampled 的失效？ | 原250-step矩阵尚待补齐；E39的200-step终止对齐变体已完成，见下行 | 变体结果与原始词表KL方案分别登记，不将单臂结果计作整个矩阵完成 |
| [Qwen3-Base 终止动作对齐](e39-base-8b/README.md) | 能否缓解Base的重复、过长输出并提高验证准确率？ | 200 outer / 3200更新 / 21轮完整验证，exit=0 | 最终MATH/AMC/AIME为68.39%/41.72%/8.54%；主要收益在前10步，完整teacher参考仍缺 |
| [Qwen3教师对照 E41/E54/E43](e41-e43/README.md) | Base与后训练student在100步内的蒸馏表现 | E41已启动，随后 E54（30B MoE teacher）、E43；每组使用8卡 | 均为100 outer、reuse16、7168训练cap；每10步valid，E41/E54无初始验证，E43有初始验证；后两组默认五集 |
| [Full-KL / MDPO / BPO 对照 E44–E47](../训练配方.md#mirror-comparison) | Anchor 约束、内层拟合目标和 weighted teacher 形状的作用 | 已补充预设，正式训练未启动；各 256 outer × 16 reuse | 计划同版本、同预算比较，已有短测仅验证实现 |
| [Polaris 跨源教师 E48](polaris-transfer/README.md) | 同源/异源迁移与KL方向如何影响拟合？ | 已完成256 outer/4096更新；[完整平台与FKL分析](polaris-transfer/2026-10-27-E48后期平台与Forward-KL及知识迁移.md) | 新批KL约0.1874；[批大小与阶段对照](polaris-transfer/2026-10-27-跨源蒸馏泛化研究总览.md#series-n)显示增大batch能降噪，但短程内未解除泛化平台 |
| [Polaris Forward KL E49](../full-vocab/forward-kl.md) | 保持E48预算时，改变KL方向能否改善迁移？ | 生产入口已接入，工程验收见[证据](polaris-transfer/data/forward-kl-implementation-20261027/README.md)；正式256 outer尚未启动 | 同时比较FKL、原生RKL、JS/TV、entropy及任务表现 |

“未发现记录”只覆盖上述项目清单和本地日志提交；不能据此判断其他机器、项目或尚未传回的作业。计划中的多 seed 和师生重测不计入已完成实验。

## 方法性能对照

| 实验 | 比较内容 | 状态 |
| --- | --- | --- |
| [Sampled token 与 full-vocab](sampled-full-performance/README.md) | 统一 MDPO eta=1，sampled/full_vocab × reuse=1/16；固定前缀与自然生成两层计时 | 方案与统一入口命令已准备；固定前缀工具待实现，GPU测量未启动 |

这里的工程性能对照不计入上方冻结的正式训练run总数。共用计时、显存和统计口径见[性能专题](../infra/profile/README.md)，实验特有的配置和执行状态只在对应目录维护。

## 优化机制小实验

[Polaris跨state拟合的优化器与精度对照](polaris-optimizer-precision/README.md)已完成22条全参数短训练、592次更新。它区分optimizer、有效步幅、权重舍入、前向精度与暖起点，评价固定未训练前缀的KL；不计入正式run，也不作为任务准确率或吞吐排名。

## 运行记录与证据

[验证结果汇总](验证结果汇总.md)集中列出统一数学消融脚本已跑实验的 mean accuracy、pass@16、majority@16、置信区间和验证预算；旧估计器独立列示。

[实际运行记录](运行记录.md)集中维护 23 个 run 的关键 setting、执行状态、更新数、最后验证坐标与分数。早期 Math/Coding 直接保留在该表中，不单设实验系列。

原始日志、完整 config/history、逐次验证与分析数据统一存放在同级 `wandb-logs` 仓库。源码仓库保留实验设计、运行表、分析正文、展示图和项目专用分析脚本，不再复制逐 run 数据。日志仓库的 `derived/` 保存可重建结果；`MirrorOPD-verl/analysis/` 保存报告明确引用的冻结输入和分析产物。

每份分析记录 run ID、来源日志提交和截断边界。日常刷新运行表不再生成一套新的配置副本；报告需要冻结输入时，再在日志仓库归档。具体目录和命令见[复现与维护](复现与维护.md)。

## 指标与分析

[指标入口](metrics/README.md)统一解释字段、横轴、验证统计和历史口径；[复现与维护](复现与维护.md)说明日志接收后如何更新记录、冻结新证据、复算分析和图表。训练脚本默认值仍见[训练配方](../训练配方.md)，各系列区分当前预设与已跑配置。

执行状态和分析状态分别维护。完成依据实际更新、末步汇总、验证与退出证据；部分日志保留有效观测。图表和分析注明所用快照；需要冻结的新报告输入写入日志仓库，日常运行表刷新不复制输入。旧报告的数据不自动替换成最新日志；当前进展以本页及运行记录为准。
