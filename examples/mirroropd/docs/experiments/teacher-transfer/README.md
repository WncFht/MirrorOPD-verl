# 教师吸收：失败复现与原因诊断

[师生配对 RKL 总览：R1、Qwen3、Llama](fixed-state-results.md)汇集已有配对的训练与留出结果。

本研究先记录哪些 pair 和 setting 下没有观察到充分吸收，再用跨 domain、跨 batch 的证据解释差距。结果按实际 pair、训练与读出条件解释。模型阶段和谱系术语见[公共资料](../../settings/模型与师生配对.md)。

## 失败复现与吸收边界

| 阅读目的                               | 正文与条件                                                                                                                    |
| -------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| 查正式训练中的配对、实际预算与学习平台 | [失败复现](failure-reproduction.md)，包含实际矩阵、E39/E41/E43；[Polaris E48/E49](cases/polaris.md)保留完整平台与 KL 方向证据 |
| 查固定状态下不同谱系、尺寸和输入模式   | [固定状态结果](fixed-state-results.md)，共用[0–320 协议](protocols/fixed-state.md)，不同支持和预算单列                        |
| 比较 SFT、SFT+RL 与直接 RL 教师        | [后训练阶段](post-training-stages.md)，Qwen/Tulu 旧证据与 MiMo/Olmo3/Light-R1 独立条件组                                      |

训练风险描述局部拟合，独立状态读出描述迁移，任务成绩描述生成表现。各表按学生、银行、支持集和预算分组。

## 原因诊断

| 读者问题                                   | 主要证据                                                                                                                   |
| ------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------- |
| 训练域改善能否推广到其他 domain            | [跨域迁移](cross-domain-transfer.md)，已完成的 JustRL 数学分支及其 Math/Code/Science/IF 读出；其他域没有完整结果时保持缺测 |
| 当前 batch 学习能否传播，旧 batch 是否遗忘 | [跨 batch 迁移](cross-batch-transfer.md)，当前/旧/未来批、永久留出及固定/fresh 状态分别解释                                |
| 可学习内容与知识遗忘的解释是否成立         | [能力筛选干预](mechanisms/capability-filter.md)，保留随机映射、温度、IF 及截断续写反例                                     |
| 共享更新方向与留出传播是否相关             | [谱诊断](mechanisms/spectral.md)，loss-gradient Gram 不等于完整 NTK                                                        |
| 平台是否仅来自数值或 optimizer             | 引用[工程实验](../../infra/numerics/optimization/README.md)，不在本研究线重复维护精度与求解报告                            |

当前批能拟合而未见批改善弱，不自动意味着旧批遗忘；换批 loss 锯齿也不能代替旧批复评。跨域改善弱首先描述迁移边界，表示与容量解释仍需对应证据。来源策略的最终效果由[Rollout 研究](../rollout-policy/README.md)维护；这里使用来源控制来解释迁移。

## 协议、复算与补充材料

[按批刷新与四域协议](protocols/online-four-domain.md)与既有 0–320 固定状态协议独立。已完成 JustRL 启用 source/history；后续关闭的分支保留自己的实际配置。

[图谱与复算](reproduction.md)定位生成结果、逐题数据和真实运行；[配对设计依据](appendix/pair-design.md)保留候选因素与尚未排除的解释；[Qwen 详细证据](appendix/qwen-lineage-evidence.md)保留 teacher-copy 等诊断。按既有约定，[RLVR-GSM8K](appendix/qwen3-rlvr-gsm8k.md)继续只在附录，不加入主比较。旧提案与撤销方案通过[历史索引](../../archive/README.md)定位，不构成自动开训清单。
