# 教师吸收：失败复现与原因诊断

[师生配对 RKL 总览：R1、Qwen3、Llama](fixed-state-results/README.md)汇集已有配对的训练与留出结果。

[Checkpoint 动力学与 30 组配对面板](checkpoint-dynamics/README.md#扩展至-30-组的配对面板)维护当前谱／frame 测量与新增 12 组的来源核查；E105–E123 已通过 CPU 协议与训练设置验收，按四卡两队列执行。

本研究先记录哪些 pair 和 setting 下没有观察到充分吸收，再用跨 domain、跨 batch 的证据解释差距。结果按实际 pair、训练与读出条件解释。模型阶段和谱系术语见[公共资料](../../settings/模型与师生配对.md)。

## 失败复现与吸收边界

| 阅读目的                                                     | 正文与条件                                                                                                                                                                                                                                                                 |
| ------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 查正式训练中的配对、实际预算与学习平台                       | [失败复现](failure-reproduction.md)，包含实际矩阵、[E39](failure-reproduction.md#qwen8)、[E41 及共享 Base 初始参考](failure-reproduction.md#qwen4)、[E43 独立分析](failure-reproduction.md#e43)；[Polaris E48/E49](appendix/polaris-long-run.md)保留完整平台与 KL 方向证据 |
| 查固定状态下不同谱系、尺寸和输入模式                         | [固定状态结果](fixed-state-results/README.md)，共用[0–320 协议](protocols/fixed-state.md)，不同支持和预算单列                                                                                                                                                              |
| 比较 SFT、SFT+RL 与直接 RL 教师                              | [后训练阶段](post-training-stages.md)，Qwen/Tulu 旧证据与 MiMo/Olmo3/Light-R1 独立条件组                                                                                                                                                                                   |
| 查看 SFT student 吸收同源 direct-RL teacher 的 MiMo 对照     | [MiMo E91](post-training-stages.md#e91)，已完成 320 updates、outer-020 评估和最终读出；包含 E72/E73 对照及证据边界                                                                                                                                                         |
| 设计共同 Base 上的 SFT/RFT teacher 吸收对照                  | [Qwen2.5-1.5B E92](cases/2026-11-07-E92-Qwen25-SFT与RFT教师吸收对照.md)，固定 teacher 来源、共同初始化和 MiMo 匹配的 OPD 预算                                                                                                                                              |
| 查看 MiMo SFT+RL teacher 到 SFT student 的对照               | [MiMo E93](post-training-stages.md#e93)，沿用 E91 student-side protocol，替换为 E71 的 `MiMo-7B-RL` teacher                                                                                                                                                                |
| 查看 MiMo position-KL、训练 loss 与 Math-bank RKL 的分离分析 | [MiMo position-KL 分析](cases/mimo-position-kl.md)，仅包含 E71/E72/E73；[R1-1.5B](cases/r1-1.5b-position-kl.md)与[Qwen](cases/qwen-position-kl.md)分别维护自己的位置图                                                                                                     |
| 查所有 position-KL 记录及早期位置差异                        | [position-KL 实验清单](appendix/position-kl-inventory.md)，区分 native teacher RKL、终止对齐原始 RKL、MDPO objective-position 和三段 fixed-state 记录                                                                                                                      |

训练风险描述局部拟合，独立状态读出描述迁移，任务成绩描述生成表现。各表按学生、银行、支持集和预算分组。

共同 Base 上的受控 teacher 构造见[专家数据中心化与教师吸收对照设计](2026-11-08-专家数据中心化与教师吸收对照设计.md)：已完成 Qwen RAW 与两个 CCS teacher 的 SFT128→OPD256、四域双 bank 与数学能力评估，并与历史 MiMo-SFT 曲线比较。RAW 的任务增益与长前缀吸收、CCS 的覆盖限制和后续能力匹配设计在同页展开；理论动机见[同架构近邻假设](2026-11-08-同架构OPD近邻假设的理论与实验设计.md)。

下一轮的共同专家候选池、完整长回答、选择/重采样方法与 SFT 预算见 [SFT 数据分布处理与教师构造](2026-11-09-SFT数据分布处理与教师构造.md)。该页区分 RAW 的任务收益恢复与绝对分布残差，后续 OPD 仅保留交叉验证接口。

## 原因诊断

[Teacher 吸收专题](absorption/README.md)统一整理输出分布、全层参数谱、函数干预、PAC 与 ISO。先读[实际设置](absorption/设置与指标.md)和[训练审计](absorption/附录-定义与复算.md)，再查[全层与功能结果](absorption/跨模型结果.md)；完成项、缺项与诊断 pilot 分开解释。

| 读者问题                                   | 主要证据                                                                                                                                                              |
| ------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 训练域改善能否推广到其他 domain            | [跨域迁移](cross-domain-transfer.md)，已完成的 JustRL 数学分支及其 Math/Code/Science/IF 读出；其他域没有完整结果时保持缺测                                            |
| 当前 batch 学习能否传播，旧 batch 是否遗忘 | [跨 batch 迁移](cross-batch-transfer.md)，当前/旧/未来批、永久留出及固定/fresh 状态分别解释                                                                           |
| 可学习内容与知识遗忘的解释是否成立         | [能力筛选干预](mechanisms/capability-filter.md)，保留随机映射、温度、IF 及截断续写反例                                                                                |
| 共享更新方向与留出传播是否相关             | [谱诊断](mechanisms/spectral.md)，loss-gradient Gram 不等于完整 NTK                                                                                                   |
| 剩余 teacher 差异为何难以跨 batch 传播     | [Teacher difference 的 PAC 与噪声底诊断](absorption/README.md)：真实训练、完整轨迹 PAC、请求方向与谱对照；已拆分 A 的局部响应与题内聚合，并完成分量传播与实际小步干预 |
| 平台是否仅来自数值或 optimizer             | 引用[工程实验](../../infra/numerics/optimization/README.md)，不在本研究线重复维护精度与求解报告                                                                       |

当前批能拟合而未见批改善弱，不自动意味着旧批遗忘；换批 loss 锯齿也不能代替旧批复评。跨域改善弱首先描述迁移边界，表示与容量解释仍需对应证据。来源策略的最终效果由[Rollout 研究](../rollout-policy/README.md)维护；这里使用来源控制来解释迁移。

[ISO 方法](absorption/附录-定义与复算.md)、[结果](absorption/跨模型结果.md)与[计算预算](absorption/附录-定义与复算.md)属于同一吸收专题，记录原论文范围、16 配对全层扩展与实测耗时。

## 协议、复算与补充材料

[按批刷新与四域协议](protocols/online-four-domain.md)与既有 0–320 固定状态协议独立。已完成 JustRL 启用 source/history；后续关闭的分支保留自己的实际配置。

[图谱与复算](reproduction.md)定位生成结果、逐题数据和真实运行；[配对设计依据](appendix/pair-design.md)保留候选因素与尚未排除的解释；[Qwen 详细证据](appendix/qwen-lineage-evidence.md)保留 teacher-copy 等诊断。按既有约定，[RLVR-GSM8K](appendix/qwen3-rlvr-gsm8k.md)继续只在附录，不加入主比较。旧提案与撤销方案通过[历史索引](../../archive/README.md)定位，不构成自动开训清单。
