# Rollout 来源与混合效果

## 背景与 motivation

本线从 2610.11291 的复现和直接扩展出发，检验在同一师生配对和 learner 配方下，不同训练数据生成策略如何影响训练收敛、成本及最终 accuracy；具体实验分别声明是否执行任务验证。当前不承担同源/异源跨 batch 迁移的微观解释。

## Setting 与实验设计

比较当前 student、初始冻结 student、teacher，以及轨迹混合和逐 token 混合。每项实验固定模型配对、learner loss、optimizer、训练与验证协议，明确更新、token 和生成成本的匹配方式；实际 setting 在具体方案中登记。数学上定义的融合中心与实际双模型后端分别见[理论](../../theory/rollout-policy/2026-10-28-Rollout策略的目标设计与求解.md)和[工程实现](../../infra/rollout/fusion/README.md)。

| 内容                       | 当前边界                                                                                               |
| -------------------------- | ------------------------------------------------------------------------------------------------------ |
| 论文设置与比较依据         | [失败复现案例调研](../../literature/opd/2026-10-24-OPD失败复现设置与Full-vocab对照.md)的 Semi-OPD 部分 |
| 来源、版本和混合实现       | 工程生命周期和短测，不能当成长程学习结果                                                               |
| 新来源/混合的最终 acc 比较 | 尚未由上述工程验收建立；完成后按独立实际运行报告                                                       |

## 正在进行的来源对照

[E20 设置下的六种前缀对照](2026-10-30-E20设置下的六种前缀对照.md)记录 R1-1.5B 配 JustRL-1.5B（J）、Polaris-7B-Preview（P）的十二组 40 outer × reuse16 实验。数据与固定前缀已冻结，S0/J/P 三套 630 题 × 4 次任务基线已完成并通过[复用检查](../../../../../../runs/prefix-fusion-scientific-v2/expansion-20261030/baseline-reuse-audit.json)。旧无验证/无保存的 S0 运行只作优化诊断。

带共同评估的两卡 v3 队列曾因共享模型路径迁移失败，J-S0/P-S0 分别保留 160/144 次成功更新，均未到可恢复 checkpoint 的保存点。这 304 次更新和原始证据[单独保留](../../../../../../runs/prefix-fusion-scientific-v2/expansion-20261030/pre-expansion-failures.json)；扩展版十二组均从原始 student 和新的 Adam 状态开始，活动进度不累计旧失败尝试。模型已改用研究目录自有绑定，20 个文件内容与冻结清单一致，四份模型回执保持不变，见[路径迁移证明](../../../../../../runs/prefix-fusion-scientific-v2/expansion-20261030/model-binding-relocation-v1.json)。

科学执行源码固定为 `c52e3eb6`。六个独立 TP1 通道使用物理 GPU 0、1、2、3、6、7，首轮分配如下；余下 LM/TM/T 条件由空闲通道独占领取。GPU 4、5 用于[工程验收](../../infra/rollout/fusion/README.md)。资源和路径迁移不改变学习、采样、题目顺序或评估预算，每个条件均保留配置差异和模型身份检查。

| GPU  | 首轮条件   | 训练前缀                                       |
| ---- | ---------- | ---------------------------------------------- |
| 0、1 | J-S0、P-S0 | 初始冻结 student                               |
| 2、3 | J-S、P-S   | 当前 student                                   |
| 6、7 | J-PM、P-PM | current student 与 teacher 的逐 token 概率混合 |

[扩展执行清单](../../../../../../runs/prefix-fusion-scientific-v2/expansion-20261030/manifest.json)固定版本、输入和条件身份；[实时状态](../../../../../../runs/prefix-fusion-scientific-v2/live-status.json)分别记录前置检查、训练、失败与完成，不以通道分配或进程启动代替成功更新。失败会停止对应通道，保留 claim 和证据，不自动重试。实验结果仍须通过逐组完成审计后计入比较。

## 实验结果与分析

策略比较以共同固定状态上的 RKL/JS 和纯 student 任务准确率为依据；不同 rollout 状态上的训练 KL/loss 单独作为优化诊断。现阶段不填入模拟或工程 smoke 的 acc 作为正式结果；终点 fresh-state 交叉评分、配对统计和最终图表另行完成，不预设混合策略更优。

Polaris 的固定来源、跨 batch 及交叉评分用于迁移假说，完整保留在[迁移机制](../teacher-transfer/transfer-mechanism/README.md)，不计作本线已完成效果实验。
