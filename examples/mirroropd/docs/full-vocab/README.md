# Full-vocab 蒸馏

| 任务 | 入口 |
| --- | --- |
| 运行Full-vocab Forward KL / E49 | [目标、支持集、优化和运行命令](forward-kl.md) |
| 理解 OPD、MDPO、BPO 目标与梯度 | [目标走读](../internals/full-kl-walkthrough.md) |
| 沿训练调用链检查数据与实现 | [执行源码路线](../internals/full-kl-execution-walkthrough.md)、[hidden/head 接入](../internals/full-vocab.md) |
| 理解分块 VJP、backend 与数值精度 | [算子走读](../internals/full-kl-kernel-walkthrough.md) |
| 比较速度、显存与容量 | [性能专题](../infra/profile/README.md) |
| 核对数值、模型和分布式验证 | [验证矩阵](validation.md) |
| 选择训练预设 | [训练配方](../训练配方.md) |
| 设计 sampled+full-KL baseline | [Baseline 扩展](baseline.md)，尚未接入训练 |

本目录保留验证证据、baseline 设计和相关复现脚本；当前机制由 internals 维护，性能结果由 profile 维护。数据按测量批次保存，正文不再按实施进度拆成多份。

用户提供的 [Full-vocab OPD 参考原文](reference/fullvocab-opd.md)保持原样。其作者、代码链接和部分图表信息不完整，原文性能数字不作为本机复现结果。
