# OPD 失败复现

固定本项目的 **DAPO → MATH-500／AMC23／AIME25**，比较外部教师文本 OPD 的 sampled-token 与 full-vocab；首轮不做 top-k、OPSD 或任务奖励混合。目录保留两份主文，研究设置和实现／验收各维护一处。

| 入口 | 内容 |
| --- | --- |
| [E21–E32 实验方案](2026-10-24-Sampled-token与Full-vocab失败复现方案.md) | 模型、长度、训练／验证预算、原论文图表、负迁移及排除设置 |
| [实现、验收与对齐诊断](2026-10-24-OPD实现验收与对齐诊断.md) | 已接入的运行时 EOS 配置与代码；teacher 原始 IDs；词表修正；已保存实验；低 KL 理论、论文证据与诊断 |
| [原图来源](figures/README.md) | 版本、原图、渲染／截取说明及权利信息 |
| [原论文参数档案](../research/2026-10-24-OPD失败复现设置与Full-vocab对照.md) | 作者原始 top-k、n=4 等设置与源码差异，不作为当前执行默认 |
| [冻结检索记录](sources/2026-10-24-OPD收敛原始检索报告.md) | 原始搜索候选，包含误召回；研究结论以两份主文为准 |

`data/` 保存原始实验结果，`scripts/` 保存对应研究探针，`figures/` 保存原图；通用测速工具位于 [benchmarks/](../../benchmarks/)。E21–E32 训练与原始师生验证使用[统一消融入口](../../run_math_opd_ablation_fsdp2.sh)，具体命令见[运行方式](2026-10-24-Sampled-token与Full-vocab失败复现方案.md#failure-launcher)。历史实验专用模型目录方案已退为验证记录；集群正常加载原始模型，运行时追加停止配置已接入并通过请求／引擎验收。正式 250 步 failure 观察尚未启动。
