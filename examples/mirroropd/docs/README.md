# MirrorOPD 文档

[实验研究](experiments/README.md)按四条问题组织：训练信号、reuse 预算、rollout 策略效果、师生迁移机制。具体报告依次说明背景与 motivation、实际 setting、结果和分析。

| 阅读目的 | 入口 |
| --- | --- |
| 研究问题、实验关系与结论 | [四条研究主线](experiments/README.md) |
| 实际运行与末次评测 | [运行记录](experiments/运行记录.md)、[验证汇总](experiments/验证结果汇总.md) |
| 公共设置与数据/指标含义 | [设置说明](settings/README.md)；具体运行配置仍在各报告 |
| 数学目标、估计器与证明 | [理论](theory/README.md) |
| 系统实现、正确性与工程性能 | [Infra](infra/README.md) |
| 外部论文、公开源码与模型卡 | [文献资料](literature/README.md) |
| 复算和维护 | [维护说明](experiments/维护说明.md) |
| 原文件的迁移位置 | [迁移对照](archive/2026-10-28-文档迁移对照.md) |

原始运行日志与大型证据继续位于同级 `wandb-logs`。日期报告保留各自输入边界；当前表格不回写历史结论。文档目录不维护活跃 Python 工具，运行入口位于 `examples/mirroropd/research`、`benchmarks` 和 `tools`。
