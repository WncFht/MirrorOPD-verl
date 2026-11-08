# MirrorOPD 文档

[师生配对 RKL 总览：R1、Qwen3、Llama](experiments/teacher-transfer/fixed-state-results/README.md)汇集已有配对的训练与留出结果。

从研究问题、公共定义或工程任务进入正文。研究报告保留实际 setting 和证据截止，运行表由本地日志生成，周报引用专题结论。

| 阅读任务                              | 入口                                                                                                 |
| ------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| 看五个 RQ、已有结论与证据关系         | [实验研究](experiments/README.md)                                                                    |
| 查实际运行进度与末次评测              | [运行记录](experiments/运行记录.md)、[验证结果](experiments/验证结果汇总.md)，注意生成视图的覆盖范围 |
| 查模型、数据、公共训练/验证参数与指标 | [模型来源](settings/模型与师生配对.md)；[设置参考](settings/README.md)                               |
| 理解目标、估计器、镜像下降与迁移理论  | [理论](theory/README.md)                                                                             |
| 准备资产、启动、恢复和扩展实现        | [工程](infra/README.md)                                                                              |
| 查论文、公开实现与模型来源            | [文献](literature/README.md)                                                                         |
| 阅读按时间冻结的汇报                  | [周报](progress/README.md)                                                                           |
| 复算研究结果与维护文档                | [实验维护](experiments/维护说明.md)、[文档维护](infra/maintenance/README.md)                         |
| 查旧接口、历史设计及迁移位置          | [历史依据](infra/maintenance/history.md)                                                             |

原始运行日志与大型证据保留在同级 wandb-logs 及原运行目录。文档不放活跃 Python 工具；复算入口位于 research、benchmarks 和 tools，冻结源码保持原样。[AGENTS.md](AGENTS.md)说明内容归属与保护边界。
