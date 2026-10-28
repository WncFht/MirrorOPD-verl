# Infra：正确实现与执行效率

| 职责 | 入口 |
| --- | --- |
| 当前架构、调用链、数据和进度合同 | [architecture](architecture/README.md) |
| 启动、资产与共享资源执行 | [training](training/README.md) |
| 模板、tokenizer、EOS及Qwen诊断 | [model-compatibility](model-compatibility/README.md) |
| Full-vocab、目标与精度的正确实现 | [numerics](numerics/README.md) |
| Rollout来源/混合执行与验收 | [rollout](rollout/README.md) |
| 判分器与验证执行 | [evaluation](evaluation/README.md) |
| 算子、生成、验证与资源测量 | [performance](performance/README.md) |

工程通过只覆盖对应设置；短smoke、数学oracle和正式任务成绩分开报告。优化器/精度改变学习效果的实验归研究线，数值实现验收归这里。
