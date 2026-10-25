# 失败复现的实现与对齐诊断

本目录保留 EOS／tokenizer 协议、teacher 输入、输出词表、真实概率探针和短测验收。正式训练设置、论文原图、运行进度与待跑项集中在[失败复现实验系列](../experiments/failure-reproduction/README.md)。

| 入口 | 内容 |
| --- | --- |
| [实现、验收与对齐诊断](2026-10-24-OPD实现验收与对齐诊断.md) | 运行时停止配置、原始 IDs、词表修正、短测与低 KL 诊断 |
| [实验数据](data/) | 已保存的探针与接口验证结果 |
| [探针脚本](scripts/) | EOS、teacher tokenizer 与输出支持检查 |

本目录中的两步训练、CPU／四卡检查和自然生成探针只用于实现诊断。正式 250-step sampled/full 配对训练的状态由实验系列维护。
