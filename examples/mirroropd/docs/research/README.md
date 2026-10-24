# 研究与实验对照

固定前缀下的 OPD、full-vocab 和镜像下降证明集中在[理论目录](../theory/README.md)。本目录保留论文的实验设置、原图、指标与适用边界。

| 文档 | 内容 |
| --- | --- |
| [当前失败复现方案](../failure-reproduction/README.md) | 使用本项目DAPO与三valid，先sampled-token再full-vocab；A–F模型、response cap、两组4B教师原始数据、对齐上限与低KL研究、排除设置附录 |
| [原论文设置核对档案](2026-10-24-OPD失败复现设置与Full-vocab对照.md) | 保留Rethinking、TIDE、Revisiting、Semi-OPD的作者参数、论文／代码差异和本仓适配边界；执行选择以当前方案为准 |
| [KL精度误差模型与诊断接口走读](2026-10-24-KL精度误差模型与诊断接口走读.md) | Logit扰动与误差相关性的推导、数值分辨率、真实初始权重probe、统一诊断接口与共享precompute VJP |
| [评测预算、KL精度与相对位置十分桶](2026-10-24-评测预算语义与KL精度及位置分桶.md) | pass/prefix mean 语义、FP32/BF16数值验证；已接入的10桶Full-KL指标与验收 |
| [E16/E17 位置指标与后期收敛](2026-10-24-E16E17位置指标与后期收敛.md) | 历史run缺少位置分桶的核对；新批起点与复用末次 KL、后期趋势、分桶与固定 probe 建议 |
| [Full-KL 收敛与师生参考](2026-10-24-Full-KL收敛与师生参考评测.md) | E16/E17 的 current + IS 含义、批内下降、历史师生准确率、长度/熵覆盖及 sequence KL 边界 |
| [五组 Math 实验对比](2026-10-24-五组Math实验对比分析.md) | E13–E17 的本地全量解码、生成/更新/时间预算对齐、2378 个指标与离线交互页面 |
| [E15 训练动态与指标分析](2026-10-23-E15训练动态与指标分析.md) | 完整 history 快照、策略差异与蒸馏信号、验证 k 曲线和成本；八组可复算图 |
| [陈旧数据训练的指标与实验对照](2026-10-21-陈旧数据训练的指标与实验对照.md) | p99、KL、log-ratio二阶矩、ESS、clipping口径；14项RL/OPD工作的设置、实测指标与原图，含VeXact、Score Centering、CIS与Distillation Dynamics；本项目实验判断 |
