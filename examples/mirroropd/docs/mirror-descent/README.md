# η、κ 控制的 MirrorOPD 目标与实现

本目录提取相邻 `MirrorOPD/theory/core` 中的形状与插值理论，并说明它如何接到本仓 full-vocab 管线。它讨论一般 $\kappa,\eta$，已有 `theory/` 文档继续维护普通对数 OPD、IS 四格和通用估计器的证明；`full-vocab/` 继续维护已经运行的工程实现。

- [形状与插值目标及 Full-vocab 离策略实现](2026-10-23-形状与插值目标及Full-vocab离策略实现.md)：目标教师、MDPO/Part/BPO/Mean、current/frozen、固定 state 下的动作 IS、clipping、内存与接入计划。
- [CPU FP64 验算](scripts/verify_mirror_targets.py)和[结果](data/finite-vocab-checks.json)：枚举完整小词表检查目标、梯度与反例，无语言模型训练。
- [现有 Full-KL 源码走读](../full-vocab/2026-10-23-Full-KL全流程源码走读.md)：已有 teacher hidden、head、FSDP、分块算子和 IO 优化。

当前生产代码只支持直接拟合原 teacher 的 full reverse KL；本目录新增的是理论、可运行数值参照与接入设计，尚未增加可启动的 `kappa/eta` 训练配置。
