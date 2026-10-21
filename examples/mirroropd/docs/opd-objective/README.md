# OPD 目标与梯度估计

统一记号见[符号约定](../符号约定.md)，与 `MirrorOPD/theory/core` 对齐。

本目录讨论固定 prefix 下的蒸馏目标、优势估计和重要性采样，不讨论 prefix 分布的校正。

| 内容 | 入口 |
| --- | --- |
| 数学推导、论文证据、实现刷新时机、实验建议 | [固定前缀下的优势估计与重要性采样](2026-10-21-固定前缀下的优势估计与重要性采样.md) |
| 实验开关、更新计数、checkpoint、启动入口与验证状态 | [优势与 IS 消融及更新计数实现方案](2026-10-21-优势与IS消融及更新计数实现方案.md) |
| Sampled-token、KL baseline、TA-OPD、ResOPD、full-vocab 与组合条件 | [学习信号与梯度估计器对照](2026-10-21-OPD学习信号与梯度估计器对照.md) |
| 共用分块算法、math 配方的显存/通信预算与条件步时估算 | [词表蒸馏的计算与显存预算](2026-10-21-词表蒸馏的计算与显存预算.md) |
| Reverse-KL 实验排序、full-vocab 计算图、top-k 支持集与 One-Shot 代码审计 | [实验排序与支持集](2026-10-21-Reverse-KL实验排序与Top-k支持集.md) |
| CPU 梯度验算 | [verify_fixed_prefix.py](scripts/verify_fixed_prefix.py) |
| 固定版本上的验算结果 | [gradient-audit.json](data/gradient-audit.json) |

系统工程分别由 [指标与验收](../infra/2026-10-21-指标与验收.md)和[同步共享八卡方案](../infra/2026-10-21-共享八卡运行与架构.md)负责。改变 GPU 调度、改变 advantage、取消 PPO clipping、切换 IS 分母是不同实验因素。
