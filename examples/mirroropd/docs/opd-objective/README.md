# OPD 目标与梯度估计

本目录讨论固定 prefix 下的蒸馏目标、优势估计和重要性采样，不讨论 prefix 分布的校正。

| 内容 | 入口 |
| --- | --- |
| 数学推导、论文证据、实现刷新时机、实验建议 | [固定前缀下的优势估计与重要性采样](2026-10-20-固定前缀下的优势估计与重要性采样.md) |
| CPU 梯度验算 | [verify_fixed_prefix.py](scripts/verify_fixed_prefix.py) |
| 固定版本上的验算结果 | [gradient-audit.json](data/gradient-audit.json) |

系统工程分别由 [最小指标方案](../infra/2026-10-21-最小指标实现方案.md)和[同步共享八卡方案](../infra/2026-10-21-同步共享八卡实现方案.md)负责。改变 GPU 调度、改变 advantage、取消 PPO clipping、切换 IS 分母是不同实验因素。
