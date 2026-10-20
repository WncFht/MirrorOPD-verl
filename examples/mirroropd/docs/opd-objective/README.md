# OPD 目标与梯度估计

本目录讨论固定 prefix 下的蒸馏目标、优势估计和重要性采样，不讨论 prefix 分布的校正。

| 内容 | 入口 |
| --- | --- |
| 数学推导、论文证据、实现刷新时机、实验建议 | [固定前缀下的优势估计与重要性采样](2026-10-20-固定前缀下的优势估计与重要性采样.md) |
| 实验开关、更新计数、checkpoint 与最小改动方案（待实施） | [优势与 IS 消融及更新计数实现方案](2026-10-21-优势与IS消融及更新计数实现方案.md) |
| CPU 梯度验算 | [verify_fixed_prefix.py](scripts/verify_fixed_prefix.py) |
| 固定版本上的验算结果 | [gradient-audit.json](data/gradient-audit.json) |

系统工程分别由 [指标与验收](../infra/2026-10-21-指标与验收.md)和[同步共享八卡方案](../infra/2026-10-21-共享八卡运行与架构.md)负责。改变 GPU 调度、改变 advantage、取消 PPO clipping、切换 IS 分母是不同实验因素。
