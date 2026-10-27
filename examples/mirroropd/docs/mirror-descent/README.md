# MirrorOPD 目标与实现

本目录维护一般 $\kappa,\eta$ 的融合目标与 MDPO/BPO 实现。所有方法从[统一实验入口](../../run_math_opd_ablation_fsdp2.sh)选择模型预算和算法，设置见[训练配方](../训练配方.md)。普通对数 OPD 的 IS 四格推导见[理论目录](../theory/README.md)，现有训练链路见[内部实现说明](../internals/README.md)。

| 入口 | 内容 |
| --- | --- |
| [形状与插值目标](形状与插值目标.md) | 融合教师、原教师目标、MDPO/Part/BPO/Mean、current/frozen、动作 IS 与参数性质 |
| [统一实现方案](实现方案.md) | Sampled MDPO、full-vocab MDPO/BPO、eta=1 Full-KL 统一、配置、目标、数据与算子 |
| [E44–E47 正式对照](../训练配方.md#mirror-comparison) | Full-KL / MDPO / BPO / 融合形状；256 outer、4096 updates，每 128 outer 保存；E45/E47 的已有结果见实验目录 |
| [实现验收结果](data/implementation-checks.json) | CPU、真实 TQ、CUDA、两卡 FSDP2 和 1.5B 短训练证据 |
| [Full-vocab 优化与实测](实现方案.md#85-已完成的数值层优化与实测)与[数据](data/full-vocab-performance.json) | Target/诊断融合、固定输入 A/B、时间与峰值显存、剩余边界 |
| [MDPO／OPD 精度报告](2026-10-27-MDPO反弹机制与浮点精度研究报告.md) | 目标解释、OPD 对照、buffer 修复、精度边界与验证范围 |
| [内层 KL 反弹审计](2026-10-26-MDPO内层KL反弹与数值链路审计.md) | 近目标梯度修正、真实四卡固定前缀重放、Adam 与计算精度对照 |
| [生产数值修复与验收](2026-10-27-蒸馏数值修复与验收.md) | BPO/Full-KL 稳定归约、分离 target、近零标量、sampled CE、回归与成本 |
| [优化器历史与前向精度对照](2026-10-27-MDPO优化器历史与前向精度因果对照.md) | 零梯度与固定更新控制；Adam/SGD/Muon 多步求解、最佳状态及中间精度干预 |
| [数值验证脚本](scripts/verify_mirror_targets.py)与[结果](data/finite-vocab-checks.json) | CPU FP64 小词表上的目标、梯度和反例检查 |

已实现 sampled MDPO、full-vocab MDPO/BPO；普通 Full-KL 是 MDPO eta=1 的直接执行分支。配置、运行约束和验证范围在统一实现方案中维护。
