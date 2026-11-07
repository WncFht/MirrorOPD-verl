# Position-KL 数据清单

图和主要数值按三个 case 维护：[MiMo E71/E72/E73](../cases/mimo-position-kl.md)、[R1-1.5B E48/E49/E35](../cases/r1-1.5b-position-kl.md)、[Qwen E39/E41/E43](../cases/qwen-position-kl.md)。

## 字段覆盖

| 记录                           | 字段 / 分桶                                          | 数量与范围                          | 使用位置                         |
| ------------------------------ | ---------------------------------------------------- | ----------------------------------- | -------------------------------- |
| E71 / E72 / E73                | `distill/full_kl_position`，十个相对 response 十分位 | 800 / 320 / 320 次逐 update 观测    | MiMo 文档                        |
| E48 / E49 / E35                | 同上，原始 teacher RKL                               | 4096 / 4096 / 4000 次逐 update 观测 | R1-1.5B 文档                     |
| E39 / E41 / E43                | 同上；冻结 CSV 保留 first / last 抽样                | 200 / 100 / 100 outer               | Qwen 文档                        |
| termination-baseline / aligned | 原始 `full_kl_position`                              | 各 20 outer，受控诊断               | 仅数据清单，不加入三个 case 的图 |
| fixed-state appendix           | `positions.csv` 的 early / middle / late             | 48 个 run，0–320 updates            | 三段记录，不能改标为 p00–10%     |
| weighted-target / MDPO         | `distill/mdpo/objective_position`                    | objective 分解                      | 与原始 teacher RKL 分开          |
| E16–E20 reuse-budget 历史      | 仅全局 `distill/full_kl`                             | 无 position 字段                    | 无法从旧日志恢复位置桶           |

## 冻结数据与复算

[完整 inventory](../data/position-kl-inventory-20261107/inventory.csv)、[初始首桶](../data/position-kl-inventory-20261107/initial-position-kl.csv)、[初始 profile](../data/position-kl-inventory-20261107/initial-position-profile.csv)由 [`plot_position_kl_inventory.py`](../figures/plot_position_kl_inventory.py)生成；[manifest](../data/position-kl-inventory-20261107/manifest.json)记录来源哈希。

```bash
uv run --no-project --with matplotlib --with numpy \
  python examples/mirroropd/docs/experiments/teacher-transfer/figures/plot_position_kl_inventory.py
```
