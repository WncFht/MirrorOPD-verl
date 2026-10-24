# E16/E17 位置与收敛分析数据

来源：2026-10-24 本地 `wandb-logs` 的 `.wandb` 解码，不读取 API 或网页曲线。完整原始日志仍在独立日志仓库；本目录只保存相关派生数据。

- `analysis.json`：来源、解码边界、窗口统计、逐批归一化复用曲线和验证点。
- `batches.csv`：每批首末 KL、相对降幅、末次梯度及可见的回答长度。
- `updates.csv`：逐 update 的 KL、actor loss、epoch index、梯度范数。
- `E16-recorded-keys.txt`、`E17-recorded-keys.txt`：解码 history 中的完整字段清单。
- `convergence.png`、`convergence.pdf`：可视化；`convergence.audit.txt` 为 figure-viz 审计结果。没有 token 位置分桶数据，不生成位置热图。

E16 完整至 outer200/update800；E17 是 partial snapshot，更新至 outer178/update2848，outer 汇总至177、验证至160。缺失值保持为空，不补零。

从源码仓库根目录重新导出时，给每个 run 分别选择不存在的新目录；导出器不会覆盖已有快照：

```bash
python3 examples/mirroropd/docs/scripts/analyze_run.py export \
  --run wncfht/MirrorOPD-verl/ujy2qx5d --out outputs/position-convergence-20261024-audit
python3 examples/mirroropd/docs/scripts/analyze_run.py export \
  --run wncfht/MirrorOPD-verl/cix5i2x2 --out outputs/position-convergence-20261024-e17
```

已有这两个快照时只运行分析命令。省略 `--plot` 可只用 Python 标准库生成统计与 CSV：

```bash
FIGURE_VIZ_MPL=/mnt/tidalfs-bdsz01/dataset/llm_dataset/zkn_data/fht_data/dotfiles/.agents/skills/figure-viz/assets/mpl \
  /root/.local/share/mirroropd/venvs/profile-cu128/bin/python \
  examples/mirroropd/docs/scripts/analyze_convergence_positions.py \
  --e16 outputs/position-convergence-20261024-audit \
  --e17 outputs/position-convergence-20261024-e17 \
  --out examples/mirroropd/docs/data/position-convergence-20261024 --plot
```

脚本核对指标声明的横轴、每批完整 epoch 序号、update 唯一性、更新成功状态以及 actor loss 与 KL 的数值一致性。图中 rolling mean 跨20个已观测 outer，末端没有补齐数据；右图固定使用141–160，先逐批除以首点再取均值。未把批间差异视为跨 seed 不确定性。
