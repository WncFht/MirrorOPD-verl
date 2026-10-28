# E48 完整训练冻结证据

日期：2026-10-27。正文为源码仓库 `examples/mirroropd/docs/experiments/teacher-transfer/cross-origin/polaris/2026-10-27-E48完整训练与后期平台分析.md`。

原始日志仓库提交：`b5cdd40f2ea42c92e2f3f25b0ca380c49e5c8f53`。E48 `grcuotmi` 的完整 `.wandb` 解码为 finished、exit_code=0，4353个history事件、256个完整outer、4096个实际optimizer更新、8018个系统记录；验证在0及32的倍数直到256。E20 `lwgb5eex`同样完整到4096更新。来源和环境分别见两个快照的 `source.json`、`hardware.json` 与包版本。

旧 `polaris-grcuotmi-20261026` 目录冻结到58个outer，不修改。新E48 history的前987行与旧快照完全相同，config也相同。`cloud-audit.json`仅为只读API身份、状态和边界核对，不是history采样来源。

| 文件 | 内容 |
| --- | --- |
| `run/` | E48完整原始事件、配置、summary、系统记录及来源 |
| `E20/` | 同更新预算的完整同源对照 |
| `statistics.json` | 32-batch窗口、逐批首末、16个reuse位置、位置桶sum/count、全部验证、系统汇总及数值一致性检查 |
| `E48-all-metrics.csv`、`E20-all-metrics.csv` | 所有有限数值字段的观测数、首末、范围和均值；稀疏缺失不补零 |
| `verification.json` | 独立的更新坐标、reuse顺序、执行状态、验证预算与文档/图形核验 |

“首次到第16次观测”之间执行了15次更新。首次和末次KL跨不同批平均；位置profile先合并sum/count再计算均值。“批间跳升/批内下降”是含首尾差值的恒等式，接近100%不意味着学习被遗忘。系统记录是周期采样，均值覆盖训练、验证和切换，不是连续trace或纯actor阶段利用率。

本目录没有模型/optimizer checkpoint或逐条验证回答。正常退出和保存checkpoint计时不证明这些文件已经同步到本机。报告只分析可见证据，未启动新训练。

从源码仓库根目录复算：

```bash
COMPLETE_DATA=examples/mirroropd/docs/experiments/teacher-transfer/cross-origin/polaris/data/polaris-complete-20261027
EARLY_DATA=examples/mirroropd/docs/archive/experiments/polaris/data/polaris-grcuotmi-20261026
python3 -m examples.mirroropd.research.transfer.analyze_completed_transfer \
  --data "$COMPLETE_DATA" --old "$EARLY_DATA"

FIGURE_VIZ_MPL=../dotfiles/.agents/skills/figure-viz/assets/mpl \
  /root/.local/share/mirroropd/venvs/profile-cu128/bin/python \
  -m examples.mirroropd.research.transfer.plot_completed_transfer \
  --data "$COMPLETE_DATA" \
  --out examples/mirroropd/docs/experiments/teacher-transfer/cross-origin/polaris/figures/completed-20261027
```

复算统计不读取网络、不改变原始history。首次建立快照使用项目 `docs/scripts/analyze_run.py export --run ... --out ...`，将两个run分别导入新目录；既有冻结快照不覆盖。报告、PNG/PDF与项目专用脚本保留在源码仓库中。
