# E15 指标快照

唯一来源为 `wncfht/MirrorOPD-verl/krkmq098`，抓取时已 finished。这里只记录 E15，完整分析见[训练动态与指标分析](../../research/2026-10-23-E15训练动态与指标分析.md)。

| 文件 | 内容 |
| --- | --- |
| `source.json` | API 抓取时间、run、600 条 history 与 `_step=599` 边界 |
| `history.jsonl.gz` | `scan_history(keys=None)` 原始正文无损 gzip，保留 null、时间戳、全部事件；解压后 61,600,895 bytes |
| `config.json` | 只读导出器保存的实际配置 |
| `hardware.json`、`package-versions.txt` | 经过字段筛选的运行环境；不是完整源码归档 |
| `system-sampled.jsonl.gz` | 2000 行 sampled system stream，其中 1018 个 GPU 采样时刻；不保证全量 |
| `metric-axes.json` | 冻结的字段横轴路由，后续复算不受默认路由变更影响 |
| `series.json.gz` | 每个指标的轴和 `(step, value)` 序列，忽略 null、拒绝冲突重复 |
| `metric-catalog.csv` | 2377 个数值指标的覆盖、常量标记、范围和窗口统计 |
| `analysis.json` | 可重算的窗口、验证、计时、终止原因和系统采样统计 |

本快照复用本机已完成的只读 API 导出 `outputs/run-comparison-20261023/krkmq098`，只读取该子目录。原导出器的旧 schema `analysis.json` 不适用于 policy-v3，没有沿用；这里由新脚本重新计算。完整 history 用 gzip 压缩而未裁剪字段；原导出证据不修改。

`first20_mean/last20_mean` 指前后 20 个观测；不足 20 点则使用全部可用点。system 中的 `sample_mean` 是采样值算术均值，不是完整时间加权均值。物理单位与有效条件查[指标字典](../../metrics/2026-10-22-指标字典.md)，不从名字自动推断平台特有字段的单位。

复现命令集中在[分析复现](../../分析复现.md#e15-policy-v3-完整指标分析)。原始归档已存在时，脚本拒绝 `--snapshot-from` 覆盖；重算时省略该参数即可。
