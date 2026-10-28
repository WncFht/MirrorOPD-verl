# 本机验证与 batch128 对照数据

分析、测量定义与限制见[本机训练与验证优化](../../../../infra/performance/2026-10-22-训练与验证性能实测.md#3-本机编译与训练优化)。实验只申请本机 GPU 0–3。W&B 为离线记录，run ID 可用于定位仓库根目录的 `wandb/offline-run-*`，不代表已经上传。

| 标记 | 子目录 | 离线 W&B run |
| --- | --- | --- |
| A | `valid-eager-w16-20261021-2` | `46ugm2mm` |
| B | `valid-graph-w16-20261021-1` | `6jpqrtw0` |
| C | `valid-graph-w32-20261021-1` | `5s25kbm8` |
| D | `valid-graph-w32-20261021-4` | `0jje8zxj` |
| E | `valid-graph-w32-s64-20261021-3` | `oqhotg75` |
| 首次省略概率的探索观察 | `valid-graph-w32-20261021-2` | 第一段 `ot1vy6w9`；意外第二次启动 `ahmji81m` 被取消 |
| 训练 batch128 | `train-b128` | `dgyajowp` |

`summary.json` 包含五个正常退出验证的汇总；各目录保存 `config.json`、`history.jsonl`、`launch.json`、`exit.json`、`gpu.csv.gz`、`engine-metrics.jsonl.gz`。`source_commit.txt` 与 `source_changes.patch.gz` 记录对应运行的基点和可执行源码差异，patch 仅保留 verl 与公共 launcher，排除同期其他任务的文档编辑。原始未筛选 patch 和训练日志仍在 outputs 对应目录。

`selection.json` 保存三个原始验证 parquet 的零起始行号，源文件为仓库同级 `MD-OPD/data/math/{MATH-500,AMC23,AIME25}/test.parquet`。完整回答在 `outputs/local-validation-20261021/<run>/training/validation/0.jsonl`，未复制到 Git。正式运行均已校验 32 个题目组、每组 16 条回答，四引擎完成计数之和为 512，generation counter 与 request token sum 一致。

`fixed-logprobs.json` 是单卡固定 32×2048 输出 token 的六次交错重复测量，全部 token ID 相同。精确诊断源码保存在同目录 `fixed-logprobs.py.txt`；这是实验源码快照，使用本机绝对路径且要求输出目录尚不存在，复跑前应修改输出路径。`batch-comparison.json` 比较旧 batch64 step2–5 与新 batch128 step2–3，两侧各 256 条回答。旧 batch64 的原始数据保持在[local-shared4-20261021](../local-shared4-20261021)。

探索目录保留首次已完成验证的 `first-history.jsonl` 与引擎采样；其进程退出状态属于意外再次启动后被取消的外层任务，不能混作正常结束的正式运行。其他在模型运行前失败或取消的启动没有性能结果，仍保存在 outputs 中。

GPU busy 按 testing 的近似时间边界取 NVML 均值；引擎采样周期约两秒，长尾是完成计数跨过 90% 与 100% 阈值的时间差。计数器求和使用四个 TP1 引擎各自最后一个成功采样，不对同一引擎的累计 counter 跨时间求和。
