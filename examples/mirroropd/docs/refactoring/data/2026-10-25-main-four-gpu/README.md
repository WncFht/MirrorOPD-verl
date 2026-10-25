# 主分支四卡验收证据

对应[验收报告](../../2026-10-25-主分支四卡验收.md)。源版本为 `5f1a0cee82dfabeaee855fc307c5208d9a9ec858`，运行时未提交差异仅为测试计数适配。

| 文件 | 内容 |
| --- | --- |
| `summary.json` | 独立读取原始输出后断言检查的结果 |
| `results.json` | 各子进程命令、退出码和用时，包含验证数据准备步骤 |
| `environment.json`、`gpu-after.txt` | 原生环境版本、使用的设备及任务结束后的 GPU 状态 |
| `full-qwen2.json`、`full-qwen3-fused.json` | 四卡小模型 loss、位置计数、参数变化和恢复检查结果 |
| `sampled-current-is.log`、`sampled-frozen-ppo.log` | 两组 sampled 四卡测试的完整输出 |
| `real-sampled/`、`real-full/` | 原始指标、完整解析配置、实际命令、源版本和运行时差异 |
| `run_validation.py` | 本机执行的冻结验证脚本，所有测试顺序运行，输出到新的唯一目录 |
| `check_results.py` | 仅用 Python 标准库重新检查指标、计数和测试结果 |

复查已保存结果，在仓库根目录执行：

```bash
python examples/mirroropd/docs/refactoring/data/2026-10-25-main-four-gpu/check_results.py \
  examples/mirroropd/docs/refactoring/data/2026-10-25-main-four-gpu
```

该命令重新生成 `summary.json`。重跑 GPU 验证时，先确认 GPU 0–3 空闲，然后运行 `run_validation.py`。脚本固定了本机仓库、原生环境、模型与数据的绝对路径；其他机器需要先修改这些路径。它创建独立 Ray session、关闭 W&B，不连接或停止已有集群。

`real-*/command.sh` 是实际命令的副本，只去掉 shell `printf` 产生的末尾分隔空格。命令中包含当时输出目录；需要重新训练时应使用生成新目录的 `run_validation.py`，避免覆写本次运行。

完整训练日志和三份单题验证 parquet 留在 `outputs/main-four-gpu-20261025-0f02312b/`；模型权重和原始数据未修改或复制到文档目录。本目录不保存 checkpoint 文件。
