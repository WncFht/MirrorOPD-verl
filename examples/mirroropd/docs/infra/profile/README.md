# Full-KL 性能分析

本目录保存 Full-KL 的测量方法、实验脚本和结果。数学目标与训练管线继续引用 [Full-KL 文档](../../full-vocab/README.md)，避免重复维护。GPU 实验限本机物理 GPU 0、1、2、3，空闲条件包含无计算进程、每卡显存低于 1 GiB、利用率低于 10%，连续满足 120 秒才启动。

## 当前状态

2026-10-22 23:17 检查时，前四卡分别占用约 73、60、59、75 GB，均有现存计算进程。旧质量消融队列尚未启动实验，已通过其 STOP 文件停止等待，保留原始计划。该目录的工程实验优先使用下一段空闲时间。

- [分层测量与优化方案](2026-10-22-Full-KL分层测量与优化方案.md)：范围、工具、实验控制和判定标准。
- `scripts/run_study.py`：空闲门控、顺序执行、按任务保存退出状态；共用 `/tmp/mirroropd-gpus-0-1-2-3.lock`。
- `scripts/benchmark.py`：合成 hidden/head 的算子对照与数值检查。
- `scripts/pipeline.py`：真实模型、固定数学回答、FSDP2、offload、activation checkpointing 的管线对照。
- `scripts/fused_candidate.py`：实验用 Triton 归一化/梯度融合，生产默认未切换。
- `scripts/analyze.py`：区间并集分析、实测 CUDA 时间线和吞吐—显存图。
- `scripts/report.py`：从成功和失败任务生成记录，保留测量范围和未完成项。

运行目录为仓库 `outputs/full-kl-profile-20261022`。`status.json` 是当前状态；`runner.log` 是队列日志；每个任务保存 `launch.json`、`stdout.log`、`exit.json`。原始 trace 和显存快照保留在运行目录，小型 JSON 和图表在运行结束后发布到本目录的 `data/`、`figures/`。自动记录尚未产生时，不能把计划写成已完成的优化。

## 启动与停止

实际使用的 Python 环境是 `/root/.local/share/mirroropd/venvs/profile-cu128/bin/python`；PyTorch 为 2.11.0+cu128。运行器必须指向已提交的冻结工作树，以免实验中途读到其他任务的修改。参数示例：

```bash
python scripts/run_study.py \
  --source /absolute/pinned/source \
  --root /absolute/repo/outputs/full-kl-profile-20261022 \
  --samples /absolute/math/validation/0.jsonl \
  --model-root /absolute/models \
  --data-root /absolute/data \
  --publish-dir /absolute/repo/examples/mirroropd/docs/infra/profile
```

在运行目录创建 `STOP` 文件，会在等待期间退出，或在当前任务结束后停止。运行器只清理自己启动并记录了创建时间的后代进程，不调用全局 `ray stop` 或 `pkill`。同一计划再次启动会读取已完成任务的退出码并跳过；失败任务需要使用新运行目录重试，以保留失败证据。

## 图表复现

```bash
python scripts/analyze.py \
  --root /absolute/repo/outputs/full-kl-profile-20261022 \
  --figures /absolute/output/figures \
  --style-dir /absolute/figure-viz/assets/mpl
```

绘图依赖 matplotlib 和 figure-viz 的 `fvstyle`，输出 PNG、PDF、SVG 及审计记录。时间线图仅展示首个 GPU kernel 开始的 80 ms 局部窗口；原始 `.json.gz` 可用 Perfetto 检查完整过程。内核分类按名称进行，未知内核保留 Other；分类求和不等同于整步墙钟时间。
