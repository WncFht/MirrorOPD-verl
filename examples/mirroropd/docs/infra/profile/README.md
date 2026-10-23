# Full-KL 性能分析

本目录保存 Full-KL 的测量方法、实验脚本和结果。数学目标与训练管线继续引用 [Full-KL 文档](../../full-vocab/README.md)，避免重复维护。GPU 实验限本机物理 GPU 0、1、2、3，空闲条件包含无计算进程、每卡显存低于 1 GiB、利用率低于 10%，连续满足 120 秒才启动。

## 当前状态

2026-10-23 新增[四卡单轮数学 reasoning 全流程观察](e2e-4gpu/README.md)，覆盖正式 graph rollout、teacher、actor、权重同步、checkpoint 与小验证窗口。14 步基线和四 rank / 四 rollout Torch trace 已完成；阶段成本、评分等待、显存和采集边界见[第一轮实测](e2e-4gpu/2026-10-23-四卡全流程实测.md)。这项观察独立于下述已完成的算子优化轮次。

2026-10-23 已完成初始 profile、同 chunk ABBA 复测、可选融合 backend 接入、四卡更新/恢复和两步非零学习率完整训练验收。最终结论、证据边界和启动命令见[Full-KL 优化结论与启动](2026-10-23-Full-KL优化结论与启动.md)。默认 E13/E14 不变；融合版本使用 `FULL_KL_BACKEND=precompute_fused`。

本机可执行的这轮测量和工程验证已完成，审阅完成后通过 `REVIEW_DONE.json` 停止每小时检查。集群八卡/A100、graph rollout 整步性能与长程准确率不包含在这轮已完成的证据中。

- [首次测量与后续验证](2026-10-23-首次测量与后续验证.md)：第一批结果与当时的待验证事项。
- [IO 对照与显存轨迹](2026-10-23-IO对照与显存轨迹.md)：完整训练 IO 数据和 allocator 生命周期。
- [自动实验记录](2026-10-23-Full-KL分层测量记录.md)：初始冻结队列的原始汇总。
- [分层测量与优化方案](2026-10-22-Full-KL分层测量与优化方案.md)：范围、工具、实验控制和判定标准。
- `scripts/run_study.py`：空闲门控、顺序执行、按任务保存退出状态；共用 `/tmp/mirroropd-gpus-0-1-2-3.lock`。
- `scripts/benchmark.py`：合成 hidden/head 的算子对照与数值检查。
- `scripts/pipeline.py`：真实模型、固定数学回答、FSDP2、offload、activation checkpointing 的管线对照。
- `scripts/fused_candidate.py`：历史候选名的兼容导入；可选 `precompute_fused` 接入见[融合算子接入与验证](2026-10-23-融合算子接入与验证.md)，生产默认未切换。
- `scripts/analyze.py`：区间并集分析、实测 CUDA 时间线和吞吐—显存图。
- `scripts/report.py`：从成功和失败任务生成记录，保留测量范围和未完成项。

运行目录为仓库 `outputs/full-kl-profile-20261022`。`status.json` 是当前状态；`runner.log` 是队列日志；每个任务保存 `launch.json`、`stdout.log`、`exit.json`。原始 trace 和显存快照保留在运行目录，小型 JSON 和图表在运行结束后发布到本目录的 `data/`、`figures/`。后续复测、生产验收和失败重试的目录见最终报告。

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

## 每小时会话检查

2026-10-23 根据用户要求启用 `scripts/hourly_monitor.py`，每 3600 秒通过本机 `codex queue` 给原会话发送后续检查任务。它使用现有会话上下文与权限，不另建模型会话、不占 GPU，也不代替实验队列。该方式是本机后台定时进程，不属于网页 Scheduled 列表；依赖本机和 Codex daemon 持续运行。

状态在运行目录的 `hourly-monitor.json`，历史在 `hourly-history.jsonl`。每条检查开始时，会话写 `hourly-ack.json`；未收到开始确认时不再积压消息。检查包括失败修复、结果分析、必要补实验、文档与阶段提交。队列退出不自动终止检查；审阅确认可做工作已完成后写 `REVIEW_DONE.json` 才结束。创建 `MONITOR_STOP` 可单独停止每小时触发，不会结束正在运行的实验。停止后已经入队的消息仍须遵守用户最新停止指令。
