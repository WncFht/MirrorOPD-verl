# 四卡单轮数学 reasoning 训练现状

本专题先测量当前完整训练过程，交付 Markdown 现状报告、测量数据与静态图表，再决定是否优化。单 teacher、FSDP2 shared-sync、SP=1、原 `precompute`、async write/prefetch 开启，使用正式 graph rollout；不改变目标、全局 batch、训练 token 长度预算或学习率，缩短总训练步数以做工程测量。GPU 仅限本机 0–3。

- [详细说明：配置、生命周期、计时边界与分析判据](2026-10-23-全流程测量与现状说明.md)。
- [第一轮实测结果](2026-10-23-四卡全流程实测.md)：14 步基线、四 rank / 四 rollout Torch trace、瓶颈排序和采集边界。
- [聚合脚本](scripts/collect.py)、[静态绘图](scripts/plot.py)。

## 测量设计

- **baseline**：14 个真实训练步；保留完整 token 预算，步骤 4–13 提供 10 个稳态候选样本；预热是否充分由实际曲线判断。最后一步额外保存一次同步 checkpoint，并做 8 题 × 2 次、最大 1024 response token 的验证窗口，单独记账，不作全量准确率或验证耗时结论。训练的模型上下文配置保持原预算。
- **torch-profile**：同一配方运行 5 步，第 4 步采 actor/teacher 四 rank trace；vLLM 单独开启 discrete capture，截取前 128 个 engine iteration。该窗口是引擎迭代，不等同于每条请求的前 128 token；不拿详细采集步当吞吐基准。
- **nsys-profile**：短程同配方的训练 worker 系统时间线，补充跨线程 NVTX/通信/传输；vLLM 不会因全局工具为 nsys 自动被其现有 adapter 采集。此边界在报告中明确。

所有运行记录阶段 host span、真实 request 起止与 token 数、后端可用时间字段、每 2 秒一次 NVML/主机观测与 HTTP 指标。观测通过 `VERL_E2E_TIMELINE_DIR` 启用，默认关闭，不插入 CUDA 同步、不记录 prompt 文本或权重。四卡数据在同一主机时间命名空间中对齐；Torch trace 的时间基准另行检查，未确认的 trace 不强行合并。

运行目录计划为 `outputs/reasoning-e2e-profile-20261023`，冻结源码和实际参数由各任务 `launch.json`/`resolved_config.yaml` 确认。此前 Full-KL 算子实验位于上级目录，作为历史材料引用，不混入本次基线。

## 交付

最终报告解释配置、模型/进程关系、完整生命周期、阶段工作量、耗时/显存/等待、采集覆盖与未知项。聚合 JSON 保存逐 step、阶段和四卡时间线、rollout 请求分布、显存及配置；静态图提供 PNG/PDF/SVG。性能优化候选与现状事实分开，第一阶段不自动修改性能配置。

## 本机运行与复现

以下命令从仓库根目录执行，使用现有 profile 环境。原队列的 baseline 和 Torch 已完成；Nsight 的补采使用独立目录，不要重复启动。

```bash
PROFILE_PY=/root/.local/share/mirroropd/venvs/profile-cu128/bin/python
PROFILE_DOC=examples/mirroropd/docs/infra/profile/e2e-4gpu
PROFILE_RUN=outputs/reasoning-e2e-profile-20261023

cat "$PROFILE_RUN/status.json"
cat "$PROFILE_RUN/report-status.json"
```

单次刷新报告只使用 CPU，不创建实验或会话任务：

```bash
"$PROFILE_PY" "$PROFILE_DOC/scripts/refresh_report.py" \
  --root "$PROFILE_RUN" --output-dir "$PROFILE_DOC" \
  --style-dir /mnt/tidalfs-bdsz01/dataset/llm_dataset/zkn_data/fht_data/dotfiles/.agents/skills/figure-viz/assets/mpl
```

`--watch` 模式在现有队列状态切换时刷新，并在队列结束后退出。它持有独立的报告锁，已有报告 worker 时再次执行会拒绝并发；它不启动训练、不发每小时消息、不改变 GPU 队列。需要停止队列时在运行目录创建 `STOP`：等待中的队列退出，运行中的任务在结束后停止推进。

`data.json` 是本地生成的测量快照，不作为源码提交；图和审计结果在有真实 baseline 后写入 `figures/`，经审阅后提交。聚合依赖 PyYAML；绘图使用 matplotlib 与 figure-viz。

准备新的复测时，先选定一个已提交的源码版本并建立独立 worktree，然后使用 [prepare.py](scripts/prepare.py)；这个命令仅做 CPU 配置导出：

```bash
"$PROFILE_PY" "$PROFILE_DOC/scripts/prepare.py" \
  --source /absolute/frozen/worktree \
  --root /absolute/new/evidence-directory \
  --model-root /absolute/models \
  --data-root /absolute/MD-OPD/data
```

它拒绝覆盖已有 `plan.json`。实际执行仍由现有 `../scripts/run_plan.py --root ...` 完成，沿用共享 GPU 锁。原始计划可在本次运行目录的 `plan.json` 复核；三个任务都使用冻结 commit `8132ca543`。

## 已做的 CPU 检查

- 埋点测试：关闭时无文件、并发线程写入、异常传播、并发 async 请求字段。
- 聚合测试：重叠区间并集、步间周期、验证请求隔离、失败请求隔离、跨时钟域拒绝、缺失数据、完整 JSONL 损坏与未完成尾行。
- 三份 Hydra 配置 CPU 导出成功；Torch 的 rollout discrete 开关和 Nsight NVTX 环境分别设置。

```bash
"$PROFILE_PY" -m pytest -q tests/utils/test_e2e_timeline.py tests/utils/test_e2e_report.py
```

14 步 GPU 基线和 Torch trace 已在实测报告中审阅；Nsight 的最终补采仍需四 rank CUDA/NVTX 产物验收。

## 实测结果复核与重画

实测文档与 `review.json` 是本轮分析入口，完整聚合数据保存在 data.json。最终 Nsight 队列位于 `outputs/reasoning-e2e-nsys-final-20261023`，继续使用共享 GPU 锁；前两次尝试的失败证据分别保留。

```bash
"$PROFILE_PY" "$PROFILE_DOC/scripts/collect.py" \
  --root "$PROFILE_RUN" --output "$PROFILE_DOC/data.json"
"$PROFILE_PY" "$PROFILE_DOC/scripts/analyze_cycle.py" \
  --root "$PROFILE_RUN" --data "$PROFILE_DOC/data.json" --output "$PROFILE_DOC/review.json"
"$PROFILE_PY" "$PROFILE_DOC/scripts/plot_review.py" \
  --data "$PROFILE_DOC/data.json" --review "$PROFILE_DOC/review.json" --output "$PROFILE_DOC/figures" \
  --style-dir /absolute/figure-viz/assets/mpl
```

`collect.py --nsys-retry-root /absolute/retry-directory` 可以将补采作为独立运行加入 HTML，保留原失败记录。`render.py --review` 将分析摘要也写进 HTML；下载成品后无需 JSON、Python、网络或图表服务。

新的 Nsight 计划固定 Python PATH、使用单区间 `stop-shutdown`，并在每 rank 的 finish hook 中等待 `.nsys-rep` 最终落盘，再允许 Ray 清理。自定义 Ray 临时目录按当前 worker session 解析。训练退出码为 0 仍需检查四份 `capture-rank*.json` 和实际 CUDA kernel 表。

`refresh_report.py --root <最终补采目录> --base-root <原始基线目录> --review <review.json> --output-dir <本目录> --watch` 会在补采结束后自动检查四 rank Nsight 文件、合并数据并刷新单文件 HTML。这个 CPU 后处理不启动训练，也不代替原始 trace 的人工审阅。
