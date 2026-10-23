# 四卡单轮数学 reasoning 训练现状

本专题先测量当前完整训练过程，交付现状报告和离线交互 HTML，再决定是否优化。单 teacher、FSDP2 shared-sync、SP=1、原 `precompute`、async write/prefetch 开启，使用正式 graph rollout；不改变目标、全局 batch、训练 token 长度预算或学习率，缩短总训练步数以做工程测量。GPU 仅限本机 0–3。

- [详细说明：配置、生命周期、计时边界与分析判据](2026-10-23-全流程测量与现状说明.md)。
- [离线交互页面](dashboard.html)：本机生成文件，随采集阶段结束更新；没有实测数据时明确显示缺失。
- [聚合脚本](scripts/collect.py)、[HTML 模板](scripts/dashboard.html)、[静态绘图](scripts/plot.py)。

## 测量设计

- **baseline**：14 个真实训练步；保留完整 token 预算，步骤 4–13 提供 10 个稳态候选样本；预热是否充分由实际曲线判断。最后一步额外保存一次同步 checkpoint，并做 8 题 × 2 次、最大 1024 response token 的验证窗口，单独记账，不作全量准确率或验证耗时结论。训练的模型上下文配置保持原预算。
- **torch-profile**：同一配方运行 5 步，第 4 步采 actor/teacher 四 rank trace；vLLM 单独开启 discrete capture，截取前 128 个 engine iteration。该窗口是引擎迭代，不等同于每条请求的前 128 token；不拿详细采集步当吞吐基准。
- **nsys-profile**：短程同配方的训练 worker 系统时间线，补充跨线程 NVTX/通信/传输；vLLM 不会因全局工具为 nsys 自动被其现有 adapter 采集。此边界在报告中明确。

所有运行记录阶段 host span、真实 request 起止与 token 数、后端可用时间字段、每 2 秒一次 NVML/主机观测与 HTTP 指标。观测通过 `VERL_E2E_TIMELINE_DIR` 启用，默认关闭，不插入 CUDA 同步、不记录 prompt 文本或权重。四卡数据在同一主机时间命名空间中对齐；Torch trace 的时间基准另行检查，未确认的 trace 不强行合并。

运行目录计划为 `outputs/reasoning-e2e-profile-20261023`，冻结源码和实际参数由各任务 `launch.json`/`resolved_config.yaml` 确认。此前 Full-KL 算子实验位于上级目录，作为历史材料引用，不混入本次基线。

## 交付

最终报告解释配置、模型/进程关系、完整生命周期、阶段工作量、耗时/显存/等待、采集覆盖与未知项。交互 HTML 提供 step 选择、阶段和四卡时间线、rollout 并发/完成分布、显存曲线和配置/原始数据入口；静态图提供 PNG/PDF/SVG。性能优化候选与现状事实分开，第一阶段不自动修改性能配置。

## 本机运行与复现

以下命令从仓库根目录执行，使用现有 profile 环境。当前 GPU 队列已经挂起等待，不要重复启动。

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

生成的 `dashboard.html` 内嵌 Plotly，浏览时无需服务端或网络，可单文件带走。`data.json` 与 HTML 是本地生成快照，不作为源码提交；图和审计结果在有真实 baseline 后写入 `figures/`，经人工审阅再提交。运行时依赖 PyYAML、Plotly；绘图另需 matplotlib 与 figure-viz。浏览器测试使用 Playwright Chromium，显式禁用 GPU。

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
- 离线 Chromium 检查：所有页签、合成测试数据的 step 筛选、桌面/手机布局，无 JavaScript 异常、无外部 HTTP 请求。合成测试截图只存于 `browser-check/`，不作为性能证据。

```bash
"$PROFILE_PY" -m pytest -q tests/utils/test_e2e_timeline.py tests/utils/test_e2e_report.py
"$PROFILE_PY" "$PROFILE_DOC/scripts/check_dashboard.py"
```

GPU 实际运行、trace 覆盖和性能结论尚待硬件空闲后的检查；CPU 通过不替代这些验证。
