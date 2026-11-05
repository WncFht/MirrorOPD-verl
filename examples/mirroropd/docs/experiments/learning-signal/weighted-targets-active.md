# TOP-D 当前实验：大 batch 与回答复用

## 假设与预算

E81（A）和 E82（B）检验：保留 E57 的目标与每批 16 次复用，将每次 optimizer update 使用的回答从 64 条增加到 256 条，能否提高相同累计回答处理量或相同更新次数下的任务表现。OPD 已有前缀是否充足仍是待验证假设；两个设置均增加 batch，不能单独识别 batch、独立前缀覆盖与总计算量的作用。已完成的长程结果和论文曲线见[主报告](weighted-targets.md)。

累计训练回答条次定义为每次 optimizer update 的回答数之和，重复使用同一回答也计数；它不等于新生成回答数，也不等于累计 token 数。

| 设置                            | 每 outer 新回答 | 每次更新回答 | 每批复用 | outer | optimizer update | 累计训练回答条次 | 新生成回答 |
| ------------------------------- | --------------: | -----------: | -------: | ----: | ---------------: | ---------------: | ---------: |
| E57，已完成                     |              64 |           64 |       16 |   100 |             1600 |           102400 |       6400 |
| E81，A                          |             256 |          256 |       16 |    25 |              400 |           102400 |       6400 |
| E82，B                          |             256 |          256 |       16 |   100 |             1600 |           409600 |      25600 |
| TOP-D，论文配方换算至 100 outer |            4096 |          256 |        1 |   100 |             1600 |           409600 |     409600 |

A 与 E57 对齐累计回答条次和新生成量，用较少次、较大 batch 的更新；B 与 E57 对齐更新次数，回答条次为其 4 倍。B 对齐论文配方 100 outer 的回答条次和更新次数，新生成量仅为其 1/16。论文公开 Figure5 的 α0.1/0.2 曲线展示到 outer410，不能把表中 100 outer 写成其最终训练预算，来源与推断见[预算核对](weighted-targets.md#top-d-与本项目的预算和评测)。

A/B 的 epoch 上限分别为 1／2，以保证 B 能达到 100 outer；实际仍以各自 outer 预算停止。DAPO 原始题池有 17917 条，正式配置经 prompt cap1024 过滤后保留 17909 条，batch256 的单 epoch 提供 69 个完整 batch，不足 100 outer，B 会重访题目并重新生成回答；25600 是新回答数，不是不同题目数。

A/B 的目标与更新配方相同，默认 seed 相同，A 是 B 的名义前缀；分开运行不保证 vLLM 随机流与浮点计算逐位相同，不构成两个独立 seed。

## 实际预设

预设由[实验目录](../../../launchers/math_experiment.py)统一解析，经[共享启动器](../../../launchers/_run_single_teacher_fsdp2.sh)生成最终配置。环境变量和末尾 Hydra 参数仍可覆盖默认值；实际运行以 resolved config 为准。

| 项目     | E81／E82 默认设置                                                                           |
| -------- | ------------------------------------------------------------------------------------------- |
| 师生     | Qwen3-1.7B-Base ← Qwen3-30B-A3B-Instruct-2507                                               |
| 目标     | full-vocab MDPO，κ1／η0.1，token-mean，future.mode=none                                     |
| 资源     | 4 卡 shared_sync，teacher 无独立卡池；rollout TP1                                           |
| Batch    | train batch=actor mini-batch=256，n1，ppo_epochs=16                                         |
| 优化器   | AdamW，LR1e-6，constant，weight decay0.01，梯度裁剪 1                                       |
| 数值执行 | 继承 E61 的 precompute_fused、chunk512、显式 BF16；EOS 合并 `[151643,151645]`               |
| 训练输入 | DAPO，本地原生 Base 前缀，enable_thinking=null；prompt cap1024，response cap16384           |
| 训练采样 | temperature1，top-p1；不额外添加 TOP-D 的 boxed 指令                                        |
| Rollout  | 每引擎 max_num_seqs64，显存比例 0.8，token budget8192                                       |
| 评测     | 不做初始评测；每 5 outer 及终点，五基准 avg@16，temperature0.7、top-p0.9、response cap31744 |
| 生成终止 | 训练与评测默认开启精确重复停止 16–128/8；可分别显式关闭                                     |
| 保存     | save_freq=-1，不保存训练 checkpoint                                                         |

E57 历史运行使用 precompute 与原混合精度配置，A/B 使用已验收的 E61 融合执行路径；这项执行差异需要在结果解释中保留。2026-11-05 起 A/B 跟随公共重复停止默认值，训练和评测均开启；E57 历史自然终止曲线的停止协议不同。需要严格配对时，应补充相同停止协议的评测，或在启动 A/B 前显式设置 `TRAIN_REPETITION_DETECTION=false VAL_REPETITION_DETECTION=false`。E81 评测节点为 outer5/10/15/20/25，E82 为 outer5/10/…/100。两者都没有新的 step0 测量，可引用主报告中有来源标识的独立 Base 参考点。

## 并发与吞吐

每个 TP1 rollout 引擎的并发上限固定为 64，CUDA graph capture sizes 同步覆盖到 64。四卡处理一轮 256 条训练回答，均衡分配时每卡约 64 条；请求路由和长度会使实际分配变化。

验证的 `data.val_batch_size=null` 保持完整题集提交，630 道题各生成 16 次，共 10080 条回答。四卡训练后 checkpoint 的 32／64 并发实测支持选择 64，条件与数值见[性能记录](../../infra/performance/2026-10-22-训练与验证性能实测.md)；重复停止协议与正式 A/B 不同，不能直接作为正式训练吞吐预测。

[并发与利用率调优](../../infra/performance/2026-10-22-批量大小与GPU利用率调优.md)区分请求供给和引擎上限；[训练与验证实测](../../infra/performance/2026-10-22-训练与验证性能实测.md)记录了请求窗口与长尾的影响。A/B 还需记录 rollout token/s、验证耗时、抢占和峰值显存，再判断是否进一步增加上限。

<a id="training-clock-comparison"></a>

## E57、E81 与 E82 的训练时钟对照

下图按 outer、optimizer update 和累计 response presentations × optimizer updates 对照 E57、E81（A）与 E82（B）的训练 rollout reward。E57 每个 outer 使用 64 条回答，E81/E82 每个 outer 使用 256 条；三组每个 outer 均执行 16 次 optimizer update。图中是训练 rollout reward，不是公共 benchmark accuracy。E82 仍在运行，曲线只显示本次数据快照中已有的训练 outer。

![按 outer step 对照](figures/training-clocks-20261105/reward-by-outer.png)

![按 optimizer update 对照](figures/training-clocks-20261105/reward-by-optimizer-update.png)

![按累计 response presentations × optimizer updates 对照](figures/training-clocks-20261105/reward-by-sample-update.png)

累计回答处理条次按 `outer × responses per outer × 16` 计算；同一回答在每次复用时再次计数。它表示回答呈现次数，不等于唯一新生成回答数或 token 数。当前曲线点与原始文件校验和记录在[图表数据](figures/training-clocks-20261105/training-clocks.csv)和[数据来源及限制](figures/training-clocks-20261105/provenance.json)，可用同目录的[绘图脚本](figures/training-clocks-20261105/plot_training_clocks.py)重建。此快照中 E57 到 outer100，E81 的训练 reward 到 outer24（outer25 的汇总指标待终点评测完成后落盘），E82 到 outer14。

TOP-D 论文 Figure 5 曲线由审计数据[复原](data/topd-e57-all-curves-20261104/curves.csv)，图文件保存在[对应图组](figures/topd-e57-all-curves-20261104/)。论文只公开图形和预算说明，没有公开逐点运行日志，因此不把论文曲线与本地训练 rollout reward 未经校准地拼接。

## 执行状态与入口

2026-11-05，E81（A）运行在物理 GPU1–4，E82（B）由独立 supervisor 并行运行在 GPU0、5–7。两者均从原始 Base 独立开始，关闭初始评测，训练与验证开启重复停止。冻结源码为 `c996d634`；[正式预算与输入](../../../../../runs/topd-large-batch-20261105/plan.json)、[最终配置](../../../../../runs/topd-large-batch-20261105/configs/)、[队列状态](../../../../../runs/topd-large-batch-20261105/queue-state.json)保留正式执行设置。CPU 测试覆盖最终 Hydra 配置解析、四卡资源、batch／reuse／预算、并发、目标和评测终止协议，见[启动器测试](../../../../../tests/examples/test_mirroropd_math_launcher_on_cpu.py)。

截至 2026-11-05 22:17（UTC+8），E81 已完成 outer25／optimizer update 400，正在进行终点评测；22:16:41 的日志显示 191/630 个评测任务完成、0 个失败。E82 已完成 optimizer update 240，最近一个已记录的训练 rollout reward 是 outer14；outer15 验证正在运行，22:17:45 的日志显示 15/630 个评测任务完成、0 个失败。两组已记录的 optimizer update 均已应用。图表数据快照只包含已落盘的训练 rollout reward，因此 E82 曲线到 outer14。首个 outer 的启动验收凭据见[启动验收凭据](../../../../../runs/topd-large-batch-20261105/startup-proof.json)。

从仓库根目录检查默认配置；应先清理会覆盖预设的旧环境变量：

```bash
DRY_RUN=1 bash examples/mirroropd/run_math_opd_ablation_fsdp2.sh E81
DRY_RUN=1 bash examples/mirroropd/run_math_opd_ablation_fsdp2.sh E82
```

E81 由 `runs/topd-large-batch-20261105/run_queue.py` 在 GPU1–4 上运行；E82 由 `runs/topd-large-batch-20261105/e82_concurrent_supervisor.py` 在 GPU0、5–7 上并行运行。两个 supervisor 都通过 `resource_guard.py` 持有 GPU 资源；E82 退出时释放自己的 broker reservation，由 broker 恢复 idle occupancy。运行日志和离线 W&B 数据保存在各自独立目录；评测与训练共享各自的四卡资源，不另占 teacher GPU。长期结果回填主报告，小规模机制核对保留在独立专题。
