# 双模型推理验收工具

这些工具服务于[双融合实现与验收记录](../../docs/refactoring/2026-10-28-Rollout双融合实现与验收记录.md)。当前验收尚未全部完成；工具可运行不等于对应 gate 已通过。

所有 GPU 入口限定物理 GPU 0–3。模型和数据均从本地加载，输出目录必须是新目录。推荐通过 shell 入口运行；它们设置了本机隔离 Python 环境与 CUDA 兼容库。

## 正式性能矩阵

```bash
bash examples/mirroropd/benchmarks/fusion/run_fusion_benchmark_fsdp2.sh dense-base my-run
bash examples/mirroropd/benchmarks/fusion/run_fusion_benchmark_fsdp2.sh dense-reference my-run
bash examples/mirroropd/benchmarks/fusion/run_fusion_benchmark_fsdp2.sh moe-scan my-run
bash examples/mirroropd/benchmarks/fusion/run_fusion_benchmark_fsdp2.sh moe-base my-run
bash examples/mirroropd/benchmarks/fusion/run_fusion_benchmark_fsdp2.sh moe-reference my-run
bash examples/mirroropd/benchmarks/fusion/run_fusion_benchmark_fsdp2.sh dense-post my-run
bash examples/mirroropd/benchmarks/fusion/run_fusion_benchmark_fsdp2.sh moe-post my-run
```

正式自然生成使用 DAPO 的前 64 条 prompt、response cap7168、α=0.5、temperature=1、top-p=1、top-k=-1，以及 `[151643,151645]` raw stop set。初始 Base 与官方后训练 Qwen3-1.7B 分别测量；后者只是另一组工作负载，不是本次 OPD 训练所得 checkpoint，也不改变 `initial` 的绑定。两者均使用原始 Base 的 prompt 协议。

每个最终对照先预热一个完整批，再测三个完整批。每轮重建请求、双 KV 和 CUDA graph，保留模型权重、CUDA allocator 缓存与磁盘 JIT 缓存。`cold_load_seconds` 只计 checkpoint 加载及模型准备，不能当作整个 Python/Ray 进程的启动耗时。原 trainer 的完整生命周期成本由训练入口另测。

`--compact 0` 保持固定行；`1` 同时做长度分桶和即时删除结束行；`2` 做长度分桶并按 2 的幂次缩小行数。参考与优化主对照使用同一个值，避免把分组变化当作 kernel 收益。每个请求的 RNG 独立，分组不会重置 seed；浮点差异仍可能使自然采样轨迹分叉，因此必须同时报告有效 token 数、长度和墙钟。

`stable` 使用 BF16 主体权重、FP32 中间激活与输出、FP16 attention/KV；MoE router 的 BF16 参数精确扩展到 FP32。Triton GEMM 使用 TF32x3，并有 FP64 小矩阵 oracle。其余 precision 选项仅用于复现被拒绝的诊断候选。

## 数值、回放与 profiling

`verify_numerics.py` 在同一前缀上比较 cached decode 与完整前缀 forward，检查 α=0/0.5/1、最终概率、选中 log-prob 和 FP64 sampler oracle。`--reference-left-padding` 改变参考路径的物理 padding；`--replay-dir`、`--data`、`--request-ids` 可重放真实生成前缀。失败结果保留为 JSON，不能在测量后放宽容差。

`probe_history.py` 进一步从最初 prompt 开始逐 token 重放正式长批，保留累积 KV，在指定位置用完整前缀 forward 检查全词表 logits 与最终分布。参考 forward 逐行执行，以限制 MoE 的激活内存，并检查 batch padding 的影响。该诊断不计入性能收益。

```bash
bash examples/mirroropd/benchmarks/fusion/run_fusion_benchmark_fsdp2.sh replay my-run
bash examples/mirroropd/benchmarks/fusion/run_fusion_benchmark_fsdp2.sh profile my-run
bash examples/mirroropd/benchmarks/fusion/run_fusion_benchmark_fsdp2.sh layouts my-run
```

固定回放使用确定的十进制 token 前缀，关闭自然 EOS 和随机抽样，只用于同工作量的算子/KV 对照。它不能代替自然生成吞吐。

Profiler 单独运行，正式吞吐不启用 profiler。短 trace 记录 prefill 和早期 decode；late trace 在第 4096 步开始采集 32 步。CUDA event 的阶段 elapsed 与 Chrome trace 一并保存；存在跨 GPU 重叠时不能直接相加。`host_stage_times` 中的普通字段是 host 提交时间，`gpu_elapsed_*` 是额外 CUDA event 计时。Profiler 的晚期诊断允许短预热，正式性能测量不允许。

停止诊断记录两模型及融合分布在两个 tokenizer 原生 EOS 坐标的概率，并比较 raw stop-set union、各自原生 EOS 映到 STOP、双方共同合并 stop set。原生事件映射保留其它 token 坐标；当 α 位于两个端点之间时，两个不同原生 EOS 坐标会退出共同支持，端点只映射活跃模型的原生 EOS。所有诊断都条件化到实际 raw-fusion 轨迹，不表示事件策略重新生成后的长度。实际 token 始终来自 raw-token fusion；诊断仅支持 temperature=1 且不截断支持集。

`paired_reference.py` 是两个独立 GPU 上的布局实验，使用 GPU peer logits 传输。它没有生产 current 权重接收路径，不能仅凭该脚本的速度宣称已交付 paired 后端。

## 原 trainer 与默认路径

```bash
bash examples/mirroropd/benchmarks/fusion/run_fusion_lifecycle_fsdp2.sh my-short-run
bash examples/mirroropd/benchmarks/fusion/run_fusion_long_lifecycle_fsdp2.sh dense my-long-run
bash examples/mirroropd/benchmarks/fusion/run_fusion_long_lifecycle_fsdp2.sh moe my-long-run
bash examples/mirroropd/benchmarks/fusion/run_fusion_regression_fsdp2.sh /absolute/baseline-checkout /absolute/new-output
```

短矩阵检查实际更新、五个验证集、保存/恢复、initial 与线性系数端点；验证 n=1、短 response cap，仅用于生命周期验收。长矩阵在 batch64/cap7168 上检查相同切换并记录全部阶段成本。

默认路径回归在两个源码版本上保持相同权重、数据、采样与编译配置，第一批预热、后三批测量。它保持 CUDA graph，显式关闭确定性模式下不兼容的 Inductor combo-kernel benchmarking。`run_fusion_trajectory_cost_fsdp2.sh` 和 `singles` 分块仅用于来源成本参照；前者禁用 actor 更新，不能列为真实训练证据。

## 输出与统计

- `rank-*.jsonl`：配置、代码指纹、原生 EOS、prompt 长度、每批输出、选中 log-prob、阶段计时、显存、GPU util 与进程 RSS。
- `summary.json`：各配置的原始耗时、有效 token 数、tokens/s、中位数与范围。
- `trace-*.json.gz`、`profile-*.txt`：独立 profiler 窗口。
- `summarize.py`：汇总已完整结束的运行；缺少 replica 或请求数量不符时拒绝生成报告。

Tokens/s 只计有效 response token，不把两模型的 forward token 相加。RSS 汇总是各进程峰值之和，不是节点去重后的物理内存。后端不提供的 preemption/recompute 字段保留为 `unavailable`。
