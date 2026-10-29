# 双模型推理验收工具

这些工具服务于[双融合实现与验收记录](../../docs/infra/rollout/fusion/2026-10-28-Rollout双融合实现与验收记录.md)。当前验收尚未全部完成；工具可运行不等于对应 gate 已通过。

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

`benchmark.py --chat-template-kwargs '{"enable_thinking":false}'` 显式选择非思考模式提示协议；默认仍为 `{}`。原始问题不变，模板在 assistant 前缀后添加空思考段。每份记录保存模板参数、实际输入 token 的 SHA-256 和原始 prompt 行的 SHA-256；reference 与优化实现必须使用相同协议。以下补充入口先运行 batch8/cap512 检查，再运行 batch64/cap7168 的三次 reference 测量，以及固定行和活跃行压紧两种优化配置：

```bash
bash examples/mirroropd/benchmarks/fusion/run_fusion_short_workload_fsdp2.sh dense my-nonthinking-run
bash examples/mirroropd/benchmarks/fusion/run_fusion_short_workload_fsdp2.sh moe my-nonthinking-run
```

非思考模式是单独标注的负载协议，不能与 `{}` 协议直接计算系统加速比。是否覆盖短响应场景，仍须根据自然 EOS、长度分布和 cap 命中率判断；入口名称和 checkpoint 名称均不构成验收证据。

每个最终对照先预热一个完整批，再测三个完整批。每轮重建请求、双 KV 和 CUDA graph，保留模型权重、CUDA allocator 缓存与磁盘 JIT 缓存。`cold_load_seconds` 只计 checkpoint 加载及模型准备，不能当作整个 Python/Ray 进程的启动耗时。原 trainer 的完整生命周期成本由训练入口另测。

`--compact 0` 保持固定行；`1` 同时做长度分桶和即时删除结束行；`2` 做长度分桶并按 2 的幂次缩小行数。参考与优化主对照使用同一个值，避免把分组变化当作 kernel 收益。每个请求的 RNG 独立，分组不会重置 seed；浮点差异仍可能使自然采样轨迹分叉，因此必须同时报告有效 token 数、长度和墙钟。

`stable` 使用 BF16 主体权重、FP32 中间激活与输出、FP16 attention/KV；MoE router 的 BF16 参数精确扩展到 FP32。Triton GEMM 使用 TF32x3，并有 FP64 小矩阵 oracle。其余 precision 选项仅用于复现被拒绝的诊断候选。

Reference 使用 eager DynamicCache，优化路径使用 append-only KV 与 CUDA graph；两者均使用 `num_splits=1` 的 Flash KV 解码算术。`benchmark.py` 默认启用 Flash KV，但只有 `--preallocate-kv` 或 `--cuda-graph` 才启用 append-only 存储。`--no-flash-kv` 保留为原生 attention 诊断入口：旧原生缓存路径在后训练模型的真实长历史上超过预设误差门槛，不能继续作为最终统一基线。生产优化路径的采样、精度和 attention kernel 未因此改变。

## 数值、回放与 profiling

`verify_numerics.py` 在同一前缀上比较 cached decode 与完整前缀 forward，检查 α=0/0.5/1、最终概率、选中 log-prob 和 FP64 sampler oracle。`--reference-left-padding` 改变参考路径的物理 padding；`--replay-dir`、`--data`、`--request-ids` 可重放真实生成前缀。失败结果保留为 JSON，不能在测量后放宽容差。

`probe_history.py` 进一步从最初 prompt 开始逐 token 重放正式长批，保留累积 KV，在指定位置用完整前缀 forward 检查全词表 logits 与最终分布。参考 forward 逐行执行，以限制 MoE 的激活内存，并检查 batch padding 的影响。该诊断不计入性能收益。

历史重放默认检查优化执行器；`--executor reference` 检查 DynamicCache 与共享 Flash KV 的参考执行器。`--source-compact 0` 可重放固定行测量，默认值为 2。结果同时记录执行器源码指纹和源轨迹配置，两个执行器都必须通过原有容差。

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

## 原生 vLLM 单卡开发

原生后端选择 `rollout_policy.execution.backend=vllm_fusion`；旧配置继续使用 HF。已验证的短训练范围包括单 actor rank 的 Dense/MoE current/initial，以及双 actor rank 的 Dense current（一个 TP2 或两个 TP1 replica）；其他双卡组合仍在验收。Learner 限 Qwen3 dense，teacher 可为 dense/MoE。原生模型保留各自配置、head 和 KV；公共 sampler 保留原策略的 top-p 边界。正式支持范围以[实现与验收记录](../../docs/infra/rollout/fusion/2026-10-28-Rollout双融合实现与验收记录.md)为准。

本机入口 `run_fusion_local_fsdp2.sh` 默认使用物理 GPU 7；`FUSION_GPUS=6`、`7` 或 `6,7` 选择已分配的卡，逐卡核对 UUID、取得排他锁并拒绝既有计算任务。它不清理其他 Ray 作业。环境位于仓外 `envs/mirroropd-fusion`；在仓库根目录运行：

```bash
bash examples/mirroropd/benchmarks/fusion/run_fusion_setup_fsdp2.sh
../envs/mirroropd-fusion/bin/python examples/mirroropd/benchmarks/fusion/prepare_vllm_workload.py \
  --data-root ../data --student ../model/Qwen3-1.7B-Base \
  --out ../runs/vllm-fusion-20261029/lifecycle-data
bash examples/mirroropd/benchmarks/fusion/run_fusion_native_lifecycle_fsdp2.sh \
  dense current my-unique-lifecycle-run
```

数据准备与运行目录必须不存在，已有 `lifecycle-data/manifest.json` 时直接复用。`dense` 可换成 `moe`，`current` 可换成 `initial`；initial 的三轮生成覆盖 α=1、0.5、0。`FUSION_RESUME_FROM` 指定现有 checkpoint 时只执行恢复后的第三轮，运行名仍须唯一。该入口默认 batch2、cap32、五来源各 2 条验证样本。`FUSION_RESPONSE_CAP`、`FUSION_TRAIN_BATCH_SIZE`、`FUSION_KV_BYTES` 和 `FUSION_MAX_NUM_SEQS` 分别设置训练响应上限、全局 batch、每卡 KV 字节预算及活跃请求上限；`FUSION_TP=2` 配合 `FUSION_GPUS=6,7` 选择一个 TP2 replica，TP1 则创建两个 replica。增加 cap 后必须重新检查训练切换容量，短验证不用于任务准确率结论。

`probe_vllm.py` 执行独立生成与全参数 sleep/wake 哈希检查，并保存配置、源码指纹、输入 token、响应、选中概率和 所选 GPU 的资源曲线。哈希检查与首次启动不计入生成计时。`run_fusion_native_benchmark_fsdp2.sh <dense|moe> <唯一目录名> <eager|graph>` 固定 batch64/cap7168、α=0.5、预热和三次测量；`FUSION_MAX_NUM_SEQS`、`FUSION_KV_GIB` 控制并发与 KV 预算。默认值是待测候选，不能由脚本存在推断容量或性能已通过。

数值工具按问题分工：`probe_vllm_math.py` 从真实采样位置取得两侧 logits 和实际过滤后的分布，与独立 FP64 oracle 比较；`probe_native_components.py` 通过独立 native student、teacher 和组合模型重放同一 token 历史。后者使用专用诊断 worker 强制后续 token，不能用于生成吞吐或研究结果。它按请求身份记录检查点，并要求批宽校准确实执行目标批宽。两个数值探针均接受 `--tensor-parallel-size 2`，所有 rank 参与模型计算，仅输出 rank 保存词表结果；不能用 TP1 限值自动宣布 TP2 通过。

`compare_native_components.py --calibrate` 将独立 native 批宽差异的最大概率误差、总变差和选中 log-prob 误差分别加 2e-6 作为数值余量，写入不可覆盖的限值文件；候选检查要求不同的冻结工作负载。该范围仅用于对应精度、输入和执行设置，不能无限外推到长历史、graph 或其他 TP。超标样例保留，不自动放宽限值。完整训练的产物由 `audit_native_lifecycle.py` 检查成功更新次数、来源版本、验证覆盖、checkpoint 分片和所声明的 GPU 资源区间。

图候选使用原生 `FULL_DECODE_ONLY`，prefill 保持 eager，并关闭模型的 torch.compile 改写；两侧模型仍各自 forward。`FUSION_ENFORCE_EAGER=false` 可让生命周期入口选择该图候选，默认仍为 eager。α 在图外的 logits 融合阶段生效，端点目前只保证数学语义，不宣称释放或跳过另一侧模型。图模式的训练与正式性能必须单独验收。

固定前缀数值探针默认 `--arrival-mode synchronized`：通过所选版本的调度暂停接口完整提交请求后统一开始，要求每一步实际批宽与声明值一致。`asynchronous` 仅用于复现到达时序差异。组件比较覆盖所有批内行；只检查首行的早期结果不能代替全批验收，失败结果保留且限值不回写。

`probe_vllm_refit.py` 使用真实 checkpoint 和 IPC 检查 student 更新、teacher 哈希及输出不变、错误版本和损坏参数流；`probe_native_coordination.py` 用两个 actor rank 调用生产权重协调方法，在小型 Qwen3 checkpoint 上检查带 collective 的参数流排空和跨 rank 版本提交。前者不替代 trainer 协调，后者不替代真实配对的容量与训练检查。两者仅是验收工具。生成与组件探针保存原生累计后端指标，`summarize_native.py` 从相邻快照计算各批抢占次数；旧记录缺少计数时仍标为 `unavailable`。

`audit_native_lifecycle.py --previous-actor-checkpoint <恢复前 actor 目录>` 可同时核对恢复前后保存的 CPU 参数分片。它检查真实数值变化并分别报告 FP32 状态和 BF16 推理视图中的变化，不能仅以更新计数推进推断权重已变化。`probe_hf_reference.py` 提供独立 HF BF16/FlashAttention 全前缀参考；其分布距离与选中概率差异用于跨后端诊断，不沿用旧 HF 阈值作为 native 引擎门槛，该新增工具仍待 GPU 执行。

Qwen2 实验兼容正在开发：R1/JustRL/Polaris 的共同 token-ID 映射已核对，组合模型保留不同宽度的原生 head，只对生成视图对齐并施加共同合法支持。RKL scorer 的完整 teacher head 不变。相关 CPU 检查已通过，真实 Qwen2 GPU 生成、refit 与训练仍待验收。R1 的独立生成探针须显式传入 `--stop-token-ids 151643`，不能沿用 Qwen3 的默认停止集合。实验设置见[E20 五种前缀对照](../../docs/experiments/rollout-policy/2026-10-30-E20设置下的五种前缀对照.md)。

## E20 五种前缀入口

`run_fusion_prefix_study_fsdp2.sh <J|P> <S0|S|LM|TM|T> <唯一标签> [formal|profile|smoke]` 复用 E20 公共配方，固定 batch64、reuse16、训练 cap7168，关闭全部验证与模型保存。`formal` 为 40 outer，`profile` 默认 2 outer，可用 `PREFIX_PROFILE_OUTERS` 指定预跑轮数。`smoke` 单独使用 batch2/cap32、两轮及 reuse16 检查接线和版本推进，不能作为正式负载证据。配置与运行目录使用不同模式前缀；正式性能参数和十组训练尚未验收。

```bash
CONFIG_ONLY=1 FUSION_GPUS=6,7 bash examples/mirroropd/benchmarks/fusion/run_fusion_prefix_study_fsdp2.sh P LM review > /tmp/prefix-P-LM.yaml
../envs/mirroropd-fusion/bin/python examples/mirroropd/benchmarks/fusion/check_prefix_study_config.py /tmp/prefix-P-LM.yaml \
  --reference ../wandb-logs/MirrorOPD-verl/analysis/batch-transfer-20261027/E20/config.json \
  --pair P --arm LM --out /tmp/prefix-P-LM-audit.json
```

`CONFIG_ONLY=1` 只解析配置，不占用 GPU、不创建训练目录；`DRY_RUN=1` 只打印命令。实际执行仍由分配入口检查 UUID、GPU 占用及锁。当前入口保持单任务独占所选 GPU，多任务共置尚未实现，不能用内部 worker 模式绕过占用保护。

`FUSION_MAX_NUM_SEQS/FUSION_KV_BYTES/FUSION_TP` 控制 LM 执行；其他来源使用 TP1 replica。`PREFIX_PPO_TOKEN_BUDGET`、`PREFIX_TEACHER_TOKEN_BUDGET`、`PREFIX_ROLLOUT_SEQS`、`PREFIX_ROLLOUT_MEMORY_FRACTION`、`PREFIX_RAY_CPUS`、`PREFIX_AGENT_WORKERS`、`PREFIX_REWARD_WORKERS`、`PREFIX_STORAGE_UNITS`、`PREFIX_OBJECT_STORE_BYTES` 和 full-vocab 的 `PREFIX_FULL_KL_BACKEND/PREFIX_TOKEN_CHUNK/PREFIX_VOCAB_CHUNK` 是待实测的执行参数。默认值是候选，不是已确认的最优配置；学习超参数由入口固定。

融合接口始终返回有限、完整的选中 token 行为 log-prob，即使 full-vocab learner 关闭了 rollout log-prob 收集。缺失、长度不符或非有限值会拒绝该响应；greedy 的行为 log-prob 在完整性检查后写为 0。此修复已做 CPU 协议回归，仍需实际服务调用复验。

独立 native 推理 trace 使用 `probe_vllm.py --profile --alphas 0.5 --repeats 1 --skip-lifecycle-audit`：先完整预热一批，再按 `--profile-delay` 与 `--profile-iterations` 记录有限的 engine 迭代窗口。迭代编号不是每条请求的 token 位置，不能据此声称所有请求都处在同一历史长度。Profile 时间不计为正式吞吐验收。

训练 profile 模式可设置 `PREFIX_TORCH_TRACE=1`，默认在 outer2 记录 rank0 的 actor 进程 trace，并保留更新前的进程事件与前两次更新。`PREFIX_TRACE_OUTER/PREFIX_TRACE_UPDATES` 选择窗口；需要后段更新时使用 `PREFIX_TRACE_DISCRETE=true` 与 `PREFIX_TRACE_SKIP`，此时 trace 的范围是对应阶段而非完整 outer。Native fusion 推理使用独立推理 trace，不能把默认 rollout manager 的 trace 当成融合引擎证据。正式模式拒绝开启该 profiler。

`PREFIX_TEACHER_LAYER_OFFLOAD=true` 是已有 FSDP2 自动逐层 CPU offload 的对照开关，默认 false 保留 E20。它会把搬运移入 forward，可能在每个计算微批重复 H2D；应比较完整 teacher 阶段及 outer，不能只看独立 transfer 字段下降。以上 profiler 与 offload 对照入口已解析配置，GPU 效果仍待实际测量。
