# 性能测量方法

本页统一 full-vocab 性能实验的测量范围、比较条件、显存口径和复算方法。数值结果分别维护在[算子性能](operators.md)与[训练性能](training.md)。

## 测量层级

| 层级 | 包含 | 不包含 | 回答的问题 |
| --- | --- | --- | --- |
| Head-loss | 给定 hidden/head 的投影、目标、归约、student hidden/head VJP | Backbone、TQ、optimizer、rollout | 数值 kernel 的计算与工作区代价 |
| 固定输入管线 | 真实模型、固定 token/mask、FSDP、teacher/actor、optimizer 与明确的上下卡范围 | 未接入的 Ray/TQ/vLLM | 优化能否传递到真实更新 |
| 自然生成 outer | Rollout、评分、teacher、actor、权重同步；按实际 timer 说明周期任务 | 初始化及 timer 外工作单列 | 训练配方的实际周期成本 |
| 完整任务 | 初始化、训练、验证、保存、清理 | 未记录的成本标为缺失 | 给定实验预算的墙钟消耗 |

`lr=0` 的固定输入实验仍可执行 optimizer 计算，目的是控制权重与轨迹；它不提供训练收敛证据。Head-loss 的耗时减少不能直接换算成完整任务的加速。

## 比较条件

固定输入 A/B 对照保持模型、token、mask、student/optimizer 初态、dtype、GPU 数、全局 batch 和目标函数相同，只改变要评估的实现或参数。更换 chunk 时需单列，不能把 chunk 改善算作融合的单项收益。MDPO/BPO 还需对齐 q、T、kappa、eta 和 diagnostics。

自然生成会改变回答长度、attention 工作量、micro-batch 数及停止原因。即使 seed 相同，也不保证两次运行产生相同前缀。自然生成对照报告工作量与阶段耗时，固定输入对照用于分辨实现本身的差异。

同一轮中的多个 inner update 共享轨迹与 frozen target，并非独立实验重复。统计时保留进程级重复、运行顺序和原始样本；不把同一进程内的多步误作独立重复来生成过窄的置信区间。

## 计时与归约

算子测量先完成预热和数值检查，再记录同步边界清楚的 wall time；CUDA event 可作为补充。编译、graph capture、首次 head 缓存与 allocator 申请可能落在启动或首轮，单独报告。详细 profiler 会扰动运行，带 profiler 的样本不混入正常速度统计。

固定输入实验采用交错 A–B–B–A 或成对 AB/BA 顺序，减少温度、时钟和后台负载的单向漂移。报告均值或中位数时说明选择、样本数和剔除规则，保留未裁剪样本。

训练周期至少区分三种时间：

- Driver timer 定义的步内区间；哪些 checkpoint/同步步骤在其中，以实际埋点为准。
- 本步起点到下一步起点的完整周期；最后一步没有下一起点时记为缺失。
- 整个进程墙钟，包括初始化与清理。

Host span 可能包含异步提交、等待及嵌套调用。Teacher transfer/forward、anchor、head、IO 等只有确认互斥后才能相加。Actor timer 已含 anchor 时，不再额外加 anchor。Full-vocab 在 forward 预计算 VJP，比较时按投影、归约、梯度计算分类，不从 forward/backward 标签推断省掉的工作。

设比较前后耗时分别为 $t_A,t_B$：

$$
\text{耗时减少比例}=1-\frac{t_B}{t_A},\qquad
\text{速度倍率}=\frac{t_A}{t_B}.
$$

固定工作量下，减少 20% 耗时等于 1.25 倍速度。自然生成的 token/s 同时记录输出 token 数；actor 的输入 token/s 与新生成 token/s 分开命名。Reuse 工作量可计入 actor token/s，但分母仍需说明是否包含 anchor 和上下卡。

## 显存口径

| 名称 | 范围 | 使用限制 |
| --- | --- | --- |
| Allocated peak | PyTorch allocator 已分配张量的峰值 | 说明 reset 位置、进程/rank 和测量窗口 |
| Extra allocated peak | 峰值减调用前 live input allocation | 适合算子工作区比较，不能与总 allocated 混用 |
| Reserved peak | PyTorch allocator 保留的内存 | 包含空闲缓存，不等于活跃 tensor |
| Requested bytes | Snapshot 中请求字节的回放 | 可能不含 allocator 对齐，范围取决于 snapshot |
| NVML 整卡采样峰值 | 同卡所有进程及外部库分配 | 采样可能漏掉短峰值，不能归因于单个 kernel |
| 日志累计高水位 | 进程自启动以来的累计计数 | 不能当成当前 actor 或 teacher 阶段峰值 |

显存统一用 GiB/MiB，分别为 $2^{30}$/$2^{20}$ bytes；容量公式不等于实测峰值。推理引擎预留的 KV 池可能决定整卡最大值，而 actor 优化只降低训练阶段峰值。禁止用两种不同口径的差值直接估计碎片或可用训练容量。

阶段 NVML 数据按实际时间窗切分，说明采样频率、边界裁剪与 rank 聚合方式。GPU busy 不是算力效率，MFU 也不自动涵盖额外 teacher/anchor head 的全部计算。

## Trace 归因

先通过 CUDA correlation 找到 CPU launch，再将 kernel 分配到最深的语义 span。仅按 autograd Function 起点归类可能把嵌套 Triton 工作归错层；未知项保留为 unattributed。

Kernel duration 求和表示工作量，按设备计算时间区间并集才得到忙碌区间。Compute 与 NCCL 的交集表示时间重叠；未重叠的 NCCL 仍不等于经过依赖分析证明的关键路径等待。跨 rank 时长不相加为整步墙钟。

多进程 host span 需要同一时钟域。Torch 的时间基准单独对齐，先验证校准误差，再叠加 host/NVML 数据。未覆盖的阶段、缺失 first-token 字段及后台线程事件用缺失语义，不填零。

已有分析工具包括 [analyze.py](../../../../benchmarks/profile/analyze.py) 的区间并集、[semantic_trace.py](../../../../benchmarks/profile/semantic_trace.py) 的 kernel 归因、[memory_timeline.py](../../../../benchmarks/profile/memory_timeline.py) 的 allocator 回放，以及 [analyze_mirror_training.py](../../../../benchmarks/profile/analyze_mirror_training.py) 的按 update 训练归因。

## 正确性门控

比较 loss、student hidden/head VJP、frozen q/T 无梯度，以及非单位 upstream 缩放。小规模使用独立 FP64/dense 参考，GPU BF16 使用实际生产投影与归约。Self-target、近目标残差、额外 teacher 输出行、空局部 mask 和不可整除 tile 分别检查。

算子数值相符后再核对 FSDP head hooks、全局 token 分母、参数更新和 checkpoint。测试通过的条件要随结果记录；一般分布下的绝对误差门槛不替代近目标相对梯度检查。测试入口见[验证矩阵](../../numerics/full-vocab/validation.md)。

## 证据组织

每个实验使用独立运行目录，至少记录源码 commit；若基线来自工作区快照，明确快照位置与 import 隔离方式。保留解析配置、实际命令、输入/有效 token 数、硬件/依赖、预热与采集规则、原始样本、退出状态和数值检查结果。失败及未完成任务保留事实，不写成成功测量。

小型审阅数据和图存入正文链接的 `data/`、`figures/`；大 trace、模型、完整日志留在对应运行目录。目录中的日期标识测量批次，正文按主题组织，不维护重复的日进度页。新运行摘要由 [report.py](../../../../benchmarks/profile/report.py) 写入运行目录的 `report.md`；发布时进入 `data/<run>/report.md`，不会覆盖主题正文。

## 复算现有结果

以下命令从仓库根目录执行，只读已有训练产物，并将复算结果写到临时目录；不会启动训练。`PROFILE_PY` 指已有的原生分析环境，其他机器按实际环境替换。原始数据需要单独传输，源码仓库不保证包含大 trace。

```bash
PROFILE_PY=/root/.local/share/mirroropd/venvs/profile-cu128/bin/python
PROFILE_DOC=examples/mirroropd/docs/infra/performance/full-vocab
RECHECK_DIR=/tmp/full-kl-performance-recheck
mkdir -p "$RECHECK_DIR"

"$PROFILE_PY" -m examples.mirroropd.benchmarks.profile.analyze_mirror_training \
  --root outputs/mirror-training-observe-20261026 \
  --output "$RECHECK_DIR/mirror-analysis.json" \
  --summary "$RECHECK_DIR/mirror-summary.json"

"$PROFILE_PY" -m examples.mirroropd.benchmarks.profile.e2e_4gpu.collect \
  --root outputs/reasoning-e2e-profile-20261023 \
  --output "$RECHECK_DIR/e2e-data.json"
"$PROFILE_PY" -m examples.mirroropd.benchmarks.profile.e2e_4gpu.analyze_cycle \
  --root outputs/reasoning-e2e-profile-20261023 \
  --data "$RECHECK_DIR/e2e-data.json" \
  --output "$RECHECK_DIR/e2e-review.json"
```

交错复测与图表的数据接口见 [summarize_confirmation.py](../../../../benchmarks/profile/summarize_confirmation.py)、[plot_step_breakdown.py](../../../../benchmarks/profile/plot_step_breakdown.py)、[plot_review.py](../../../../benchmarks/profile/e2e_4gpu/plot_review.py)。绘图使用既有 figure-viz 样式；图中只能出现对应输入文件支持的读数。

## 发起新测量

先选择冻结源码、输入和单项变量，再建立新的运行目录。工具入口如下；实际参数以各脚本 `--help` 为准。

| 工作 | 工具 |
| --- | --- |
| Endpoint synthetic head-loss | [benchmark.py](../../../../benchmarks/profile/benchmark.py) |
| MDPO/BPO 优化前后固定输入 A/B | [benchmark_mirror.py](../../../../benchmarks/profile/benchmark_mirror.py) |
| 真实模型固定输入管线 | [pipeline.py](../../../../benchmarks/profile/pipeline.py) |
| 分层实验队列 | [run_study.py](../../../../benchmarks/profile/run_study.py) |
| 完整 graph 训练计划与执行 | [prepare.py](../../../../benchmarks/profile/e2e_4gpu/prepare.py)、[run_plan.py](../../../../benchmarks/profile/run_plan.py) |
| CPU 聚合快照刷新 | [refresh_report.py](../../../../benchmarks/profile/e2e_4gpu/refresh_report.py) |

本机测量限获分配的物理 GPU 0–3，沿用 `/tmp/mirroropd-gpus-0-1-2-3.lock`。启动前确认无计算进程、每卡显存低于 1 GiB、利用率低于 10%，持续 120 秒；共享锁不替代运行期间的干扰记录。队列停止使用所属运行目录的 `STOP`，不终止其他任务。

新实验先通过数值门控、完成充分预热，再做无 profiler 配对计时；最后补少量真实训练周期验证。默认关闭自动会话轮询；测量工具与人工审阅各自负责运行事实和结论。
