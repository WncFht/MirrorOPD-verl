# Full-vocab 训练性能

本页维护固定输入管线、自然生成训练、reuse 与 full/sample 成本。计时、显存和 trace 口径见[测量方法](methodology.md)，head-loss 微基准见[算子性能](operators.md)。各节保留测量身份，当前代码或预设改变后不据此修改旧读数。

## 固定输入管线

四张 H800，student 为 DeepSeek-R1-Distill-Qwen-1.5B、teacher 为 JustRL-DeepSeek-1.5B。固定同一批 32 条数学回答，输入 token=163573、有效 loss token=161701，token chunk=512、每卡 token budget=16384、activation checkpointing 开启、lr=0。测量包含真实 FSDP2 worker 和上下卡，不含 Ray/TQ/vLLM。

冻结源码 `fc81ec6a9`，A 为 precompute，B 为 fused candidate；每次独立进程预热 3 步、测量 9 步，采用 A–B–B–A 顺序。每个 backend 共 18 个非预热样本，合并后取中位数；不把这些步当独立实验计算置信区间。[原始记录](data/final/confirmation)、[聚合数据](figures/final/confirmation.json)。

| 测量范围 | precompute | fused | 差异 |
| --- | ---: | ---: | --- |
| Actor 更新中位数 | 3.719 s | 3.480 s | 耗时减少 6.4% |
| Teacher→actor 中位数 | 5.080 s | 4.885 s | 耗时减少 3.9% |
| 固定输入 allocated 峰值 | 17.031 GiB | 约 16.600 GiB | 减少约 0.43 GiB |

两次融合进程的 actor 中位数分别为 3.472、3.499 s，两次基线为 3.774、3.663 s；单步仍有波动。数值对照的 loss 绝对差约 $6.71\times10^{-8}$、首轮梯度范数最大绝对差约 $1.06\times10^{-5}$。它们验证该输入与实现的相符程度，不证明长期训练效果。

![配对 actor 测量](figures/final/confirmation-actor_s.png)

额外的 checkpointing 关闭测试将 allocated 推至约 61.3 GiB、reserved 推至约 77.7 GiB，短窗口没有明确速度收益。该结果仅适用于当时的输入/token budget，不排除其他预算下的取舍；证据见[固定管线原始数据](data/review-004)。

## 单次更新的完整训练

冻结源码 `8132ca543`，四张 H800、同一 1.5B 师生、batch=mini-batch=64、n=1、每批更新一次、prompt/response cap=4096/7168、precompute、async write/prefetch、graph rollout。共 14 个训练 outer，步骤 4–13 为正常统计窗口；另有独立 5 步 Torch 运行，第 4 步采四 rank 训练与四 rollout replica trace。

依赖为 PyTorch 2.11.0+cu128、vLLM 0.25.1+cu129、Ray 2.56.1、Transformers 5.12.1。数据见 [review.json](e2e-4gpu/review.json)；完整输入、配置和 trace 位于 `outputs/reasoning-e2e-profile-20261023`。

| Driver 阶段 | 十步均值（s） | 步内占比 |
| --- | ---: | ---: |
| Rollout pipeline | 23.247 | 69.12% |
| Teacher 上下卡 | 0.742 | 2.20% |
| Teacher forward + hidden 发布 | 1.745 | 5.19% |
| Actor update | 6.866 | 20.41% |
| 权重同步 / rollout wake | 0.998 | 2.97% |
| 其余步内工作 | 0.035 | 0.10% |

步内均值为 33.632 s，中位数 32.721 s，范围 29.956–40.823 s；完整步间周期均值 33.716 s。初始化至第一步开始为 254.68 s，首步 89.95 s，最后一次 checkpoint 10.139 s；末尾 8 题×2 次、response cap=1024 的小验证耗时 4.117 s，不能代表全量数学验证。

![完整训练阶段](e2e-4gpu/figures/baseline-phases.png)

正常窗口生成 640 条回答、3956383 个输出 token，平均约 6182 token/回答。生成请求区间平均 19.94 s，而 driver rollout pipeline 为 23.25 s；第 6、9 步有约 8 s 的生成后评分 RPC 等待。63.59% 回答触及长度上限，最后 10% 请求集中结束，这组证据不支持优先把瓶颈归为 decode 长尾。

HTTP 采样观察到每引擎最多 16 个运行请求、KV 使用率最大约 5.49%；first-token 字段缺失。Rollout trace 只覆盖 128 次引擎 iteration，约 0.38–0.40 s 的局部窗口。训练 trace 的 host 时钟校准最大偏差 0.053 ms，且停止点在步末权重同步之前；空的同步 GPU 事件不表示同步耗时为零。

独立 profile 步的 actor wall time 为 7.284 s，四 rank compute 区间并集约 4.67–4.74 s，无 compute 重叠的 NCCL 约 0.21–0.33 s。仅从这些区间不能把其余时间全部标为浪费。Actor 日志 allocated 高水位为 14.14–15.07 GiB、reserved 为 15.48–16.34 GiB；没有该周期的 allocator snapshot，不能分解为精确的参数/激活/碎片占比。

14 次更新全部 applied，loss/梯度均有限，checkpoint 和小验证链路完成。Nsight 最终补采的队列已完成，但 [产物状态](../../../../../../outputs/reasoning-e2e-nsys-final-20261023/report-status.json)仍为未通过采集验收；本节 GPU 结论使用已审阅的 Torch trace，不把队列退出当成 Nsight 证据。

## 多次复用的完整训练

E44 Full-KL、E45 MDPO（kappa=0、eta=0.3）、E46 BPO（同 kappa/eta）各完成 4 outer×16 reuse、64 次成功更新。冻结源码为 `8c3dda8cb91f0a9778e32d78f3959ac9a79f8218`，四张 H800、同一 1.5B 师生、DAPO、seed=42、batch=mini-batch=64、n=1、prompt/response cap=4096/7168。

三组均使用 precompute_fused、head chunk=128、actor/teacher token budget=32768、FP32 参数与 AdamW、BF16 混合精度、activation checkpointing、hidden async write/prefetch。Mirror 开启 diagnostics；梯度累积 no-sync 关闭。Rollout 为四个 TP1 副本、graph、并发上限128、token budget=8192、显存比例0.65。验证与保存关闭，这不是正式 256 outer run 的完成报告。

Outer 1 单列预热，outer 2–3 为无 Torch profiler 的正常窗口，outer 4 在 rank 0 采 teacher/anchor 与前两次 actor 更新。以下数值只使用标明的窗口；每方法只有两个正常 outer，轨迹不同，不做同输入速度排名或置信区间。[摘要与检查](data/mirror-observe-20261026/summary.json)、[完整运行资料](../../../../../../outputs/mirror-training-observe-20261026/README.md)。

### 正常窗口的阶段与工作量

| 方法 | Outer（s） | Rollout（s） | Teacher 含搬运（s） | Actor（s） | Actor 内 anchor（s） | 同步（s） | Actor 占比 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E44 | 109.26 | 25.20 | 2.47 | 80.48 | — | 1.08 | 73.7% |
| E45 | 126.99 | 22.84 | 4.55 | 98.40 | 3.26 | 1.16 | 77.5% |
| E46 | 121.03 | 20.22 | 2.45 | 97.21 | 3.09 | 1.13 | 80.3% |

| 方法 | 平均新 response token | 平均输入 token | 每更新每 rank micro-batches：outer 2/3 | Actor 输入 token/s，含16次reuse |
| --- | ---: | ---: | --- | ---: |
| E44 | 366300 | 372548 | 3/3 | 74067 |
| E45 | 378890 | 385138 | 4/3 | 62625 |
| E46 | 374845 | 381094 | 4/3 | 62727 |

Actor token/s 的分母包含 anchor 和调用开销；它不是新生成速度，也没有消除 attention 对序列长度分布的依赖。E44 两轮 rollout pipeline 为30.21/20.19 s，生成请求区间都约19.58 s；E45 teacher forward 为4.47/1.71 s。波动保留在均值中，不能只选最快一轮。

### Actor 内部成本

下表先在每个 rank/outer 内合并同名 host span 的区间并集，再对4 rank×2 outer取均值；各行存在嵌套，不能求和重建 actor wall time。

| 每 rank 每 outer 的 host 区间（s） | E44 | E45 | E46 |
| --- | ---: | ---: | ---: |
| Forward/backward | 77.79 | 91.36 | 90.97 |
| 其中 head-loss | 13.61 | 20.60 | 21.11 |
| Optimizer，含 clipping/非有限检查 | 0.57 | 0.56 | 0.63 |
| Hidden read 显式等待 | 1.30 | 2.09 | 2.51 |
| Micro-batch H2D，含 anchor | 1.01 | 1.80 | 1.65 |
| Engine transfer | 1.53 | 3.91 | 2.67 |

Rank 0、前两次 actor 更新的 GPU kernel 工作量如下。根据 launch correlation 限定训练区间，排除前面的 teacher/anchor backbone；每行互斥归因，单位 ms。

| Head 内工作 | E44 | E45 | E46 |
| --- | ---: | ---: | ---: |
| Student 投影 | 300.9 | 301.3 | 301.0 |
| Teacher 投影；mirror 含归一化 | 292.8 | 476.9 | 477.4 |
| Anchor 投影与归一化 | — | 474.3 | 474.2 |
| Weighted target | — | 123.4 | 123.5 |
| Objective/诊断归约 | 188.2 | 270.2 | 372.1 |
| dH 累计 | 277.0 | 280.0 | 278.2 |
| dW 累计 | 1404.5 | 1419.3 | 1422.7 |
| 其他 head / backward apply | 54.0 | 44.9 | 44.0 |
| Head 合计 | 2517.4 | 3390.2 | 3493.2 |
| Head 占全部训练 kernel 工作量 | 22.3% | 28.2% | 28.9% |

dW 是 head 最大项；C=128 会频繁读写完整 FP32 累计缓冲，支持对更大 chunk 做配对试验。MDPO/BPO 的 target+归约仅占全部训练 kernel 工作量约3.3%/4.1%；继续只优化逐元素归约的整轮收益受限。约50%–56%的 kernel 工作量未被 full-vocab 标签归因，其中还含 NCCL 等，不能全部算作 backbone。

### 通信与显存

| Rank 0 前两次更新的区间（ms） | E44 | E45 | E46 |
| --- | ---: | ---: | ---: |
| Compute 并集 | 9232.4 | 10250.6 | 10323.9 |
| NCCL 并集 | 2002.0 | 1710.9 | 1684.9 |
| Compute 与 NCCL 重叠 | 1101.6 | 1222.4 | 773.2 |
| NCCL 无 compute 重叠 | 900.4 | 488.5 | 911.7 |

当前测量每个 micro-batch 同步梯度；启用 no-sync 的实际收益与未分片梯度内存尚需固定输入检查，不能按 micro-batch 数线性推算。该窗口 pageable H2D 为141/292/278 ms、约1.00/2.00/2.01 GiB，包含首次 frozen head 上卡，不能直接乘8外推整轮。

NVML 约1 Hz，按阶段边界各裁0.25 s；下表是采样峰值，可能漏掉短尖峰。

| 方法 | 正常 actor 整卡峰值（GiB） | Rollout 整卡峰值（GiB） | Actor GPU busy 均值 | 第4轮 rank-0 trace allocated 峰值（GiB） |
| --- | ---: | ---: | ---: | ---: |
| E44 | 19.56 | 54.04 | 93.6% | 13.53 |
| E45 | 20.07 | 54.11 | 89.3% | 14.05 |
| E46 | 19.87 | 54.11 | 90.0% | 14.03 |

Trace 峰值包含此前评分阶段，NVML 包含同卡其他分配，二者不能相减得到碎片。任务顺序执行、启动前遵守共享锁和空闲门槛，但没有持续记录所有 GPU PID，不能严格排除瞬时外部干扰。部分 rollout preemption 字段缺失，不据此宣称零抢占。

三组更新坐标1–64连续、全部applied，无非有限梯度或跳步，位置桶计数与目标重建通过。它们确认短程执行与观测完整，不证明目标单调下降或长期质量。下一项可判别对照统一见[性能入口](README.md#下一组有判别力的测量)。

## IO 与模型驻留

同步/异步 IO 的独立对照使用 `39d8be84a`、四卡 H800、eager rollout，各4步，第3步profile；两组生成 token 不相同。因此记录等待变化，不宣称整步 IO 的因果加速比例。[原始证据](data/review-005)。

| 主线程观测 | 同步（ms） | Async write + prefetch（ms） |
| --- | ---: | ---: |
| Hidden read wait | 228.934 | 68.034 |
| Hidden write wait | 未启用 | 0.077 |
| Hidden write flush | 未启用 | 53.465 |
| Teacher hidden D2H host span | 213.741 | 197.056 |

该组 PyTorch trace 未保留后台 read/write 主体，不能以零补齐；D2H host span 包含此前 CUDA 等待，不能全算 PCIe 传输。Graph 单次更新窗口中，teacher D2H约0.213 s、后台write约0.210 s、最终flush约0.056 s、后台read约0.205 s；这些嵌套区间不相加为关键路径成本，详见[已聚合观测](e2e-4gpu/review.json)。

固定管线 rank-0 snapshot 回放的 requested 峰值为16.822 GiB、reserved为20.670 GiB；它与四卡最大 allocated17.031 GiB不同。原始snapshot位于 `outputs/full-kl-profile-20261022/pipeline-baseline/memory-rank0.pickle`，图见[allocator轨迹](figures/review-005/memory-baseline.png)。

## Full-KL 与 sampled

下一组配对实验的矩阵、统一启动命令和固定前缀工具需求维护在[公平性能对照方案](../sampled-full/README.md)，尚未产生新的测量结果。

比较配置与全部训练成本的唯一数据表维护在[五组 Math 实验对比](../../../experiments/learning-signal/sampled-vs-full/2026-10-24-五组Math实验对比分析.md#7-时间成本显存和资源活动)。该组包括八卡 A100 上的 Full-KL E14 与 sampled vanilla E15，full 使用 precompute，二者学习率、生成轨迹和实际源码状态未完全控制。

该观察支持：full 的 actor 稍慢，但通过省去 old-log-prob/advantage、使用 hidden teacher 通路，普通 outer 更短；actor 进程 allocated 高水位较低，而 NVML 整卡峰值接近。它不构成当前 fused 与显式 token-IS 的固定输入 benchmark。复用多次的 actor 时间应同时报告每 outer 与每次更新，不能把开销摊薄解释为生成周期变快。

完整实验还包含大规模验证，其生成预算可超过训练；验证协议与成本分布继续引用原实验分析，避免在性能页重复维护 run 级报表。推理参数选择见[学生模型推理性能](../2026-10-27-学生模型推理性能与参数选择.md)。
