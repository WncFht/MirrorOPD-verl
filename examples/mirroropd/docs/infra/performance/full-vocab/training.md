# Full-vocab 训练性能

本页维护固定输入管线、自然生成训练、reuse 与 full/sample 成本。计时、显存和 trace 口径见[测量方法](methodology.md)，head-loss 微基准见[算子性能](operators.md)。各节保留测量身份，当前代码或预设改变后不据此修改旧读数。

## 固定输入管线

四张 H800，student 为 DeepSeek-R1-Distill-Qwen-1.5B、teacher 为 JustRL-DeepSeek-1.5B。固定同一批 32 条数学回答，输入 token=163573、有效 loss token=161701，token chunk=512、每卡 token budget=16384、activation checkpointing 开启、lr=0。测量包含真实 FSDP2 worker 和上下卡，不含 Ray/TQ/vLLM。

冻结源码 `fc81ec6a9`，A 为 precompute，B 为 fused candidate；每次独立进程预热 3 步、测量 9 步，采用 A–B–B–A 顺序。每个 backend 共 18 个非预热样本，合并后取中位数；不把这些步当独立实验计算置信区间。[原始记录](data/final/confirmation)、[聚合数据](figures/final/confirmation.json)。

| 测量范围                | precompute |         fused | 差异            |
| ----------------------- | ---------: | ------------: | --------------- |
| Actor 更新中位数        |    3.719 s |       3.480 s | 耗时减少 6.4%   |
| Teacher→actor 中位数    |    5.080 s |       4.885 s | 耗时减少 3.9%   |
| 固定输入 allocated 峰值 | 17.031 GiB | 约 16.600 GiB | 减少约 0.43 GiB |

两次融合进程的 actor 中位数分别为 3.472、3.499 s，两次基线为 3.774、3.663 s；单步仍有波动。数值对照的 loss 绝对差约 $6.71\times10^{-8}$、首轮梯度范数最大绝对差约 $1.06\times10^{-5}$。它们验证该输入与实现的相符程度，不证明长期训练效果。

![配对 actor 测量](figures/final/confirmation-actor_s.png)

额外的 checkpointing 关闭测试将 allocated 推至约 61.3 GiB、reserved 推至约 77.7 GiB，短窗口没有明确速度收益。该结果仅适用于当时的输入/token budget，不排除其他预算下的取舍；证据见[固定管线原始数据](data/review-004)。

## 单次更新的完整训练

冻结源码 `8132ca543`，四张 H800、同一 1.5B 师生、batch=mini-batch=64、n=1、每批更新一次、prompt/response cap=4096/7168、precompute、async write/prefetch、graph rollout。共 14 个训练 outer，步骤 4–13 为正常统计窗口；另有独立 5 步 Torch 运行，第 4 步采四 rank 训练与四 rollout replica trace。

依赖为 PyTorch 2.11.0+cu128、vLLM 0.25.1+cu129、Ray 2.56.1、Transformers 5.12.1。数据见 [review.json](e2e-4gpu/review.json)；完整输入、配置和 trace 位于 `outputs/reasoning-e2e-profile-20261023`。

| Driver 阶段                   | 十步均值（s） | 步内占比 |
| ----------------------------- | ------------: | -------: |
| Rollout pipeline              |        23.247 |   69.12% |
| Teacher 上下卡                |         0.742 |    2.20% |
| Teacher forward + hidden 发布 |         1.745 |    5.19% |
| Actor update                  |         6.866 |   20.41% |
| 权重同步 / rollout wake       |         0.998 |    2.97% |
| 其余步内工作                  |         0.035 |    0.10% |

步内均值为 33.632 s，中位数 32.721 s，范围 29.956–40.823 s；完整步间周期均值 33.716 s。初始化至第一步开始为 254.68 s，首步 89.95 s，最后一次 checkpoint 10.139 s；末尾 8 题×2 次、response cap=1024 的小验证耗时 4.117 s，不能代表全量数学验证。

![完整训练阶段](e2e-4gpu/figures/baseline-phases.png)

正常窗口生成 640 条回答、3956383 个输出 token，平均约 6182 token/回答。生成请求区间平均 19.94 s，而 driver rollout pipeline 为 23.25 s；第 6、9 步有约 8 s 的生成后评分 RPC 等待。63.59% 回答触及长度上限，最后 10% 请求集中结束，这组证据不支持优先把瓶颈归为 decode 长尾。

HTTP 采样观察到每引擎最多 16 个运行请求、KV 使用率最大约 5.49%；first-token 字段缺失。Rollout trace 只覆盖 128 次引擎 iteration，约 0.38–0.40 s 的局部窗口。训练 trace 的 host 时钟校准最大偏差 0.053 ms，且停止点在步末权重同步之前；空的同步 GPU 事件不表示同步耗时为零。

独立 profile 步的 actor wall time 为 7.284 s，四 rank compute 区间并集约 4.67–4.74 s，无 compute 重叠的 NCCL 约 0.21–0.33 s。仅从这些区间不能把其余时间全部标为浪费。Actor 日志 allocated 高水位为 14.14–15.07 GiB、reserved 为 15.48–16.34 GiB；没有该周期的 allocator snapshot，不能分解为精确的参数/激活/碎片占比。

14 次更新全部 applied，loss/梯度均有限，checkpoint 和小验证链路完成。Nsight 最终补采的队列已完成，但 [产物状态](../../../../../../outputs/reasoning-e2e-nsys-final-20261023/report-status.json)仍为未通过采集验收；本节 GPU 结论使用已审阅的 Torch trace，不把队列退出当成 Nsight 证据。

## 多次复用的完整训练

E44 Full-KL、E45 MDPO（kappa=0、eta=0.3）、E46 BPO（同 kappa/eta）各完成 4 outer×16 reuse、64 次成功更新。冻结源码为 `8c3dda8cb91f0a9778e32d78f3959ac9a79f8218`，四张 H800、同一 1.5B 师生、DAPO、seed=42、batch=mini-batch=64、n=1、prompt/response cap=4096/7168。

三组均使用 precompute_fused、head chunk=128、actor/teacher token budget=32768、FP32 参数与 AdamW、BF16 混合精度、activation checkpointing、hidden async write/prefetch。Mirror 开启 diagnostics；梯度累积 no-sync 关闭。Rollout 为四个 TP1 副本、graph、并发上限 128、token budget=8192、显存比例 0.65。验证与保存关闭，这不是正式 256 outer run 的完成报告。

Outer 1 单列预热，outer 2–3 为无 Torch profiler 的正常窗口，outer 4 在 rank 0 采 teacher/anchor 与前两次 actor 更新。以下数值只使用标明的窗口；每方法只有两个正常 outer，轨迹不同，不做同输入速度排名或置信区间。[摘要与检查](data/mirror-observe-20261026/summary.json)、[完整运行资料](../../../../../../outputs/mirror-training-observe-20261026/README.md)。

### 正常窗口的阶段与工作量

| 方法 | Outer（s） | Rollout（s） | Teacher 含搬运（s） | Actor（s） | Actor 内 anchor（s） | 同步（s） | Actor 占比 |
| ---- | ---------: | -----------: | ------------------: | ---------: | -------------------: | --------: | ---------: |
| E44  |     109.26 |        25.20 |                2.47 |      80.48 |                    — |      1.08 |      73.7% |
| E45  |     126.99 |        22.84 |                4.55 |      98.40 |                 3.26 |      1.16 |      77.5% |
| E46  |     121.03 |        20.22 |                2.45 |      97.21 |                 3.09 |      1.13 |      80.3% |

| 方法 | 平均新 response token | 平均输入 token | 每更新每 rank micro-batches：outer 2/3 | Actor 输入 token/s，含16次reuse |
| ---- | --------------------: | -------------: | -------------------------------------- | ------------------------------: |
| E44  |                366300 |         372548 | 3/3                                    |                           74067 |
| E45  |                378890 |         385138 | 4/3                                    |                           62625 |
| E46  |                374845 |         381094 | 4/3                                    |                           62727 |

Actor token/s 的分母包含 anchor 和调用开销；它不是新生成速度，也没有消除 attention 对序列长度分布的依赖。E44 两轮 rollout pipeline 为 30.21/20.19 s，生成请求区间都约 19.58 s；E45 teacher forward 为 4.47/1.71 s。波动保留在均值中，不能只选最快一轮。

### Actor 内部成本

下表先在每个 rank/outer 内合并同名 host span 的区间并集，再对 4 rank×2 outer 取均值；各行存在嵌套，不能求和重建 actor wall time。

| 每 rank 每 outer 的 host 区间（s） |   E44 |   E45 |   E46 |
| ---------------------------------- | ----: | ----: | ----: |
| Forward/backward                   | 77.79 | 91.36 | 90.97 |
| 其中 head-loss                     | 13.61 | 20.60 | 21.11 |
| Optimizer，含 clipping/非有限检查  |  0.57 |  0.56 |  0.63 |
| Hidden read 显式等待               |  1.30 |  2.09 |  2.51 |
| Micro-batch H2D，含 anchor         |  1.01 |  1.80 |  1.65 |
| Engine transfer                    |  1.53 |  3.91 |  2.67 |

Rank 0、前两次 actor 更新的 GPU kernel 工作量如下。根据 launch correlation 限定训练区间，排除前面的 teacher/anchor backbone；每行互斥归因，单位 ms。

| Head 内工作                   |    E44 |    E45 |    E46 |
| ----------------------------- | -----: | -----: | -----: |
| Student 投影                  |  300.9 |  301.3 |  301.0 |
| Teacher 投影；mirror 含归一化 |  292.8 |  476.9 |  477.4 |
| Anchor 投影与归一化           |      — |  474.3 |  474.2 |
| Weighted target               |      — |  123.4 |  123.5 |
| Objective/诊断归约            |  188.2 |  270.2 |  372.1 |
| dH 累计                       |  277.0 |  280.0 |  278.2 |
| dW 累计                       | 1404.5 | 1419.3 | 1422.7 |
| 其他 head / backward apply    |   54.0 |   44.9 |   44.0 |
| Head 合计                     | 2517.4 | 3390.2 | 3493.2 |
| Head 占全部训练 kernel 工作量 |  22.3% |  28.2% |  28.9% |

dW 是 head 最大项；C=128 会频繁读写完整 FP32 累计缓冲，支持对更大 chunk 做配对试验。MDPO/BPO 的 target+ 归约仅占全部训练 kernel 工作量约 3.3%/4.1%；继续只优化逐元素归约的整轮收益受限。约 50%–56% 的 kernel 工作量未被 full-vocab 标签归因，其中还含 NCCL 等，不能全部算作 backbone。

### 通信与显存

| Rank 0 前两次更新的区间（ms） |    E44 |     E45 |     E46 |
| ----------------------------- | -----: | ------: | ------: |
| Compute 并集                  | 9232.4 | 10250.6 | 10323.9 |
| NCCL 并集                     | 2002.0 |  1710.9 |  1684.9 |
| Compute 与 NCCL 重叠          | 1101.6 |  1222.4 |   773.2 |
| NCCL 无 compute 重叠          |  900.4 |   488.5 |   911.7 |

当前测量每个 micro-batch 同步梯度；启用 no-sync 的实际收益与未分片梯度内存尚需固定输入检查，不能按 micro-batch 数线性推算。该窗口 pageable H2D 为 141/292/278 ms、约 1.00/2.00/2.01 GiB，包含首次 frozen head 上卡，不能直接乘 8 外推整轮。

NVML 约 1 Hz，按阶段边界各裁 0.25 s；下表是采样峰值，可能漏掉短尖峰。

| 方法 | 正常 actor 整卡峰值（GiB） | Rollout 整卡峰值（GiB） | Actor GPU busy 均值 | 第 4 轮 rank-0 trace allocated 峰值（GiB） |
| ---- | -------------------------: | ----------------------: | ------------------: | -----------------------------------------: |
| E44  |                      19.56 |                   54.04 |               93.6% |                                      13.53 |
| E45  |                      20.07 |                   54.11 |               89.3% |                                      14.05 |
| E46  |                      19.87 |                   54.11 |               90.0% |                                      14.03 |

Trace 峰值包含此前评分阶段，NVML 包含同卡其他分配，二者不能相减得到碎片。任务顺序执行、启动前遵守共享锁和空闲门槛，但没有持续记录所有 GPU PID，不能严格排除瞬时外部干扰。部分 rollout preemption 字段缺失，不据此宣称零抢占。

三组更新坐标 1–64 连续、全部 applied，无非有限梯度或跳步，位置桶计数与目标重建通过。它们确认短程执行与观测完整，不证明目标单调下降或长期质量。下一项可判别对照统一见[性能入口](README.md#下一组有判别力的测量)。

## IO 与模型驻留

同步/异步 IO 的独立对照使用 `39d8be84a`、四卡 H800、eager rollout，各 4 步，第 3 步 profile；两组生成 token 不相同。因此记录等待变化，不宣称整步 IO 的因果加速比例。[原始证据](data/review-005)。

| 主线程观测                   | 同步（ms） | Async write + prefetch（ms） |
| ---------------------------- | ---------: | ---------------------------: |
| Hidden read wait             |    228.934 |                       68.034 |
| Hidden write wait            |     未启用 |                        0.077 |
| Hidden write flush           |     未启用 |                       53.465 |
| Teacher hidden D2H host span |    213.741 |                      197.056 |

该组 PyTorch trace 未保留后台 read/write 主体，不能以零补齐；D2H host span 包含此前 CUDA 等待，不能全算 PCIe 传输。Graph 单次更新窗口中，teacher D2H 约 0.213 s、后台 write 约 0.210 s、最终 flush 约 0.056 s、后台 read 约 0.205 s；这些嵌套区间不相加为关键路径成本，详见[已聚合观测](e2e-4gpu/review.json)。

固定管线 rank-0 snapshot 回放的 requested 峰值为 16.822 GiB、reserved 为 20.670 GiB；它与四卡最大 allocated17.031 GiB 不同。原始 snapshot 位于 `outputs/full-kl-profile-20261022/pipeline-baseline/memory-rank0.pickle`，图见[allocator 轨迹](figures/review-005/memory-baseline.png)。

## Full-KL 与 sampled

下一组配对实验的矩阵、统一启动命令和固定前缀工具需求维护在[公平性能对照方案](../sampled-full/README.md)，尚未产生新的测量结果。

比较配置与全部训练成本的唯一数据表维护在[五组 Math 实验对比](../../../experiments/learning-signal/sampled-vs-full/2026-10-24-五组Math实验对比分析.md#7-时间成本显存和资源活动)。该组包括八卡 A100 上的 Full-KL E14 与 sampled vanilla E15，full 使用 precompute，二者学习率、生成轨迹和实际源码状态未完全控制。

该观察支持：full 的 actor 稍慢，但通过省去 old-log-prob/advantage、使用 hidden teacher 通路，普通 outer 更短；actor 进程 allocated 高水位较低，而 NVML 整卡峰值接近。它不构成当前 fused 与显式 token-IS 的固定输入 benchmark。复用多次的 actor 时间应同时报告每 outer 与每次更新，不能把开销摊薄解释为生成周期变快。

完整实验还包含大规模验证，其生成预算可超过训练；验证协议与成本分布继续引用原实验分析，避免在性能页重复维护 run 级报表。推理参数选择见[学生模型推理性能](../2026-10-27-学生模型推理性能与参数选择.md)。

## 无 teacher 缓存的批量 RKL 评分实测

2026-10-31，在物理 GPU5 的单张 H800 上测量固定 Qwen3-1.7B student 配 Qwen3-1.7B／4B／8B teacher 的即时评分。每配置 128 条合成 token 序列，prompt256，response 分别 2,048／7,168；microbatch1、词表 chunk256。student 权重 FP32、teacher 权重 BF16，骨干执行 BF16 autocast 和 FA2，输出层 FP32 且 TF32 关闭。每配置预热两条，再完整评分 128 条；每个配置测一次完整批，不报告为多次重复的中位数。

包含 student 前向、teacher 前向、两者词表投影、log-softmax、full-vocab RKL 及归约；不包含模型加载、rollout、磁盘读取、optimizer、分布式通信或训练／验证权重切换。不持久缓存 teacher hidden 或 log-prob，单条评分工作张量计算后释放。RKL-only 实现与仓库 `native_metrics` 在每配置预热的首个 chunk 做了数值一致性检查并通过；这不等于完整分布式实现验收。合成输入只用于测执行成本，RKL 数值不用于科学结论。

| Teacher    | Response token／条 | Student 前向／秒 | Teacher 前向／秒 | 输出层与 RKL／秒 | 整批／秒 | 峰值 allocated／GiB |
| ---------- | ------------------ | ---------------- | ---------------- | ---------------- | -------- | ------------------- |
| Qwen3-1.7B | 2048               | 4.80             | 3.92             | 7.36             | 16.10    | 11.85               |
| Qwen3-1.7B | 7168               | 13.57            | 12.15            | 25.71            | 51.44    | 11.91               |
| Qwen3-4B   | 2048               | 4.79             | 7.74             | 8.15             | 20.68    | 16.43               |
| Qwen3-4B   | 7168               | 13.69            | 25.81            | 28.99            | 68.50    | 16.49               |
| Qwen3-8B   | 2048               | 4.79             | 10.91            | 10.69            | 26.40    | 25.07               |
| Qwen3-8B   | 7168               | 13.58            | 36.35            | 37.43            | 87.36    | 25.14               |

完整原始读出见[result.json](data/rkl-throughput-20261031/result.json)。复算脚本位于仓库同级 `wandb-logs/MirrorOPD-verl/analysis/rkl-throughput-20261031/probe.py`。此结果是未做 batch 吞吐调优的单卡前向评分基线，不能外推为任意 student 尺寸或原生分布式训练中断点的实测耗时。

新协议每 outer 评 128 条当前来源留出和 128 条动态历史，两者均 cap7,168。若全部达到 cap，按本次吞吐线性估计，每 outer 分别约 103／137／175 秒，50outer 分别约 86／114／146 分钟。公共 benchmark cap16,384 与生成时间另计，未在此测试；实际短回答会改变开销，prompt 长度、模型共驻和训练切换也需重新验收。

仓库已有 `research/backends/hf_rkl.py::score`，按 microbatch 与 token chunk 评分，但它接收预先准备的 teacher hidden states，并额外计算 JS、TV、熵等。原生 verl 也有 full-vocab 训练算子、forward-only 执行与并行引擎；尚未接通本轮要求的仅 token 输入、即时双模型评分、动态历史集合和每 outer 评估调度。这次脚本复用了模型 hidden 提取与指标校验，不是已交付的生产评分入口。

## 真实轨迹 RKL 瓶颈诊断与四卡测试状态

2026-10-31 请求四卡串行共享与固定 2＋2 并行的完整切换测试。资源检查仅确认物理 GPU4／5 原为 gpu-occupy，GPU0–3 仍属 scientific-expanded-v1 活动实验；未停止这些模型任务，四卡对比未启动。为继续定位计算瓶颈，仅回收 GPU4 执行下述单卡诊断，不能将其作为四卡加载／卸载／恢复基线。

输入为已有 Qwen3 thinking 银行的 128 条原始轨迹，response 总计 131,072 token（每条 1,024），prompt 共 12,588 token；未重新 tokenize 或重复拼接。student Qwen3-1.7B FP32、teacher Qwen3-8B BF16，骨干 BF16 autocast／FA2，输出层 FP32、TF32 关闭，microbatch1。每个 chunk 预热 2 条后测 3 次完整 128 条，单独对 4 条轨迹采集 CPU／CUDA trace。各分块结果与 chunk256 比较，单题累计 RKL 最大绝对差小于 5.1e-6。

| Token chunk | 128 条墙钟中位数／秒 | 峰值 allocated／GiB |
| ----------- | -------------------- | ------------------- |
| 128         | 15.190               | 24.543              |
| 256         | 14.696               | 25.050              |
| 512         | 14.609               | 26.064              |
| 1024        | 14.580               | 28.096              |

chunk256 中位数对应那次测量的顺序 CUDA event 阶段时间如下；它是阶段 elapsed，不是所有 kernel duration 的简单相加。详细 profiler 运行与无 profiler 计时分开，不把 profiler 中的父子事件重复求和。

| 阶段                   | 秒    | 阶段合计占比 |
| ---------------------- | ----- | ------------ |
| student_backbone       | 3.603 | 24.5%        |
| teacher_backbone       | 5.839 | 39.8%        |
| student_projection     | 1.603 | 10.9%        |
| teacher_projection     | 3.168 | 21.6%        |
| log_softmax_pair       | 0.218 | 1.5%         |
| rkl_elementwise_reduce | 0.254 | 1.7%         |

在这组短前缀、单卡、microbatch1 读出中，teacher／student 骨干约占 40%／25%，双方词表投影约 33%，log-softmax 与 RKL 逐元素归约约 3%。详细 trace 的矩阵乘是主要 CUDA kernel 类别；这是整个评分路径的观察，不是只针对 KL 减法／乘法的算子结论。

chunk256→1024 墙钟中位数只下降约 0.8%，峰值 allocated 增加约 3.05GiB。测试为顺序 chunk sweep、每配置 3 次，未做多轮交错确认，不据这一小差值宣称确定的加速。后续优先测试骨干 microbatch／packing 及 FP32 词表投影效率，再评估 KL 归约融合；精度改变必须单独数值验收。不能从此推断 16K、FSDP 通信或完整_validate 切换的瓶颈占比。

数据见[single-profile.json](data/rkl-diagnostic-20261031/single-profile.json)、[inputs.json](data/rkl-diagnostic-20261031/inputs.json)。脚本、独立 trace、kernel 表与资源状态保存在仓库同级 `wandb-logs/MirrorOPD-verl/analysis/rkl-four-gpu-20261031/`。其中 `rkl-four-real-rows.trace.json.gz` 名称的 four 指四条轨迹，不是四张 GPU。

初次资源不足时的单卡诊断已由下文四卡完整对照补齐；14B 权重、768 条性能输入及恢复窗口现均已完成。正式四域题目清单与 Trainer 集成另行推进。

## 四卡验证评分：串行与固定 2＋2 的完整对照

2026-10-31，在物理 GPU0、2、4、5 的 H800 上完成三组师生对照。固定 2＋2 在三组尺寸上都可运行，但没有统一的速度优势：4B teacher 组串行略快，8B teacher 组并行中位数较低，7B student／14B teacher 组持平。完整 18 个正式计时点的逐题 RKL 及恢复后 student 检查输出均一致，最大观测差为 0。

![四卡验证完整窗口耗时](figures/validation-placement-20261031.png)

### 测试合同与可比较范围

- 串行：teacher 骨干在四卡加载并评分，真实 TQ SimpleStorage 暂存 hidden；teacher 卸载，student 四卡加载后评分与归约。
- 并行：teacher 使用物理 0／2，student 使用 4／5；固定两个 2-rank FSDP2 组，以最多两个在途 hidden 张量的 NCCL 队列流水处理。teacher head 在 student 侧做 FP32 投影。
- 两者计时均从 CPU 布局已准备、开始 GPU 加载，到 student 四卡 FSDP2 骨干和独立输出头驻留并通过一次就绪 forward；包括装卸、hidden 传输、清理和四卡 student 恢复。无 optimizer／梯度，无 vLLM 权重同步，不含生成与磁盘下载。
- 每策略预热后测三次，顺序为串行、并行、并行、串行、串行、并行。正式计时保留轻量阶段 CUDA events；完整 torch.profiler trace 独立采集，不混入中位数。每组另采串／并各四个 rank 的 trace，共 24 份。
- Student 主权重 FP32，teacher 主权重 BF16；FSDP2 计算参数 BF16，两个输出头使用严格 FP32 operands／FP32 输出，TF32 关闭；microbatch2、token chunk256，按长度排序后轮转分配。GPU 通过 NV8 互联，CPU 未做 NUMA 绑核。

这是基于 verl 装卸工具、FrozenTeacherHead 和 TQ 的 FSDP2 评分原型，**尚非正式 Trainer 的_validate 集成验收**。输出头在评分 rank 上复制，CPU 侧四卡及两卡布局预先构造；当前训练 checkpoint 的 4→2 权重重分片／同步成本尚未计入，也不能把一次性冷初始化当作已经被生产运行自动摊销。没有使用完整 verl remove-padding／动态 token 批处理。源代码和数据身份均已保存，三组主结果使用同一 benchmark 代码 hash。

### 真实输入

| 模型家族        | 轨迹数 | 回答 token | Prompt token | 最大回答长度 | 超过 7,168 的轨迹 |
| --------------- | ------ | ---------- | ------------ | ------------ | ----------------- |
| Qwen3           | 768    | 3,819,455  | 68,957       | 16,384       | 189               |
| R1-Distill-Qwen | 768    | 2,772,962  | 67,117       | 16,384       | 113               |

每套输入含 256 条已有银行轨迹及 512 条已有数学 benchmark 回答；前者保留原始 prompt／response IDs，后者保留原始 response IDs 并用对应固定 tokenizer 重新编码已记录的 rendered prompt。公共部分截到 16,384，银行部分实际多为 1,024。它们是执行性能用的真实数学轨迹，**不是新四域科学评估清单，也不是所有 768 条都达到 cap**。两种布置在同一模型对上使用完全相同的 token；跨模型家族的 token 量不同，只做同行布置比较，不据此拟合模型规模速度规律。

### 完整窗口结果

单位秒，范围是三次实际读数的最小值至最大值，不是置信区间。正的耗时变化表示并行更慢。显存是单 rank 最大 PyTorch allocated，不含所有 NCCL 外部分配或冷构造峰值。

| Student／teacher                                          | 串行中位数〔范围〕      | 2＋2 中位数〔范围〕     | 并行耗时变化 | 串行／并行峰值 GiB |
| --------------------------------------------------------- | ----------------------- | ----------------------- | ------------ | ------------------ |
| Qwen3-1.7B／Qwen3-4B                                      | 98.67〔98.20–98.75〕    | 101.27〔99.35–103.43〕  | +2.64%       | 6.73／8.33         |
| Qwen3-1.7B／Qwen3-8B                                      | 129.61〔125.71–130.27〕 | 121.92〔121.70–130.97〕 | -5.93%       | 8.85／12.54        |
| DeepSeek-R1-Distill-Qwen-7B／DeepSeek-R1-Distill-Qwen-14B | 159.21〔159.05–164.03〕 | 159.53〔156.92–166.65〕 | +0.20%       | 17.89／24.52       |

不能从 8B 组的中位数下降推出普遍优势：并行重复有波动，且另外两组没有改善。独立分组还增加当前 checkpoint 同步的工程成本，正式部署收益需在集成后复测。

### 阶段瓶颈

下表来自各策略中位数对应运行。数值为角色内 rank 的平均 CUDA range elapsed，包含等待，不是纯 kernel 时间；并行 teacher 与 student 各行工作会重叠，不能相加当总墙钟。

| Student／teacher                                          | 串行 teacher 骨干 | 串行 student 骨干 | 串行两 head | 串行归一化＋RKL | 串行 TQ＋hidden 搬运 |
| --------------------------------------------------------- | ----------------- | ----------------- | ----------- | --------------- | -------------------- |
| Qwen3-1.7B／Qwen3-4B                                      | 35.08             | 18.00             | 26.44       | 3.45            | 10.59                |
| Qwen3-1.7B／Qwen3-8B                                      | 48.88             | 18.08             | 35.01       | 3.45            | 15.66                |
| DeepSeek-R1-Distill-Qwen-7B／DeepSeek-R1-Distill-Qwen-14B | 59.33             | 31.19             | 36.07       | 2.51            | 15.54                |

| Student／teacher                                          | 并行 teacher 骨干 | Teacher 等待消费 | 并行 student 骨干 | Student 侧两 head | 归一化＋RKL | Student 等待 teacher |
| --------------------------------------------------------- | ----------------- | ---------------- | ----------------- | ----------------- | ----------- | -------------------- |
| Qwen3-1.7B／Qwen3-4B                                      | 67.97             | 25.80            | 33.24             | 53.05             | 6.90        | 0.73                 |
| Qwen3-1.7B／Qwen3-8B                                      | 93.23             | 17.90            | 33.06             | 70.27             | 6.90        | 1.10                 |
| DeepSeek-R1-Distill-Qwen-7B／DeepSeek-R1-Distill-Qwen-14B | 117.04            | 20.04            | 57.79             | 73.00             | 5.01        | 1.45                 |

骨干 range 还包含输入准备与 FSDP 通信。当前主要代价在骨干与词表投影，log-softmax 及 RKL 逐元素归约较小。2＋2 将两套输出头都放到 student 两卡，使该侧承担更大的 head 工作量；teacher 存在等待消费，而 student 等 teacher 数据较少。trace 里较长的 NCCL kernel 驻留不能单独证明链路带宽不足，它也可能包含等待接收方和流水背压。

| Teacher                      | 串行 hidden D2H／H2D，各 GiB | 并行 hidden 单向发送 GiB |
| ---------------------------- | ---------------------------- | ------------------------ |
| Qwen3-4B                     | 18.21／18.21                 | 18.21                    |
| Qwen3-8B                     | 29.14／29.14                 | 29.14                    |
| DeepSeek-R1-Distill-Qwen-14B | 26.45／26.45                 | 26.45                    |

这些是 hidden 本体的逻辑字节，不含模型／head 参数装卸、TQ 元数据、CPU 副本及 FSDP all-gather。串行 D2H 与 H2D 是两次真实搬运；并行 send／recv 表示同一次传输两端，不能重复相加。

### 现成投影快路径的独立小测

主表使用严格 FP32 operands 投影作为参考，不能冒充生产 verl 所有已优化算子的最佳速度。`verl.utils.linear.project`已有 BF16 operands、FP32 output 路径。另在单 H800 上用 8 条真实前缀产生的 8,192 个 hidden 位置比较 teacher 投影；同一 BF16 hidden 和 BF16 teacher 权重，参考先转 FP32，候选直接调用 project。关闭 BF16 reduced-precision reduction，不降低训练后 FP32 student head 精度。

| Chunk | 严格 FP32 投影中位秒 | BF16 输入／FP32 输出中位秒 | 仅 teacher 投影加速 | 最大 token RKL 绝对变化 |
| ----- | -------------------- | -------------------------- | ------------------- | ----------------------- |
| 256   | 0.19945              | 0.01767                    | 11.28×              | 9.36e-05                |
| 512   | 0.20141              | 0.01670                    | 12.06×              | 9.36e-05                |
| 1024  | 0.19969              | 0.01717                    | 11.63×              | 9.36e-05                |

最大 logit 绝对差约 0.000362；teacher-copy 单 token 读出在约[-2.31e-07, 2.78e-07]，均值约 -4.32e-10。微小负值是浮点读出误差，不是真实负 KL。快路径有明显性能潜力，也存在可见数值差异；这是投影微基准，不是完整验证 11 倍加速。尚未用它替换主表计算，也未完成全部模型、16K 和训练后 FP32 checkpoint 的数值验收。

### 串行与并行优化后的配对结果

Qwen3-1.7B student／Qwen3-8B teacher 在相同四卡上完成了配置筛选和完整数据确认。筛选使用按长度抽取的 128 条真实轨迹，比较 36,864／73,728／147,456／294,912 token 预算、最大 microbatch16、串行异步 TQ 写入与预取、并行最多 32 个在途 hidden 张量，以及 padded／packed 输入。完整确认复用上述 768 条、3,819,455 个 response token；每个配置重复两次，计时仍到四卡 student 恢复并完成固定前向探针。

| 投影及执行配置                                  | 串行两次耗时（s） | 串行中位（s） | 并行两次耗时（s） | 并行中位（s） | 串行／并行峰值 allocated（GiB） |
| ----------------------------------------------- | ----------------- | ------------: | ----------------- | ------------: | ------------------------------- |
| FP32 head；36k 动态批处理，串行预取／并行深流水 | 111.06／109.96    |        110.51 | 121.39／117.45    |        119.42 | 9.09／16.35                     |
| BF16 head 乘法、FP32 输出；packed；其余同上     | 72.41／79.15      |         75.78 | 94.30／91.01      |         92.66 | 8.98／16.33                     |

在 FP32 参考下，串行比并行节省 7.5% 时间；在第二行精度设置下，节省 18.2%。旧保守基线为串行 129.61 s、并行 121.92 s；只调整批处理与传输，串行改善更大，两个方案的排序发生变化。第二行另外改变投影精度与 packing，不能把全部收益归因于串行布局。

128 条筛选中，串行 36k／74k／147k／295k 预算的中位耗时分别约 25.47／26.28／30.59／36.33 s。增加微批并未持续加速；36k 是已测试配置中更合适的预算，不是所有模型与输入上的全局最优。固定 2＋2 的深流水减少了发送阻塞，但 teacher 8B 只占两卡，head 加速后 teacher 骨干及模型切换成为更主要的成本。串行预取的 TQ 等待接近零；CUDA 区间中的骨干时间包含 FSDP 通信及相关等待，不能直接解释为纯算术 kernel 时间。

数值核对使用相同题目、原始 token 和每题有效长度。FP32 调优与旧参考的逐题 RKL 一致；第二行相对 FP32 的最大逐题平均 RKL 差为 $2.565\times10^{-6}$。四卡恢复后的固定前向探针输出一致；此探针不是全部参数的逐字节哈希。原始 student head 来自 BF16 权重，第二行显式采用原生 BF16 混合精度的投影口径；后续训练读出仍需与训练实现保持相同精度，不能将任意训练后 FP32 权重降精度当作严格等值变换。

另以 64 个真实状态的 FP64 投影作参考，teacher 严格 FP32 与 BF16 输入／FP32 输出的最大 token RKL 误差分别约 $4.51\times10^{-5}$ 和 $3.43\times10^{-5}$。两个实现之间的差异不全来自快路径；FP32 参考本身也有舍入误差。原生只读 RKL 复用 `weighted_rows(..., gradient=False)`：GPU 与训练 fused 算子的最大 token 差为 $1.19\times10^{-7}$，teacher-copy 为零，额外 teacher 词表项的 FP64 小规模核对误差小于 $4.5\times10^{-8}$。

优化比较限定于这组 Qwen3 模型和真实数学轨迹。已有三种尺寸的保守基线保留；其他模型、当前训练后 student、四域题池和完整训练切换不沿用上述加速比。accuracy、各类 RKL 频率和训练 loss 平均方式继续待定。

[配置与逐项审计](data/validation-optimization-20261031/audited-comparison.json)保留全部重复值、峰值、逐题差异和分角色阶段时间。原始配置扫描、脚本快照、逐题输出和日志位于仓库同级 `wandb-logs/MirrorOPD-verl/analysis/rkl-optimization-20261031/`。

### 原生 verl 的 JustRL → R1-1.5B 完整设置验收

四卡串行 `shared_sync` 已完成两轮完整设置验收：teacher 为 `hbx/JustRL-DeepSeek-1.5B`，student 为 `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`，沿用原生 V1、FSDP2 和 vLLM。训练池准备 6,400 道去重 DAPO 题，实际使用 256 道不同题；每轮 batch128、n=1、fresh student rollout、reuse32，共 64 次有效更新。Prompt cap2048、response cap7168；teacher／actor 的 packed token budget 均为 32768，head chunk256，`precompute_fused`，hidden async write／prefetch 开启。

AdamW 使用 LR=$10^{-6}$、constant scheduler、betas=(0.9,0.999)、weight decay=0.01、grad clip=1。该次训练及以下来源留出数值采用当时的 token-mean；新 Parquet 验收的评估题均值另表报告，不回写这些旧数值。任务奖励不参与蒸馏 loss。

来源留出为独立 128 题，由 initial student 一次生成，共 808,935 个 response token；三次读出使用相同银行哈希。历史集合使用真实训练 token，第二轮后从两批各保留 64 题。

| 有效更新数 | 固定来源留出 RKL | 历史 RKL（逐批等权 token-mean） |
| ---------: | ---------------: | ------------------------------: |
|          0 |         0.114618 |                               — |
|         32 |         0.060328 |                        0.056084 |
|         64 |         0.026837 |                        0.023065 |

固定来源留出 RKL 下降 **76.6%**。两批的首／末次训练 forward loss 分别为 0.115172→0.057652、0.060034→0.022055；末次 forward 发生在该批最后一次参数更新之前。历史集合成员随批数变化，因此不能直接把表中两个历史均值之差当作固定集合改善。单独匹配两次读出都出现的第 1 批 64 条轨迹，其 RKL 从 0.055724 降至 0.024888，下降 55.3%。

| 每轮阶段（s）                   | 第 1 轮 | 第 2 轮 |
| ------------------------------- | ------: | ------: |
| Fresh rollout 管线              |   26.25 |   26.21 |
| Teacher 加载／卸载＋评分        |    4.01 |    4.05 |
| Actor 的 32 次更新              |  311.49 |  303.01 |
| 来源＋历史共 256 条固定轨迹读出 |   26.58 |   24.89 |
| 权重同步回 rollout              |    1.15 |    1.10 |
| 整个 outer                      |  371.19 |  360.85 |

训练循环合计约 **12.2 分钟**；actor 更新平均约 9.6 s／次。首次验证另耗时 140.2 s，包含 initial student 银行生成及首次生成服务调用。表中的固定轨迹读出启用了完整参数／Adam 哈希验收，包含该额外成本；正式频率仍需按选定评估任务和关闭哈希后的耗时决定。训练阶段 NVML 采样多次达到 88%–100% GPU utilization；原始时间线保留在运行目录，生成、权重切换及初始化分开观察。

验收脚本核对了 64 条连续有效更新记录、finite loss／梯度、256 道唯一训练题、来源留出无重叠、rollout 版本 0→1、历史 64＋64 配额，以及每次读出前后四个 rank 的参数／Adam 内容哈希不变。两轮实际生成 1,544,472 个训练 response token。Teacher hidden 只在当前评分或 reuse 阶段存放于 CPU TQ，SSD offload 关闭；运行没有保存 checkpoint 或 optimizer 文件，日志和原始轨迹等产物约 83 MB。

短验收为 batch8、reuse2、cap256、两轮，已先通过相同接口；完整设置另起新运行，从原始 student 权重和新的 128 题初始银行开始。该工程验收覆盖来源留出和历史轨迹，公共四域／teacher-rollout 银行及断点恢复仍需接入。

[完整验收摘要](data/native-rkl-acceptance-20261031/acceptance-audit.json)可由[audit_fixed_rkl.py](../../../../benchmarks/profile/audit_fixed_rkl.py)复算。完整配置、64 次更新指标、逐题 RKL、轨迹、状态哈希、源码快照与 NVML 时间线位于仓库同级 `wandb-logs/MirrorOPD-verl/analysis/rkl-optimization-20261031/native-acceptance-a01/`；源码快照另见相邻 `source-snapshots/native-acceptance-a01/`。

### Parquet 基础链路的四卡 GPU 验收

新`FixedRKLReadout`已完成实际 GPU 端到端验收：物理 GPU0–3，JustRL→R1-1.5B，原生 V1 同步／FSDP2／vLLM；batch128、reuse32、LR1e-6、训练与来源 cap7168，共两轮 64 次有效更新。训练用 token-mean；以下评估先题内 token 均值再按题平均，history 外层按 batch 等权。这是基础训练链路验收，不是四域 50outer 正式实验。

| 更新数 | 固定来源 heldout RKL | 动态 history RKL |
| ------ | -------------------: | ---------------: |
| 0      |             0.112802 |                — |
| 32     |             0.059306 |         0.055371 |
| 64     |             0.025966 |         0.022233 |

固定来源风险下降 76.98%。相邻两次 history 共同保留的第 1 批 64 条轨迹，从 0.055064 降到 0.024597；动态 history 整体均值包含换样，不直接解释成相同状态的降幅。该结果与上一节 token-mean 旧验收口径不同。

三次读出分别覆盖 128／256／256 条轨迹，均无无效评分行，四个 rank 的参数／Adam 只读哈希检查均通过。保存了 256 道不同训练题的真实 token；第 1／2 批 rollout 版本分别为 0／1，末轮 history 每批 64 条。银行、训练轨迹、history 引用和逐题 RKL 均为 Parquet，manifest 完整发布；终止原因也已落盘。

| 阶段（秒）                     | 第 1 轮 | 第 2 轮 |
| ------------------------------ | ------: | ------: |
| Rollout 管线                   |   26.21 |   26.21 |
| Teacher 加载／卸载＋前向       |    4.13 |    3.69 |
| Actor 的 32 次更新             |  316.56 |  299.28 |
| 来源＋history 读出，含状态审计 |   26.21 |   24.02 |
| 最终 checkpoint 保存           |       — |    5.29 |
| 权重同步                       |    1.01 |    0.89 |
| 完整 outer                     |  375.75 |  361.06 |

两轮训练约 12.3 分钟；包含冷启动、初始验证与进程清理的实际运行耗时约 19.0 分钟。初始生成＋读出 129.33 秒，其中固定 token 评分与状态恢复约 15.82 秒。各阶段有额外编排开销，不能将局部计时简单当作完整 outer。

仅保存`global_step_2`一份原生 FSDP model-only checkpoint，包含 4 个权重分片，总计 7,109,058,476 字节；无 optimizer 和 extra-state 权重文件，保留了原生保存器的小型配置／进度元数据。CPU 读回四个分片通过：每片 339 项，全为 FP32 且有限，合计 1,777,088,000 个本地参数元素。此次未重新加载模型做前向／RKL 对齐，也未验证拟议的统一 HF 导出器。

首个实际训练尝试在初始银行阶段暴露`KVBatchMeta.fields=None`兼容问题，当时尚无参数更新。修复后将回调测试改用真实 KVBatchMeta，15 项相关 CPU 回归通过，再从原始权重重新运行本次成功验收。失败尝试保留为工程证据，不混入上述指标。

[完整验收摘要](data/parquet-gpu-acceptance-20261031/summary.json)、[逐阶段验收](data/parquet-gpu-acceptance-20261031/acceptance-audit.json)与[checkpoint 读回检查](data/parquet-gpu-acceptance-20261031/checkpoint-inspection.json)对应仓库`runs/parquet-gpu-acceptance-20261031/native-a03/`。运行期间源码 hash 无变化。GPU0–3 已确认恢复为 gpu-occupy，其他实际任务未停止；[恢复记录](data/parquet-gpu-acceptance-20261031/occupancy-restored.json)保留物理映射及进程状态。

未覆盖：公共 benchmark 全集 accuracy、公共 initial／teacher／current 银行、32K 评分窗口、推理专用 padding、四域数据清单与 50outer 预算。已有基础链路通过不代表这些功能已实现。

### 证据与复算

结果见[审计摘要](data/validation-placement-20261031/audited-summary.json)、[投影小测](data/validation-placement-20261031/projection-probe.json)、[依赖与源码身份](data/validation-placement-20261031/dependencies.json)。输入统计见[Qwen3](data/validation-placement-20261031/qwen3-inputs.json)与[R1](data/validation-placement-20261031/r1-inputs.json)。

配置扫描审计：[audit_validation_sweep.py](../../../../benchmarks/profile/audit_validation_sweep.py)。执行原型：[validation_placement.py](../../../../benchmarks/profile/validation_placement.py)；分析：[analyze_validation_placement.py](../../../../benchmarks/profile/analyze_validation_placement.py)；绘图：[plot_validation_placement.py](../../../../benchmarks/profile/plot_validation_placement.py)。原始逐题读出、24 份主结果 trace、启动命令、脚本快照和日志保存在仓库同级 `wandb-logs/MirrorOPD-verl/analysis/rkl-four-gpu-20261031/`。

主结果统一采用 microbatch TQ 读写。同目录的`qwen4-n128`及早期`qwen4-n768`逐题 TQ 版本只作为诊断，不混入主表。首次 shape／memory profiler 与 FSDP 参数 swap 冲突后，正式 trace 改为不记录 shape／memory；峰值显存由非 profiler 阶段的 allocator 计数提供。三个主运行均正常退出、通过逐题及恢复检查，绘图使用真实重复读数。

## 统一四域评估与模型导出验收

原生 V1／FSDP2／vLLM 的统一协调器完成四卡集成：JustRL→DeepSeek-R1-Distill-Qwen-1.5B，batch8、reuse2、两轮 4 次更新；source8 题，公共 RKL 面板 10 题，各模型／节点 16 条公共回答。训练与来源回答 cap7168，公共 cap32768；评估 hidden 窗口 128MiB。这里的小预算用于覆盖执行路径，不作为正式 50 轮结果。

| 完成 outer | 逻辑评分行 | RKL 窗口 | Source 按题 RKL | History 批等权 RKL |
| ---------- | ---------: | -------: | --------------: | -----------------: |
| 0          |         28 |        3 |        0.106107 |                  — |
| 1          |         46 |        5 |        0.105909 |           0.102227 |
| 2          |         46 |        5 |        0.105296 |           0.104221 |

每次读出都合并同一边界到期的轨迹，再按字节窗口依次进行 teacher 与 student 前向；padding 不计入逻辑行数。全部行有效，四 rank 参数／Adam 只读审计通过。outer0 的 current 结果复用 initial 评分，没有重复 forward。训练轨迹的权重版本分别为 0、1，两个 outer 确实使用 fresh student rollout。

最终保存 FP32 master 权重，不保存 Adam。原生 model-only 重载的固定输入 RKL 最大误差为 0，之后删除临时 FSDP 分片；独立新进程从 HF 权重初始化，再评分原 8 条 source 轨迹，最大误差同样为 0。完整运行墙钟 660.94 秒，包含模型启动、训练、公共生成、判分、评分与导出；不能将它当作单轮训练吞吐。

[验收摘要](data/unified-core-acceptance-20261031/summary.json)、[实际配置](data/unified-core-acceptance-20261031/config.yaml)和[启动时源码 hash](data/unified-core-acceptance-20261031/source-hashes.json)保留了条件与证据。原始轨迹、日志和逐 run 图表位于仓库`runs/unified-evaluation-20261031/`；后续规模验收状态由[实施页](../../architecture/evaluation/代码走读与实施计划.md#8-统一评估实现与待验收边界)维护。

### 规模与六卡检查

Code、Science、IF 分别完成两轮集成，每组使用 128 条对应来源留出。7B 同权重参考与 7B student／14B teacher 各对 7 条 32768 token 合成诊断前缀评分，验证长序列、补零和窗口变化，不作为自然 rollout 上的拟合结果。

| Student／Teacher | window／token chunk | 窗口数 | 评分墙钟（含审计） |
| ---------------- | ------------------- | -----: | -----------------: |
| R1-7B／R1-7B     | 512MiB／128         |      4 |           99.04 秒 |
| R1-7B／R1-7B     | 1GiB／256           |      2 |           52.71 秒 |
| R1-7B／R1-14B    | 512MiB／128         |      7 |          176.36 秒 |
| R1-7B／R1-14B    | 1GiB／256           |      3 |           80.43 秒 |

两种执行配置逐题 RKL 完全一致；同权重组 RKL 为 0。窗口和 token chunk 同时改变，因此这些时延不是单因素比较。14B teacher 每条 32K 前缀的 hidden＋positions 实际传输约 320.3MiB，与 BF16 hidden 宽度 5120 的估算一致。

六卡使用 batch128／reuse32 完成 32 次真实更新，物理补齐到 132 行，统计仍为 128 题。Actor 更新阶段 202.41 秒；固定 source128 的 RKL 从 0.106403 降至 0.056139，降幅 47.24%。该检查只有一个 outer，数据也不同于旧四卡基准，不能据此作严格吞吐或收敛比较。最终模型重载 RKL 误差为 0，无 Adam 文件。[规模验收摘要](data/unified-scale-acceptance-20261101/summary.json)保存逐题集合的汇总与审计。
