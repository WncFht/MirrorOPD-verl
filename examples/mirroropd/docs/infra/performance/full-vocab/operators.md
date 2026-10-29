# Full-vocab 算子性能与容量

本页只比较给定 hidden/head 后的投影、目标、归约与一阶 VJP。真实模型更新与完整训练见[训练性能](training.md)，统计与显存定义见[测量方法](methodology.md)。数值实现及推导由[算子走读](../../architecture/full-kl-kernel-walkthrough.md)维护。

## 当前算子复测入口

[benchmark_full_kl.py](../../../../benchmarks/full_vocab/benchmark_full_kl.py) 用合成 BF16 hidden/head 比较生产实现，student 固定为 hidden1536、vocab151936。`--teacher-hidden`、`--teacher-vocab` 可分别选择 JustRL 的 1536/151936 或 Polaris 的 3584/152064；后者保留额外 teacher 输出行的归一化贡献。`--backends precompute precompute_fused --reference-backend dense` 比较普通与 fused 归约，`--chunk` 控制 token 分块。

工具在每个 backend 计时前检查 loss 和 hidden/head VJP，沿用绝对 loss 误差小于 1e-4、梯度相对 L2 误差小于 0.02 的随机输入门槛；近目标和小梯度仍由独立 FP64 CUDA 测试覆盖，不能用随机输入检查代替。每个形状预热两次、测量五次，已有输出文件会被拒绝。`--profile` 在计时结束后另做一次 CPU/CUDA trace，输出独立的 `.trace.json.gz`，trace 时间不混入性能样本。扩展的异构形状与 trace 入口已完成下述开发检查；历史性能数字保持原测量身份。

## R1 配对的算子开发测量

2026-10-30，在物理 GPU 6 的独占开发队列中复测当前 precompute 与 fused 实现。Learner head 为 vocab151936、hidden1536，BF16 输入与 FP32 归约；先通过 79 项 CUDA 数值测试，覆盖近目标、小梯度、BF16、非整齐词表和额外 teacher 输出行等案例。运行证据位于工作区 `runs/vllm-fusion-20261029/r1-fused-operator-checks-01/` 与 `r1-fused-shapes-02/`。

4096 位置的一次性 dense autograd 参考在 80GB H800 上 OOM，原始失败保留。后续用 512 位置的 dense autograd 检查 loss/VJP，大规模以普通 precompute 检查系统一致性；后者共享部分实现，不能单独充当独立数学 oracle。原误差门槛保持不变。

| JustRL head 形状，4096 位置 | 中位墙钟（ms） | Allocated 峰值（GiB） |
| --------------------------- | -------------: | --------------------: |
| precompute，chunk512        |        165.861 |                 5.731 |
| fused，chunk512             |         26.491 |                 2.834 |
| precompute，chunk1024       |        159.188 |                 9.649 |
| fused，chunk1024            |         23.234 |                 3.850 |

每个 backend 预热两次、测量五次，保留全部样本。单独的 chunk512 trace 中，GPU kernel 数为 1299/115，kernel duration 总和为 157.774/26.070 ms；这是 head-only trace，不是完整 outer 时间。512 位置的独立 dense 对照及 4096 位置的一致性检查均通过。Polaris 的 hidden3584/vocab152064 异构 head 也完成全部对照：4096 位置、chunk512 的普通/fused 中位数为 185.635/46.651 ms，chunk1024 为 176.078/40.662 ms；fused allocated 峰值分别为 3.429/4.447 GiB。两组 × 两种 chunk × 512/4096 位置共八组均通过所声明范围的检查。J 的 fused/chunk1024 两轮完整训练 profile 已完成，更新计数和版本审计通过；端到端观测与差异范围见[融合验收记录](../../rollout/fusion/2026-10-28-Rollout双融合实现与验收记录.md#qwen2-实验配对)。最终参数仍需结合其他来源检查、正式重复及剩余候选验收确定。

## 证据范围

| 测量                       | 输入与实现身份                                                                        | 原始证据                                                                                                                                                                            |
| -------------------------- | ------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Endpoint backend 比较      | 2026-10-22 的生产实现快照；H800 单卡，synthetic BF16，V=151936、d=1536                | [kernel-final.json](../../numerics/full-vocab/data/implementation-20261022/kernel-final.json)、[32K 容量](../../numerics/full-vocab/data/implementation-20261022/capacity-32k.json) |
| Precompute 四 backend 复测 | 2026-10-26 工作区模块快照；共享 H800，N=4096、V=151936、d=1536、chunk=512；探索性测量 | [原始样本](data/precompute-20261026/benchmark.json)、[协议与源码身份](data/precompute-20261026/README.md)                                                                           |
| Endpoint 融合              | `39d8be84a` 冻结源码，H800、N=32768；相同 chunk 对比 precompute 与融合候选            | [benchmark.json](data/review-004/kernel/benchmark.json)、[kernel 归因](data/review-004/reviews/004-semantic-traces.json)                                                            |
| MDPO/BPO target 与诊断融合 | 2026-10-25 优化前工作区模块快照，imports 隔离；四张 H800 各运行一个独立组合           | [完整对照](../../../experiments/teacher-transfer/optimizer-precision/mechanism/data/full-vocab-performance.json)                                                                    |
| 原型与 value-only 数值接口 | 2026-10-22 prototype，H800 单卡，N=512/4096，含自有原型和本机 Liger                   | [CPU 审计](../../numerics/full-vocab/data/probe-20261022/cpu.json)、[GPU 样本](../../numerics/full-vocab/data/probe-20261022/h800.json)                                             |

这些是不同实现阶段的测量，不能把同名 backend 的不同结果当成独立重复样本。它们都不包含 backbone、FSDP 通信、TQ、optimizer 或 rollout。2026-10-26 复测对应明确保存的工作区模块；其他批次仍按各自的历史源码解释，后续代码变化不自动刷新这些结论。

## Endpoint backend 比较

4096 个位置、token chunk=512、vocab chunk=8192 的 head-loss 前反向中位数如下。显存为包含常驻输入的 allocated 峰值。

| Backend    | 时间（ms） | Allocated 峰值（GiB） |
| ---------- | ---------: | --------------------: |
| dense      |      37.40 |                 12.55 |
| recompute  |      61.87 |                  3.30 |
| cached     |      71.59 |                  2.30 |
| streaming  |      70.93 |                  2.30 |
| precompute |      37.40 |                  3.15 |
| liger      |     130.58 |                  2.47 |

该形状下 precompute 达到接近 dense 的耗时，allocated 峰值低约 75%。Cached/streaming 的峰值接近，是因为梯度累计与返回缓冲仍占固定容量。Liger 数字对应本机特定入口、chunk 与 FP32 累积设置，不能推广为库整体的速度排名。

在 N=32768 的容量测量中，dense 未作为性能参照，以 streaming 检查大形状的一致性，小规模独立 dense/FP64 仍负责数学验证：

| Backend    | 时间（ms） | Allocated 峰值（GiB） |
| ---------- | ---------: | --------------------: |
| cached     |     567.36 |                  2.71 |
| streaming  |     562.79 |                  2.71 |
| precompute |     288.00 |                  3.48 |

更低的 head 工作区不保证更短的模型更新；选择时还需加入实际 backbone、冻结 head 与 optimizer 状态。

<a id="precompute-20261026"></a>

## Precompute 四 backend 复测

2026-10-26 对普通 teacher-only Full-KL 比较 dense、recompute、precompute、precompute_fused。Precompute 在本次 forward 内按 token 块计算解析 logits 梯度 G，并累计 $dH=GW$、$dW=G^\top H$；backward 只施加上游缩放并继续 backbone 反传。缓存不跨 optimizer 更新复用。它相对 dense 减少全词表中间图，相对 recompute 避免 backward 再次投影师生 logits；fused 在这个框架上进一步优化词表归约。

输入为 synthetic BF16 hidden/head，N=4096、V=151936、d=1536，token chunk=512、vocab chunk=8192；PyTorch 2.11.0+cu128、单张 H800，TF32 关闭。源码基点为 `141f26f81a1310dd8e89e67cd721bdf993bd1ca8` 加未提交修改；实际使用的 [full_vocab.py 快照](data/precompute-20261026/full_vocab.py) 和[测量脚本](data/precompute-20261026/benchmark.py)均已保留，不能仅用该 commit 的纯净 checkout 代替。

每个 backend 预热 2 次，然后执行 9 轮交替正序/逆序测量，wall time 前后显式 CUDA synchronize；报告全部样本的中位数和范围，不裁剪异常值。显存另做一次调用前 reset，报告本进程包含常驻输入的 allocated 峰值，不是 NVML 整卡显存。完整数据见 [benchmark.json](data/precompute-20261026/benchmark.json)。

| Backend          | Wall time 中位数（ms） | 最小—最大（ms） | Allocated 峰值（GiB） |
| ---------------- | ---------------------: | --------------: | --------------------: |
| dense            |                  53.81 |     51.40—60.21 |                 12.55 |
| recompute        |                  86.92 |    83.98—132.78 |                  3.30 |
| precompute       |                  54.61 |     51.49—86.19 |                  3.01 |
| precompute_fused |                  30.74 |     29.05—47.23 |                  2.57 |

在这组观测中，precompute 与 dense 耗时接近，allocated 峰值减少约 76.0%；相对 recompute，precompute 耗时减少约 37.2%。Fused 相对普通 precompute 的中位数速度为 1.78×，耗时减少约 43.7%，allocated 峰值再减少约 0.434 GiB。这些比值只使用同一批测量，不能与 2026-10-22 的读数拼接成版本加速。

该测量使用已有进程驻留的共享 GPU 7，未按本专题当前要求的 GPU 0–3 独占锁、持续空闲 120 秒门槛执行，也没有持续记录外部进程、锁频或跨进程重复。因此作为探索性算子数据保留，不作为正式性能验收；范围内的波动不能忽略。首次计时与最终计时之间核对到临时量释放变更，最终表只使用重新冻结源文件后的复测，未合并首次样本。详细身份与复算命令见[测量记录](data/precompute-20261026/README.md)。

同次[数值审计](../../../../../../outputs/full-kl-softmax-audit-20261026/README.md)中，已有 118 项 CPU、41 项 CUDA 测试通过，但新增近目标检查仍复现了 endpoint fused 与 BPO 的梯度问题。Benchmark 自身没有在计时前内置逐例独立 VJP 门控；它记录的随机输入 loss 接近，不能覆盖上述反例。性能表不是数值正确性验收，待稳定化补丁完成后需同时复核 VJP 和性能。

真实更新的历史收益仍引用[固定输入管线](training.md#固定输入管线)：precompute → fused 的 actor 更新中位数 3.719 → 3.480 秒，耗时减少约 6.4%。本次没有重新执行 backbone/FSDP/TQ/rollout，不能把 1.78× 换成整轮训练速度。

## OPD endpoint 融合

H800、BF16 hidden/head、FP32 logits/梯度累计，V=151936、d=1536、N=32768；每个配置先核对 loss/VJP，再计时 7 次，报告中位数。`fused_candidate` 是数据中的实现标签，对应后来接入的 `precompute_fused` endpoint。

| Backend    | Chunk | 时间（ms） | Allocated 峰值（GiB） |
| ---------- | ----: | ---------: | --------------------: |
| precompute |   512 |    287.785 |                 3.917 |
| fused      |   512 |    162.675 |                 3.482 |
| precompute |  1024 |    260.098 |                 5.223 |
| fused      |  1024 |    137.127 |                 4.353 |

同 chunk=512 时耗时减少 43.5%，速度约 1.77×，allocated 峰值减少 0.434 GiB。Chunk=1024 的耗时减少 47.3%，同时使用更多显存。不能把改变 chunk 的收益合并进同 chunk 融合收益。

![Head chunk 的耗时与显存](figures/review-004/head-chunk-n32768.png)

在独立 trace 中，chunk=512 的归一化与 logits 梯度 kernel 工作量由 157.231 ms 降至 29.879 ms；student/teacher 投影为 44.439→45.180 ms，hidden 梯度矩阵乘为 19.702→20.130 ms，head 梯度矩阵乘为 61.009→61.081 ms。收益主要来自减少逐元素/归约及中间张量读写；投影和梯度 GEMM 没有相应加速。这些是 kernel duration 求和，不能与无 profiler wall time 混用。

从该算子优化到真实 actor 的收益，见[固定输入管线](training.md#固定输入管线)。

## MDPO BPO target 与诊断融合

2026-10-25 的 A/B 使用相同 BF16 hidden/head、seed=71、N=512、V=151936、d=1536、kappa=0.5、eta=0.3，前后都为 `precompute_fused`。每个版本预热 2 次，交替执行 9 组 AB/BA；每组 objective/diagnostics 在同一张 H800 上比较，四组分别占用 GPU 0–3。GPU 4–7 当时有其他任务，没有做锁频或多次独立进程的置信区间研究。

优化范围是 weighted target、行归约、诊断复用与临时张量生命周期，不包括模型或调度。下表时间包含三组 head 投影、target、objective 与 hidden/head VJP；显存为扣除 live input 后的额外 allocated 峰值。

| 目标 | 诊断 | Chunk | 优化前→后（ms） | 速度倍率 | 额外峰值前→后（MiB） |
| ---- | ---- | ----: | --------------: | -------: | -------------------: |
| MDPO | 关   |   512 |      12.65→5.66 |    2.24× |        3829.6→1783.5 |
| MDPO | 开   |   512 |      18.78→5.90 |    3.18× |        4009.1→2229.2 |
| BPO  | 关   |   512 |      12.17→5.88 |    2.07× |        3829.6→1932.0 |
| BPO  | 开   |   512 |      18.43→6.16 |    2.99× |        4602.6→2229.2 |

Chunk=128 的四种组合速度倍率为 1.66–2.34×，优化后额外峰值约 1340.8 MiB；chunk=32 为 1.31–1.62×，峰值受固定 head 梯度缓冲限制，前后未下降。12 个比较中，BF16 hidden/head VJP 相对 L2 差异最大约 $3.75\times10^{-5}$。完整样本见[测量 JSON](../../../experiments/teacher-transfer/optimizer-precision/mechanism/data/full-vocab-performance.json)。

普通 Full-KL eta=1 使用独立 endpoint kernel，不能将此表的 2–3 倍套给它。2026-10-26 的[近目标审计](../../../experiments/teacher-transfer/optimizer-precision/mechanism/2026-10-26-MDPO内层KL反弹与数值链路审计.md)又修正了 MDPO 大公共常数下的梯度消减误差；上表测量早于该修正，当前代码需要重新配对计时，旧 A/B 误差也不替代近目标验证。

复测入口为 [benchmark_mirror.py](../../../../benchmarks/profile/benchmark_mirror.py)。基线位于 `outputs/mirror-performance/baseline/mirror_perf_baseline`，包含隔离 imports 的数值模块；保留该基线，新的代码比较使用独立快照和输出目录，不能让 baseline 导入 candidate。

## 容量预算

令 N 为全局有效预测行数，V 为输出词表大小，$d_T,d_s$ 为师生 hidden 宽度，C 为 head token chunk，$b$ 为元素字节数。运输、常驻 head 与单块工作区分别计算：

$$
M_{H_T}=Nd_Tb,\qquad M_{W_T}=Vd_Tb,\qquad
M_{\mathrm{dense}}=NVb,\qquad M_{\mathrm{tile,FP32}}=4CV.
$$

Teacher hidden 按样本分布在 CPU/TQ，actor 按 micro-batch 读取；冻结 teacher head 的副本按 actor rank 持有。需要 q 时增加 anchor hidden/head。DP 分摊样本不意味着完整 frozen head 自动按 DP 大小分片。

以 64 条满长回答、每条 7168 个有效位置、V=151936、$d_T=d_s=1536$ 为例：

| 对象                             |         容量 | 范围                                                     |
| -------------------------------- | -----------: | -------------------------------------------------------- |
| Sampled FP32 log-prob            |     1.75 MiB | 全局有效位置；复用已有 response ID                       |
| Sampled INT32 ID + FP32 log-prob |     3.50 MiB | 全局有效位置；实际 sampled 序列布局还可能含 prompt/dummy |
| BF16 teacher hidden              |   1.3125 GiB | 全局 payload；不代表同时全量驻留 GPU                     |
| BF16 teacher head                |   0.4347 GiB | 每个持有完整 head 的 actor rank                          |
| BF16 完整 logits                 | 129.8281 GiB | 全局有效位置                                             |
| FP32 完整 target                 | 259.6563 GiB | 全局有效位置                                             |
| 一份 FP32 token tile，C=512      |   0.2898 GiB | 当前块，仍覆盖完整词表                                   |
| FP32 student head 梯度累计       |   0.8694 GiB | 完整 head 工作缓冲                                       |

若保存 top-k 反馈，INT32 ID 与 FP32 log-prob 的容量为 $8NK$ bytes；额外 sampled 项是否可以复用取决于反馈协议。这只描述监督 payload，不减少 student softmax 对全词表的依赖，也不能用 V/K 推导整个训练加速。

对 N=32768，一份 BF16 logits 为 9.2734 GiB，一份 FP32 logits/log-prob 为 18.5469 GiB。若朴素表达式同时保留两份 BF16 logits 与三份 FP32 数组，这五份共 74.1875 GiB，尚未计入模型与激活；它是明确生命周期假设下的例子，不是所有 full 实现的最低需求。

Precompute 将词表数组限制到当前 token chunk，但仍保留完整 dW 累积与当前 micro-batch 的 dH。减小 C 不会消除固定 head 缓冲；`vocab_chunk_size` 也不会把一般 MDPO/BPO 的 precompute 变成二维 streaming。具体分配和反传见[算子实现](../../architecture/full-kl-kernel-walkthrough.md#4-precompute-的前向与反向分工)。

容量表适用于这里声明的模型与长度。其他 teacher hidden size、额外词表行、长回答和 dtype 按公式重算，不从模型名称猜测容量。

## Value-only 与近似方案的边界

原型的 value-only KL 在 4096 位置时，token 分块为 31.35 ms、额外 allocated 2.029 GiB；在线词表分块为 34.83 ms、额外 allocated 0.109 GiB。该实验只计算双方 head 投影和 KL 数值，不含 sampled student backward，不能用它与完整 loss 前反向相减来估计 baseline 训练速度。原始输入及样本见[原型 GPU 数据](../../numerics/full-vocab/data/probe-20261022/h800.json)，实现探针为 [probe_full_kl.py](../../../../benchmarks/full_vocab/probe_full_kl.py)。

完整 sampled+full-KL baseline 仍需自己的训练接入和测量，设计见[baseline 扩展](../../numerics/full-vocab/baseline.md)。SVD、量化、top-k 截断改变数值目标，应作为独立方法比较；不能与 exact full-vocab 的透明优化混在同一速度排名中。
