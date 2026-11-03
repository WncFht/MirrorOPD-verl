# 验证资源、并发与容量的冻结测量

本页集中维护资源采样、逐 checkpoint 离线估计和模型子集标定。三组证据的模型、版本、窗口和负载不同；当前执行默认由[公共训练设置](../../../settings/训练设置.md)维护，不能把子集可运行或离线估计当作完整负载最优吞吐。

<a id="observed"></a>

<a id="observed-并发-64-的完整验证资源实测"></a>

## 长程验证：并发 64 的资源实测

本记录分析 Qwen3-1.7B-Base 当前 8 卡、每卡 max_num_seqs=64、gpu_memory_utilization=0.8 的全部 21 轮完整验证窗口（step 0–200）。验证边界由 driver 的 timer/testing 事件确定；显存和 GPU utilization 来自约每 30 秒一次的 nvidia-smi 采样。下表的平均值按采样点与设备等权统计，包含初始化验证内部开销和收尾等待；显存峰值是采样峰值，不是连续采集的绝对峰值。

| Outer step | Valid 耗时（分） | 采样点数 | GPU 利用率均值 | 每卡平均显存（GiB） | 采样显存峰值（GiB） |
| ---------- | ---------------: | -------: | -------------: | ------------------: | ------------------: |
| 0          |            192.4 |      371 |          94.4% |               65.83 |               66.17 |
| 10         |             29.1 |       56 |          82.8% |               66.13 |               66.48 |
| 20         |             32.8 |       64 |          78.7% |               66.25 |               66.58 |
| 30         |             21.8 |       42 |          85.1% |               66.08 |               66.41 |
| 40         |             23.8 |       47 |          89.4% |               66.10 |               66.98 |
| 50         |             24.4 |       48 |          77.8% |               66.18 |               66.69 |
| 60         |             25.4 |       50 |          79.2% |               66.13 |               66.61 |
| 70         |             25.1 |       49 |          86.0% |               66.32 |               66.75 |
| 80         |             23.0 |       45 |          86.3% |               66.06 |               66.39 |
| 90         |             21.2 |       41 |          80.2% |               66.23 |               66.63 |
| 100        |             21.7 |       42 |          83.6% |               66.22 |               66.51 |
| 110        |             16.4 |       33 |          83.4% |               66.02 |               66.64 |
| 120        |             19.1 |       37 |          83.4% |               66.17 |               66.55 |
| 130        |             28.4 |       55 |          83.2% |               66.19 |               66.65 |
| 140        |             20.1 |       40 |          85.1% |               66.23 |               66.48 |
| 150        |             19.2 |       38 |          86.9% |               66.11 |               66.33 |
| 160        |             24.5 |       47 |          80.9% |               66.12 |               66.35 |
| 170        |             22.6 |       44 |          83.5% |               66.20 |               66.43 |
| 180        |             24.8 |       48 |          73.7% |               65.98 |               66.46 |
| 190        |             18.2 |       35 |          93.2% |               66.14 |               66.72 |
| 200        |             21.0 |       41 |          82.8% |               66.05 |               66.49 |

初始 checkpoint 的整轮 GPU 利用率均值为 94.4%；训练后为 73.7%–93.2%。这与生成期间的瞬时 100% 不矛盾。显存平均约 66.0–66.3 GiB/卡，采样峰值最高约 67.0 GiB；这些是 GPU 已分配显存，不代表 KV block 的使用率。

Step 160 的验证耗时 24.47 分钟。逐卡最后一条生成完成时间，与 GPU 利用率采样对应如下：

| GPU | 最后一条生成完成（距验证开始，分） | 完成后等待至验证结束（分） | 整轮 GPU 利用率均值 |
| --- | ---------------------------------: | -------------------------: | ------------------: |
| 0   |                              23.93 |                       0.54 |               97.9% |
| 1   |                              18.96 |                       5.51 |               76.6% |
| 2   |                              16.81 |                       7.66 |               68.1% |
| 3   |                              22.50 |                       1.97 |               91.5% |
| 4   |                              20.03 |                       4.44 |               80.9% |
| 5   |                              19.95 |                       4.52 |               80.9% |
| 6   |                              18.28 |                       6.19 |               74.5% |
| 7   |                              18.79 |                       5.69 |               76.6% |

在这轮采样中，各卡最后一条生成完成前的 GPU utilization 均值为 100%，完成后的采样均值为 0%。因此，整轮利用率下降与长尾请求造成的设备收尾时间差一致。最后等待还包含评分和结果整理等开销，不能把所有尾部时长都归为可消除的调度开销。这个观察也不能排除生成满载期间的 KV 重算损失。

当前运行没有可靠的 KV 竞争计数。实际配置为 `actor_rollout_ref.rollout.disable_log_stats=True`；现有输出没有完整的逐请求抢占字段，训练器明确记录：

```text
Pipeline metrics omitted: backend did not report preemptions for every request; unknown is not zero
```

结构化 history 中没有 preemption、KV occupancy 或 eviction 指标。由此不能断言“没有抢占”，也无法给出本轮 KV 清理次数。此前按回答长度回放得到的压力比例属于容量模型估计，不是本轮实测 preemption 比例。正常请求完成后释放 KV、权重切换时清空缓存，与容量不足导致的抢占及重算需要分别统计。

后续运行应开启 `actor_rollout_ref.rollout.disable_log_stats=False`，并按 replica 和验证窗口采集 `vllm:kv_cache_usage_perc`、`vllm:num_preemptions_total`、running/waiting 请求数以及生成 token 计数。抢占计数应取验证开始与结束之间的增量；仅开启日志不保证当前 verl 适配器会自动生成逐请求 preemption 指标，需从 vLLM 指标接口单独收集。

数据：[全部 21 轮资源 JSON](../../../../../../../wandb-logs/MirrorOPD-verl/analysis/e39-qwen3base-final-20261027/resources.json)、[step 160 逐卡收尾明细](../../../../../../../wandb-logs/MirrorOPD-verl/analysis/e39-validation-resources-20261027/validation-resources.json)。完整训练结论见[最终分析](../../../experiments/teacher-transfer/failure-reproduction.md#qwen8)，容量模型另见[逐 checkpoint 并发估计](#estimated)。

<a id="native-scheduler"></a>

## MiMo 原生评估调度与参数验证

MiMo 的原生公共评估原先按 benchmark 顺序逐集排空。每个数据集内部已经滚动补位，但跨集等待、固定轮转副本分配，以及同步等待 CPU 判分提交，会在长短回答混合时降低可用并发。调度改动将同一模型版本的题目合并成一个有限请求流，复用 least-inflight 路由；生成与写盘之间设置有界队列，判分独立消费已发布的 Parquet。本节结果对应下述实测源码版本；当前组件接口由[评估包合同](../../../../../../verl/trainer/evaluation/README.md)维护。

架构对照使用 MiMo-7B-RL，输入 tokenizer 为同谱系 Base；两张 GPU，各一个 TP1 replica。比较固定`max_num_seqs=16`、`gpu_memory_utilization=0.65`、`max_num_batched_tokens=16384`、全局请求窗口 32、发布 chunk 16，并开启 vLLM 统计。数学、代码和科学的 response cap 为 24,576，IFEval 为 8,192；温度 1、top-p 0.95、seed 42。两组使用相同模型、问题及每题采样数，仅替换调度实现。

| Benchmark    | 问题数 | 每题 n | 回答数 |
| ------------ | -----: | -----: | -----: |
| MATH-500     |      8 |      1 |      8 |
| AIME24       |      1 |     16 |     16 |
| LCBv5        |      2 |     10 |     20 |
| GPQA-Diamond |      2 |      4 |      8 |
| IFEval       |     12 |      1 |     12 |
| 合计         |     25 |      — |     64 |

题目按准备文件顺序取子集，用于观察跨集补位和长尾，不估计完整 benchmark 的准确率。小子集放大了每集排空的占比，性能结果不能直接外推到完整 6,703 个回答。比较同时核对完整样本覆盖、输出 token 数、截断比例、生成吞吐和端到端耗时；只减少墙钟时间不足以证明相同工作量下吞吐提高。

架构对照已经完成，两个进程均正常退出，64 个样本的判分全部完成并绑定对应 generation manifest。模型／题目／engine contract／采样协议、sample ID 和 seed 均匹配。56 项 CPU 测试通过，覆盖跨集补位、慢消费者的有界背压、动态路由、判分按完成顺序消费、恢复生成、token 映射，以及取消时远端请求与后台写盘的确认。

| 条件                         | 生成耗时 | 输出 token |         生成吞吐 | 撞 cap 回答 | 含模型启动与判分的进程耗时 |
| ---------------------------- | -------: | ---------: | ---------------: | ----------: | -------------------------: |
| 逐集排空／固定轮转           | 327.69 s |    301,267 |   919.36 token/s |        1/64 |                   569.75 s |
| 统一请求流／least-inflight   | 174.79 s |    281,196 | 1,608.80 token/s |        0/64 |                   411.35 s |
| 统一请求流，32 序列／窗口 64 | 149.18 s |    295,519 | 1,980.95 token/s |        1/64 |                   417.07 s |

生成耗时下降 46.66%，吞吐为原来的 1.75 倍，进程总耗时下降 27.80%。生成耗时按 generation 的未写入锁文件创建时间至完成 manifest 的修改时间计算，包含预检和发布，排除模型启动与最后的 CPU 判分排空；新实现另在 manifest 中记录内部计时。

两组只有 3/64 个回答逐 token 相同，新组总输出 token 少 6.66%。批处理顺序变化会影响浮点执行及后续采样，这组自然生成对照支持实际执行收益，不能解释成固定 token 工作量下的纯调度加速。小子集的跨集等待占比和不同回答长度都会影响效果，完整 6,703 条评估的收益仍需实测。

vLLM 日志中可见的 KV 使用率峰值分别为 16.5% 与 20.2%；这是离散日志观测值。新架构收尾至全部请求结束的三次 Prometheus snapshot 显示两副本累计 preemption 均为 0；完整采样位于运行目录。统一请求流仍有最后一批长题的收尾阶段，没有用丢弃慢题或缩短 cap 消除它。

架构对照通过后，每卡序列上限 32／全局窗口 64 的组合也完成了 64 条生成与判分，显存预算仍为 0.65，CUDA Graph 捕获大小增加 32。与新架构的 16 序列组相比，输出 token 增加 5.09%，生成耗时再下降 14.65%，吞吐提高 23.13%。该组生成阶段比旧架构快 54.48%，吞吐为旧架构的 2.15 倍；含启动和判分的进程总耗时却略高于 16 序列组，因此短任务仍需考虑启动成本。

32 序列组的日志 KV 峰值为 23.9%，没有 OOM 或 preemption 告警。其实时指标探针未在集群退出前取到 snapshot，不能把无告警写成已测得的零抢占计数。参数组源码已包含后台线程取消确认辅助函数；最终审阅又将普通 shard 写盘接入此函数，并补充 NativeGenerator 在写盘中取消的端到端 CPU 回归。GPU 实测冻结版本与最终源码因此存在这一处调用差异，性能表保留冻结版本结果。三组模型／题目／协议、sample ID 和 seed 全部匹配。最终 CPU 检查为 56 项通过，三个 GPU 进程均返回 0。

32 序列／窗口 64 是两张 GPU 上独立评估的已验证候选。这批子集不足以覆盖完整评估的稳态 KV 压力，也未验证所有 student checkpoint 或另外两组 model pair；通用 MiMo 训练／评估默认值保留 16 序列。正式队列继续使用既有配置与冻结源码，下一次完整评估可优先比较此候选。

配置、题目 ID、源码 hash、日志、判分及分析结果位于[`runs/eval-scheduler-20261101`](../../../../../../runs/eval-scheduler-20261101)，汇总为[`analysis.json`](../../../../../../runs/eval-scheduler-20261101/analysis.json)。原有 MiMo 正式队列继续使用其冻结源码。后续按请求窗口与引擎序列上限的组合、KV 使用率、waiting／running 与 preemption 计数选点；生成上限、采样次数和训练批量不作为吞吐调参项。

<a id="estimated"></a>

<a id="estimated-qwen3-17b-base-各-checkpoint-的验证并发估计"></a>

## 按真实请求长度估计候选并发

2026-10-27，使用完整验证 step 0–160 的逐请求输入长度、输出长度、实际 replica 分配和提交顺序，估计每卡验证并发的候选范围。每个 checkpoint 都包含 9,120 条回答，提取后的平均长度逐项核对原始验证指标。当前运行的 student 验证上限已经是每卡 64；此前为 Qwen3-8B teacher 准备的并发 16 属于另一模型。

平均回答变短意味着可以提高相对初始 checkpoint 的并发，但均值不足以判断当前 64 是否偏小。Step 140 平均回答为 2.26k token，撞 31,744-token 上限的回答仅占 3.56%，却贡献了约 50.0% 的生成 token。请求在解码队列中的停留时间随长度增长，长尾请求在活跃集合中被放大；文本重复也不会让其历史 KV 自动消失。

估计使用 Qwen3-1.7B-Base 的 BF16 KV、TP1、80% 显存预算。容量参考值是此前并发 24/32 完整测量得到的每卡 554,208 token，即约 59.2 GiB KV，额外留出 10% 余量。该容量来自历史同模型测量，并非当前并发 64 引擎的实时容量读数；CUDA Graph、同卡其他组件占用等差异可能改变实际容量。

设一条回答的输入长度为 $P_i$，输出长度为 $L_i$。在“每条活跃请求每个 decode 轮次生成一个 token”的近似下，一条请求占用的累计 KV-token 轮次为：

$$
A_i=\sum_{t=0}^{L_i-1}(P_i+t)=P_iL_i+\frac{L_i(L_i-1)}{2}.
$$

以请求活跃轮次数加权，每个活跃请求的平均 KV 量估计为：

$$
K_{\mathrm{live}}\approx\frac{\sum_i A_i}{\sum_i L_i},\qquad
C_{\mathrm{mean}}\approx\frac{0.9B}{K_{\mathrm{live}}},\quad B=554208.
$$

这与直接计算 $0.9B/\mathbb E[L]$ 不同。Step 140 的 $K_{\mathrm{live}}$ 约为 8.86k token，平均容量比值约 56，而朴素长度均值法会给出约 221。平均容量比值仍未约束瞬时峰值，不能直接视为安全并发或吞吐最优值。

进一步按每个 replica 的提交顺序回放固定长度请求：模拟连续补充请求、逐 token 增长 KV、请求结束释放 KV。使用等速 decode 轮次，不模拟真实 GPU kernel 时间、prefill 延迟、KV 共享、抢占与重算。为近似 block 对齐，每条请求加 8 token 的预留。所得“压力比例”是反事实回放中 KV 超过 90% 参考容量的 decode 轮次比例，不是实际抢占率或实际 GPU 时间占比。

候选区间下端选择模型压力比例不超过 5% 的最大候选并发；上端选择不超过初始 checkpoint 并发 24 的参考压力（约 17.4%）的最大候选并发。初始 24 是已有实测中吞吐接近 32、抢占更少的选择；把其模型压力当作后续选点标尺是一项启发式。该区间用于减少试验点，不是统计置信区间，也不证明区间内哪一点最快。

| Checkpoint outer step | 平均输出长度 | 撞上限比例 | 活跃请求平均 KV 估计 | 优先试验并发／卡 |
| --------------------- | -----------: | ---------: | -------------------: | ---------------: |
| 0                     |       14.45k |      43.8% |               15.64k |            16–24 |
| 10                    |        2.79k |       5.3% |               10.29k |            32–40 |
| 20                    |        2.92k |       5.7% |               10.60k |            32–40 |
| 30                    |        2.40k |       3.9% |                9.19k |            40–48 |
| 40                    |        2.60k |       4.6% |                9.80k |            32–40 |
| 50                    |        2.42k |       4.0% |                9.23k |            40–48 |
| 60                    |        2.52k |       4.3% |                9.51k |               40 |
| 70                    |        2.66k |       4.6% |                9.65k |            32–40 |
| 80                    |        2.47k |       4.2% |                9.54k |               40 |
| 90                    |        2.29k |       3.4% |                8.63k |            40–48 |
| 100                   |        2.34k |       3.7% |                9.04k |            40–48 |
| 110                   |        1.99k |       2.7% |                7.82k |            48–64 |
| 120                   |        2.15k |       3.2% |                8.46k |            40–56 |
| 130                   |        2.73k |       5.1% |               10.26k |            32–40 |
| 140                   |        2.26k |       3.6% |                8.86k |            40–48 |
| 150                   |        2.25k |       3.4% |                8.65k |            40–48 |
| 160                   |        2.52k |       4.2% |                9.31k |            40–48 |

多数训练后 checkpoint 适合优先比较 32、40、48；step 110 可以上探 64。Step 130 长尾增加，应回到较低区间。当前 64 不能据此断言一定比 48 慢，但这些估计不支持直接升到 128。最终吞吐取决于少量抢占带来的成本、kernel 效率、Graph padding、负载不均与收尾阶段；容量估计需用至少一个训练后 checkpoint 的受控性能比较校准。

在当前 8 卡、全局训练 batch 64、每题一个 rollout 的设置中，训练生成每卡平均只有约 8 条请求。提高引擎上限不会凭空增加训练批量。本记录主要针对每次 9,120 条回答的验证阶段。训练回答上限为 7,168，验证为 31,744；训练长度已被较短上限截断，不能直接代替验证长尾数据。

以后可复用已有初始 checkpoint 的 24/32 测量，避免重复完整扫描；对训练中的后续验证，根据上一次 valid 的完整长度分布更新估计。若需要更早预测，使用少量相同验证协议的长输出样本估计尾部，并给稀有长尾留余量。新 checkpoint 的分布可能波动，因此只按步数单调增加并发不可靠。可以先在一个已有训练后 checkpoint 比较 32、48、64，随后把得到的压力与吞吐关系用于相邻 checkpoint，而不再从初始点重扫全部参数。

当前长程实验的并发未修改。本记录是离线估计，没有增加 GPU benchmark。

数据与复算入口：[逐 checkpoint 表](../../../../../../../wandb-logs/MirrorOPD-verl/analysis/e39-concurrency-estimate-20261027/checkpoint-estimates.csv)、[估计与回放结果](../../../../../../../wandb-logs/MirrorOPD-verl/analysis/e39-concurrency-estimate-20261027/analysis.json)、[逐请求长度](../../../../../../../wandb-logs/MirrorOPD-verl/analysis/e39-concurrency-estimate-20261027/request-lengths.json.gz)、[计算脚本](../../../../research/studies/qwen_base/estimate_validation_concurrency.py)。历史实测参见[学生推理性能记录](../2026-10-27-学生模型推理性能与参数选择.md#qwen3-17b-base)。

<a id="calibration"></a>

<a id="calibration-模型验证资源与并发标定"></a>

## 历史模型资源与子集标定

本页保留 2026-10-24 的冻结资源测量，区分短子集可运行、KV 容量选点和完整验证吞吐。表中的 C student 为 R1-Distill-Qwen-1.5B，D/E student 共用 Qwen3-1.7B-Base；这次独立 student 验证不加载 teacher。

模型协议、EOS 与输出支持的正确性见[对齐诊断](../../model-compatibility/probes/2026-10-24-OPD实现验收与对齐诊断.md)。下面的数值不作为当前配置默认值或全量吞吐最优值。

<a id="calibration-validation-resources"></a>

<a id="calibration-7-统一模型目录与-cde-验证资源实测"></a>

### 7. 统一模型目录与 C/D/E 验证资源实测

2026-10-24 的资产归集是历史操作。当前模型按 repo／revision／subfolder 统一到身份目录，公共入口仍以 `MODEL_ROOT=仓库同级/model` 指定总根目录，具体版本通过注册表解析。下方命令已更新为当前位置；原始实验记录保留当时路径，经迁移清单投影读取。`prepare_models.py`／`check_assets.py` 支持 `--models all` 或指定登记别名，默认仍选 A/B 的两个模型；资产与检查合同见[资产准备](../../training/资产准备.md)。

<a id="calibration-71-窗口修正与测试口径"></a>

#### 7.1 窗口修正与测试口径

公共脚本原来把 student 窗口算成 `prompt + max(train_response, valid_response) + 1`。D/E 即使使用 1024 prompt 和 31744 valid response，也会得到 32769，超过 Base 模型的 32768 上限。现已取消 student 的额外 1；teacher 的独立 vLLM 评分需要在完整训练序列后申请一个 dummy token，其余量仍保留。启动器回归加入了 `1024 + 31744 == 32768` 检查。

真实验证使用 GPU 0–3，通过现有 `run_math_validation_fsdp2.sh student` 执行。每个验证集取前两题，每题 16 回答，共 **6 题、96 回答**；response cap 保持 **31744**，T=0.7、top-p=0.9、TP1、Graph、每引擎 `max_num_seqs=128`、GPU memory utilization=0.65、token budget=8192。C 用 R1-1.5B、prompt cap4096；D/E 共用 Qwen3-1.7B-Base、prompt cap1024、non-thinking、追加停止 ID151645。两组均直接从统一模型目录加载初始权重，不执行 optimizer 更新。

D/E 只在 teacher 不同。学生验证不加载 teacher，因此共用同一次结果；本节覆盖 student 的 val-only 资源，teacher 独立生成及训练评分/反传需要各自测量。

| 指标                          | C student | D/E student |
| ----------------------------- | --------: | ----------: |
| 验证总耗时，秒                |    130.65 |      362.06 |
| 生成阶段，秒                  |    120.32 |      359.98 |
| GPU 0–3 中最大显存采样值，MiB |     54629 |       54785 |
| 自然 stop / 回答数            |     95/96 |       52/96 |
| 撞 response cap / 回答数      |      1/96 |       44/96 |
| OOM／失败请求                 |     无／0 |       无／0 |

显存按 2 秒采样，是观察到的峰值，不能保证捕捉所有瞬时峰值。两组池大小都按 0.65 预分配，因此 `nvidia-smi` 占用接近，并不代表每条回答的 KV 成本接近。

以下准确率仅用于核对小样本链路，**每个基准只有两道题，不能当作完整基准成绩**：

| 基准          | C 小样本 accuracy / 平均长度 | D/E 小样本 accuracy / 平均长度 |
| ------------- | ---------------------------- | ------------------------------ |
| MATH-500 子集 | 90.625% / 7129.63            | 31.25% / 17598.97              |
| AMC23 子集    | 100% / 2016.56               | 43.75% / 12158.34              |
| AIME25 子集   | 62.5% / 10376.97             | 6.25% / 15364.75               |

D/E 的 44 条 length 截断回答全部 format_score=0、score=0，抽查可见循环重复片段。追加 EOS 只修复终止协议，不保证 Base 学会正确作答或及时生成终止符。不能通过缩短验证 cap 隐藏此现象，也不能把小样本时长简单线性外推到完整 570×16 验证。

<a id="calibration-72-kv-容量与并发"></a>

#### 7.2 KV 容量与并发

按实际模型配置，BF16 KV 每 token 的字节数为 `2 × layers × kv_heads × head_dim × 2 bytes`。C student 为 28 KiB/token，一条完整32K约0.875 GiB；D/E student 为 112 KiB/token，一条约3.5 GiB。只看 1.5B 和 1.7B 参数量会漏掉这个 4 倍差异。

D/E 的真实四卡验证日志显示，每卡约 47.35 GiB 可用 KV cache、约 443312 个 token，完整 32K 并发约 13.53；观测到的最高 KV 使用率为 89.1%。该测量包含 96 条回答，每引擎 128 请求的上限未持续压满，日志中未记录抢占；全量负载单独标定。

为隔离缓存压力，另用真实 Base 模型分别在单卡比较 128、12、8 的并发上限：48 个互不共享缓存的 28672-token 合成前缀，各强制生成 1024 个普通 token；固定 0.65 显存比例、8192 token 预算、Graph，关闭 prefix cache。这是调度压力测试，不是数学评测。计时包含 prefill 和生成，不包含模型初始化。

| max_num_seqs | 耗时，秒 | 最大 KV 池使用率 | vLLM 抢占次数 |
| ------------ | -------: | ---------------: | ------------: |
| 128          |    78.80 |          99.989% |             3 |
| 12           |    77.58 |          80.361% |             0 |
| 8            |    81.75 |          53.598% |             0 |

这是一次不同 GPU 上的对照，不据此宣称某个值全局最优。D/E 建议先用 **12**；更保守可用 8。C 的 128 默认值在本轮子集通过，且同一 student 的既有全量调优支持该默认。保留 `gpu_memory_utilization=0.65`、token budget8192、Graph 和 valid n=16。`n` 控制每题样本数，`max_num_seqs` 控制每引擎同时执行的回答数，降低后者不会减少验证样本数。

运行 D/E 学生验证时，除了正常模型路径，还要显式指定 Qwen 协议与 32K 窗口预算：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 STUDENT_GPUS=4 \
STUDENT_MODEL="$(python -m examples.mirroropd.assets.paths --alias Qwen3-1.7B-Base)" \
MAX_PROMPT_LENGTH=1024 CONTINUOUS_TOKEN_MODEL_FAMILY=qwen3 \
ROLLOUT_MAX_NUM_SEQS=12 \
bash examples/mirroropd/run_math_validation_fsdp2.sh student \
  +data.apply_chat_template_kwargs.enable_thinking=False \
  actor_rollout_ref.rollout.stop_token_ids='[151645]'
```

`MODEL_ROOT`、`DATA_ROOT`、`PYTHON_BIN` 按机器实际目录设置；完整验证仍为 570×16。并发 12 的四卡完整负载尚缺复测，单卡合成压力测试用于容量参考。

<a id="calibration-73-结果与复现"></a>

#### 7.3 结果与复现

[验证汇总](../../model-compatibility/probes/data/validation-resources-20261024/validation-summary.json)、[压力汇总](../../model-compatibility/probes/data/validation-resources-20261024/stress-summary.json)、[D/E 缓存日志](../../model-compatibility/probes/data/validation-resources-20261024/DE-vllm-cache-stats.txt)、[C GPU 采样](../../model-compatibility/probes/data/validation-resources-20261024/C-gpu.jsonl)、[D/E GPU 采样](../../model-compatibility/probes/data/validation-resources-20261024/DE-c128-gpu.jsonl)。原始[配置 C](../../model-compatibility/probes/data/validation-resources-20261024/C-config.yaml)／[配置 D/E](../../model-compatibility/probes/data/validation-resources-20261024/DE-config.yaml)、[指标 C](../../model-compatibility/probes/data/validation-resources-20261024/C-metrics.jsonl)／[指标 D/E](../../model-compatibility/probes/data/validation-resources-20261024/DE-metrics.jsonl)、[命令 C](../../model-compatibility/probes/data/validation-resources-20261024/C-command.sh)／[命令 D/E](../../model-compatibility/probes/data/validation-resources-20261024/DE-command.sh)一并保留。完整回答在相应 `outputs/failure-*-valid31k-*/validation/records/0.jsonl`。

压力测试脚本为 [probe_rollout_capacity.py](../../../../benchmarks/probe_rollout_capacity.py)，原始结果为[128](../../model-compatibility/probes/data/validation-resources-20261024/stress-c128.json)、[12](../../model-compatibility/probes/data/validation-resources-20261024/stress-c12.json)、[8](../../model-compatibility/probes/data/validation-resources-20261024/stress-c8.json)。例如：

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 PYTHONPATH=. "$PYTHON_BIN" \
  examples/mirroropd/benchmarks/probe_rollout_capacity.py \
  --model "$(python -m examples.mirroropd.assets.paths --alias Qwen3-1.7B-Base)" --max-num-seqs 12 \
  --output /tmp/rollout-capacity-12.json
```

<a id="calibration-concurrency-calibration"></a>

<a id="calibration-74-全部模型的并发标定历史-af-代号"></a>

#### 7.4 全部模型的并发标定（历史 A–F 代号）

以下冻结记录保留当时的 A–F 代号，依次对应现行 E15/E13、E23/E24、E25/E26、E27/E28、E29/E30、E31/E32；运行时使用 E 编号。随后在 GPU 0–3 完成 22 组首轮测试和 8 组独立子集复核，共 2016 条真实回答。每张卡运行独立 TP1 实例，固定 BF16、Graph、显存比例 0.65、token budget 8192、T=0.7、top-p=0.9、response cap31744。首轮每个基准两题；Qwen 后训练模型及三个 Qwen teacher 的复核每个基准八题、每题四条，共 96 条。自然生成的回答长度会随数值计算和采样变化，不能把所有耗时差归因于调度。

| 角色          | 模型                    | 已测候选起点 | 该配置真实题目 KV 峰值 |
| ------------- | ----------------------- | -----------: | ---------------------: |
| A/B/C student | R1-Distill-Qwen-1.5B    |          128 |                 14.56% |
| D/E student   | Qwen3-1.7B-Base         |           12 |                 84.78% |
| F student     | Qwen3-1.7B Non-thinking |           32 |                 28.61% |
| A/B teacher   | JustRL-DeepSeek-1.5B    |           64 |                  8.00% |
| C teacher     | R1-Distill-Qwen-7B      |           32 |                 11.75% |
| D teacher     | Qwen3-8B Non-thinking   |           32 |                 48.43% |
| E teacher     | Qwen3-4B Non-thinking   |           32 |                 15.44% |
| F teacher     | Qwen3-4B-Instruct-2507  |           32 |                 39.78% |

这些配置的真实子集均无抢占或 OOM。Base 的并发 8/12/16 分别测得 1008/1101/1197 tokens/s；16出现5次抢占，12无抢占，因此保留12作为保守起点。后训练 Qwen3-1.7B 不能直接套用 Base 的 12：其独立复核中 32 的 KV 峰值仅 28.61%，队列未耗尽期间的吞吐更高。该吞吐窗口与其他并发的上下文长度分布不完全相同，不直接外推全量加速倍数。

表内数值是已测起点，**不是吞吐最优值或完整验证的无抢占保证**。仍有 KV 余量的模型尚需继续测更高并发，并保证持续请求供给。R1 student 的 128 还参考了既有 570×16 完整验证；其他模型没有完成相同规模的完整基线。Teacher 数值适用于独立自回归验证，不控制训练中的 shared-sync FSDP 评分。

[30 组汇总](../../model-compatibility/probes/data/validation-concurrency-20261024/summary.json)与[候选配置及边界](../../model-compatibility/probes/data/validation-concurrency-20261024/recommended.json)保存完整数值；同目录 `runs/` 保存逐次结果、命令和压缩遥测。完整回答保留在仓库根 `outputs/validation-concurrency-calibration-20261024/`。复现脚本为 [benchmark_validation_concurrency.py](../../../../benchmarks/benchmark_validation_concurrency.py)。

后续 [benchmark_saturated_rollout.py](../../../../benchmarks/benchmark_saturated_rollout.py) 改用持续补充队列、20 秒预热、60 秒计时，并在截止时终止剩余请求；用于吞吐筛选，不计算准确率，也不证明 32K 稳态稳定性。该脚本只通过 CPU 配置与 prompt 准备检查：GPU 被其他任务占用后，用户暂停了后续实验，尚无对应 GPU 结果。
