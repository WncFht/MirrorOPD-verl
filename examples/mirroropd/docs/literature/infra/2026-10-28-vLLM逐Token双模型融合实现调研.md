# vLLM 逐 Token 双模型融合：公开实现与适配判断

更新日期：2026-10-28。调研范围：vLLM 官方文档、GitHub issue/PR、作者公开仓库和论文中的实现说明。本文进行了源码阅读，没有安装第三方推理环境、运行模型或测量 GPU 性能。作者的运行记录与本项目验收分开表述。

对应需求见[Rollout 双融合实现细节与验收计划](../../infra/rollout/fusion/2026-10-28-Rollout双融合实现细节与验收计划.md)。本文新增调研证据与工程建议，不改变该计划的验收状态。

## 1. 结论

公开代码中已经有 vLLM 逐 token 双模型协同生成。最有用的四份实现是 CoS、vllm-logits、CoDIT 和 DoubtLess。它们分别覆盖通用 ensemble sampler、单引擎双 backbone、双引擎同步，以及异构大小模型在同一 vLLM 请求内执行。

建议保留现有 HF decoder 作为数值参考，把现代 vLLM 的单调度器双模型路径列为首个生产候选。设计可以借鉴 DoubtLess 的模型包装与独立小模型配置、vllm-logits 的注册方式，以及 CoS 的 raw-logits sampler。CoDIT 适合研究分离 GPU 子组的同步协议，但其 CPU/ZeroMQ 整词表传输不宜直接成为生产实现。

这些仓库均不能直接视为已经满足 MirrorOPD 的交付要求：需要补上异构 dense/MoE 配对、current student 在线权重更新、teacher 冻结、训练/验证资源切换、behavior log-prob 和正式长 cap 验收。当前本地验收记录使用 vLLM 0.25.1；公开实现分别针对不同版本，迁移必须固定版本并逐项验证。

## 2. 实现对照与阅读快照

| 项目               | 公开实现的性质                                     | vLLM 版本依据                        | 对本项目的价值                                    | 主要限制                                                     |
| ------------------ | -------------------------------------------------- | ------------------------------------ | ------------------------------------------------- | ------------------------------------------------------------ |
| CoS，ICML 2025     | 修改 vLLM，含普通 ensemble 与 speculative ensemble | 安装说明指定 0.6.5 预编译 wheel      | 有明确的 raw-logits 融合分支与共享请求执行        | 老 worker/sampler 接口；默认 weighted ensemble 是概率平均    |
| vllm-logits        | 外部注册的单引擎双 backbone                        | 依赖限定 0.15.x；作者称测试于 0.15.1 | 直接实现加权 logits；独立 KV 层编号               | 两侧复用同一主体配置，只改层数；默认改写 EOS                 |
| CoDIT，COLM 2026   | 两个 vLLM 引擎，patch sampler 和 scheduler         | README 和 patch 指定 0.11.0          | 有 Qwen3 dense/MoE 脚本，显式回传同一 token       | 每步 CPU/ZeroMQ 传输分布，锁步调度                           |
| DoubtLess，2026-10 | 单个 vLLM 请求驱动大模型与小模型两个视图           | 依赖固定 0.29.0                      | 独立模型尺寸、GPU 融合、CUDA stream、decode graph | 原算法为三路 contrast；含专用 attention/state/scheduler 逻辑 |
| Co-LLM，2024       | vLLM logits 服务与外部逐步协调                     | 依赖固定 0.2.1.post1                 | 历史实现，可参考 raw logits 输出                  | 接口老，基于服务调用，不优先迁移                             |

上述版本均来自仓库依赖或安装说明，不代表与当前任意 vLLM 版本兼容。[^cos-readme][^vl-package][^codit-readme][^dl-package][^collm-req]

| 仓库                                      | 阅读的提交                                 | 提交日期（UTC） |
| ----------------------------------------- | ------------------------------------------ | --------------- |
| Kamichanw/CoS                             | `b7c555324539bb476d39a4002922973e296d1e18` | 2025-06-15      |
| NizarIslah/vllm-logits                    | `a987755aad1a94689745843676d491bc77201e51` | 2026-08-19      |
| Tatsuya736482/contrastive_decoding_public | `a73844a928ba0017605a388425e088b9047a7836` | 2026-07-26      |
| leoluolol/DoubtLess                       | `4f2b3f9e4cf4a5ea6c6ace4b148f5699895bb372` | 2026-10-15      |
| clinicalml/co-llm                         | `d83ab8cf19ceb49c7814383a897089e51cdeebfe` | 2024-05-04      |

## 3. CoS：已有通用 raw-logits ensemble 接口

CoS 的 `EnsembleDecodeWorker` 将同一 `ExecuteModelRequest` 交给多个 worker；每个模型保存自己的 KV，收集各模型 logits 后调用一个 `EnsembleSampler`，只向外返回一个采样结果。缓存预算按各模型每块 KV 的字节数之和计算，并初始化相同数量的块。该执行骨架直接对应“同一前缀、双 KV、统一采样”。[^cos-worker]

`EnsembleSampler` 明确区分三种融合位置：[^cos-sampler]

| 参数         | 实际顺序                                                       | 对本项目的适用性                                               |
| ------------ | -------------------------------------------------------------- | -------------------------------------------------------------- |
| `raw-logits` | 先 ensemble，再公共 penalty/temperature/filter，最后归一化采样 | 与现有 raw-token logit fusion 目标相符；仍需统一精度和合法词表 |
| `logits`     | 各模型分别做处理，再 ensemble                                  | 有过滤时与本项目目标不同                                       |
| `probs`      | 各模型分别处理并 softmax，再 ensemble                          | 属于概率层融合                                                 |

默认 `configs/method/we.yaml` 选择 `probs`，计算 $(1-\lambda)p_0+\lambda p_1$。它还把第一个模型的 `logprobs[0]` 原样返回，因此默认配置返回的 log-prob 不能作为混合行为策略的概率。raw-logits 分支则在融合后重新计算 softmax/log-softmax。迁移时应选择该分支，并核对实际采样实现中最终过滤与 log-prob 的一致性。[^cos-we][^cos-sampler]

建议复用其接口划分和共同请求语义，不整体移植旧 vLLM fork。CoS 的安装版本早于当前项目环境，核心执行代码仍使用旧 `Worker`、`SequenceGroupMetadata` 和 `SamplingMetadata.prepare`。README 中的 1.11–2.23 倍性能收益是作者对其协同解码设置的报告，不能推导为本项目 batch64/cap7168 的收益。[^cos-readme][^cos-worker]

CoS 同时提供 speculative 路径。这个方向可留到普通双模型路径完成之后；普通 speculative decoding 保持某个目标分布，不能通过简单开启 draft model 配置就自动变成目标为 fusion 分布的采样。

## 4. vllm-logits：直接 logits 加权，但异构配置和 EOS 需要修改

该包通过 `vllm.general_plugins` 注册双模型类。两个 backbone 对同一 input IDs 和 positions 执行 forward；返回拼接的 hidden states，在 `compute_logits` 中分别经过各自 LM head，最后计算

$$
z=(1-\alpha)z_A+\alpha z_B.
$$

模型层编号做偏移，给两侧 attention 分配不同的 KV 层身份。这个实现证明可以把双模型包装成一个 vLLM 模型，沿用引擎的请求和采样路径。[^vl-package][^vl-backbones]

源码中的边界比 README 的概括更重要：

- 两侧配置从同一个 `vllm_config` 深拷贝，只分别修改 `num_hidden_layers`。两个 LM head 也取同一个 `hf_config.hidden_size`。它没有为任意尺寸配对加载两份独立完整配置，不能据 README 列出 0.6B、1.7B、4B、8B 就认为这些尺寸可以任意交叉配对。
- 默认保护 Qwen 的 151645 等 token：融合之后直接使用 specialist 对这些位置的 logits。这改变 raw-token fusion 的停止概率。迁移时需要显式取消这种保护，或把它声明为独立算法。
- 虽然 README 使用“no extra forward pass”的说法，源码在非端点时分别执行 `model_a(...)` 与 `model_b(...)`，然后计算两个 head。应理解为一次引擎调用内部完成两模型计算，不能理解为双模型只消耗单模型 FLOPs。
- `load_weights` 从两份本地 checkpoint 读取。它不是现成的 current student 流式更新接口；要单独设计训练权重到 student 子模块的映射和版本发布。

这些判断来自 `backbones.py`；保留该包装思路时，应重新实现独立模型配置和权重接收路径。[^vl-backbones]

## 5. CoDIT：真实双引擎协调代码，通信和请求身份仍需重做

CoDIT 的 expert/amateur 各有 vLLM 引擎。Amateur 计算完整 log-probs，经 coordinator 传给 expert；expert 构造 contrastive logits 后调用原 sampler，把采样 token 传回 amateur。Amateur 将该 token 放入自己的 `SamplerOutput`，两边下一步继续消费同一生成 token。[^codit-sample]

其 sampler 实现的目标为带 expert plausibility mask 的 contrastive 分布，核心分数是 $\log p_E-\beta\log p_A$。改成本项目的正权重 logit fusion，需要去掉该算法专用 mask，替换系数，并保留一次公共温度与过滤。

完整、未截断的 log-probs 可以用于同样的线性融合。令 $\log p_M(v)=z_M(v)-\log Z_M$，则

$$
(1-\alpha)\log p_S(v)+\alpha\log p_T(v)
=(1-\alpha)z_S(v)+\alpha z_T(v)-C,
$$

其中 $C=(1-\alpha)\log Z_S+\alpha\log Z_T$ 与 token 无关，经过公共温度与 softmax 后常数消去。这个等价关系不允许各模型先做不同的 top-k/top-p 截断，也不保证任意非线性 logits 处理与加常数可交换。

公开 Qwen3-30B-A3B 脚本明确使用 instruct/base 两个 30B-A3B checkpoint，每边 TP2，分别占 GPU 0–1 和 2–3；设置 batch200、max_tokens4096。这个配置是公开运行入口的证据，不是我们已经复现该负载的证据，也不是 1.7B + 30B-A3B-Instruct-2507 的配对验收。[^codit-script]

源码中的迁移重点如下：

| 项目     | 已核实实现                                                                       | 本项目需要的处理                                      |
| -------- | -------------------------------------------------------------------------------- | ----------------------------------------------------- |
| 分布传输 | `.cpu()` → NumPy → ZeroMQ → tensor 上 GPU                                        | 优先同执行侧融合；确需跨 GPU 时使用设备侧通信并实测   |
| 调度     | patch Scheduler 与 KVCacheManager，通过共享内存/信号量协调 running 与 preemption | 明确共同请求计划、重排、取消及排空协议                |
| 请求身份 | running IDs 转为数值并放入 float32 数组；sampler hash 基于嵌套列表形状和 step    | 使用稳定 request/session ID，不把形状相同视为身份一致 |
| 初始阶段 | 根据“全部 output IDs 为空且达到历史最大 batch”跳过同步                           | 单独检查真实第一 token 与 warmup 是否会混淆           |
| 参数规模 | 启动脚本共用两边 TP 参数                                                         | 异构 GPU 子组和不同 TP 必须扩展                       |

上述事实分别见 sampler 和 scheduler patch。[^codit-sample][^codit-scheduler]

按 $B=64,V=151936,\text{FP32}$ 估算，一个完整词表张量为 37.09 MiB。假设 64 个请求都活跃并持续 7168 步，仅每步一个这样的张量就累计约 259.66 GiB 数据；这不含额外拷贝和协议开销，真实 EOS/活跃请求减少会降低总量。这是通信量估算，不能换算为未经测量的耗时或加速倍数。

## 6. DoubtLess：最值得参考的现代单引擎异构执行路径

DoubtLess 原始算法使用一个大模型和一个小模型的 brief/verbose 两种视图：

$$
z=z_{\mathrm{base}}+\alpha(z_{\mathrm{brief}}-z_{\mathrm{verbose}}).
$$

它同时有“两个模型均在 vLLM 中”和“vLLM 大模型 + Transformers 小模型”两条后端。这里讨论前一条，不能把两个后端的模型覆盖范围混在一起。README 的纯 vLLM 示例为 Qwen3-32B + Qwen3-0.6B，依赖固定 vLLM 0.29.0。[^dl-readme][^dl-package]

`ContrastForCausalLM` 从 small checkpoint 单独读取 `AutoConfig`，分别保存大、小模型 hidden size。两个模型吃同一请求的生成 token，独立产生 hidden states 和 logits，再把融合分数交回 vLLM sampler。小模型放在第二个 CUDA stream；engine 配置 `FULL_DECODE_ONLY` graph。请求状态还配合 batch 更新、instruction KV 和 preemption replay 处理。[^dl-model][^dl-engine][^dl-sched]

因此可以从中借鉴以下设计：一个请求只保留一份生成序列身份；每个模型拥有自己的结构、权重和 KV；生成循环内部完成两侧 forward 与融合；统一交给 vLLM sampler。实现简单的共同前缀双模型 fusion 时，可以省去其 brief/verbose 指令插入与三视图部分。是否真的能省去所有专用 hook，需要针对选定 vLLM 版本验证。

该代码也包含 `Qwen3MoeForCausalLM` 的大模型类映射，但这只证明代码有该入口。论文附录描述的主布局利用了相同 KV head 数和 head dimension；小模型实现限定 full-attention Qwen3/Llama/Qwen2。不能据映射存在就宣布 dense + 30B MoE、不同 KV 几何形状和任意 TP 已经验收。[^dl-model][^dl-paper]

此外，DoubtLess 只在 thinking 区间应用 contrast，保留 base 的 EOS 与 closing-tag 分数，并使用 base 的候选 mask。复用执行结构时应删除这些算法特定规则，保持本项目既定 raw-token fusion。其 `load_weights` 支持分开加载 base/thinker 权重，可作为命名路由参考，但 current student 的原子 refit、teacher 保护和 FSDP2 集成仍需实现。[^dl-model]

## 7. 官方能力与其他相关工作

vLLM 官方支持外部模型注册，也有批级 logits processor。processor 的 `apply(logits)` 接收当前 batch 的词表张量，并通过 `BatchUpdate` 维护加入、删除和重排的请求状态。这个接口适合承载融合运算，但第二模型的加载、forward、KV 和同步仍需由自定义模型或执行器提供。[^vllm-registration][^vllm-logits]

官方 Controlled Decoding issue #9541 明确讨论每 timestep 多模型执行，目前页面状态为 closed / not planned。结合文档和公开 issue/PR 检索，没有核实到上游提供可直接配置的通用双 checkpoint token-fusion 产品接口。这个结论不否定外部插件的可行性，也不把单个关闭的 issue 当作全部上游能力的证明。[^vllm-issue]

其他来源的作用如下：

| 来源       | 可确认的关联                                                                                           | 不能据此推断的能力                                                      |
| ---------- | ------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------- |
| Co-LLM     | 提供 vLLM raw-logits server，contrastive/proxy 基线在外部协调                                          | 不等于现代引擎内、低通信开销的双模型执行                                |
| ThinkMerge | 论文说明两阶段 vLLM pipeline，答案阶段平均 pre-softmax logits 并反馈同一 token；one-step 版本当时用 HF | 主要是同一模型多个推理前缀；未核实到可直接移植的公开 one-step vLLM 仓库 |
| UltraFuser | 论文提出多专家 token-level gating，并讨论高效推理                                                      | 不据论文描述认定已找到可运行的现代 vLLM 交付包                          |
| TRB        | 提供行为分布构造方法                                                                                   | 此次未定位到可核实的作者 vLLM 双模型实现                                |

Co-LLM 的接口与依赖见作者仓库；ThinkMerge 的 one-step 发布承诺见论文第 4.1 节；UltraFuser 与 TRB 仅在这里作为方法相关来源。[^collm-readme][^collm-server][^thinkmerge][^ultrafuser][^trb]

## 8. 对 MirrorOPD 的具体建议

把原计划“优化 HF 或接 vLLM/SGLang，由测量决定”进一步落实为：HF reference 保留；先实现并验收一个现代 vLLM 双模型候选；采用哪个生产实现仍由同负载测量决定。

首选候选使用一个调度器和组合模型类。Teacher 与 student 分别从完整配置构建，使用独立参数命名与 KV 身份，输入相同真实 token IDs。在生成执行侧计算 FP32 加权 logits，之后只有一套公共温度、top-k/top-p 与 sampler。这个建议依据 DoubtLess 与 vllm-logits 的代码结构，不是已完成的本地设计或性能结论。

| 实施事项          | 可参考来源                                       | 必须补齐的项目要求                                              |
| ----------------- | ------------------------------------------------ | --------------------------------------------------------------- |
| 外部模型类注册    | vllm-logits、DoubtLess、官方 registry            | 在 actor/Ray/vLLM 子进程中一致加载；固定 vLLM 版本              |
| 两份独立模型配置  | DoubtLess                                        | 1.7B + 8B，以及 dense + MoE；分别处理 hidden size、KV heads、TP |
| 同前缀与统一采样  | CoS raw-logits 分支                              | 合法 token-ID 映射、FP32 融合、公共过滤、最终 behavior log-prob |
| current 权重更新  | 本地命名 target 协议；DoubtLess 的子模块权重路由 | 仅更新 student；完整接收后发布版本；失败保持不可用              |
| KV 与请求生命周期 | 单引擎结构；CoDIT 调度同步作为对照               | batch 重排、preemption、取消、refit 清 KV、恢复不串请求         |
| 资源切换          | 本地 shared-sync runtime                         | sleep/wake、teacher scoring、validation 引擎和训练内存预算      |
| 正式吞吐          | 原计划 G7                                        | 两组配对、batch64/cap7168、重复完整批测量、全部同步/切换成本    |

如果组合模型在所选版本上不能可靠处理异构 KV 或所需 GPU 布局，再比较双引擎方案。该候选需要一个共同调度决定、设备侧 logits 传输、一次采样和 token 广播。不能让两个独立 continuous-batching scheduler 只凭 batch 行号对应请求。

建议用以下顺序做下一阶段验证：

1. 固定版本；先以 1.7B + 8B、短 cap、共同前缀回放比较 HF/vLLM logits 与最终分布。验证 $\alpha=0,0.5,1$，并测试第一步 EOS。
2. 验证不同长度、请求重排、preemption/recompute、取消和停止；只比较相同前缀上的数值，不把相同 seed 必须生成完全相同轨迹当作跨后端正确性的唯一判据。
3. 接入 current refit：更新前后 student 确实改变，teacher 不变；失败不发布新版本。完成训练→验证→再训练与恢复。
4. 扩展到 30B-A3B-Instruct-2507，并按原 G7 测正式 cap。单独统计有效 token 数和长度分布，避免把更早停止解释成推理内核加速。
5. 普通双模型路径完成后，再评估 CoS 风格 speculative fusion；验收目标应是混合分布本身及其行为概率。

### 参考文献

[^cos-readme]: Jiale Fu et al. CoS README，阅读提交 b7c5553，2025. [安装、方法与代码入口](https://github.com/Kamichanw/CoS/blob/b7c555324539bb476d39a4002922973e296d1e18/readme.md).

[^cos-worker]: CoS. EnsembleDecodeWorker，阅读提交 b7c5553. [执行与 KV 预算源码](https://github.com/Kamichanw/CoS/blob/b7c555324539bb476d39a4002922973e296d1e18/vllm/vllm/ensemble_decode/ensemble_decode_worker.py).

[^cos-sampler]: CoS. EnsembleSampler，阅读提交 b7c5553. [三种融合位置与 sampler 源码](https://github.com/Kamichanw/CoS/blob/b7c555324539bb476d39a4002922973e296d1e18/vllm/vllm/ensemble_decode/ensemble_sampler.py).

[^cos-we]: CoS. Weighted ensemble 配置，阅读提交 b7c5553. [we.yaml](https://github.com/Kamichanw/CoS/blob/b7c555324539bb476d39a4002922973e296d1e18/configs/method/we.yaml).

[^vl-package]: Nizar Islah. vllm-logits 包配置，阅读提交 a987755，2026. [依赖与插件注册](https://github.com/NizarIslah/vllm-logits/blob/a987755aad1a94689745843676d491bc77201e51/pyproject.toml).

[^vl-backbones]: Nizar Islah. vllm-logits 双模型实现，阅读提交 a987755. [backbones.py](https://github.com/NizarIslah/vllm-logits/blob/a987755aad1a94689745843676d491bc77201e51/src/vllm_logits/backbones.py).

[^codit-readme]: Tatsuya Ichinose et al. CoDIT README，阅读提交 a73844a，2026. [安装、支持模型与使用说明](https://github.com/Tatsuya736482/contrastive_decoding_public/blob/a73844a928ba0017605a388425e088b9047a7836/README.md).

[^codit-sample]: CoDIT. vLLM sampler patch，阅读提交 a73844a. [分布交换、融合与 token 回传](https://github.com/Tatsuya736482/contrastive_decoding_public/blob/a73844a928ba0017605a388425e088b9047a7836/src/patch_vllm_sample.py).

[^codit-scheduler]: CoDIT. vLLM scheduler patch，阅读提交 a73844a. [调度、请求身份与 preemption](https://github.com/Tatsuya736482/contrastive_decoding_public/blob/a73844a928ba0017605a388425e088b9047a7836/src/patch_vllm_scheduler.py).

[^codit-script]: CoDIT. Qwen3-30B-A3B 脚本，阅读提交 a73844a. [模型、TP、batch 与 cap](https://github.com/Tatsuya736482/contrastive_decoding_public/blob/a73844a928ba0017605a388425e088b9047a7836/scripts/run_Qwen3-30B-A3B.sh).

[^dl-readme]: Minghao Luo, Liang Chen. DoubtLess README，阅读提交 4f2b3f9，2026. [双后端与模型配对](https://github.com/leoluolol/DoubtLess/blob/4f2b3f9e4cf4a5ea6c6ace4b148f5699895bb372/README.md).

[^dl-package]: DoubtLess. 包配置，阅读提交 4f2b3f9. [固定 vLLM 0.29.0](https://github.com/leoluolol/DoubtLess/blob/4f2b3f9e4cf4a5ea6c6ace4b148f5699895bb372/pyproject.toml).

[^dl-model]: DoubtLess. 组合模型、attention 与权重加载，阅读提交 4f2b3f9. [model.py](https://github.com/leoluolol/DoubtLess/blob/4f2b3f9e4cf4a5ea6c6ace4b148f5699895bb372/doubtless/vllm/model.py).

[^dl-engine]: DoubtLess. 引擎与请求配置，阅读提交 4f2b3f9. [engine.py](https://github.com/leoluolol/DoubtLess/blob/4f2b3f9e4cf4a5ea6c6ace4b148f5699895bb372/doubtless/vllm/engine.py).

[^dl-sched]: DoubtLess. preemption replay hook，阅读提交 4f2b3f9. [sched.py](https://github.com/leoluolol/DoubtLess/blob/4f2b3f9e4cf4a5ea6c6ace4b148f5699895bb372/doubtless/vllm/sched.py).

[^dl-paper]: Minghao Luo, Liang Chen. DoubtLess: Training-Free Contrastive Decoding for Efficient Reasoning. arXiv:2610.18862v1，2026，附录 D. [实现细节](https://arxiv.org/html/2610.18862v1#A4).

[^collm-readme]: Shannon Zejiang Shen et al. Co-LLM README，阅读提交 d83ab8c，2024. [服务与推理流程](https://github.com/clinicalml/co-llm/blob/d83ab8cf19ceb49c7814383a897089e51cdeebfe/README.md).

[^collm-req]: Co-LLM. requirements，阅读提交 d83ab8c. [vLLM 版本依赖](https://github.com/clinicalml/co-llm/blob/d83ab8cf19ceb49c7814383a897089e51cdeebfe/requirements.txt).

[^collm-server]: Co-LLM. raw logits 服务，阅读提交 d83ab8c. [api_server_logits.py](https://github.com/clinicalml/co-llm/blob/d83ab8cf19ceb49c7814383a897089e51cdeebfe/collm/inference/api_server_logits.py).

[^vllm-registration]: vLLM 官方文档。Registering a Model，2026-10-28 访问。[外部模型注册](https://docs.vllm.ai/en/latest/contributing/model/registration/).

[^vllm-logits]: vLLM 官方文档。Logits Processors，2026-10-28 访问。[批级接口与 BatchUpdate](https://docs.vllm.ai/en/latest/design/logits_processors/).

[^vllm-issue]: vLLM. Support for Controlled Decoding，issue #9541，2024 创建，2026-10-28 访问。[需求与状态](https://github.com/vllm-project/vllm/issues/9541).

[^thinkmerge]: Haonan Wang et al. Think in Parallel, Answer as One: Logit Averaging for Open-Ended Reasoning. arXiv:2512.02874v1，2025，第 4.1 节。[实现说明](https://arxiv.org/html/2512.02874v1#S4.SS1).

[^ultrafuser]: Ning Ding et al. Fusing Highly Specialized Language Models for Comprehensive Expertise. ACL，2025. [论文与作者信息](https://aclanthology.org/2025.acl-long.42/).

[^trb]: Daniil Plyusov et al. Trust-Region Behavior Blending for On-Policy Distillation. arXiv:2605.31159v1，2026. [论文](https://arxiv.org/html/2605.31159v1).
