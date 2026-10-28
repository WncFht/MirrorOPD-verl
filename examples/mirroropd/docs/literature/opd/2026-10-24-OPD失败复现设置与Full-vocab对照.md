# OPD 原论文设置与 Full-vocab 核对档案

**当前执行方案已独立整理到[失败复现目录](../../infra/model-compatibility/probes/README.md)：使用本项目 DAPO 与三个 valid，先 sampled-token，再 full-vocab，首轮不做 top-k。** 本文保留此前的作者参数核对、候选比较和配置解析记录；以下 n=4、top-k 及 200 步示例不代表当前执行选择。

日期：2026-10-24。目标是先复现 sampled-token 或 top-k OPD 的停滞、负迁移或长程失稳，再检验完整词表 reverse KL 是否改善。本文核对论文正文、附录、公开脚本和本仓实现；不把未公开的参数补成“论文设置”。本仓源码核对基于 `376e8185e` 及当前工作区，尚未启动训练。

证据标记：**P** 为论文明确写出；**C** 为公开代码；**L** 为本仓源码或本机资产；**R** 为建议的复现选择。P 与 C 不一致时分别保留。现有 E13–E20 的结果与预算仍由[训练配方](../../settings/训练设置.md)和[五组实验分析](../../experiments/learning-signal/sampled-vs-full/2026-10-24-五组Math实验对比分析.md)维护，本文不改现有队列。

## 1. 优先复现什么

| 设置 | 要复刻的失败 | 对 Full-vocab 的研究价值 | 当前工程状态 |
| --- | --- | --- | --- |
| R1-Distill-1.5B student，JustRL-1.5B teacher，训练窗口从 7K 扩到 15K | Rethinking OPD 的长窗口后期失稳，默认为 student-top-16 | 区分单词元／支持截断问题与坏前缀问题 | 两个模型已在本机，架构与词表满足现有 full-KL；严格 top-k 基线尚未实现，可先做 sampled/full 长窗预实验 |
| R1-Distill-1.5B student，R1-Distill-7B teacher | 强教师几乎不能带来提升；换 JustRL-1.5B teacher 则成功 | 检查完整监督是否能挖出 top-k 没利用的信号 | 缺 7B 权重；输出 head 分别 151936／152064 行，现有 full-KL 拒绝，需要词表对齐 |
| JustRL-1.5B student，R1-Distill-7B teacher | 较强 7B 教师使学生回退，损失其 RL 收益 | 检验 full-vocab 能否避免错误能力覆盖 | 同样有词表行数差异；比前项更强的负迁移测试 |
| Qwen3-1.7B-Base student，Qwen3-8B teacher | TIDE 中 sampled OPD 九项平均 Avg@8 仍为 6.9%，长度膨胀、答案缺失 | 失败明确、checkpoint 公开，最值得作为主失败 benchmark | 当前 full-KL 仅支持 untied Qwen2；Qwen3 与 tied embedding 尚需适配 |
| Qwen2.5-7B-Instruct student，OpenThinker3-7B teacher | Revisiting 的局部负迁移与不稳定；teacher-top-32 有改善 | 可以直接比较 sampled、teacher-top-32、full-rKL | 两者是 untied Qwen2、输出维数相同；缺权重与完整验证数据，需核 token-ID 和特殊标记 |
| Qwen3-1.7B Non-thinking student，Qwen3-4B-Instruct-2507 teacher | Semi-OPD 中低重叠配置：在线采样不如初始学生缓存 | 检验 full-vocab 能否缓解在线前缀退化 | 需 Qwen3/tied-head 适配，且 DAPO 过滤口径不同 |

表内保留原始候选范围。当前 A–F 选择、sampled→full 顺序和后置设置由[新方案](../../experiments/teacher-transfer/teacher-comparison/failure-matrix/2026-10-25-Sampled-token与Full-vocab失败复现方案.md)统一维护。R1 的跨规模模型仍需要词表对齐，不能只把 `TEACHER_MODEL` 换成 7B 就直接跑现有 full-KL。

这些是不同的失败：不提升、覆盖学生已有能力、长序列崩溃、某个基准退化，应该分别报告。只要 full-KL 有收益，不能就声称它解释了所有 OPD failure。

## 2. Batch、rollout 与 optimizer update 的口径

设 $B$ 为一次外层步骤的 prompt 数，$n$ 为每题生成数，$M$ 为 `ppo_mini_batch_size` 配置值，$E$ 为 `ppo_epochs`。在本仓 V1 路径中，实际序列 mini-batch 会乘上 $n$：

$$
B_{\rm seq}=Bn,\qquad M_{\rm seq}=Mn,\qquad
U_{\rm outer}=E\,\frac{B}{M}.
$$

以上假设整除、无补齐额外步骤且没有跳过更新。[V1 `_update_actor`](../../../../../verl/trainer/ppo/v1/trainer_base.py)明确把 mini-batch 乘以 rollout.n。

| 配置 | 每次生成回答数 | 每次 optimizer update 的回答数 | 每个 outer 的更新数 |
| --- | ---: | ---: | ---: |
| 当前默认：B=64，n=1，M=64，E=1 | 64 | 64 | 1 |
| Rethinking／TIDE：B=64，n=4，M=64，E=1 | 256 | 256 | 1 |
| 将 Revisiting 的 128 条回答、64 条回答 mini-batch 映射到本仓：B=16，n=8，M=8，E=1 | 128 | 64 | 2 |

因此，**64 prompts × 4 rollouts 不是总共 64 条回答**；也不要因为有 256 条回答，就误以为 `mini_batch_size=64` 一定产生 4 次更新。本仓与 Rethinking 的 verl worker 都有乘 $n$ 的逻辑。Revisiting 使用 `env.rollout.n=8`，其 actor 的 `rollout.n` 默认 1，参数语义需要单独翻译。

## 3. Rethinking OPD：最贴近当前模型的系列设置

论文：arXiv:2604.13016，重点 §3、§4.1、§5.1、§6.1、附录 A.2。公开代码核对到 `ac26e38d6f1572eb027597b48a9f4e01f6915ef8`。[^rethinking][^rethinking-code]

### 3.1 模型对与失败定义

| 子设置 | Student | Teacher | 论文现象 |
| --- | --- | --- | --- |
| 无收益 | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | §4.1 中验证曲线基本不改善，overlap 及梯度动态停滞 |
| 成功对照 | 同上 | `hbx/JustRL-DeepSeek-1.5B` | 明显提升，恢复较大比例的师生性能差距 |
| 负迁移 | `hbx/JustRL-DeepSeek-1.5B` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | §3.3 中学生回退，和向自身 pre-RL 检查点蒸馏的方向相似 |
| 思考模式不兼容 | `Qwen/Qwen3-1.7B-Base` | `Qwen/Qwen3-4B`，Non-thinking | §3.1／§5.1 中直接 OPD 较差，教师回答 SFT 冷启动后改善 |
| 模式兼容教师对照 | `Qwen/Qwen3-1.7B-Base` | `lllyx/Qwen3-4B-Base-GRPO` | 与 Qwen3-4B Non-thinking 比较；不能把它与另一篇论文的“4B-GRPO”混为同一个 checkpoint |
| 长程失稳 | R1-Distill-1.5B | JustRL-1.5B | §6.1 中 3K／7K 较稳定，10K／15K 出现后期失稳，异常从后缀向前传播 |

### 3.2 公共训练配置

| 参数 | 数值／设置 | 证据 |
| --- | --- | --- |
| Train dataset | DAPO-Math-17K | P/C |
| Prompt batch | 64 | P/C |
| Rollouts per prompt | 4 | P/C |
| 实际序列 batch | 256 | 由代码语义换算 |
| PPO mini-batch 配置 | 64；worker 乘 4 后为 256 条回答 | P/C |
| 每批优化轮数 | 1；与上述 batch 配置合起来每 outer 一次更新 | C |
| LR | `1e-6` | P/C |
| Scheduler | 公开代码默认 constant；未打开 cosine 分支 | C |
| Train epochs | 1 | P/C |
| 长窗口消融步数 | §6.1 正文写 200 steps；图 13 讨论 step180–250，需保留这一报告差异 | P |
| Max prompt length | 1024 | P/C |
| 默认训练响应上限 | 7168 | P/C |
| 长窗口变体 | 3072、7168、10240、15360 等；原脚本注释明确列出 15360 | P/C |
| Sampling temperature | 1.0 | P/C |
| Sampling top-p | 1.0 | P；配置继承需保留 |
| Sampling top-k | 不截断；不同于蒸馏支持集 K | C/迁移时显式固定 |
| 蒸馏支持集 | student top-16；`LOG_PROB_TOP_K=16, TOP_K_STRATEGY=only_stu` | P/C |
| 采样词元版本 | `LOG_PROB_TOP_K=0` | C |
| Teacher temperature | 1.0 | C |
| Loss aggregation | token-mean | C |
| Reference KL／format reward | 关闭 | P/C |
| 数据打乱 | `data.shuffle=False` | C |
| Actor 权重初始化 dtype | `model_dtype=fp32`；不能据此声称所有前向乘法均 FP32 | C |
| Hardware | 8×A800 80GB | C README |

Adam 系数、weight decay、混合精度、gradient clipping 等未在论文表 2 逐项报告，严格复现应解析该 commit 的完整 Hydra 配置。下面的本仓提案采用本仓已明确的 AdamW `(0.9,0.999)`、wd=0.01、grad clip=1、无 warmup，标为 R/L，不冒充论文表 2 给定值。

### 3.3 验证配置

| 参数 | 设置 |
| --- | --- |
| Valid/test benchmarks | AIME2024、AIME2025、AMC2023 |
| 每题生成数 | 16 |
| Temperature／top-p | 0.7／0.95 |
| Max response length | 31744 |
| 主指标 | 每个基准 avg@16；不要替换为 pass@16 |
| 问题模板 | `{problem} Please reason step by step, and put your final answer within \boxed{}.` |
| Chat template | 模型自身模板；公开评测脚本默认 `enable_thinking=False` |
| 验证流程 | 公开仓库建议单独跑 `scripts/val/eval/`，而非直接依赖其旧 verl 内置验证 |

作者 README 指出，verl v0.7.0 内置验证曾在其运行中低估 5–7 个百分点，建议训练时关闭该路径、离线评测 checkpoint。**这不是本仓也必然存在同一 bug 的结论**，但意味着需要先用共同评测流程重测原始 student 与 teacher，不能把不同验证脚本的分数直接拼接。[^rethinking-code]

训练模板也重要。论文的标准 DAPO 设置包含原始“最后一行 Answer:”指令，§5.2 专门比较了它与 teacher-aligned 的 boxed 模板。本仓导入脚本去掉了原 DAPO 前后缀，现有 parquet 保存的是裸题目；严格复现需生成单独的论文格式数据，不能只对齐模型和 LR。

### 3.4 论文 top-k 等式与公开实现需要分别标注

论文将 top-k 写成支持集内部重新归一化的 reverse KL。公开代码 `dp_actor.py::compute_reward_weights` 的默认 `only_stu` 分支则：

1. 用选出的学生 log-prob 在 K 维做 softmax，得到归一化学生权重；
2. 用**原始全词表归一化的**师生 log-prob 之差构造优势；
3. 将停止梯度的候选优势交给 PG 更新，候选维度求和。

因此迁移时应区分 **作者代码的 student-top-k PG** 与 **对支持内归一化 KL 直接 backward**，不能未经梯度核对就当成同一个实现。若目的是复刻作者 failure，优先保持作者代码路径；若目的是研究支持截断偏差，可另加直接归一化 KL 作为机制对照。[^rethinking-actor][^rethinking-core]

## 4. TIDE：强失配、停滞和重复最明确的主测试

论文：arXiv:2608.09836。主文表 1、表 3 和附录 D 给出了较完整的设置。代码核对到 `b54a9e15b2064a8d18c36f3cd3f73ec8906cefc5`。[^tide][^tide-code]

### 4.1 模型与目标现象

- **Failure pair**：student=`Qwen/Qwen3-1.7B-Base`，teacher=`Qwen/Qwen3-8B`。注意 student 必须是 Base，不能换成`Qwen3-1.7B`的 post-trained 版本。
- **成功对照**：student=`deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`，teacher=`hbx/JustRL-DeepSeek-1.5B`。
- **Failure 定义**：强失配配置中，未训练 student 与普通 sampled OPD 九项宏平均 Avg@8 均为 6.9%；OPD 平均响应长度 22395、无 boxed 答案比例 65.5%。TIDE 相应为 20.3%、7294 词元、5.4%。复现目标是停滞伴随生成退化，不只是要求某个小基准分数低。

### 4.2 训练配置

| 参数 | 设置 | 证据 |
| --- | --- | --- |
| Train dataset | DAPO-Math-17K | P/C |
| Data shuffle | False | P/C |
| Prompt batch／n | 64／4 | P/C |
| 实际序列 batch | 256 | P/C 换算 |
| Mini-batch 配置 | 64；实际 256 条回答 | P/C 换算 |
| Micro-batch per GPU | 1，允许动态 token batching／梯度累积 | P/C |
| Update per rollout batch | 1 | P，附录 D.1 |
| Train epochs | 1，最终 checkpoint 报告，不按 test 挑最好 checkpoint | P |
| Optimizer | AdamW | P |
| LR／schedule／warmup | `1e-6`／constant／none | P/C |
| Adam β／weight decay | `(0.9,0.999)`／0.01 | P |
| Gradient clip | 1.0 | P |
| Prompt／response cap | 1024／7168，总训练窗口 8192 | P/C |
| Student／teacher temperature | 1.0／1.0 | P/C |
| Train top-p／top-k | 1.0／-1；由公开 rollout YAML 继承，迁移时显式固定 | C |
| Repetition penalty | 1.0 | P |
| 普通 OPD 基线 | sampled-token log-ratio，单步更新，不加任务 reward | P |
| Reference KL／format reward | 均无 | P |
| Loss aggregation | 全批有效 response token 的 token-mean | P |
| Precision | 论文写 FP32 student、FP32/BF16 teacher；公开脚本控制 model 加载 dtype，需区分混合精度前向 | P/C |
| Hardware | 8 GPUs，FSDP、checkpointing 与 activation offload | P/C |

本仓若使用 17917 条记录、B=64、drop-last，单 epoch 约 279 个 outer／optimizer updates；这是本仓数据计算结果，不是论文保证的精确 step 数。应先核对作者数据行数与过滤规则，再决定结束条件。

### 4.3 验证配置

| 参数 | 设置 |
| --- | --- |
| 九项基准 | AIME24、AIME25、AMC23、MATH-500、Minerva、OlympiadBench、BRUMO、CMIMC、HMMT |
| 每题生成数 | 8 |
| Temperature／top-p | 0.7／0.95 |
| Max response length | 31744 |
| Prompt | `{problem} Please reason step by step, and put your final answer within \boxed{}.` |
| Thinking mode | 评测明确 disabled |
| Grading | 规则式 boxed 提取与数学等价；无法解析 boxed 答案判错，不用 LLM verifier |
| 报告指标 | 每基准 Avg@8、Pass@8；九项等权宏平均；长度、答案缺失与重复 |
| Checkpoint | one-epoch 的最终 checkpoint |

公开 `gen_vllm.py` 当前默认只列 AIME24/AIME25/AMC23；复现九项主表必须补齐其余基准并核版本，不能把三个基准均分标成论文的 6.9%。本仓已有 MATH-500/AMC23/AIME25，还缺 AIME24 和其他五项。[^tide-eval]

本仓数学判分器在没有 boxed 格式时，会对最后 2000 字符进行 `math_verify` 回退；TIDE 论文把没有可解析 boxed 的回答判错。因此严格主指标应另用匹配的评分口径，并同时保留宽松数学正确率辅助诊断，避免把格式恢复误报为推理恢复。实现见[math.py](../../../../../verl/utils/reward_score/one_shot_opd/math.py)。

### 4.4 Full-vocab 对照

建议首先保持该 pair 的全部训练与验证设置，比较：sampled-token → student-top-16 reverse KL → teacher-top-16 reverse KL → full-vocab reverse KL。TIDE 本身作为已有修复参考，不是本轮必须实现的训练方法。

Full-vocab 若改善 Avg@8 且同时降低重复、长度上限停止和答案缺失，才支持“单词元／有限支持造成的监督问题能被完整词表部分修复”。仅 full KL 下降、不改善任务指标，仍可能是局部目标与正确性不一致。

## 5. Revisiting OPD：Teacher-top-32 的直接复现

论文：arXiv:2603.25562，§4.2 与附录表 A1；代码核对到 `f32f284f25bae5b16d2d44ee336b52851dccc736`。[^revisiting][^revisiting-code]

| 参数 | 论文／代码设置 | 说明 |
| --- | --- | --- |
| Student | `Qwen/Qwen2.5-7B-Instruct` | P/C |
| Teacher | `open-thoughts/OpenThinker3-7B` | P/C |
| Train dataset | DAPO-Math-17K 的英文部分 | P；公开入口需要自行准备 train parquet |
| Prompt batch | 16 | C 脚本 `train_data_size=16` |
| Group size | 8 | P/C，使用 `env.rollout.n=8` |
| 序列 batch | 128 | P 表 A1；与 16×8 一致 |
| 序列 mini-batch | 64 | P/C |
| 本仓翻译 | B=16、rollout.n=8、ppo_mini_batch_size=8 | 本仓会再乘 n，得 64 条回答 |
| LR | `2e-6` | P/C |
| Optimizer／warmup | AdamW／0 | P |
| 每批优化轮数 | 1 | C 默认 |
| 总预算 | 表 A1 写 400 training updates | 按 2 updates/outer翻译为200outer；公开脚本只写1 epoch，未显式锁 400updates，必须记清实际更新数 |
| Prompt／response cap | 2048／16384 | P/C |
| Train temperature | 1.0 | P/C |
| Train top-p | 原 sampled 基线 1.0；LSM 修复 0.9 | P/C，直接比较时存在第二个变化因素 |
| 蒸馏支持集 | teacher top-32，双方在支持内重归一化 | P/C |
| Special-token mask | unmasked 原基线 False；mask 为独立消融 | C |
| Valid/test | MATH500、AIME24、AIME25、Minerva、OlympiadBench | P |
| Eval temperature／top-p | 1.0／0.9 | P/C |
| Eval 采样数 | 主表 pass@1；AIME24 另报 avg@32 消融 | 主表不能替换为当前项目 avg@16 后直接比数值 |
| Hardware | 8×H100 | P |

这组的失败证据需要准确表述：AIME24 从原 student 的 13.3 降到 sampled OPD 的 10.0，但五项均分从 28.2 升到 36.4；不是整体完全没有学习。Mask 后均分 40.7，teacher-top-K 约 41.5–41.7；比 full-vocab 时需要保持 mask 和采样条件相同。[^revisiting]

建议用两个采样条件块分别比较：在 top-p=1.0 下比较 sampled／teacher-top-32／full；再在 top-p=0.9 下重复。不能把“sampled+p=1”与“full+p=0.9”之间的全部差异归给 full-vocab。

公开原始与 LSM 脚本还分别设置了环境 seed21／42、不同 test_freq。受控比较应统一种子与验证频率，并把这些与作者现成脚本的差异写入结果。README 对 `k1` 的一个文字说明写成了其他估计器的表达式；实际 `core_algos.py` 中 `kl/k1` 是 `logprob-ref_logprob`，应以执行代码为准。[^revisiting-original][^revisiting-core]

这对模型的 head 行数均为 152064，架构为 untied Qwen2，符合现有 full-KL 的结构限制，但 token-ID 语义、特殊标记和 chat 格式仍需在下载元数据后逐项对齐。仅 head 形状相同不是充分条件。

## 6. Semi-OPD：低重叠在线失败的补充测试

论文：arXiv:2610.11291，附录 A、表 2–4。[^semi]

| 参数 | 设置 |
| --- | --- |
| 较小的模型对 | student=`Qwen/Qwen3-1.7B`，Non-thinking；teacher=`Qwen/Qwen3-4B-Instruct-2507` |
| 更强退化但更贵的 pair | Qwen3-1.7B Thinking student → Qwen3-30B-A3B-Thinking teacher |
| Train dataset | DAPO-Math-17K，经去重后 14116 条英文 prompt |
| Prompt batch／n | 128／1 |
| Epoch／steps | 1 epoch，约 110 步 |
| LR／optimizer | `1e-6` constant；AdamW β=(0.9,0.999)、wd=0.01 |
| Warmup／grad clip | 无／1.0 |
| Precision | BF16 |
| Prompt cap | 2048 |
| Response cap | 图 1／图 2 及 17 对模型的非 thinking 对照为 4096；thinking 为 16384 |
| 重要报告差异 | **表 1 的标准 OPD 对所有 pair 均使用 16K，以区别 FastOPD；不能将该表与匹配 4K 的主曲线混为一个设置** |
| Sampling | temperature=1、top-p=1 |
| Eval | AIME24、AIME25、AIME26、HMMT Feb26；avg@32；temperature=1、top-p=1；response cap32768，Nemotron 例外 40960 |
| 基础实现 | NeMo-RL DTensor＋vLLM；按模型规模使用 1–8 节点、每节点 8×H100 |

若要检验 full-vocab 是否救回在线采样，应在同一 4K 或 16K cap 内比较 sampled online、full online、固定初始学生缓存；不要同时改 cap。本仓当前 DAPO 有 17917 行，不能把它直接当作论文的 14116 英文去重版本。

Qwen3-30B-A3B 以及更大的 MoE 教师暂不适合本仓首批实验：当前共享 teacher 显式拒绝 MoE，full-head 路径也不支持 Qwen3。4B dense teacher 的模型对更适合作为后续适配后的起点。

## 7. Demystifying 与其他案例的取舍

Demystifying OPD（arXiv:2607.13399）有明确的强教师失配与长度异常现象，但复现信息不如前面几组完整。正文与附录可确认：[^demystifying]

| 参数 | 已确认内容 |
| --- | --- |
| Student | Qwen3-1.7B-Base；另有 Qwen3-1.7B post-trained 学生 |
| Teacher | Qwen3-1.7B／4B，以及作者的 1.7B-GRPO／4B-GRPO 版本 |
| Train dataset | Nemotron-Cascade Math，**不是 DAPO** |
| Batch／n | 128／1 |
| 每批优化轮数 | 1 epoch per rollout batch |
| Train length | 文中写 maximum sequence length16384，未清楚拆 prompt/response 预算 |
| Sampling | temperature=1、top-p=1 |
| Eval | MATH500、Minerva 用 pass@1；AMC23、AIME24/25/26、HMMT25 用 avg@32 |
| avg@32 估计 | 生成 40 个候选，10 次抽取 32 个子集后平均 |
| 仍缺 | LR、精确 mini-batch 与总更新数、GRPO 教师 artifact、完整数据处理和 clip 阈值等 |

不能把 THUNLP 的 `lllyx/Qwen3-4B-Base-GRPO` 直接当作这篇的 4B-GRPO。当前不建议把它作为第一个“精确复现”目标，也不应拿邻近论文常用的 `1e-6` 填成其已知 LR。

ESR 等工作的基线粒度还需核实现；Systematic Study 的主实验本来就在算 full-vocabulary KL。它们适合讨论更广义的 OPD 失败，但不应未经区分就拿来宣称“sampled/top-k 失败后换 full 能救”。

## 8. 本仓当前能做什么，哪些需要先适配

### 8.1 Full-vocab 的实际实现范围

| 项目 | 当前代码结论 | 对实验的影响 |
| --- | --- | --- |
| Full loss | 完整条件 reverse KL，token-mean；teacher 固定 | 与 sampled 当前信号的条件动作期望比较；不消除 prefix 偏移 |
| 架构 | 只允许 `model_type=qwen2`，`tie_word_embeddings=False` | Qwen3／Qwen3-Base 直接被拒绝；不是改 MODEL_PATH 就可用 |
| 执行 | FSDP2、SP=1、packed input、shared_sync | 现有常用数学配方满足 |
| LoRA／静态 padding／fused PPO | full 路径不支持 | ESR 的 LoRA 设置不能直接搬到当前 full 路径 |
| Teacher head | 仅加载 untied Qwen2 `lm_head.weight` | 后续支持 tied embedding 时，需要同时处理加载与 backward，而不只是放宽断言 |
| Vocab 形状 | student 与 teacher head 行数必须相同 | R1-1.5B→R1-7B 当前不兼容 |
| Teacher 架构 | shared_sync 拒绝 MoE 及多模态 | 30B-A3B 教师暂不可直接用 |
| Train temperature | full 路径要求单位温度 | 上述首批设置 T=1 符合 |
| 现有 top-k | `forward_kl_topk`为 teacher 支持上的截断 FKL，还会 clamp 负值 | 不能替代 S-top16／T-top32 normalized RKL |
| Shared teacher top-k | 当前 shared_sync 仅支持 k1+PG 或 full-rKL；评分阶段关掉 top-k | 原版 top-k 复现需要新 payload/loss 接线 |

代码依据：[full-vocab adapter](../../../../../verl/models/transformers/full_vocab.py)、[FSDP engine](../../../../../verl/workers/engine/fsdp/transformer_impl.py)、[冻结 teacher head](../../../../../verl/trainer/distillation/hidden.py)、[full-KL 算子](../../../../../verl/trainer/distillation/full_vocab.py)、[蒸馏配置验证](../../../../../verl/workers/config/distillation.py)、[loss 注册与聚合](../../../../../verl/trainer/distillation/losses.py)。数学与计算图复用[既有理论文档](../../theory/2026-10-22-Full-vocab计算图与四种更新.md)。

### 8.2 实际核对过的模型元数据

| 模型 | HF 架构 | Vocab size | Tied embedding | 本仓 full 结构兼容性 |
| --- | --- | ---: | --- | --- |
| R1-Distill-Qwen-1.5B | Qwen2 | 151936 | False | 支持 |
| JustRL-DeepSeek-1.5B | Qwen2 | 151936 | False | 支持；本机与 student 的基础 vocab 及 added-token ID 映射一致 |
| R1-Distill-Qwen-7B | Qwen2 | 152064 | False | 单模型支持，但与 1.5B 配对 head 行数不等 |
| Qwen3-1.7B-Base | Qwen3 | 151936 | True | 不支持 |
| Qwen3-8B | Qwen3 | 151936 | False | 不支持 |
| Qwen3-1.7B | Qwen3 | 151936 | True | 不支持 |
| Qwen3-4B-Instruct-2507 | Qwen3 | 151936 | True | 不支持 |
| Qwen2.5-7B-Instruct | Qwen2 | 152064 | False | 结构支持，尚未在该 pair 运行 |
| OpenThinker3-7B | Qwen2 | 152064 | False | 同上 |

元数据来自本机 config 与官方 HF config；没有下载新模型权重。[^model-configs]

R1 的 151936／152064 差异不能直接用矩阵裁切“解决”后继续称原始 full-vocab KL。应先确定实际 token 语义、padding 行与生成时的有效词表，再统一支持及归一化；这会成为额外方法条件。

Qwen3-1.7B-Base 的 config 最大位置为 32768。本仓会把 `max_prompt + max(train_response,val_response) + 1` 作为 vLLM 窗口；即使 1024+31744 也会派生 32769。适配 Qwen3 时要明确总窗口与 EOS 预算，不能继续沿用当前 4096 prompt 默认值。

### 8.3 本机资产与默认配方差异

- 本机现有权重在工作区 `models/`，含 R1-Distill-1.5B 与 JustRL-1.5B；脚本默认的同级 `model/` 在本机不存在。开发机的单数目录是另一个部署状态。
- 本机数据在 `MD-OPD/data/math/`，有 DAPO 17917 条、MATH-500 500 条、AMC23 40 条、AIME25 30 条；没有 AIME24 目录。
- 默认是 B=64、n=1、prompt cap4096、验证 top-p=0.9、打乱训练数据。Rethinking/TIDE 分别需要 n=4、prompt cap1024、验证 top-p=0.95、固定数据顺序。
- 原生 Python 默认指 `.venv-native/bin/python`，该路径在本机不存在；本机已有可解析配置的 `../../../../../../../../../../../../root/.local/share/mirroropd/venvs/profile-cu128/bin/python`。不据此推断开发机环境或资产已经同步。
- DeepSeek 模型应保持 `CONTINUOUS_TOKEN_MODEL_FAMILY=deepseek`；Qwen2.5 模型要改为 `qwen25`／经核对的 Qwen 协议，Qwen3 要用其协议与明确的 thinking 设置。不要让默认 DeepSeek builder 处理 Qwen3。

## 9. 原候选 Full-vocab 对照设计（非当前执行计划）

### 9.1 每个 failure pair 的最小矩阵

| Arm | 目标 | 用途 |
| --- | --- | --- |
| 原论文基线 | 保持作者的 sampled 或 top-k PG／直接 KL 实现 | 先确认 failure 确实出现 |
| Sampled-current | 当前 log-ratio、停止梯度、无额外任务 reward；fresh batch 一次更新 | 轻量条件 reverse-KL 基线 |
| Student-top-k RKL | K=16，明确是否支持内重归一化、是否 score surrogate | 检验学生支持截断 |
| Teacher-top-k RKL | K=16 或原论文 32，同样明确归一化与梯度 | 检验教师支持选择 |
| Full-vocab RKL | 当前学生概率、完整词表求和，直接精确条件梯度 | 检验取消动作抽样及支持截断是否改善 |

第一轮保持 `ppo_epochs=1`，不要同时引入 E16/E17 的多次复用。最初可以只做原基线与 Full 两条，确认现象后再补机制臂；若原基线是 PPO／冻结 advantage 且多次更新，则应补同样梯度语义的 sampled-current 对照，避免把取消 clipping 或改变信号刷新归因于完整词表。

Normalized top-k 与 full-vocab 在支持内外的概率质量处理不同，full 并非单纯把同一个随机估计器的样本数增大。**sampled→full**主要检验条件动作积分；**normalized top-k→full**还检验截断与归一化目标变化。

### 9.2 固定的实验条件

- 相同 student 初始化、teacher checkpoint、prompt 顺序、数据范围、训练长度、n、有效序列 batch、LR、优化器和 loss 的全局归约。
- 相同验证题、prompt/chat 模板、thinking 开关、采样温度、top-p、回答上限和 grader；先独立重测未训练 student 与 teacher。
- 公平比较以相同 optimizer 更新数和训练 prompt 预算为主，再单列 wall-clock／GPU-hours。不要把 full 做更多复用后的收益称为“同预算 full 救回”。
- 第一轮可用 seed42 验证现象，正式报告至少 3 个种子（R）。原论文未给 seed 处要标作我们的选择。
- 一个 run 训练后轨迹自然会变化，相同随机种子不意味着两种 loss 始终看到相同 prefix。若要分离局部梯度差异，另用固定 prefix 做诊断，不能把它与在线效果混为一个实验。

主比较先固定论文 LR。若 full 未救回，可以给所有方法相同的 0.5×／1×／2×LR 预算做补充，区分优化尺度与目标问题；不能只替 full 挑最优 LR 再与未调参的失败基线比较。

### 9.3 判断“救回”的指标

| 失败类型 | 首要判据 | 辅助诊断 |
| --- | --- | --- |
| 不提升 | 最终／固定 checkpoint 窗口的 avg@k 显著优于原始 student 与失败 baseline | 师生差距恢复比例、梯度范数 |
| 后期崩溃 | 相同训练长度下避免从峰值持续退化 | 长度分位数、停止原因、重复、缺失答案 |
| 负迁移 | 保住原 student 能力并改善目标任务 | 不要只报告 teacher matching loss |
| 局部负迁移 | 对原退化基准改善，且其余基准不过度牺牲 | 每个基准单列，不只看 macro 均值 |

Full-vocab 可能消除动作抽样噪声、恢复遗漏的监督，但不会让错误前缀上的教师分布自动变成正确监督，也不会保证教师具有可迁移的新能力。若 failure 在 full 下仍存在，这本身就是支持“状态／目标问题超出支持集估计”的有价值结果。

## 10. 原 n=4 长窗配置解析记录（非当前执行计划）

下面是已有资产上的配置预览，采用 Rethinking 的 B=64、n=4、LR 与长度设置，但暂沿用本机已有数据和三项验证。因此它是**可先进行的长窗工程预实验**，不是已经准备完毕的论文精确复现。严格版本需要补 AIME24、论文 prompt 和 top-k 基线。

从仓库根执行。代码块保留 `DRY_RUN=1`，只展示命令；真正训练仍需按现有作业方式显式分配 GPU。

```bash
export MODEL_ROOT=/mnt/tidalfs-bdsz01/dataset/llm_dataset/zkn_data/fht_data/models
export DATA_ROOT=/mnt/tidalfs-bdsz01/dataset/llm_dataset/zkn_data/fht_data/MD-OPD/data
export PYTHON_BIN=/root/.local/share/mirroropd/venvs/profile-cu128/bin/python
export VERL_USE_UV=0
export TRAIN_BATCH_SIZE=64 PPO_EPOCHS=1 LEARNING_RATE=1e-6
export MAX_PROMPT_LENGTH=1024 MAX_RESPONSE_LENGTH=15360
export MAX_VAL_RESPONSE_LENGTH=31744 VAL_N=16
export TEST_FREQ=20 SAVE_FREQ=20 SEED=42

DRY_RUN=1 EXPERIMENT_NAME="failure-sampled-L${MAX_RESPONSE_LENGTH}-s${SEED}" \
OPD_ADVANTAGE_SOURCE=current OPD_POLICY_LOSS_MODE=reinforce \
bash examples/mirroropd/run_math_opd_ablation_fsdp2.sh custom --method sampled \
  actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.actor.ppo_mini_batch_size=64 \
  actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
  actor_rollout_ref.actor.fsdp_config.seed=42 \
  actor_rollout_ref.actor.data_loader_seed=42 \
  actor_rollout_ref.rollout.seed=42 \
  actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
  data.shuffle=False trainer.total_training_steps=200

DRY_RUN=1 EXPERIMENT_NAME="failure-full-L${MAX_RESPONSE_LENGTH}-s${SEED}" \
bash examples/mirroropd/run_math_opd_ablation_fsdp2.sh custom --method full_kl \
  actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.actor.ppo_mini_batch_size=64 \
  actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
  actor_rollout_ref.actor.fsdp_config.seed=42 \
  actor_rollout_ref.actor.data_loader_seed=42 \
  actor_rollout_ref.rollout.seed=42 \
  actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
  data.shuffle=False trainer.total_training_steps=200
```

把 `MAX_RESPONSE_LENGTH` 改为 7168，得到对应短窗对照；其余参数不改。完整 4 臂是 sampled-7K、sampled-15K、full-7K、full-15K。论文后期图包含到 250 步，若 200 步未出现失败，可预先约定延长至 250／一个完整 epoch，标为延长复现，不能事后只选最差节点。

本轮已对 15K sampled-current／reinforce 与 full-rKL 配置运行 `CONFIG_ONLY=1`：解析结果为 B=64、n=4、M=64、E=1、LR=1e-6、response15360、val n16/top-p0.95、200outer。代码块额外显式固定相同seed与零warmup。**未启动 Ray 训练、未加载 GPU 权重、未声称 failure 已复刻。**

## 参考文献与源码

[^rethinking]: Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe. arXiv:2604.13016，§3–6、附录 A.2。[论文](https://arxiv.org/html/2604.13016)
[^rethinking-code]: Thinking-Space/Rethinking-OPD，commit `ac26e38d6f1572eb027597b48a9f4e01f6915ef8`。[训练脚本](https://github.com/Thinking-Space/Rethinking-OPD/blob/ac26e38d6f1572eb027597b48a9f4e01f6915ef8/on_policy_distillation.sh)；[评测脚本](https://github.com/Thinking-Space/Rethinking-OPD/blob/ac26e38d6f1572eb027597b48a9f4e01f6915ef8/scripts/val/eval/gen_vllm.py)；[README 与验证说明](https://github.com/Thinking-Space/Rethinking-OPD/blob/ac26e38d6f1572eb027597b48a9f4e01f6915ef8/README.md)。
[^rethinking-actor]: 同仓 `dp_actor.py` 的 `compute_reward_weights`、`only_stu`及更新分支。[源码](https://github.com/Thinking-Space/Rethinking-OPD/blob/ac26e38d6f1572eb027597b48a9f4e01f6915ef8/verl/verl/workers/actor/dp_actor.py)
[^rethinking-core]: 同仓 `core_algos.py` 的 3 维候选 PG 及候选求和。[源码](https://github.com/Thinking-Space/Rethinking-OPD/blob/ac26e38d6f1572eb027597b48a9f4e01f6915ef8/verl/verl/trainer/ppo/core_algos.py)
[^tide]: Mismatch Matters: On-Policy Distillation Beyond Token Agreement. arXiv:2608.09836，表 1、表 3、表 9 与附录 D。[论文](https://arxiv.org/html/2608.09836)
[^tide-code]: yzc-666/TIDE，commit `b54a9e15b2064a8d18c36f3cd3f73ec8906cefc5`。[训练脚本](https://github.com/yzc-666/TIDE/blob/b54a9e15b2064a8d18c36f3cd3f73ec8906cefc5/on_policy_distillation.sh)；[rollout 默认配置](https://github.com/yzc-666/TIDE/blob/b54a9e15b2064a8d18c36f3cd3f73ec8906cefc5/verl/verl/trainer/config/rollout/rollout.yaml)。
[^tide-eval]: 同仓独立评测脚本。[源码](https://github.com/yzc-666/TIDE/blob/b54a9e15b2064a8d18c36f3cd3f73ec8906cefc5/scripts/val/eval/gen_vllm.py)
[^revisiting]: Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes. arXiv:2603.25562，§4.2、表 1、附录表 A1。[论文](https://arxiv.org/html/2603.25562)
[^revisiting-code]: hhh675597/revisiting_opd，commit `f32f284f25bae5b16d2d44ee336b52851dccc736`。[Teacher-top-K 数学脚本](https://github.com/hhh675597/revisiting_opd/blob/f32f284f25bae5b16d2d44ee336b52851dccc736/examples/opd/opd_math_qwen2.5-7b_it.sh)
[^revisiting-original]: 同仓 sampled-token 数学脚本。[源码](https://github.com/hhh675597/revisiting_opd/blob/f32f284f25bae5b16d2d44ee336b52851dccc736/examples/opd/opd_original_math_qwen2.5_7b_it.sh)
[^revisiting-core]: 同仓 `kl_penalty`、`compute_opd_advantage` 与数据批处理。[core_algos.py](https://github.com/hhh675597/revisiting_opd/blob/f32f284f25bae5b16d2d44ee336b52851dccc736/verl/trainer/ppo/core_algos.py)；[trainer](https://github.com/hhh675597/revisiting_opd/blob/f32f284f25bae5b16d2d44ee336b52851dccc736/verl/trainer/ppo/ray_trainer_multitask.py)。
[^semi]: When Do We Need On-Policy Distillation? Distilling on Offline Student Rollouts Is Often Better. arXiv:2610.11291，附录 A、表 2–4，特别注意表 1 与图 2 的 rollout cap 差异。[论文](https://arxiv.org/html/2610.11291)
[^demystifying]: Demystifying On-Policy Distillation: Roles, Pathologies, and Regulations. arXiv:2607.13399，§3–5 与附录 A。[论文](https://arxiv.org/html/2607.13399)
[^model-configs]: 2026-10-24 核对官方 HF config：[R1-7B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B/blob/main/config.json)、[Qwen3-1.7B-Base](https://huggingface.co/Qwen/Qwen3-1.7B-Base/blob/main/config.json)、[Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B/blob/main/config.json)、[Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B/blob/main/config.json)、[Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/blob/main/config.json)、[Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/config.json)、[OpenThinker3-7B](https://huggingface.co/open-thoughts/OpenThinker3-7B/blob/main/config.json)。本机两份 1.5B 模型另直接读取 config 与 tokenizer 做语义映射比较，没有新增权重下载或文件摘要清单。
