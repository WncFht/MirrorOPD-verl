# OPD 原论文设置与 Full-vocab 核对档案

**当前执行方案已独立整理到[失败复现目录](../failure-reproduction/README.md)：使用本项目DAPO与三个valid，先sampled-token，再full-vocab，首轮不做top-k。** 本文保留此前的作者参数核对、候选比较和配置解析记录；以下n=4、top-k及200步示例不代表当前执行选择。

日期：2026-10-24。目标是先复现 sampled-token 或 top-k OPD 的停滞、负迁移或长程失稳，再检验完整词表 reverse KL 是否改善。本文核对论文正文、附录、公开脚本和本仓实现；不把未公开的参数补成“论文设置”。本仓源码核对基于 `376e8185e` 及当前工作区，尚未启动训练。

证据标记：**P** 为论文明确写出；**C** 为公开代码；**L** 为本仓源码或本机资产；**R** 为建议的复现选择。P 与 C 不一致时分别保留。现有 E13–E20 的结果与预算仍由[训练配方](../训练配方.md)和[五组实验分析](2026-10-24-五组Math实验对比分析.md)维护，本文不改现有队列。

## 1. 优先复现什么

| 设置 | 要复刻的失败 | 对 Full-vocab 的研究价值 | 当前工程状态 |
| --- | --- | --- | --- |
| R1-Distill-1.5B student，JustRL-1.5B teacher，训练窗口从7K扩到15K | Rethinking OPD 的长窗口后期失稳，默认为 student-top-16 | 区分单词元／支持截断问题与坏前缀问题 | 两个模型已在本机，架构与词表满足现有 full-KL；严格 top-k 基线尚未实现，可先做 sampled/full 长窗预实验 |
| R1-Distill-1.5B student，R1-Distill-7B teacher | 强教师几乎不能带来提升；换 JustRL-1.5B teacher 则成功 | 检查完整监督是否能挖出 top-k 没利用的信号 | 缺7B权重；输出 head 分别151936／152064行，现有 full-KL 拒绝，需要词表对齐 |
| JustRL-1.5B student，R1-Distill-7B teacher | 较强7B教师使学生回退，损失其RL收益 | 检验 full-vocab 能否避免错误能力覆盖 | 同样有词表行数差异；比前项更强的负迁移测试 |
| Qwen3-1.7B-Base student，Qwen3-8B teacher | TIDE 中 sampled OPD 九项平均 Avg@8 仍为6.9%，长度膨胀、答案缺失 | 失败明确、checkpoint公开，最值得作为主失败 benchmark | 当前 full-KL 仅支持 untied Qwen2；Qwen3 与 tied embedding 尚需适配 |
| Qwen2.5-7B-Instruct student，OpenThinker3-7B teacher | Revisiting 的局部负迁移与不稳定；teacher-top-32 有改善 | 可以直接比较 sampled、teacher-top-32、full-rKL | 两者是 untied Qwen2、输出维数相同；缺权重与完整验证数据，需核 token-ID 和特殊标记 |
| Qwen3-1.7B Non-thinking student，Qwen3-4B-Instruct-2507 teacher | Semi-OPD 中低重叠配置：在线采样不如初始学生缓存 | 检验 full-vocab 能否缓解在线前缀退化 | 需 Qwen3/tied-head 适配，且 DAPO 过滤口径不同 |

表内保留原始候选范围。当前A–F选择、sampled→full顺序和后置设置由[新方案](../failure-reproduction/2026-10-24-Sampled-token与Full-vocab失败复现方案.md)统一维护。R1的跨规模模型仍需要词表对齐，不能只把 `TEACHER_MODEL` 换成7B就直接跑现有full-KL。

这些是不同的失败：不提升、覆盖学生已有能力、长序列崩溃、某个基准退化，应该分别报告。只要 full-KL 有收益，不能就声称它解释了所有 OPD failure。

## 2. Batch、rollout 与 optimizer update 的口径

设 $B$ 为一次外层步骤的 prompt 数，$n$ 为每题生成数，$M$ 为 `ppo_mini_batch_size` 配置值，$E$ 为 `ppo_epochs`。在本仓 V1 路径中，实际序列 mini-batch 会乘上 $n$：

$$
B_{\rm seq}=Bn,\qquad M_{\rm seq}=Mn,\qquad
U_{\rm outer}=E\,\frac{B}{M}.
$$

以上假设整除、无补齐额外步骤且没有跳过更新。[V1 `_update_actor`](../../../../verl/trainer/ppo/v1/trainer_base.py)明确把 mini-batch 乘以 rollout.n。

| 配置 | 每次生成回答数 | 每次 optimizer update 的回答数 | 每个 outer 的更新数 |
| --- | ---: | ---: | ---: |
| 当前默认：B=64，n=1，M=64，E=1 | 64 | 64 | 1 |
| Rethinking／TIDE：B=64，n=4，M=64，E=1 | 256 | 256 | 1 |
| 将 Revisiting 的128条回答、64条回答 mini-batch 映射到本仓：B=16，n=8，M=8，E=1 | 128 | 64 | 2 |

因此，**64 prompts × 4 rollouts 不是总共64条回答**；也不要因为有256条回答，就误以为 `mini_batch_size=64` 一定产生4次更新。本仓与 Rethinking 的 verl worker 都有乘 $n$ 的逻辑。Revisiting 使用 `env.rollout.n=8`，其 actor 的 `rollout.n` 默认1，参数语义需要单独翻译。

## 3. Rethinking OPD：最贴近当前模型的系列设置

论文：arXiv:2604.13016，重点 §3、§4.1、§5.1、§6.1、附录A.2。公开代码核对到 `ac26e38d6f1572eb027597b48a9f4e01f6915ef8`。[^rethinking][^rethinking-code]

### 3.1 模型对与失败定义

| 子设置 | Student | Teacher | 论文现象 |
| --- | --- | --- | --- |
| 无收益 | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | §4.1中验证曲线基本不改善，overlap及梯度动态停滞 |
| 成功对照 | 同上 | `hbx/JustRL-DeepSeek-1.5B` | 明显提升，恢复较大比例的师生性能差距 |
| 负迁移 | `hbx/JustRL-DeepSeek-1.5B` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | §3.3中学生回退，和向自身pre-RL检查点蒸馏的方向相似 |
| 思考模式不兼容 | `Qwen/Qwen3-1.7B-Base` | `Qwen/Qwen3-4B`，Non-thinking | §3.1／§5.1中直接OPD较差，教师回答SFT冷启动后改善 |
| 模式兼容教师对照 | `Qwen/Qwen3-1.7B-Base` | `lllyx/Qwen3-4B-Base-GRPO` | 与Qwen3-4B Non-thinking比较；不能把它与另一篇论文的“4B-GRPO”混为同一个checkpoint |
| 长程失稳 | R1-Distill-1.5B | JustRL-1.5B | §6.1中3K／7K较稳定，10K／15K出现后期失稳，异常从后缀向前传播 |

### 3.2 公共训练配置

| 参数 | 数值／设置 | 证据 |
| --- | --- | --- |
| Train dataset | DAPO-Math-17K | P/C |
| Prompt batch | 64 | P/C |
| Rollouts per prompt | 4 | P/C |
| 实际序列 batch | 256 | 由代码语义换算 |
| PPO mini-batch配置 | 64；worker乘4后为256条回答 | P/C |
| 每批优化轮数 | 1；与上述batch配置合起来每outer一次更新 | C |
| LR | `1e-6` | P/C |
| Scheduler | 公开代码默认constant；未打开cosine分支 | C |
| Train epochs | 1 | P/C |
| 长窗口消融步数 | §6.1正文写200 steps；图13讨论step180–250，需保留这一报告差异 | P |
| Max prompt length | 1024 | P/C |
| 默认训练响应上限 | 7168 | P/C |
| 长窗口变体 | 3072、7168、10240、15360等；原脚本注释明确列出15360 | P/C |
| Sampling temperature | 1.0 | P/C |
| Sampling top-p | 1.0 | P；配置继承需保留 |
| Sampling top-k | 不截断；不同于蒸馏支持集K | C/迁移时显式固定 |
| 蒸馏支持集 | student top-16；`LOG_PROB_TOP_K=16, TOP_K_STRATEGY=only_stu` | P/C |
| 采样词元版本 | `LOG_PROB_TOP_K=0` | C |
| Teacher temperature | 1.0 | C |
| Loss aggregation | token-mean | C |
| Reference KL／format reward | 关闭 | P/C |
| 数据打乱 | `data.shuffle=False` | C |
| Actor权重初始化dtype | `model_dtype=fp32`；不能据此声称所有前向乘法均FP32 | C |
| Hardware | 8×A800 80GB | C README |

Adam系数、weight decay、混合精度、gradient clipping等未在论文表2逐项报告，严格复现应解析该commit的完整Hydra配置。下面的本仓提案采用本仓已明确的 AdamW `(0.9,0.999)`、wd=0.01、grad clip=1、无warmup，标为R/L，不冒充论文表2给定值。

### 3.3 验证配置

| 参数 | 设置 |
| --- | --- |
| Valid/test benchmarks | AIME2024、AIME2025、AMC2023 |
| 每题生成数 | 16 |
| Temperature／top-p | 0.7／0.95 |
| Max response length | 31744 |
| 主指标 | 每个基准avg@16；不要替换为pass@16 |
| 问题模板 | `{problem} Please reason step by step, and put your final answer within \boxed{}.` |
| Chat template | 模型自身模板；公开评测脚本默认 `enable_thinking=False` |
| 验证流程 | 公开仓库建议单独跑 `scripts/val/eval/`，而非直接依赖其旧verl内置验证 |

作者README指出，verl v0.7.0内置验证曾在其运行中低估5–7个百分点，建议训练时关闭该路径、离线评测checkpoint。**这不是本仓也必然存在同一bug的结论**，但意味着需要先用共同评测流程重测原始student与teacher，不能把不同验证脚本的分数直接拼接。[^rethinking-code]

训练模板也重要。论文的标准DAPO设置包含原始“最后一行 Answer:”指令，§5.2专门比较了它与teacher-aligned的boxed模板。本仓导入脚本去掉了原DAPO前后缀，现有parquet保存的是裸题目；严格复现需生成单独的论文格式数据，不能只对齐模型和LR。

### 3.4 论文 top-k 等式与公开实现需要分别标注

论文将 top-k 写成支持集内部重新归一化的 reverse KL。公开代码 `dp_actor.py::compute_reward_weights` 的默认 `only_stu` 分支则：

1. 用选出的学生log-prob在K维做softmax，得到归一化学生权重；
2. 用**原始全词表归一化的**师生log-prob之差构造优势；
3. 将停止梯度的候选优势交给PG更新，候选维度求和。

因此迁移时应区分 **作者代码的student-top-k PG** 与 **对支持内归一化KL直接backward**，不能未经梯度核对就当成同一个实现。若目的是复刻作者failure，优先保持作者代码路径；若目的是研究支持截断偏差，可另加直接归一化KL作为机制对照。[^rethinking-actor][^rethinking-core]

## 4. TIDE：强失配、停滞和重复最明确的主测试

论文：arXiv:2608.09836。主文表1、表3和附录D给出了较完整的设置。代码核对到 `b54a9e15b2064a8d18c36f3cd3f73ec8906cefc5`。[^tide][^tide-code]

### 4.1 模型与目标现象

- **Failure pair**：student=`Qwen/Qwen3-1.7B-Base`，teacher=`Qwen/Qwen3-8B`。注意student必须是Base，不能换成`Qwen3-1.7B`的post-trained版本。
- **成功对照**：student=`deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`，teacher=`hbx/JustRL-DeepSeek-1.5B`。
- **Failure定义**：强失配配置中，未训练student与普通sampled OPD九项宏平均Avg@8均为6.9%；OPD平均响应长度22395、无boxed答案比例65.5%。TIDE相应为20.3%、7294词元、5.4%。复现目标是停滞伴随生成退化，不只是要求某个小基准分数低。

### 4.2 训练配置

| 参数 | 设置 | 证据 |
| --- | --- | --- |
| Train dataset | DAPO-Math-17K | P/C |
| Data shuffle | False | P/C |
| Prompt batch／n | 64／4 | P/C |
| 实际序列 batch | 256 | P/C换算 |
| Mini-batch配置 | 64；实际256条回答 | P/C换算 |
| Micro-batch per GPU | 1，允许动态token batching／梯度累积 | P/C |
| Update per rollout batch | 1 | P，附录D.1 |
| Train epochs | 1，最终checkpoint报告，不按test挑最好checkpoint | P |
| Optimizer | AdamW | P |
| LR／schedule／warmup | `1e-6`／constant／none | P/C |
| Adam β／weight decay | `(0.9,0.999)`／0.01 | P |
| Gradient clip | 1.0 | P |
| Prompt／response cap | 1024／7168，总训练窗口8192 | P/C |
| Student／teacher temperature | 1.0／1.0 | P/C |
| Train top-p／top-k | 1.0／-1；由公开rollout YAML继承，迁移时显式固定 | C |
| Repetition penalty | 1.0 | P |
| 普通OPD基线 | sampled-token log-ratio，单步更新，不加任务reward | P |
| Reference KL／format reward | 均无 | P |
| Loss aggregation | 全批有效response token的token-mean | P |
| Precision | 论文写FP32 student、FP32/BF16 teacher；公开脚本控制model加载dtype，需区分混合精度前向 | P/C |
| Hardware | 8 GPUs，FSDP、checkpointing与activation offload | P/C |

本仓若使用17917条记录、B=64、drop-last，单epoch约279个outer／optimizer updates；这是本仓数据计算结果，不是论文保证的精确step数。应先核对作者数据行数与过滤规则，再决定结束条件。

### 4.3 验证配置

| 参数 | 设置 |
| --- | --- |
| 九项基准 | AIME24、AIME25、AMC23、MATH-500、Minerva、OlympiadBench、BRUMO、CMIMC、HMMT |
| 每题生成数 | 8 |
| Temperature／top-p | 0.7／0.95 |
| Max response length | 31744 |
| Prompt | `{problem} Please reason step by step, and put your final answer within \boxed{}.` |
| Thinking mode | 评测明确disabled |
| Grading | 规则式boxed提取与数学等价；无法解析boxed答案判错，不用LLM verifier |
| 报告指标 | 每基准Avg@8、Pass@8；九项等权宏平均；长度、答案缺失与重复 |
| Checkpoint | one-epoch的最终checkpoint |

公开 `gen_vllm.py` 当前默认只列AIME24/AIME25/AMC23；复现九项主表必须补齐其余基准并核版本，不能把三个基准均分标成论文的6.9%。本仓已有MATH-500/AMC23/AIME25，还缺AIME24和其他五项。[^tide-eval]

本仓数学判分器在没有boxed格式时，会对最后2000字符进行 `math_verify` 回退；TIDE论文把没有可解析boxed的回答判错。因此严格主指标应另用匹配的评分口径，并同时保留宽松数学正确率辅助诊断，避免把格式恢复误报为推理恢复。实现见[math.py](../../../../verl/utils/reward_score/one_shot_opd/math.py)。

### 4.4 Full-vocab 对照

建议首先保持该pair的全部训练与验证设置，比较：sampled-token → student-top-16 reverse KL → teacher-top-16 reverse KL → full-vocab reverse KL。TIDE本身作为已有修复参考，不是本轮必须实现的训练方法。

Full-vocab若改善Avg@8且同时降低重复、长度上限停止和答案缺失，才支持“单词元／有限支持造成的监督问题能被完整词表部分修复”。仅full KL下降、不改善任务指标，仍可能是局部目标与正确性不一致。

## 5. Revisiting OPD：Teacher-top-32 的直接复现

论文：arXiv:2603.25562，§4.2与附录表A1；代码核对到 `f32f284f25bae5b16d2d44ee336b52851dccc736`。[^revisiting][^revisiting-code]

| 参数 | 论文／代码设置 | 说明 |
| --- | --- | --- |
| Student | `Qwen/Qwen2.5-7B-Instruct` | P/C |
| Teacher | `open-thoughts/OpenThinker3-7B` | P/C |
| Train dataset | DAPO-Math-17K的英文部分 | P；公开入口需要自行准备train parquet |
| Prompt batch | 16 | C脚本 `train_data_size=16` |
| Group size | 8 | P/C，使用 `env.rollout.n=8` |
| 序列 batch | 128 | P表A1；与16×8一致 |
| 序列 mini-batch | 64 | P/C |
| 本仓翻译 | B=16、rollout.n=8、ppo_mini_batch_size=8 | 本仓会再乘n，得64条回答 |
| LR | `2e-6` | P/C |
| Optimizer／warmup | AdamW／0 | P |
| 每批优化轮数 | 1 | C默认 |
| 总预算 | 表A1写400 training updates | 按2 updates/outer翻译为200outer；公开脚本只写1 epoch，未显式锁400updates，必须记清实际更新数 |
| Prompt／response cap | 2048／16384 | P/C |
| Train temperature | 1.0 | P/C |
| Train top-p | 原sampled基线1.0；LSM修复0.9 | P/C，直接比较时存在第二个变化因素 |
| 蒸馏支持集 | teacher top-32，双方在支持内重归一化 | P/C |
| Special-token mask | unmasked原基线False；mask为独立消融 | C |
| Valid/test | MATH500、AIME24、AIME25、Minerva、OlympiadBench | P |
| Eval temperature／top-p | 1.0／0.9 | P/C |
| Eval采样数 | 主表pass@1；AIME24另报avg@32消融 | 主表不能替换为当前项目avg@16后直接比数值 |
| Hardware | 8×H100 | P |

这组的失败证据需要准确表述：AIME24从原student的13.3降到sampled OPD的10.0，但五项均分从28.2升到36.4；不是整体完全没有学习。Mask后均分40.7，teacher-top-K约41.5–41.7；比full-vocab时需要保持mask和采样条件相同。[^revisiting]

建议用两个采样条件块分别比较：在top-p=1.0下比较sampled／teacher-top-32／full；再在top-p=0.9下重复。不能把“sampled+p=1”与“full+p=0.9”之间的全部差异归给full-vocab。

公开原始与LSM脚本还分别设置了环境seed21／42、不同test_freq。受控比较应统一种子与验证频率，并把这些与作者现成脚本的差异写入结果。README对 `k1` 的一个文字说明写成了其他估计器的表达式；实际 `core_algos.py` 中 `kl/k1` 是 `logprob-ref_logprob`，应以执行代码为准。[^revisiting-original][^revisiting-core]

这对模型的head行数均为152064，架构为untied Qwen2，符合现有full-KL的结构限制，但token-ID语义、特殊标记和chat格式仍需在下载元数据后逐项对齐。仅head形状相同不是充分条件。

## 6. Semi-OPD：低重叠在线失败的补充测试

论文：arXiv:2610.11291，附录A、表2–4。[^semi]

| 参数 | 设置 |
| --- | --- |
| 较小的模型对 | student=`Qwen/Qwen3-1.7B`，Non-thinking；teacher=`Qwen/Qwen3-4B-Instruct-2507` |
| 更强退化但更贵的pair | Qwen3-1.7B Thinking student → Qwen3-30B-A3B-Thinking teacher |
| Train dataset | DAPO-Math-17K，经去重后14116条英文prompt |
| Prompt batch／n | 128／1 |
| Epoch／steps | 1 epoch，约110步 |
| LR／optimizer | `1e-6` constant；AdamW β=(0.9,0.999)、wd=0.01 |
| Warmup／grad clip | 无／1.0 |
| Precision | BF16 |
| Prompt cap | 2048 |
| Response cap | 图1／图2及17对模型的非thinking对照为4096；thinking为16384 |
| 重要报告差异 | **表1的标准OPD对所有pair均使用16K，以区别FastOPD；不能将该表与匹配4K的主曲线混为一个设置** |
| Sampling | temperature=1、top-p=1 |
| Eval | AIME24、AIME25、AIME26、HMMT Feb26；avg@32；temperature=1、top-p=1；response cap32768，Nemotron例外40960 |
| 基础实现 | NeMo-RL DTensor＋vLLM；按模型规模使用1–8节点、每节点8×H100 |

若要检验full-vocab是否救回在线采样，应在同一4K或16K cap内比较sampled online、full online、固定初始学生缓存；不要同时改cap。本仓当前DAPO有17917行，不能把它直接当作论文的14116英文去重版本。

Qwen3-30B-A3B以及更大的MoE教师暂不适合本仓首批实验：当前共享teacher显式拒绝MoE，full-head路径也不支持Qwen3。4B dense teacher的模型对更适合作为后续适配后的起点。

## 7. Demystifying 与其他案例的取舍

Demystifying OPD（arXiv:2607.13399）有明确的强教师失配与长度异常现象，但复现信息不如前面几组完整。正文与附录可确认：[^demystifying]

| 参数 | 已确认内容 |
| --- | --- |
| Student | Qwen3-1.7B-Base；另有Qwen3-1.7B post-trained学生 |
| Teacher | Qwen3-1.7B／4B，以及作者的1.7B-GRPO／4B-GRPO版本 |
| Train dataset | Nemotron-Cascade Math，**不是DAPO** |
| Batch／n | 128／1 |
| 每批优化轮数 | 1 epoch per rollout batch |
| Train length | 文中写maximum sequence length16384，未清楚拆prompt/response预算 |
| Sampling | temperature=1、top-p=1 |
| Eval | MATH500、Minerva用pass@1；AMC23、AIME24/25/26、HMMT25用avg@32 |
| avg@32估计 | 生成40个候选，10次抽取32个子集后平均 |
| 仍缺 | LR、精确mini-batch与总更新数、GRPO教师artifact、完整数据处理和clip阈值等 |

不能把THUNLP的 `lllyx/Qwen3-4B-Base-GRPO` 直接当作这篇的4B-GRPO。当前不建议把它作为第一个“精确复现”目标，也不应拿邻近论文常用的 `1e-6` 填成其已知LR。

ESR等工作的基线粒度还需核实现；Systematic Study 的主实验本来就在算full-vocabulary KL。它们适合讨论更广义的OPD失败，但不应未经区分就拿来宣称“sampled/top-k失败后换full能救”。

## 8. 本仓当前能做什么，哪些需要先适配

### 8.1 Full-vocab 的实际实现范围

| 项目 | 当前代码结论 | 对实验的影响 |
| --- | --- | --- |
| Full loss | 完整条件reverse KL，token-mean；teacher固定 | 与sampled当前信号的条件动作期望比较；不消除prefix偏移 |
| 架构 | 只允许 `model_type=qwen2`，`tie_word_embeddings=False` | Qwen3／Qwen3-Base直接被拒绝；不是改MODEL_PATH就可用 |
| 执行 | FSDP2、SP=1、packed input、shared_sync | 现有常用数学配方满足 |
| LoRA／静态padding／fused PPO | full路径不支持 | ESR的LoRA设置不能直接搬到当前full路径 |
| Teacher head | 仅加载untied Qwen2 `lm_head.weight` | 后续支持tied embedding时，需要同时处理加载与backward，而不只是放宽断言 |
| Vocab形状 | student与teacher head行数必须相同 | R1-1.5B→R1-7B当前不兼容 |
| Teacher架构 | shared_sync拒绝MoE及多模态 | 30B-A3B教师暂不可直接用 |
| Train temperature | full路径要求单位温度 | 上述首批设置T=1符合 |
| 现有top-k | `forward_kl_topk`为teacher支持上的截断FKL，还会clamp负值 | 不能替代S-top16／T-top32 normalized RKL |
| Shared teacher top-k | 当前shared_sync仅支持k1+PG或full-rKL；评分阶段关掉top-k | 原版top-k复现需要新payload/loss接线 |

代码依据：[full-vocab adapter](../../../../verl/models/transformers/full_vocab.py)、[FSDP engine](../../../../verl/workers/engine/fsdp/transformer_impl.py)、[冻结teacher head](../../../../verl/trainer/distillation/hidden.py)、[full-KL算子](../../../../verl/trainer/distillation/full_vocab.py)、[蒸馏配置验证](../../../../verl/workers/config/distillation.py)、[loss注册与聚合](../../../../verl/trainer/distillation/losses.py)。数学与计算图复用[既有理论文档](../theory/2026-10-22-Full-vocab计算图与四种更新.md)。

### 8.2 实际核对过的模型元数据

| 模型 | HF架构 | Vocab size | Tied embedding | 本仓full结构兼容性 |
| --- | --- | ---: | --- | --- |
| R1-Distill-Qwen-1.5B | Qwen2 | 151936 | False | 支持 |
| JustRL-DeepSeek-1.5B | Qwen2 | 151936 | False | 支持；本机与student的基础vocab及added-token ID映射一致 |
| R1-Distill-Qwen-7B | Qwen2 | 152064 | False | 单模型支持，但与1.5B配对head行数不等 |
| Qwen3-1.7B-Base | Qwen3 | 151936 | True | 不支持 |
| Qwen3-8B | Qwen3 | 151936 | False | 不支持 |
| Qwen3-1.7B | Qwen3 | 151936 | True | 不支持 |
| Qwen3-4B-Instruct-2507 | Qwen3 | 151936 | True | 不支持 |
| Qwen2.5-7B-Instruct | Qwen2 | 152064 | False | 结构支持，尚未在该pair运行 |
| OpenThinker3-7B | Qwen2 | 152064 | False | 同上 |

元数据来自本机config与官方HF config；没有下载新模型权重。[^model-configs]

R1的151936／152064差异不能直接用矩阵裁切“解决”后继续称原始full-vocab KL。应先确定实际token语义、padding行与生成时的有效词表，再统一支持及归一化；这会成为额外方法条件。

Qwen3-1.7B-Base的config最大位置为32768。本仓会把 `max_prompt + max(train_response,val_response) + 1` 作为vLLM窗口；即使1024+31744也会派生32769。适配Qwen3时要明确总窗口与EOS预算，不能继续沿用当前4096 prompt默认值。

### 8.3 本机资产与默认配方差异

- 本机现有权重在工作区 `models/`，含R1-Distill-1.5B与JustRL-1.5B；脚本默认的同级 `model/` 在本机不存在。开发机的单数目录是另一个部署状态。
- 本机数据在 `MD-OPD/data/math/`，有DAPO 17917条、MATH-500 500条、AMC23 40条、AIME25 30条；没有AIME24目录。
- 默认是B=64、n=1、prompt cap4096、验证top-p=0.9、打乱训练数据。Rethinking/TIDE分别需要n=4、prompt cap1024、验证top-p=0.95、固定数据顺序。
- 原生Python默认指 `.venv-native/bin/python`，该路径在本机不存在；本机已有可解析配置的 `/root/.local/share/mirroropd/venvs/profile-cu128/bin/python`。不据此推断开发机环境或资产已经同步。
- DeepSeek模型应保持 `CONTINUOUS_TOKEN_MODEL_FAMILY=deepseek`；Qwen2.5模型要改为 `qwen25`／经核对的Qwen协议，Qwen3要用其协议与明确的thinking设置。不要让默认DeepSeek builder处理Qwen3。

## 9. 原候选 Full-vocab 对照设计（非当前执行计划）

### 9.1 每个failure pair的最小矩阵

| Arm | 目标 | 用途 |
| --- | --- | --- |
| 原论文基线 | 保持作者的sampled或top-k PG／直接KL实现 | 先确认failure确实出现 |
| Sampled-current | 当前log-ratio、停止梯度、无额外任务reward；fresh batch一次更新 | 轻量条件reverse-KL基线 |
| Student-top-k RKL | K=16，明确是否支持内重归一化、是否score surrogate | 检验学生支持截断 |
| Teacher-top-k RKL | K=16或原论文32，同样明确归一化与梯度 | 检验教师支持选择 |
| Full-vocab RKL | 当前学生概率、完整词表求和，直接精确条件梯度 | 检验取消动作抽样及支持截断是否改善 |

第一轮保持 `ppo_epochs=1`，不要同时引入E16/E17的多次复用。最初可以只做原基线与Full两条，确认现象后再补机制臂；若原基线是PPO／冻结advantage且多次更新，则应补同样梯度语义的sampled-current对照，避免把取消clipping或改变信号刷新归因于完整词表。

Normalized top-k与full-vocab在支持内外的概率质量处理不同，full并非单纯把同一个随机估计器的样本数增大。**sampled→full**主要检验条件动作积分；**normalized top-k→full**还检验截断与归一化目标变化。

### 9.2 固定的实验条件

- 相同student初始化、teacher checkpoint、prompt顺序、数据范围、训练长度、n、有效序列batch、LR、优化器和loss的全局归约。
- 相同验证题、prompt/chat模板、thinking开关、采样温度、top-p、回答上限和grader；先独立重测未训练student与teacher。
- 公平比较以相同optimizer更新数和训练prompt预算为主，再单列wall-clock／GPU-hours。不要把full做更多复用后的收益称为“同预算full救回”。
- 第一轮可用seed42验证现象，正式报告至少3个种子（R）。原论文未给seed处要标作我们的选择。
- 一个run训练后轨迹自然会变化，相同随机种子不意味着两种loss始终看到相同prefix。若要分离局部梯度差异，另用固定prefix做诊断，不能把它与在线效果混为一个实验。

主比较先固定论文LR。若full未救回，可以给所有方法相同的0.5×／1×／2×LR预算做补充，区分优化尺度与目标问题；不能只替full挑最优LR再与未调参的失败基线比较。

### 9.3 判断“救回”的指标

| 失败类型 | 首要判据 | 辅助诊断 |
| --- | --- | --- |
| 不提升 | 最终／固定checkpoint窗口的avg@k显著优于原始student与失败baseline | 师生差距恢复比例、梯度范数 |
| 后期崩溃 | 相同训练长度下避免从峰值持续退化 | 长度分位数、停止原因、重复、缺失答案 |
| 负迁移 | 保住原student能力并改善目标任务 | 不要只报告teacher matching loss |
| 局部负迁移 | 对原退化基准改善，且其余基准不过度牺牲 | 每个基准单列，不只看macro均值 |

Full-vocab可能消除动作抽样噪声、恢复遗漏的监督，但不会让错误前缀上的教师分布自动变成正确监督，也不会保证教师具有可迁移的新能力。若failure在full下仍存在，这本身就是支持“状态／目标问题超出支持集估计”的有价值结果。

## 10. 原 n=4 长窗配置解析记录（非当前执行计划）

下面是已有资产上的配置预览，采用Rethinking的B=64、n=4、LR与长度设置，但暂沿用本机已有数据和三项验证。因此它是**可先进行的长窗工程预实验**，不是已经准备完毕的论文精确复现。严格版本需要补AIME24、论文prompt和top-k基线。

从仓库根执行。代码块保留 `DRY_RUN=1`，只展示命令；真正训练仍需按现有作业方式显式分配GPU。

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
bash examples/mirroropd/run_math_deepseek_r1_1_5b_fsdp2.sh \
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
bash examples/mirroropd/run_full_reverse_kl.sh \
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

把 `MAX_RESPONSE_LENGTH` 改为7168，得到对应短窗对照；其余参数不改。完整4臂是 sampled-7K、sampled-15K、full-7K、full-15K。论文后期图包含到250步，若200步未出现失败，可预先约定延长至250／一个完整epoch，标为延长复现，不能事后只选最差节点。

本轮已对15K sampled-current／reinforce与full-rKL配置运行 `CONFIG_ONLY=1`：解析结果为B=64、n=4、M=64、E=1、LR=1e-6、response15360、val n16/top-p0.95、200outer。代码块额外显式固定相同seed与零warmup。**未启动Ray训练、未加载GPU权重、未声称failure已复刻。**

## 参考文献与源码

[^rethinking]: Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe. arXiv:2604.13016，§3–6、附录A.2。[论文](https://arxiv.org/html/2604.13016)
[^rethinking-code]: Thinking-Space/Rethinking-OPD，commit `ac26e38d6f1572eb027597b48a9f4e01f6915ef8`。[训练脚本](https://github.com/Thinking-Space/Rethinking-OPD/blob/ac26e38d6f1572eb027597b48a9f4e01f6915ef8/on_policy_distillation.sh)；[评测脚本](https://github.com/Thinking-Space/Rethinking-OPD/blob/ac26e38d6f1572eb027597b48a9f4e01f6915ef8/scripts/val/eval/gen_vllm.py)；[README与验证说明](https://github.com/Thinking-Space/Rethinking-OPD/blob/ac26e38d6f1572eb027597b48a9f4e01f6915ef8/README.md)。
[^rethinking-actor]: 同仓 `dp_actor.py` 的 `compute_reward_weights`、`only_stu`及更新分支。[源码](https://github.com/Thinking-Space/Rethinking-OPD/blob/ac26e38d6f1572eb027597b48a9f4e01f6915ef8/verl/verl/workers/actor/dp_actor.py)
[^rethinking-core]: 同仓 `core_algos.py` 的3维候选PG及候选求和。[源码](https://github.com/Thinking-Space/Rethinking-OPD/blob/ac26e38d6f1572eb027597b48a9f4e01f6915ef8/verl/verl/trainer/ppo/core_algos.py)
[^tide]: Mismatch Matters: On-Policy Distillation Beyond Token Agreement. arXiv:2608.09836，表1、表3、表9与附录D。[论文](https://arxiv.org/html/2608.09836)
[^tide-code]: yzc-666/TIDE，commit `b54a9e15b2064a8d18c36f3cd3f73ec8906cefc5`。[训练脚本](https://github.com/yzc-666/TIDE/blob/b54a9e15b2064a8d18c36f3cd3f73ec8906cefc5/on_policy_distillation.sh)；[rollout默认配置](https://github.com/yzc-666/TIDE/blob/b54a9e15b2064a8d18c36f3cd3f73ec8906cefc5/verl/verl/trainer/config/rollout/rollout.yaml)。
[^tide-eval]: 同仓独立评测脚本。[源码](https://github.com/yzc-666/TIDE/blob/b54a9e15b2064a8d18c36f3cd3f73ec8906cefc5/scripts/val/eval/gen_vllm.py)
[^revisiting]: Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes. arXiv:2603.25562，§4.2、表1、附录表A1。[论文](https://arxiv.org/html/2603.25562)
[^revisiting-code]: hhh675597/revisiting_opd，commit `f32f284f25bae5b16d2d44ee336b52851dccc736`。[Teacher-top-K数学脚本](https://github.com/hhh675597/revisiting_opd/blob/f32f284f25bae5b16d2d44ee336b52851dccc736/examples/opd/opd_math_qwen2.5-7b_it.sh)
[^revisiting-original]: 同仓sampled-token数学脚本。[源码](https://github.com/hhh675597/revisiting_opd/blob/f32f284f25bae5b16d2d44ee336b52851dccc736/examples/opd/opd_original_math_qwen2.5_7b_it.sh)
[^revisiting-core]: 同仓 `kl_penalty`、`compute_opd_advantage` 与数据批处理。[core_algos.py](https://github.com/hhh675597/revisiting_opd/blob/f32f284f25bae5b16d2d44ee336b52851dccc736/verl/trainer/ppo/core_algos.py)；[trainer](https://github.com/hhh675597/revisiting_opd/blob/f32f284f25bae5b16d2d44ee336b52851dccc736/verl/trainer/ppo/ray_trainer_multitask.py)。
[^semi]: When Do We Need On-Policy Distillation? Distilling on Offline Student Rollouts Is Often Better. arXiv:2610.11291，附录A、表2–4，特别注意表1与图2的rollout cap差异。[论文](https://arxiv.org/html/2610.11291)
[^demystifying]: Demystifying On-Policy Distillation: Roles, Pathologies, and Regulations. arXiv:2607.13399，§3–5与附录A。[论文](https://arxiv.org/html/2607.13399)
[^model-configs]: 2026-10-24核对官方HF config：[R1-7B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B/blob/main/config.json)、[Qwen3-1.7B-Base](https://huggingface.co/Qwen/Qwen3-1.7B-Base/blob/main/config.json)、[Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B/blob/main/config.json)、[Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B/blob/main/config.json)、[Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/blob/main/config.json)、[Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/config.json)、[OpenThinker3-7B](https://huggingface.co/open-thoughts/OpenThinker3-7B/blob/main/config.json)。本机两份1.5B模型另直接读取config与tokenizer做语义映射比较，没有新增权重下载或文件摘要清单。
