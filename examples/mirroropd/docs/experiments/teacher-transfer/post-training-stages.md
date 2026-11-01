# 后训练阶段的教师可学习性

比较同一学生对不同后训练阶段教师的拟合。Qwen/Tulu 提供固定 S0、0–320 更新的历史结果；MiMo 及后续阶段链采用按批刷新协议。各组共享哪些条件、改变哪些因素，在对应设置中说明。

完整配对结果见[师生 RKL 总览](fixed-state-results.md)，阶段来源和公开权重见[模型资料](../../settings/模型与师生配对.md)。

<a id="fixed-stage"></a>

<a id="fixed-stage-qwen-与-llama-阶段对照0320-更新"></a>

## Qwen SFT 深度与 Tulu 阶段

这组比较采用学生 0–320 更新。Qwen 的 step80／step640 来自同一 SFT run；六个新增 teacher 和额外 Base-GRPO 条件各训练 320 次，共 2240 次更新。

<a id="fixed-stage-固定模型与证据"></a>

### 固定模型与证据

| 编号 | Teacher 仓库                                      | 固定 revision                              | 子目录    |
| ---- | ------------------------------------------------- | ------------------------------------------ | --------- |
| Q1   | `jasonrqh/Qwen3-4B_Math-CoT-20k_lr5e-5_ep8_bs256` | `aaf1c51a6a2ee319750cafbd7cb8be0df10018b8` | `step80`  |
| Q2   | `jasonrqh/Qwen3-4B_Math-CoT-20k_lr5e-5_ep8_bs256` | `aaf1c51a6a2ee319750cafbd7cb8be0df10018b8` | `step640` |
| L1   | `allenai/Llama-3.1-Tulu-3-8B-SFT`                 | `f2a0b46b0cfda21003c6141b1ff837b7e165524d` | `根目录`  |
| L2   | `allenai/Llama-3.1-Tulu-3-8B-DPO`                 | `a7beb67e33ffd01cc87ac3b46cadc1000985b8db` | `根目录`  |
| L3   | `allenai/Llama-3.1-Tulu-3-8B`                     | `666943798adbde0b1aff34626007e26986a3c107` | `根目录`  |
| L4   | `hkust-nlp/Llama-3.1-8B-SimpleRL-Zoo`             | `d6e668fcf8db53ff55cc39cf347b3494f56dc7e9` | `根目录`  |

[模型注册表](../../../research/studies/fixed_transfer/plans/stage-lineage-models.json)定义身份；[逐模型清单](transfer-mechanism/data/stage-lineage-20261030/Q1.json)及同目录 Q2、L1–L4 文件保存路径、字节数、SHA256 与下载完成回执。[Qwen 验收配置](../../../research/studies/fixed_transfer/plans/qwen-stage-lineage-acceptance.json)与[Llama 验收配置](../../../research/studies/fixed_transfer/plans/llama-stage-lineage-acceptance.json)的 `probes` 阶段已完成；两份配置的用途为输出验收；正式训练条件见下文。模型身份包含 SFT checkpoint 子目录。

Qwen step80／step640 的训练来源和 Tulu／SimpleRL 的阶段链统一见[模型介绍](../../settings/模型与师生配对.md#qwen3)及[Tulu 阶段链](../../settings/模型与师生配对.md#llama)。六组均为跨尺寸蒸馏，补充教师侧阶段与谱系；不能替代同构 RL／SFT 对照。公开父型号不证明精确父权重、数据和预算完全受控。

<a id="fixed-stage-比较条件"></a>

### 比较条件

每条分支独立从原始 student 开始，同一输入条件共享冻结题目和 S0 前缀。Qwen 使用共同 thinking 前缀，Llama 使用固定 3B student 的共享 off 输入。Tulu 原生模板不同，未静默换成 teacher 私有前缀；SimpleRL 没有原生聊天模板，保留输入不匹配标记。

Tulu teacher 输出头 128264，student 为 128256；保留原生 teacher 归一化并记录额外质量。

<a id="fixed-stage-等预算结果"></a>

### 等预算结果

| Teacher                       | 学生更新数 | 整体留出初始 RKL | 整体留出终点 RKL |   降幅 |
| ----------------------------- | ---------: | ---------------: | ---------------: | -----: |
| Qwen3-4B Math-CoT SFT step80  |        320 |         0.386144 |         0.287476 | 25.55% |
| Qwen3-4B Math-CoT SFT step640 |        320 |         0.419072 |         0.370862 | 11.50% |
| Qwen3-4B-Base-GRPO            |        320 |         0.664417 |         0.335392 | 49.52% |
| Tulu-3-8B-SFT                 |        320 |         0.192855 |         0.104783 | 45.67% |
| Tulu-3-8B-DPO                 |        320 |         0.428990 |         0.227525 | 46.96% |
| Tulu-3-8B，最终 RLVR          |        320 |         0.453277 |         0.236005 | 47.93% |
| Llama-3.1-8B-SimpleRL-Zoo     |        320 |         0.263856 |         0.117807 | 55.35% |

Qwen 和 Llama 分别比较，结合初始风险、绝对残差与相对降幅解释结果。SimpleRL 还采用共享 student 前缀这一特定输入条件。

[附录中的等预算曲线与阶段偏好诊断](#stage-curves)保留训练全池、旧批和三域留出。阶段投影的定义与结果见下节。

<a id="stage-curves"></a>

<a id="stage-curves-附录阶段对照的-0320-图表"></a>

### 曲线与偏好方向

所有曲线限定共同的 0–320 更新。固定训练全池早期包含尚未访问的题，A–D 是其子集；训练与三域留出同时展示。

<a id="stage-curves-qwen-三组"></a>

#### Qwen 三组

![绝对 RKL](transfer-mechanism/data/equal-budget-20261030/qwen-sft-grpo/equal-budget-rkl.png)

![相对降幅](transfer-mechanism/data/equal-budget-20261030/qwen-sft-grpo/equal-budget-drop_percent.png)

[逐点数据](transfer-mechanism/data/equal-budget-20261030/qwen-sft-grpo/learning-curves.csv)。

<a id="stage-curves-llama-四组"></a>

#### Llama 四组

![绝对 RKL](transfer-mechanism/data/equal-budget-20261030/llama-stages/equal-budget-rkl.png)

![相对降幅](transfer-mechanism/data/equal-budget-20261030/llama-stages/equal-budget-drop_percent.png)

[逐点数据](transfer-mechanism/data/equal-budget-20261030/llama-stages/learning-curves.csv)。SimpleRL 为共享前缀下的输入不匹配对照。

<a id="stage-curves-tulu-阶段偏好方向的评分结果"></a>

#### Tulu 阶段偏好方向的评分结果

三条 320 更新 student 终点在同一 448 题固定状态上完成评分，不产生科学更新。[逐题矩](transfer-mechanism/data/stage-preferences-20261030/prompt-moments.csv)覆盖训练 A–T 和 Math／Code／IF 留出；[完成回执](transfer-mechanism/data/stage-preferences-20261030/complete.json)锁定权重、银行、精度与实现。由逐题矩重算全部 16 个配对投影差，最大差为 1.33e-15，见[复算记录](transfer-mechanism/data/stage-preferences-20261030/recomputed-contrasts.json)；这是汇总核对，不是第二次独立 forward 验证。

![Tulu 相邻阶段的学生偏好投影差](transfer-mechanism/data/stage-preferences-20261030/paired-stage-preferences.png)

| Teacher 阶段方向 | 训练全池：后阶段 student 减前阶段 student | 整体留出：后阶段 student 减前阶段 student |
| ---------------- | ----------------------------------------- | ----------------------------------------- |
| SFT→DPO          | +0.893 [0.883, 0.903]                     | +0.697 [0.677, 0.715]                     |
| DPO→RLVR         | +0.778 [0.761, 0.795]                     | +0.465 [0.421, 0.505]                     |

方括号为逐题配对 bootstrap 95% 区间；完整域别见[配对结果](transfer-mechanism/data/stage-preferences-20261030/paired-stage-projection.csv)。两条方向上，后阶段教师训练出的 student 均比前阶段对应 student 更沿后阶段的 token 偏好变化，且该现象出现在训练和固定留出上。

**该读数为有符号的投影系数差，取值可超过 1。** 整体留出在 DPO→RLVR 方向的投影为 1.914→2.378，差为 0.465。相应余弦从 0.285 提升到 0.339；SFT→DPO 方向余弦从 0.136 提升到 0.676，见[几何量表](transfer-mechanism/data/stage-preferences-20261030/geometry.csv)。两个阶段方向具有不同的尺度和相关性，各自解释。

在共享 student 输入、固定 S0、H1024 和单训练 seed 下，学生在训练与留出上都追随了部分教师偏好变化。独立任务表现由对应任务评测回答。

<a id="mimo"></a>

<a id="mimo-mimo-三类后训练教师吸收对照"></a>

## MiMo：三类后训练教师

固定 MiMo-7B-Base student，比较 SFT、Base 直接 RL、SFT 后继续 RL 三个冻结教师在相同数学训练与公共评估条件下的可拟合性。原 E69／E70 DeepSeekMath 方案已撤销；新 E71／E72／E73 分别对应 SFT+RL、直接 RL、SFT。模型阶段与固定 revision 统一见[阶段链选型](../../settings/模型与师生配对.md#mimo)；BayesRL／Olmo3 与 Light-R1 仅为后续候选。

<a id="mimo-1-实验分支"></a>

<a id="1-实验分支"></a>

### 实验分支

| 编号 | Student      | Teacher         | 教师训练路径 |
| ---- | ------------ | --------------- | ------------ |
| E71  | MiMo-7B-Base | MiMo-7B-RL      | Base→SFT→RL  |
| E72  | MiMo-7B-Base | MiMo-7B-RL-Zero | Base→RL      |
| E73  | MiMo-7B-Base | MiMo-7B-SFT     | Base→SFT     |

三组从同一 Base 权重和 fresh Adam 独立开始，不顺序续接。E71 对 E73 比较完整 SFT+RL 与 SFT 教师的拟合难度；E72 提供不经 SFT 的 RL 路径参照。公开教师的数据、监督和计算预算不同，这里研究三个真实目标，不宣称只改变唯一算法变量。固定银行、current 轨迹和任务表现分别报告，不新增“完全吸收”的数值门槛。

<a id="mimo-2-共同训练设置"></a>

<a id="2-共同训练设置"></a>

### 共同训练设置

沿用[JustRL 按批刷新实验](cross-domain-transfer.md)的数据、优化预算和读出类别，关闭 source heldout 及 history RKL。训练回答长度保留 7168，公共评估长度按下表记录，并受 MiMo 原生 32K 上限约束。

| 项目     | 默认值                                                                        |
| -------- | ----------------------------------------------------------------------------- |
| 框架     | main 原生 verl V1／FSDP2、单节点 shared-sync，师生串行使用同一 GPU 池         |
| 数据     | 同一冻结`questions-v2`中的 6400 道 DAPO 数学训练题，题序相同，`shuffle=False` |
| Rollout  | current student 每 outer 刷新，每题 1 答；T1、top-p1、top-k−1、seed42         |
| 更新     | batch=mini-batch128、reuse32、50 outer；计划 1600 次 optimizer 更新           |
| 目标     | 原生 151680 维 Full-vocab RKL，无任务奖励、无 anchor、无 EOS 合并             |
| 优化器   | 全参数 AdamW，LR1e-6、constant、无 warmup、β=(0.9,0.999)、wd0.01、clip1       |
| 训练长度 | prompt2048＋response7168，共 9216；超长 prompt 报错                           |
| 精度     | FP32 学生主权重、BF16 混合计算；teacher 和 rollout 为 BF16                    |
| 后端     | `precompute_fused`、token chunk256、vocab chunk8192、异步运输及预取           |
| 默认资源 | 4 卡共享池、TP1、参数／optimizer offload、激活重算；microbatch 不改变科学预算 |
| 保存     | 终点 HF 模型及重载校验，无 optimizer 或中途恢复点                             |

所有分支使用 MiMo 发布的相同聊天模板及 token-ID 映射，continuous-token family 为`qwen25`。生成共同停止集合为`[151643,151645]`，覆盖两类原生 EOS；teacher 评分仍区分这两个动作。Thinking 内容保留在训练及 RKL 中，accuracy 取`</think>`后的答案；无 thinking 标签时保留整个回答。

MTP 推测解码和辅助训练关闭。原生模型配置将 MTP 层数设为 0，只加载下一 token 主干与 head；直接 HF 适配保留 MTP 模块时冻结其参数。四类公开 checkpoint 保持原位，三个训练分支采用同一执行约定，只比较普通下一 token 策略。

<a id="mimo-3-公共评估长度与采样"></a>

<a id="3-公共评估长度与采样"></a>

### 公共评估长度与采样

MiMo 论文§3.5.1 使用 T0.6、top-p0.95，Math／Code／Science 最大生成长度 32768，其他任务 8192；§3.3.3 和发布配置的总上下文 32768。[^mimo-mimo] 本实验保留既定公共 prompt 上限 8192，将推理域回答 cap 设为 24576，二者之和严格为 32768；IF 回答 cap 采用官方 8192。原定 32768 回答＋8192 prompt 的 40960 总预算不进入本轮原生上下文配置。

| 领域    | Prompt cap | Response cap | Accuracy 采样                         |
| ------- | ---------: | -----------: | ------------------------------------- |
| Math    |       8192 |        24576 | MATH-500×1；AMC23、AIME24/25/26 各×16 |
| Code    |       8192 |        24576 | LCBv5×10                              |
| Science |       8192 |        24576 | GPQA-Diamond×4                        |
| IF      |       8192 |         8192 | IFEval×1                              |

公共生成继续使用已定 T1、top-p0.95、top-k−1、seed42；官方 T0.6 仅作来源说明。Accuracy 共 1648 题、每模型每节点 6703 答，沿用 JustRL 题量与 n，不替换成 MiMo 论文的重复数。实际结果不宣称复现官方分数。

<a id="mimo-4-公共-rkl-与调度"></a>

<a id="4-公共-rkl-与调度"></a>

### 公共 RKL 与调度

| 读出                             | 工作量与调度                                                |
| -------------------------------- | ----------------------------------------------------------- |
| Source heldout RKL               | `source_enabled=False`，不生成、不评分                      |
| Training history RKL             | `history_enabled=False`，不抽样、不评分；真实训练轨迹仍保存 |
| 固定公共 initial／teacher bank   | 四域各 128 题、sample0；outer0／1／2 及每 5 outer 评分      |
| Current RKL 与全量 accuracy      | outer0／25／50；current sample0 复用同轮 accuracy 轨迹      |
| 初始 student 与 teacher accuracy | 各自 bootstrap 时完成                                       |

RKL 方向始终为 student 对 teacher。三个分支共享公共题目、panel IDs、采样协议和原始 Base；initial 银行复用完全相同的 token 产物，经各组 bootstrap finalize 校验。Teacher 银行归各自教师，不把不同前缀上的绝对 KL 混作主排序。训练 loss 为全局 token-mean；评估题内按有效 token 平均，再按题平均，四域外层等权。

<a id="mimo-5-入口与状态"></a>

<a id="5-入口与状态"></a>

### 入口与状态

本节命令针对[MiMo 接入验收](../../infra/model-compatibility/mimo/2026-11-01-MiMo原生下一token接入验收.md)所记录的源码版本。阶段专用入口 `launchers/mimo.py` 随模型接入变更维护；可用预设以当前 checkout 的[公共 shell 入口](../../../run_math_opd_ablation_fsdp2.sh) `--list-presets` 输出为准。`MIMO_QUESTIONS`指定 prepared 题目，`MIMO_BOOTSTRAP`指定各 teacher 绑定的完整银行。

```bash
DRY_RUN=1 bash examples/mirroropd/run_math_opd_ablation_fsdp2.sh E71
DRY_RUN=1 bash examples/mirroropd/run_math_opd_ablation_fsdp2.sh E72
DRY_RUN=1 bash examples/mirroropd/run_math_opd_ablation_fsdp2.sh E73
```

四份固定版本权重已落地，[原生 MiMo 工程验收](../../infra/model-compatibility/mimo/2026-11-01-MiMo原生下一token接入验收.md)已通过更新、公共读出、32K 上下文、终点原生重载及独立 HF 加载检查。短程验收使用独立的小样本、长度和更新预算，不替代本页 50 outer 设置，也不计入正式更新数。

三组已按 E71→E72→E73 提交持久运行队列，从原始 Base 分别开始。[实际完整配置与冻结计划](../../../../../runs/mimo-stage-transfer-20261101/formal-plan.json)固定各组师生权重、题集和源码；[运行状态](../../../../../runs/mimo-stage-transfer-20261101/formal-queue-state.json)区分 bootstrap、训练和已完成分支。正式完整预算的训练结果尚未产生；BayesRL／Olmo3 和 Light-R1 的接入独立记录在[阶段链验收](../../infra/model-compatibility/stage-chains/2026-11-01-Olmo3与LightR1阶段链接入验收.md)中。

<a id="mimo-参考文献"></a>

#### 参考文献

[^mimo-mimo]: Xiaomi. MiMo: Unlocking the Reasoning Potential of Language Model – From Pretraining to Posttraining. 2025. [论文§3.3.3、§3.5.1](https://arxiv.org/html/2505.07608)、[官方阶段及评估说明](https://huggingface.co/XiaomiMiMo/MiMo-7B-SFT)。

## Olmo3 与 Light-R1 的阶段条件

以下是阶段链预设的科学条件，工程接入和短程验收仍见[模型兼容性](../../infra/model-compatibility/stage-chains/2026-11-01-Olmo3与LightR1阶段链接入验收.md)。完整训练的完成范围按对应运行回执解释，不以短测代替。

<a id="stage-chains-1-预设与模型"></a>

### 模型与实际前序

| 预设 | Student 权重   | Teacher 权重                 | 阶段         |
| ---- | -------------- | ---------------------------- | ------------ |
| E74  | Olmo-3-1025-7B | Olmo3-C3PO-7B                | SFT+RL       |
| E75  | Olmo-3-1025-7B | Olmo3-IVON-SFT-7B            | SFT          |
| E76  | Qwen2.5-14B    | Light-R1-14B-DS              | SFT1+SFT2+RL |
| E77  | Qwen2.5-14B    | DeepSeek-R1-Distill-Qwen-14B | SFT1         |

两条链均没有在这里登记独立的 Base→RL checkpoint，不另造 RL-only 分支。每组计划从各自共同 Base 与 fresh Adam 开始。BayesRL 的 C3PO 保留其参数探索与 Seq-MIS 训练来源，不标成 vanilla GRPO；Light-R1 的实际前序为 R1-Distill→额外 3K SFT→GRPO。当前 E77 的 R1-Distill 不等于 E76 的 RL 直接起点，因此 E76／E77 不构成同一 SFT checkpoint 的严格 RL 增量对照；缺少的 SFT-only 中间权重及原始论文证据见[阶段链核查](../../settings/模型与师生配对.md#light-r1-lineage)。

本节命令针对[阶段链接入验收](../../infra/model-compatibility/stage-chains/2026-11-01-Olmo3与LightR1阶段链接入验收.md)所记录的源码版本；`launchers/stage_chains.py` 复用 `launchers/mimo.py` 的共同预算与读出，两个入口随模型接入变更维护。可用预设以当前 checkout 的 `--list-presets` 输出为准。E74–E77 的共享题集和 teacher-specific 银行分别由 `STAGE_CHAIN_QUESTIONS`、`STAGE_CHAIN_BOOTSTRAP` 指定。完整入口需要先完成 initial、teacher、finalize 三个原生 bootstrap 阶段。

<a id="stage-chains-3-默认预算与长度"></a>

### 训练与读出配置

| 项目            | 默认值                                                                         |
| --------------- | ------------------------------------------------------------------------------ |
| 数学训练        | 同一冻结题集的 6400 道 DAPO 题，题序固定，batch／mini-batch128                 |
| 优化预算        | reuse32、50 outer、计划 1600 次更新；AdamW LR1e-6、seed42                      |
| 训练长度        | prompt2048＋response7168；超长报错                                             |
| 公共长度        | prompt8192；Math／Code／Science response24576，IF response8192；总长最多 32768 |
| 公共读出        | 四域各 128 题固定银行；accuracy 保留 MiMo 的题量、n 和 T1／top-p0.95           |
| 调度            | 固定银行 outer0／1／2 及每 5 outer；current RKL、全量 accuracy 在 0／25／50    |
| 已关闭读出      | source heldout、history RKL；训练轨迹仍保存                                    |
| 精度与保存      | FP32 主权重、BF16 混合计算；终点 HF FP32 模型及重载检查                        |
| 默认资源        | 4 卡共享师生池；Olmo3／14B 每 rollout 副本最多 8／4 个序列                     |
| microbatch 预算 | 训练最多 16384 token／GPU；学生与教师推理预算 32768，可容纳完整公共轨迹        |

这里沿用 MiMo 的公共长度，不增加上下文外推。Olmo3 的模型配置为 65536，Light-R1 链的模型配置为 131072。R1 与 Light 原版 tokenizer 的长度元数据分别为 16384、20000，共同协议继续保留 SFT 的元数据；实际生成与评分按显式 32K 预算执行并验收。长序列工程检查不代替长推理准确率评估。

完整题集已做模板长度审计：Olmo3／R1 协议的 6400 道训练题最长分别为 1381／1374 token，公共题最长分别为 2856／2788 token，均在既定上限内；[审计数据](../../../../../runs/stage-chain-adaptation-20261101/prompt-lengths.json)。
