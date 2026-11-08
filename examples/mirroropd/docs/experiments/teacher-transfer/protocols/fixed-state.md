# 固定状态 RKL 实验协议

本协议用于[师生配对 RKL 总览](../fixed-state-results/README.md)中的 0–320 更新比较。训练与评分使用冻结的初始学生轨迹；按批刷新实验采用[另一份协议](online-four-domain.md)。

<a id="common"></a>

<a id="common-当前展示协议固定状态0320-更新"></a>

<a id="common-数据与状态"></a>

## 题目与状态银行

题目来自 DAPO 数学、TACO 代码和 MultiIF 英文指令。完整银行共 640 题：数学训练池 A–T 为 320 题，数学留出 192 题，Code 与 IF 各 64 题。常规留出读出使用 Math64、Code32、IF32，共 128 题。

每个 student／输入模式由原始权重生成自己的 S0 回答，temperature=1、固定逐题 seed、H=1024，EOS 后停止。同一条件内的 teacher 共用这些 token 前缀。训练更新时重算 student 分布，冻结 teacher 可缓存 hidden/head。

留出表示本轮未参与梯度更新；题目来自原数据集的 train split，部分题曾用于其他研究。独立任务评测覆盖 MATH500、AMC23、AIME24/25/26 共 630 题，采用 avg@4。

<a id="common-预算和优化"></a>

## 更新与数值条件

| 参数                      | 设置                                                          |
| ------------------------- | ------------------------------------------------------------- |
| Batch／reuse／horizon     | 16／16／1024                                                  |
| Optimizer                 | AdamW，LR1e-6，β=(0.9,0.999)，ε=1e-8，weight decay0.01，clip1 |
| 参数、梯度、Adam 与输出头 | FP32                                                          |
| 主干与位置编码            | BF16 autocast；RoPE buffers 保留 FP32；TF32 关闭              |
| 固定风险读出              | 每 16 次更新；记录实际数值执行条件                            |

| 学生更新数 | 已访问训练题组 | 已访问独立题数 |
| ---------- | -------------- | -------------: |
| 64         | A–D            |             64 |
| 128        | A–H            |            128 |
| 320        | A–T            |            320 |

<a id="common-风险和画图"></a>

## 风险与图表

训练目标是在固定前缀上对完整动作支持求和的 RKL，即 KL(student ∥ teacher)。每题使用固定 H1024 分母，EOS 之后没有损失项。整体留出先合并 Math64、Code32、IF32 的风险，再计算相对初始降幅。

终点图的训练栏只包括已经访问的题；固定全训练池曲线可以包括尚未访问的计划题，图中注明范围。A–D 为训练池子集。跨 batch 图分别标出已见、当前、未来训练批和永久留出。

<a id="common-已保存的数据"></a>

每次更新保存更新前的当前 batch loss、梯度范数、实际 token 数和耗时；固定快照对应已完成更新后的权重。既有读出通常包括 RKL、JS、TV、student/teacher entropy、有效 token 数、适用配对的额外支持质量及位置汇总；完整逐 token KL 和全词表概率的可用性以产物为准。

<a id="r1"></a>

<a id="r1-r1-同尺寸教师0320-结果与执行条件"></a>

## R1 同尺寸：确定性副本

Student 固定为原始 DeepSeek-R1-Distill-Qwen-1.5B，六个训练 teacher 为 JustRL、Nemotron v1、MiniMath、DeepScaleR、BroRL 和 DeepCoder；Math-1.5B 仅做模型基线评测。[执行配置](../../../../research/studies/boundary/plans/r1-samesize-deterministic320.json)绑定各模型身份。

六条副本从原始 student 与 fresh Adam 独立开始，采用确定性 FA2 backward。每次更新前核对 analytic VJP 与 autograd，并检查重复 backward。[重复性记录](../transfer-mechanism/r1/data/budget-audit-20261030/gradient-repeatability.json)和[S0 核对](../transfer-mechanism/r1/data/budget-audit-20261030/deterministic-CONT-J-gate.json)保存验证结果。旧 checkpoint 的严格重放未通过，因此新副本作为独立轨迹记录。

<a id="small-to-large"></a>

<a id="small-to-large-r1-小-teacher大-student0320-比较"></a>

## R1 小教大：共同输出支持

原始 R1-7B 分别学习 R1-1.5B 与 JustRL-1.5B，使用同一 R1-7B S0 银行和预算，各分支独立初始化。双方输出头相差 128 行；目标采用共同 151936 维条件 RKL，保留完整 7B head 并记录额外概率质量。

主比较采用确定性 FA2 的 320 更新副本，旧 64 更新任务评测保留其 checkpoint 身份。Polaris／Light 旧行使用原生支持，结果表将两种口径标开。

<a id="qwen"></a>

<a id="qwen-qwen3-双模式当前-0320-比较协议"></a>

## Qwen：输入模式与模板

同一 Qwen3-1.7B 权重分别建立 thinking 与 non-thinking S0 银行，各银行内 teacher 共用 student 前缀。两种模式在各自银行内比较，模型版本与实际输入见[配置](../../../../research/studies/fixed_transfer/plans/qwen3-mode-protocol.json)及[详细诊断](../appendix/qwen-lineage-evidence.md)。

普通 Qwen3-0.6B／4B／8B 具有双模式对照。e3、GRPO-KL 保留 thinking 参照；Thinking-Space 的 SFT/Base-OPD 采用与公开训练条件匹配的 non-thinking 输入，旧 thinking 结果作为模式不匹配条件保留。其他 teacher 的模式能力按发布说明核对。

2507 两版使用各自原生模板，单列比较。Polaris-4B 同时保留 YaRN 差异。阶段研究中的 teacher step80／step640 是模型身份，相关结果见[后训练阶段](../post-training-stages.md#fixed-stage)。Teacher-copy 提供同条件参考点。

<a id="llama"></a>

<a id="llama-llama-固定学生0320-教师比较"></a>

## Llama：共享输入与支持差异

Student 为 Unsloth 来源的 Llama-3.2-3B-Instruct。原六组 teacher 为 GT、Co-rewarding-I、arithmetic-full SFT、Llama-3.2-1B-Instruct、Llama-3.1-8B-Instruct、Nemotron-Nano-8B-v1；新增组为 Tulu-SFT／DPO／RLVR 与 SimpleRL。身份见[原教师配置](../../../../research/studies/fixed_transfer/plans/llama-lineage-protocol.json)和[阶段配置](../../../../research/studies/fixed_transfer/plans/stage-lineage-models.json)。精确父 revision 的已知范围按各模型资料记录。

各组采用共享 student 模板、固定 off 系统消息与日期、冻结题目和 S0 前缀。Nemotron 的来源为 SFT＋RL＋合并；算术 SFT 保留数据披露范围；Tulu 记录额外 teacher 输出质量；SimpleRL 为输入不匹配条件。原六组的[绝对曲线](../transfer-mechanism/data/equal-budget-20261030/llama-original/equal-budget-rkl.png)与[相对曲线](../transfer-mechanism/data/equal-budget-20261030/llama-original/equal-budget-drop_percent.png)、新增阶段证据均使用各自 0–320 读出。

全部分组数值在[配对总览](../fixed-state-results/README.md)维护，逐 run 图和重绘方式见[图谱与复算](../reproduction.md)。
