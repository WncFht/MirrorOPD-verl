# 固定状态蒸馏：数据、指标与读图口径

本页维护谱系实验共用的题池、固定状态、指标与图形规则。每个历史实验的实际预算、支持集、模板和数值实现仍以其报告为准；新公共约定不改写旧实验。

## 训练数据与 held-out 的具体含义

题目来自本地固定抽样，不是当前 student 在线不断采新题：数学来自 `data/math/DAPO/train.parquet`，代码来自 `data/code/TACO/train.parquet`，指令遵循来自 `data/if/MultiIF/train.parquet`。原始抽样固定 seed2026102711，并排除当时更早实验已用题；按原 R1 tokenizer 筛选 24–384 token 的 prompt，过滤中文、链接和 interactive problem，IF 保留 English 条目。后续模型复用这些题目 ID，重新按自身 student 模板编码，不重新筛选一批更容易的题。

| 用途          | 完整银行题数 | 分组            |                                    当前常规固定评分 |
| ------------- | -----------: | --------------- | --------------------------------------------------: |
| 数学训练池    |          320 | A–T，每组 16 题 | 64 步实验评分 A–H 128 题；320 步实验评分 A–T 320 题 |
| 数学 held-out |          192 | M01–M12         |                                   M01–M04，共 64 题 |
| Code held-out |           64 | Code01–Code04   |                             Code01–Code02，共 32 题 |
| IF held-out   |           64 | IF01–IF04       |                                 IF01–IF02，共 32 题 |
| 合计          |      **640** | 40 组×16 题     |                 64 步实验 256 题；320 步实验 448 题 |

训练只用数学池。64 次更新访问 A–D 共 64 题，128 次访问 A–H 共 128 题，320 次访问 A–T 共 320 题；每组 16 题连续 reuse16 次。64 步时 E–H 属于训练池中尚未访问的批，不应与永久保留的 M/Code/IF 混为同一 held-out 定义。640/1280 阶段循环同一训练池，不增加数据覆盖。

每个 student/模式先生成一条对应的 S0 回答，H1024、独立逐题 seed、temperature1，遇 EOS 提前停止；随后所有 teacher 共享这些逐题前缀。每次更新用 teacher 在前缀上的完整概率分布计算 RKL，不用标准答案做监督，也不需要 teacher 生成一套训练答案。更新后的 student 不重新 rollout，因此当前是固定初始状态上的蒸馏拟合诊断，而非每步 fresh rollout 的生产 OPD。

数学留出衡量同领域未用于更新的状态；Code/IF 衡量跨领域分布拟合迁移。目前这些面板报告 RKL/JS，不等价于代码通过率或指令任务成功率。当前图同时展示Math、Code、IF及整体留出风险，按整体留出降幅排序。

“held-out”只相对于本轮蒸馏更新而言。这些题本身取自上游数据的 train split，部分曾用于更早研究，不能保证 teacher 预训练/后训练未见，也不是完全未接触的盲测集。独立的 630 题任务评测由 MATH500、AMC23、AIME24/25/26 组成；训练题与这 630 题做了规范化文本去重，但它们不等于这里的 192 道数学 held-out。

抽样及分组可复核[题池构建源码](../../../../research/studies/cross_batch/prepare_long_batch_transfer.py)与[R1 冻结/去重源码](../../../../research/studies/boundary/prepare.py)。Thinking 与 non-thinking 分别生成状态银行；Llama 也必须用固定 Llama student 重新生成，不能复用 Qwen/R1 的回答 token。

## 整体 held-out 的平均与排序

当前共同评分面板包含 Math64、Code32、IF32，共 128 题。对每道题先按固定 H1024 预算归一化，再合并求平均；提前 EOS 后的不存在位置没有额外损失项。给定三个领域的初始/终点平均风险，整体量为：

$$
R_t^{\mathrm{all}}=\frac{64R_t^{\mathrm{Math}}+32R_t^{\mathrm{Code}}+32R_t^{\mathrm{IF}}}{128},
\qquad
D^{\mathrm{all}}=100\left(1-\frac{R_T^{\mathrm{all}}}{R_0^{\mathrm{all}}}\right).
$$

图保留负降幅，以向左的柱形表示 RKL 上升；不截掉不利结果。图从上到下按整体降幅排序，不直接平均三个领域的百分比，也不将三个领域等权。整体初始风险为零时不强行给出排序。各领域的绝对风险、降幅和题数保存在 CSV，便于改用其他明确声明的汇总方式。

每个 student／输入模式单独一张图，左右面板共用教师顺序和百分比刻度。左侧展示实际参与更新的完整训练题集（`train_seen`）与旧批 A–D；右侧展示整体、Math、Code、IF 四组留出条形。64／128／320 步分别只有 64／128／320 道题参与训练，之后固定 A–T 循环仍为 320 题；未参与更新的题目不算训练数据。A–D 是训练题集的子集，64 步时两条训练柱相同，不能当作两套独立证据。颜色表示面板；实心表示 teacher 与 student 标称尺寸相同，斜线纹理表示跨尺寸。同尺寸只表示规模相同，不自动证明参数布局同构；teacher-copy 仅用于严格布局可比的模型。

模型标签使用完整名称与尺寸，并区分 Base、后训练双模式、专用 Thinking／Instruct checkpoint，以及实际输入模式和原生训练模式。普通 Qwen3 的官方名称不带 Instruct 后缀，图中保留原名并标为 post-trained dual-mode，不人为改名。Qwen3-1.7B-Base 标为祖先分布及 thinking 输入；普通 Qwen3-4B/8B 标为双模式 checkpoint、本轮 thinking 输入；Thinking-2507 标为 thinking-only 及前缀差异。SFT/Base-OPD 的历史 thinking 输入与其 non-thinking 训练条件不匹配，标签明确保留这一点。

同一图可能包含 64/128/320 步终点，行标签记录实际更新数。R1 小教大两组采用共同 151936 维条件 KL，其他原生支持结果不能直接当成完全相同训练目标。排序用于定位现象，不构成忽略预算、状态来源、模式和支持差异的因果排名。

## 优化与状态协议

主设置为 batch16、reuse16、H1024、LR1e-6、AdamW β=(0.9,0.999)、ε=1e-8、weight decay0.01、clip1。参数/梯度/Adam/head 为 FP32，backbone 使用原生 BF16 autocast，RoPE buffers 保留 FP32，TF32 关闭。不同执行 microbatch 须通过数值核验。固定 seed 不保证 GPU 反向逐值可重复；R1 受控探针已测得默认 FlashAttention 反向的微小差异，新 R1 长训显式开启确定性反向并重新验收，历史条件保留原记录。题目 bootstrap 不包含这种训练轨迹变动的不确定性。

固定初始 S0 与 fresh rollout 是不同条件。当前谱系主实验用固定 S0；Thinking/non-thinking 和不同 student 分别建立银行，同一条线内的 teacher 共享前缀。320→640→1280 续训要求固定题池覆盖并保持 Adam/RNG/批次位置；旧 weights-only 终点不得冒充连续恢复。

## 证据与复算入口

- 当前模型清单、三域结果和图：[实验总览](2026-10-29-教师谱系实验总览与RKL进展.md)。
- 数值路径、命令、缓存和恢复合同：[研究引擎](../../../../research/studies/fixed_transfer/README.md)。
- 汇总与绘图：[overview.py](../../../../research/studies/fixed_transfer/overview.py)。
- 每个实验的精确模式、预算、支持与限制分别保留在[R1](r1/README.md)、[Qwen3](qwen3/README.md)、[Llama](llama/README.md)。
