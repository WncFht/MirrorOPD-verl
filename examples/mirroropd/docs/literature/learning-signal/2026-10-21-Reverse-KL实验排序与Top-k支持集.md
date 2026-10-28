# Reverse-KL 实验排序、数值与梯度实现、Top-k 支持集

2026-10-22 将第 2 节的重复推导迁入[理论目录](../../theory/README.md)；本页维护实验排序、支持集选择与作者实现审计。

沿用[统一符号](../../theory/符号约定.md)：教师 $T$、冻结学生锚点 $q_k$、当前学生 $p_{k,n}=p_\theta$、实际行为分布 $\beta$，行为 IS 为 $w=p/\beta$。固定一个前缀，省略条件变量，定义

$$
\ell_a=\log\frac{p_a}{T_a},\qquad K=\mathrm{KL}(p\Vert T)=\sum_a p_a\ell_a,
\qquad \psi_a=\nabla_\theta\log p_a.
$$

以下取正概率与完整支持，所有梯度身份均针对固定前缀，不包含前缀访问测度的导数。

本页补充实验决策和支持集审计；基础方法见[条件梯度估计理论](../../theory/2026-10-22-条件梯度估计与控制变量.md)，hidden/head 容量见[算子性能与容量](../../infra/performance/full-vocab/operators.md#容量预算)。2026-10-21 核对本机 `One-Shot-OPD` 的干净 checkout `6b6b1782cc3f04749dd077de19353f839f6bc048`，并读取相关论文和固定版本源码。这里只做研究与 CPU 审计，没有启动新增训练实验，也没有修改 One-Shot-OPD。

## 1. Reverse-KL 实验优先级

本页维护方法比较的科学问题和控制变量，不维护第二份运行预算。

| 资料                                                                                       | 维护职责                                          |
| ------------------------------------------------------------------------------------------ | ------------------------------------------------- |
| 本页第 1、6 节                                                                             | 方法比较顺序、必要负对照与支持集数据合同          |
| [训练配方](../../experiments/learning-signal/sampled-controls/README.md)                   | E1–E15 编号、学习率、复用次数、外层预算与评测频率 |
| [Full-KL 数学对照](../../experiments/learning-signal/sampled-vs-full/README.md)            | 对照设置、实际运行与结果分析                      |
| [估计器审计](2026-10-21-OPD学习信号与梯度估计器对照.md#9-在当前-math-setting-下的实验安排) | 作者实现差异、资源公平性与需记录的量              |

排序依据是解释力、对照必要性与工程成本，不是未经实测的准确率预测。

| 顺序 | 方法                                        | 为什么重要                                                                      |
| ---- | ------------------------------------------- | ------------------------------------------------------------------------------- |
| 1    | Sampled-token，current signal + 正确行为 IS | 最便宜且已有实现的参照；本仓对应 E4，但首轮显式 `PPO_EPOCHS=1`                  |
| 2    | 精确 full-vocab rKL                         | 固定前缀的数值/梯度 oracle；先做小批审计，再评估流式端到端组                    |
| 3    | Sampled-token + KL baseline                 | 最直接检验“只降低抽样噪声是否足够”；区分 full baseline 与 top-k 近似 baseline   |
| 4    | ResOPD，并配套 HETS                         | 同一 sparse payload 下检验尾部向量控制；HETS 是归因所必需的消融                 |
| 5    | TA-OPD                                      | 检验只匹配头部与尾部总质量是否已经足够；它是粗粒度目标，不是完整 rKL 的无偏梯度 |
| 6    | SC-TA                                       | 与 ResOPD 具有相同完整 rKL 期望，但随机残差不同；补充这一边界对照               |
| 7    | Normalized / truncated top-k 与更多支持集   | 机制对照和论文复现，放在主线明确之后；不同 top-k loss 不能混名                  |

预算只够四种方法时，保留 **sampled、full-vocab、sampled+baseline、ResOPD**。如果 full-vocab 暂时只能做固定前缀审计，端到端先用 HETS 占第四组，与 sampled、baseline、Res 配对。要正式解释 Res 的控制项收益，必须增加 HETS；完整第一轮建议六组为 sampled、full-vocab、baseline、HETS、Res、TA，再补 SC-TA。

这里把 full-vocab 提到前面，是因为它既是被比较的方法，也是验证其他实现的工具；并不要求第一天就跑完整 200-step 大预算。

### 1.1 第一轮固定什么

2026-10-21 的方法筛选方案固定 JustRL→R1-Distill-1.5B、DAPO、batch=mini-batch=64、n=1、LR=$10^{-6}$、采样 temperature=1/top-p=1/top-k=-1。使用一次更新，先排除 clipping、优势冻结、重复复用和额外任务奖励的混杂。

先固定 teacher top-16 支持集比较 baseline/HETS/Res/TA/SC-TA，最大化数据通路复用；此处 baseline 是 **teacher-top-k 变体**。原版 vOPD 的 student-top-k baseline 随后单列，不能换了支持集却沿用“原版复现”的名称。[^vopd-code]

先做固定输入的梯度与数值验收，再做短程验证和多种子比较；执行步数、验证频率和 seed 在具体实验计划中确定。Full-vocab 组不额外套 sampled PPO ratio；它已对当前 next-token 分布积分。随机估计器的 ratio 数值保护触发和 optimizer 跳步应为零。算法选定后再研究四轮复用；不把整个方法矩阵与 E1–E6、所有支持集立即做笛卡尔积。

## 3. Top-k 支持集不是只有两种

为避免占用核心稿的冻结中心 $K_T,K_S$，统一用 $K_{\rm head}$ 表示候选数。定义

$$
S_T=\operatorname{TopK}(T,K_{\rm head}),\quad
S_q=\operatorname{TopK}(q_k,K_{\rm head}),\quad
S_p=\operatorname{TopK}(p_{k,n},K_{\rm head}).
$$

| 支持集                      | 定义                                                                        | 倾向与遗漏                                                                                  |
| --------------------------- | --------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| Teacher top-k               | $S_T$                                                                       | 包含 teacher 偏好的候选；可能遗漏 student 过度集中的错误候选                                |
| Student anchor top-k        | $S_q$                                                                       | 关注 rollout/update 起点的高概率候选，便于缓存；多次更新后可能过时                          |
| Student current top-k       | $S_p$                                                                       | 随当前学生刷新；新候选需要可取得 teacher 分数                                               |
| 交集                        | $S_q\cap S_T$                                                               | 聚焦共同高概率区域；会丢弃双方不一致处，可能为空或仅一项                                    |
| 并集                        | $S_q\cup S_T$                                                               | 同时覆盖 student 和 teacher 的主要模式；去重后大小在 $K_{\rm head}$ 到 $2K_{\rm head}$ 之间 |
| 对称差                      | $(S_q\cup S_T)\setminus(S_q\cap S_T)$                                       | 只研究不一致区域，适合作机制消融，不是默认主方案                                            |
| Student-only / teacher-only | $S_q\setminus S_T$ / $S_T\setminus S_q$                                     | 能进一步区分“压低 student 多余质量”和“补 teacher 偏好”                                      |
| 固定预算混合集              | $\operatorname{TopK}((q_k+T)/2,K_{\rm budget})$，或去重后按固定规则补齐配额 | 保持统一查询预算；混合分布仅用于选集，不自动改变 teacher 目标                               |
| Correction-aware            | 根据 $\lvert\delta_a\rvert$ 或其正负部分选择                                | 更直接面向 logit 梯度；需要额外 teacher 信息，不能当作廉价 top-k 查询                       |

上表的集合运算也可使用 $S_p$，但必须在实验名中注明 current / anchor，以及选集刷新频率。Student vLLM 的 top-k 是 $\beta$ 的支持，又是一个需要辨认的版本，不应笼统标为“student top-k”。

**支持集与 loss 是独立轴。** 同一个 $S$ 可以用于 normalized subset KL、TA 粗粒度 KL、KL baseline、HETS/Res/SC 的精确头部。使用 $T$ 选集仍可以优化 reverse KL；使用 student 权重也不自动保证实现是完整 reverse-KL 梯度。

### 3.1 我们应先比较哪些支持集

针对当前 JustRL→R1-Distill-1.5B，第一优先级为 **student、teacher、intersection、union** 四种；对称差放在解释机制的后一轮。Part I 的实验恰好使用相同模型家族配对，报告交集在其 top-16 配方下接近 student-top-k、对称差更弱；因此 intersection 值得优先，但不能将这个结果推广为所有 teacher/student 的定律。[^rethinking]

分两种预算报告：

- **机制对照：** 两边各取 top-16，真实计算交集、并集和对称差，并报告每位置实际集合大小和 padding 后的实际传输宽度。并集最多 32 项，不能把收益全部归因于更好的选集。
- **成本对照：** student-16、teacher-16、固定总预算 16 的混合集。Union 8+8 去重后可能不足 16，要预先规定补齐规则并记录实际数量；补齐后的集合不能再标成原始 intersection。

固定同一个 loss、一次更新和全局 token-mean。记录实际 $\lvert S\rvert$、student/teacher mass、交集为空和 singleton 的比例，以及相对 full-KL 的梯度方向与误差。交集为空时保持原有全局 response-token 分母，将该位置贡献记为零；若改成只对非空支持归一化，就额外改变了位置权重。只记录 overlap ratio 不够；同样的 overlap 可以对应不同概率质量和不同梯度。

### 3.2 Normalized top-1 是特殊退化情况

若 $S$ 只有一个元素，$p(\cdot\mid S)=T(\cdot\mid S)=(1)$，因此 normalized subset KL 的值和梯度都恒为零。Empty support 需要明确返回零及 coverage 指标，不能依靠 `NaN→0` 隐藏未覆盖位置。

某些名为 top-1 的加权 PG 实现仍能更新，因为它并非这个 normalized-KL loss。Part I 报告的 top-1 不稳定现象不能被用来证明纯 singleton normalized KL 会产生非零梯度；公式定义、具体代码和实验标签必须分别核对。

### 3.3 两种 union 的工程含义完全不同

逐位置的 $S_q(s)\cup S_T(s)$ 最多有 $2K_{\rm head}$ 项。另一些推理 API 只接受一份共享 token-ID 列表，于是先构造整条回答的

$$
\mathcal U=\bigcup_{t=1}^{H}S_q(s_t),
$$

再在每个位置查询 $\mathcal U$。查询量从 $HK_{\rm head}$ 变为 $H\lvert\mathcal U\rvert$，最坏 $\lvert\mathcal U\rvert=\min(V,HK_{\rm head})$。**跨位置 union 不是 teacher/student union**。Many Faces 论文详细记录了这一 SGLang 接口问题；已有上游 fixed-token prefill 工作也要按实际 API 返回形状核对。[^manyfaces]

## 4. One-Shot-OPD 的实际实现

### 4.1 版本与入口

本机 checkout 干净，HEAD 为 `6b6b1782cc3f04749dd077de19353f839f6bc048`，remote 为作者 `Thinking-Space/One-Shot-OPD`。已通过 GitHub 固定 commit 文件读取交叉核对核心 helper。该版本的配方使用 `trainer.use_v1=False`，不能直接等同于本项目 V1 路径。

`recipe/_common.sh` 默认：`LOG_PROB_TOP_K=0`、`METRIC_TOP_K=0`、`TOP_K_STRATEGY=only_stu`、`REWARD_WEIGHT_MODE=student_p`。前者开启 top-k 训练 reward，后者只开启统计；默认二者为零，普通 sampled-token 路径不受下面 top-k 分支发现的影响。[^oneshot-recipe]

### 4.2 从字段生产到 loss

| 位置                                             | 审阅到的行为                                                                                        |
| ------------------------------------------------ | --------------------------------------------------------------------------------------------------- |
| `FSDPEngineWithLMHead._emit_opd_student_top_k()` | 从该次 student logits 重新取 top-k，并保存完整词表归一化的 log-prob                                 |
| `ray_trainer.py` 的 old-log-prob 处理            | 将这些 IDs 与 log-probs 作为冻结候选字段传给 teacher                                                |
| `compute_top_k_teacher_log_probs()`              | 计算 teacher 自己的 top-k；通常 gather student IDs 上的 teacher 分数，intersection 分支屏蔽不重合项 |
| `TeacherWorker._forward_micro_batch()`           | 曾返回 teacher 自己的 IDs/分数，但与 student 候选上的 teacher 分数是不同字段                        |
| `TeacherWorker.compute_rm_score()`               | 汇总并传给 reward helper 的仍是 student 候选上的 K 列分数；未完整接入 teacher-only 候选列           |
| `compute_top_k_reward()`                         | 构造权重和 mask；union/symmetric-difference 构造 2K mask 后，按输入分数宽度切片                     |
| `token_reward_direct` / `vanilla_topk`           | 固定 reward 为 advantage，对候选轴逐项 PPO ratio/clip，再沿候选轴求和                               |

关键文件分别见固定版本引用。[^oneshot-worker][^oneshot-engine][^oneshot-loss]

### 4.3 五个开关不等于五种支持集都已端到端接通

在该调用链的 K 列输入下：

| 开关                 | 意图         | 当前源码路径的行为                                                                               |
| -------------------- | ------------ | ------------------------------------------------------------------------------------------------ |
| `only_stu`           | Student 支持 | 与实际 student 候选输入一致                                                                      |
| `only_tch`           | Teacher 支持 | helper 与 `only_stu` 使用相同全真 mask，调用方又未切换到 teacher 候选，因此仍作用于 student 候选 |
| `intersection`       | 交集         | 在 student 列上应用交集 mask；这一集合选择可以实现                                               |
| `union`              | 去重并集     | 2K mask 被截到 K 列，teacher-only 列缺失；在审计例子中与 `only_stu` 输出相同                     |
| `union-intersection` | 对称差       | teacher-only 半边缺失，剩余 student-only 候选；还关闭了概率权重归一化                            |

这是针对**当前固定版本的字段链路和 helper 输入形状**的结论，不是对论文全部实验或普通 sampled-token 实现的评价。Helper 若收到正确拼接的 2K 列分数，可以表达去重 union；缺的是当前路径完整提供并消费这些列，不能只改一个策略字符串。

CPU 例子取

$$
q=(0.4,0.3,0.2,0.1),\qquad T=(0.1,0.25,0.45,0.2),\qquad K_{\rm head}=2.
$$

此时 $S_q=\{0,1\}$、$S_T=\{2,1\}$，正确 union 为 $\{0,1,2\}$、对称差为 $\{0,2\}$。沿当前 K 列调用形状，`only_stu`、`only_tch`、`union` 的 reward 完全相同；补入完整两侧分数后，teacher-only 的 token 2 才获得相应列。结果保存在 [audit.json](data/support-selection-20261021/audit.json)。

### 4.4 权重归一化不等于 normalized subset KL

默认 `student_p` 在支持内归一化冻结学生概率，再乘**完整分布**的 log-prob gap，生成固定 reward。没有 clip、且第一次更新 $p=q$ 时，其成本梯度为

$$
g_{\rm repo}=\sum_{a\in S}\frac{q_a}{q(S)}\log\frac{q_a}{T_a}\,\psi_a.
$$

而直接 normalized subset KL 在同一点的梯度为

$$
g_{\rm norm}=\sum_{a\in S}\frac{q_a}{q(S)}
\left(\ell_a-\mathbb E_{q(\cdot\mid S)}\ell\right)\psi_a.
$$

二者一般不同。独立 CPU 例子中最大 logit 梯度差约 0.1741。`teacher_p` 或 `none` 又改变候选系数；不能只改权重后就称改成 forward KL。`union-intersection` 同时改变选集和归一化，直接拿它与 normalized union 比，还混入了 loss scale 因素。

### 4.5 多次更新的候选 ID 对齐

该仓库训练 forward 会重新取当前 student top-k，旧 advantage 则来自旧 top-k；第二次更新可能按 rank 位置配上不同 token IDs。作者的 `validate_opd_top_k_config()` 因而明确要求 `ppo_epochs=1`、mini-batch 等于整个 train batch，拒绝多次更新。这是已存在的保护，不能说该版本无条件允许错误复用。[^oneshot-guard]

移植时建议将固定 `support_ids` 传入 actor，每次 gather 当前概率，而不是按新 top-k 排名位置与旧数组相减。即使只做一次更新，也应检验输入/输出 ID 对应与 causal shift；后续若选用 current support，必须同步重新取得新 ID 的 teacher 分数。

## 5. 相关论文与实现证据

| 来源                              | 对支持集问题的贡献                                                                    | 本轮边界                                                                                               |
| --------------------------------- | ------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| Rethinking OPD Part I，2604.13016 | 同一 JustRL/R1-Distill 配对上比较 student、交集、对称差；交集接近 student，对称差较弱 | 经验结论限于该配方；论文定义与现有代码的 weighted-PG 需要分别核对                                      |
| One-Shot OPD Part II 仓库         | 提供五种策略名、概率权重、reward/metric 两种开关                                      | 本轮核对当前公开 checkout，不以它自动证明所有历史论文实验的实现                                        |
| vOPD，2605.07865                  | 原版 baseline 在 current student top-k 上将双方归一化                                 | Baseline 与直接 subset KL 目标不同；teacher-top-k 变体另列                                             |
| TA-OPD / ResOPD                   | 原版使用 teacher top-k，分别补粗尾部和随机尾部残差                                    | 正确补偿可推广到其他固定集合，但效果和 API 不会自动相同                                                |
| Many Faces，2605.11182v2          | 区分普通截断、stop-gradient、归一化 top-k；讨论跨位置查询 union 膨胀和交集方案        | 本轮读取论文；未核验其完整作者训练实现                                                                 |
| SparseOPD，2609.34386v1           | 先形成完整 $\delta_a=p_a(\ell_a-K)$，再按修正幅度/正负质量选取并补偿 sparse logits    | 需要完整 teacher correction 信息；保留总正负质量不等于恢复每个坐标的精确梯度；本轮未核验到作者发布代码 |

SparseOPD 提醒我们：大概率并不一定意味着大梯度，teacher-only 的低 student 概率候选也不能一律忽略。但同一个 reverse-KL 目标仍有 $p_a$ 权重，使用 teacher 集合不会自动变成 mode-covering 的 forward KL。评价支持集可额外记录

$$
\mathrm{energy}(S)=\frac{\sum_{a\in S}\delta_a^2}{\sum_a\delta_a^2},
$$

在分母非零时比较它与 head mass；这只是 logit 空间诊断，不替代全参数梯度比较。[^sparse]

## 6. 建议的最小实验矩阵与接线

先固定 teacher-top-16，对比 baseline/HETS/Res/TA/SC；sampled 与 full-vocab 不依赖这套支持集。再固定一个直接 loss 和一个无偏估计器（例如 TA 与 Res），分别比较 student/teacher/intersection/union。这样能区分“选了哪些候选”与“如何处理剩余尾部”，而不是只跑五种开关后把差异都解释为支持集。

统一 sparse 数据合同可以是：`support_ids`、`teacher_support_log_probs`、`support_mask`、`teacher_sampled_logprob`，另保留真实 `rollout_log_probs`。所有模型必须在同一套去重后的 IDs 上 gather；padding 在求差、指数和归一化前屏蔽。Sampled action 另存，不直接并入 baseline 或头部支持集：让支持集依赖本次抽样动作会破坏原来按固定集合推导的无偏性，必须另行推导。

- Teacher support：teacher 直接产出支持和分数，actor gather 当前 $p$，最适合先接当前共享池。
- Frozen student / union support：先复用 old-log-prob forward 产生 $S_q$，teacher 返回 union 上的分数；actor 在固定 IDs 上算当前概率。对于 HETS/Res 的精确头部，不需要给每个候选再套 PPO ratio；行为 IS 只用于随机尾项。
- Current student support：teacher 返回 hidden states 和所需归一化信息，actor 可查询后来新增的 IDs；或付出重新评分成本。只缓存旧 sparse 分数，不能支持任意新的 current top-k。

任务评测采用所选实验配方的完整数据集、回答数与长度预算，并保持生成 token 预算与更新数可对照；具体字段以[当前指标字典](../../settings/指标口径.md)为准。多次复用、top-k 大小扫参、SparseOPD 风格梯度压缩放在核心方法通过后再做。

## 7. 本轮验证与复现

[脚本](../../../research/objectives/verify_fullkl_and_support.py)包含 20 个 FP64 分布案例，检查 dense KL、stopped-gap、score surrogate 和 analytic-logit VJP 的一阶梯度相等，验证两种 IS 数值估计的期望梯度，以及错误 detach 和 singleton normalized KL 的零梯度。最大正确梯度误差约 $2.22\times10^{-16}$。

可选从已审阅 One-Shot checkout 抽取三个纯 tensor helper，按生产调用链的 K 列形状构造上述反例，并验证正确 2K 输入能恢复 teacher-only 列。没有运行作者的 Ray/FSDP GPU 栈，未将这个测试包装成端到端性能结论。

```bash
CUDA_VISIBLE_DEVICES='' python -m examples.mirroropd.research.objectives.verify_fullkl_and_support \
  --oneshot-root ../One-Shot-OPD --output /tmp/fullkl-support-audit.json
```

[完整检索记录](data/support-selection-20261021/paper-search.md)包含 24 条去重结果，保留全部原始排序和总结；两个检索来源零诊断，两个明确 arXiv ID 均解析成功。检索列表包含噪声，不代表全部条目经过正文审阅。

## 旧理论章节索引

原理论章节的锚点集中保留在下表；正文只维护实验与源码证据。

| 原章节                                                                                                       | 现维护位置                                                                                                                          |
| ------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------- |
| <a id="2-full-vocab-kl一个真值不同计算图"></a>2. Full-vocab KL：一个真值，不同计算图                         | [Full-vocab 计算图与四种更新](../../theory/2026-10-22-Full-vocab计算图与四种更新.md#1-full-vocab-kl-的计算图)                       |
| <a id="21-只计算数值作为诊断或-baseline"></a>2.1 只计算数值，作为诊断或 baseline                             | [Full-vocab 计算图与四种更新](../../theory/2026-10-22-Full-vocab计算图与四种更新.md#11-只计算数值作为诊断或-baseline)               |
| <a id="22-作为训练-loss直接自动微分"></a>2.2 作为训练 loss，直接自动微分                                     | [Full-vocab 计算图与四种更新](../../theory/2026-10-22-Full-vocab计算图与四种更新.md#12-作为训练-loss直接自动微分)                   |
| <a id="23-只传递精确梯度可以使用代理或自定义-backward"></a>2.3 只传递精确梯度，可以使用代理或自定义 backward | [Full-vocab 计算图与四种更新](../../theory/2026-10-22-Full-vocab计算图与四种更新.md#13-只传递精确梯度可以使用代理或自定义-backward) |
| <a id="24-两个容易写错的-detach"></a>2.4 两个容易写错的 detach                                               | [Full-vocab 计算图与四种更新](../../theory/2026-10-22-Full-vocab计算图与四种更新.md#14-两个容易写错的-detach)                       |

### 参考文献

[^vopd-code]: holi-lab. vOPD，`d4146d5015216c6f8e89a7243c24912b634f531a`。[current student top-k baseline](https://github.com/holi-lab/vOPD/blob/d4146d5015216c6f8e89a7243c24912b634f531a/src/opd_trainer.py#L147)。

[^rethinking]: Li et al. Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe. arXiv:2604.13016, 2026，overlap/non-overlap 与 support-size 实验。[论文](https://arxiv.org/html/2604.13016)。

[^manyfaces]: Zhu et al. The Many Faces of On-Policy Distillation: Pitfalls, Mechanisms, and Fixes. arXiv:2605.11182v2, 2026，第 6.1 节与附录 A.6。[论文](https://arxiv.org/html/2605.11182v2)。

[^oneshot-recipe]: Thinking-Space. One-Shot-OPD，`6b6b1782cc3f04749dd077de19353f839f6bc048`。[配方](https://github.com/Thinking-Space/One-Shot-OPD/blob/6b6b1782cc3f04749dd077de19353f839f6bc048/recipe/_common.sh)；[README](https://github.com/Thinking-Space/One-Shot-OPD/blob/6b6b1782cc3f04749dd077de19353f839f6bc048/README.md)。

[^oneshot-worker]: 同一版本。[teacher top-k helper 与实际字段汇总](https://github.com/Thinking-Space/One-Shot-OPD/blob/6b6b1782cc3f04749dd077de19353f839f6bc048/verl/verl/workers/fsdp_teacher_worker.py)。

[^oneshot-engine]: 同一版本。[student top-k 生产](https://github.com/Thinking-Space/One-Shot-OPD/blob/6b6b1782cc3f04749dd077de19353f839f6bc048/verl/verl/workers/engine/fsdp/transformer_impl.py#L1303)。

[^oneshot-loss]: 同一版本。[advantage 与候选轴 PPO](https://github.com/Thinking-Space/One-Shot-OPD/blob/6b6b1782cc3f04749dd077de19353f839f6bc048/verl/verl/trainer/ppo/opd_core_algos.py)；[loss 字段选择](https://github.com/Thinking-Space/One-Shot-OPD/blob/6b6b1782cc3f04749dd077de19353f839f6bc048/verl/verl/workers/utils/losses.py)。

[^oneshot-guard]: 同一版本。[候选漂移校验](https://github.com/Thinking-Space/One-Shot-OPD/blob/6b6b1782cc3f04749dd077de19353f839f6bc048/verl/verl/trainer/ppo/opd_trainer_mixin.py#L145)。

[^sparse]: Miao et al. Look Before You Select: Rethinking Vocabulary Sparsification in On-Policy Distillation. arXiv:2609.34386v1, 2026，第 2–3 节。[论文](https://arxiv.org/html/2609.34386v1)。
