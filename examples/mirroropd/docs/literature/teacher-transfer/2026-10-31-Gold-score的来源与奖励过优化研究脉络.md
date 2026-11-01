# Gold score 的来源与奖励过优化研究脉络

本文解释 gold reward／gold score 在 RLHF、直接偏好优化和 OPD 中分别指什么，以及为什么研究者常把它画在 KL 或平方根 KL 坐标上。它不是一个跨论文固定定义的指标，也不是一个可以调用同一实现就复现的 benchmark。

## 1. 先把符号读懂

用统一记号写：

$$
G_{\mathrm{gold}}(\pi)
=\mathbb E_{x\sim D,\ y\sim\pi(\cdot\mid x)}
[r_{\mathrm{gold}}(x,y)].
$$

$r_{\mathrm{gold}}$ 是研究者选择的较独立的评价信号，$G_{\mathrm{gold}}$ 是策略输出在该信号下的平均表现。本文用上述符号统一讨论；原论文的记号分别为：Gao 等使用 gold RM score／$R$，2609.32722 使用 gold score／$G$。[^gao][^bao]

| 名称          | 在论文中承担的角色               | 是否必然是真实正确率                   |
| ------------- | -------------------------------- | -------------------------------------- |
| Proxy reward  | 实际用于优化的代理信号           | 否                                     |
| Gold reward   | 用来检查优化效果的参考信号       | 否，可能仍是一个模型                   |
| Gold score    | 策略在 gold 信号下的平均分       | 取决于 gold 的定义                     |
| Accuracy      | 对／错判定下的成功概率或经验比例 | 是一种特定的 gold score                |
| Teacher score | teacher 自己生成时的任务成绩     | 不是 student 收到的所有 token 监督质量 |

如果 $r_{\mathrm{gold}}\in\{0,1\}$，则 $G_{\mathrm{gold}}$ 就是期望采样正确率。如果 $r_{\mathrm{gold}}$ 是 reward model 的实值输出，量纲、中心化和缩放都依赖该模型。它们不能直接作数值比较。

gold 表示“本研究使用的参考标准”，不意味着这个标准无偏、无噪声或已经等于真实人类意图。研究者有时故意指定一个大模型为合成实验的 oracle，目的正是让代理误差可测量。

## 2. 早期背景：训练代理与真实评估分开

### 2.1 2017：从人类偏好学习奖励

Christiano 等用人类对轨迹片段的偏好训练奖励预测器，再用该预测器驱动 RL。它建立了一个重要区分：优化过程中使用的是学习到的信号，任务是否实现人类目标仍需要另行判断。本文把它作为概念背景，不声称该文首次创造了 gold reward 术语。[^christiano]

### 2.2 2020：语言模型的学习奖励与独立质量评价

Stiennon 等将人类摘要偏好用于训练奖励模型，再优化摘要策略，并以独立评价检查质量。这个阶段已经能看到“训练 reward 更高”与“人类更喜欢”的区别，也为后来的过优化研究提供了语言模型场景。[^stiennon]

这一阶段的重点是评价角色分离，还不是“正确率必然随平方根 KL 线性”的定律。

## 3. 2022：平方根 KL 与 gold/proxy 两条曲线成为直接研究对象

### 3.1 Bai 等：奖励与平方根 KL 的经验关系

Bai 等的 §4.3 专门研究奖励与相对初始策略的平方根 KL 之间的近似线性关系。KL 在参数空间局部是二次量，取平方根后更像局部距离，这为选择横轴提供直觉。[^bai]

这里的经验关系依赖奖励定义、训练路径和观测区间；仅凭选择平方根坐标，不会强制所有任务都呈正线性。它也不等同于 2609.32722 后来使用的数学正确率。

### 3.2 Gao、Schulman、Hilton：系统测量 reward overoptimization

为了降低大量人工复评的成本，Gao 等指定一个较大的 gold RM，利用它产生标签训练较小 proxy RM，再让策略优化 proxy，同时用 gold RM 评分。其设计把“优化了多少”与“是否还改善参考目标”分开。[^gao]

他们比较 RL 与 best-of-n，发现 gold 分数的经验形状随优化方式不同，而 proxy 分数继续改善时 gold 可能回落。它们用相对初始化的 KL 表示优化强度，并对奖励作中心化／尺度处理；不能把该分数直接读成百分比正确率。[^gao]

这项研究的关键贡献是把 Goodhart 式分离变成可测量曲线。它并未规定 gold 必须来自某种特定模型，也没有保证 gold RM 就是人类真值。其“优化方式影响曲线形状”的结果，本身就提示不能把一条全局直线当作普遍规律。

## 4. 2024：独立裁判、信息论上界与最优策略

### 4.1 直接偏好优化也可能发生过优化

Rafailov 等研究 DPO、IPO、SLiC 等直接对齐算法。虽然这些方法不一定显式训练独立的 proxy RM，仍可能在更大 KL 预算下出现质量下降。他们的部分实验用 GPT-4 判断的胜率替代早期工作中的 gold RM score。[^daa]

因此，gold 的“实现”已经从一个标量 reward model 扩展到独立裁判的比较结果。评估时还需固定比较对象、裁判版本、提示和顺序处理；胜率不是与模型本身无关的绝对质量刻度。

### 4.2 Mroueh：把平方根关系分成上界和可达性

Mroueh 在适当奖励尾部条件下证明

$$
G(P)-G(Q)\leq \sqrt{2\sigma^2D_{KL}(P\Vert Q)}.
$$

论文进一步讨论 best-of-n、不同 divergence，以及 proxy 误差如何影响 gold 改善。[^mroueh]

这给出了“为什么会出现平方根 KL 尺度”的严格依据，但上界不保证实际算法达到它。对正确率，我们可以进一步证明：固定 prompt 分布、直接优化真实正确性且允许充分丰富的回答分布时，小预算的最优前沿确实为平方根形式；沿任意平滑参数路径，则只得到可能为正、负或零的一阶展开。完整证明见[正确率与序列／Token KL](../../theory/accuracy-and-kl.md)。

## 5. 2025：蒸馏中的 gold 可以是 oracle 分布距离

Tiapkin 等的 Teacher Hacking 构造 oracle→teacher→student，用 teacher 距离作为代理诊断、oracle 距离作为较真实的目标诊断。于是，即使 teacher 拟合继续改善，也能检测 student 是否反而远离 oracle。[^hacking]

这仍沿用 gold/proxy 的角色划分，但“gold 指标”此时可以是越小越好的距离，而非越大越好的 reward。若统一写成 $G_{\mathrm{gold}}$，必须先说明是否取负距离；否则画图方向都可能被混淆。

对我们的影响是：若研究目标明确是复制 teacher，teacher 距离就是直接的 fidelity 指标；若研究目标是数学正确性，就需要额外的答案验证。teacher 在第一个问题中是目标，在第二个问题中只是代理，两种角色不能混用。

## 6. 2026：OPD 把 gold 具体化为任务正确率

### 6.1 2609.32722：gold score 是 held-out accuracy

Bao 等明确把任务正确率作为 gold score，把 teacher 诱导的 token 信号作为 proxy，并以 student 相对自身 SFT 初始化的 token-mean KL 平方根索引训练过程。[^bao]

这里沿用了 Gao 等的研究框架，但换了 gold 的具体实现和 KL 的聚合方式。因此：

- “gold”不是 teacher 的 token 概率，也不是 teacher 自己的正确率。
- 图上的 $G$ 是正在训练的 student 的任务成绩。
- 横轴参照是 student 初始化；它衡量改变幅度，不是 teacher 拟合残差。
- 早期直线描述收益与策略位移的关系，不是训练步数与收益的关系。

### 6.2 同期研究：数据量与 KL 预算也要一起考虑

Aouad 等的 2609.38526 将偏好比较数据量与 divergence 预算共同纳入信息论模型，研究可达奖励改善的上界与构造，并用 gold/proxy 模型做经验验证。其理论对象与一般 OPD 训练轨迹不同。[^revisit]

该文发表于 2026-09-29，晚于 Bao 等的 2026-09-26，因此这里把它列为同期发展，不声称它是 Bao 论文的历史来源。

更一般地，“允许使用更大的 KL 预算时，最优可达成绩”不会因为预算放宽而下降，因为可行策略集合只会扩大。实际沿 proxy 训练得到的成绩却可以下降。最优前沿与某个训练过程的轨迹，必须分别讨论。

## 7. 阅读 gold 曲线时的检查顺序

| 要核对的项  | 需要写清楚的内容                                            |
| ----------- | ----------------------------------------------------------- |
| Gold 是什么 | 真值判定、独立人类评价、LLM 裁判、gold RM，还是 oracle 距离 |
| 分数的单位  | 概率、百分比、胜率、中心化 reward 或负距离                  |
| 评价谁      | 当前 student、teacher 自身，还是 best-of-n 选择后的输出分布 |
| 评价分布    | 哪批 prompt、何种采样参数、完整输出还是截断输出             |
| KL 参照     | 初始化、teacher、其他 checkpoint 或解码策略                 |
| KL 聚合     | 序列累计、固定 horizon 平均、逐响应平均或有效 token 平均    |
| 结论形式    | 无条件上界、有条件最优前沿、局部展开，还是经验拟合          |
| 可重复性    | 多训练 seed、采样误差、峰值选择与独立确认                   |

对于 2609.32722，可以简写成：当前 student 的数学期望正确率／其经验估计，对相对初始化的 token-mean 位移作图。对于我们，“固定前缀下 teacher-RKL 的下降”应保留自己的名称，不必为了沿用 gold 术语而把分布拟合硬改成一个任务成绩。

## 8. 证据与检索记录

本页是一条与当前讨论有关的研究脉络，不宣称考证了 gold 一词在统计学或机器学习中的首次出现。概念背景、直接引用链、同期相关工作已经分开标记。

[完整原始检索](sources/gold-history-20261031/完整原始检索.md)按精确 ID 保存；[来源清单](sources/gold-history-20261031/sources.json)记录核对版本与章节。历史各文献的分数不能直接汇总成一条数值曲线。

### 参考文献

[^christiano]: Christiano et al. Deep Reinforcement Learning from Human Preferences. 2017；核对 arXiv 版本入口。[原文](https://arxiv.org/abs/1706.03741)。概念背景，不作术语首创归属。

[^stiennon]: Stiennon et al. Learning to Summarize from Human Feedback. 2020；arXiv:2009.01325. [原文](https://arxiv.org/abs/2009.01325)。学习奖励与独立质量评价。

[^bai]: Bai et al. Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback. arXiv:2204.05862, 2022. [§4.3](https://arxiv.org/html/2204.05862)。奖励与平方根 KL 的近似关系。

[^gao]: Gao, Schulman, Hilton. Scaling Laws for Reward Model Overoptimization. arXiv:2210.10760v1, 2022；后发表于 ICML 2023. [原文](https://arxiv.org/html/2210.10760v1)。§1–3、gold/proxy 构造与分数校准。

[^daa]: Rafailov et al. Scaling Laws for Reward Model Overoptimization in Direct Alignment Algorithms. arXiv:2406.02900v2, 2024. [§3、图 1](https://arxiv.org/html/2406.02900v2)。独立裁判胜率替代 gold RM 分数。

[^mroueh]: Mroueh. Information Theoretic Guarantees For Policy Alignment In Large Language Models. arXiv:2406.05883v1, 2024. [§3–4](https://arxiv.org/html/2406.05883v1)。奖励改善上界与 proxy/gold 转移。

[^hacking]: Tiapkin et al. On Teacher Hacking in Language Model Distillation. ICML 2025；核对正文 arXiv:2502.02671v1. [原文](https://arxiv.org/html/2502.02671v1)。oracle 与 teacher 的双距离诊断。

[^bao]: Bao et al. Scaling Properties of Same-Family On-Policy Distillation. arXiv:2609.32722v1, 2026-09-26. [§1、§3、附录 D](https://arxiv.org/html/2609.32722v1)。当前单篇笔记研究对象。

[^revisit]: Aouad, El Gadarri, Farias. Revisiting Scaling Laws for Reward Optimization. arXiv:2609.38526v1, 2026-09-29. [原文](https://arxiv.org/html/2609.38526v1)。同期数据量与 divergence 预算研究。
