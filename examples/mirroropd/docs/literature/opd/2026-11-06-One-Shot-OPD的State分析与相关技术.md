# One-Shot OPD 的 State 分析与相关技术

本文解读 arXiv:2609.04172《Rethinking On-Policy Distillation of Large Language Models II: One Training Example》，聚焦 state 分析，并比较相邻的状态跟踪与内部表征分析方法。该文于 2026-09-03 提交 arXiv；以下结论按论文报告的实验理解，不视作独立复现。

## 论文如何定义和分析 state

论文中的 state 不是模型内部的循环状态，也不是单独的题目，而是自回归上下文：

$$s_i=(x,y_{<i})$$

其中 $x$ 是输入，$y_{<i}$ 是截至当前位置之前的生成前缀。学生从自身策略采样轨迹，teacher 在轨迹的每个 state 上给出下一 token 分布。因此，query 的作用之一是诱导学生进入不同前缀；训练信号则是每个前缀上的 teacher-student 分布差异。论文把 OPD 的数据效率问题重述为：有限 query 的 rollout 能覆盖多少 full-data OPD 会访问的 state，以及模型吸收这些 state 上监督的速度如何。

论文的核心状态覆盖率按以下步骤计算：

1. 在数学任务上收集 full-data OPD 的 rollout，并从每条轨迹均匀抽取 8 个相对位置的 state。
2. 用 teacher 在 state 最后 token 处的最终层 hidden vector $h_T(s)$ 作为 state 表示。该实验所用表示维度为 1536。
3. 以 full-data rollout 中 3/5 的样本构建参考集；另 2/5 留作 held-out full-data 对照。
4. 对表示做 PCA 降到 50 维（保留 71.9% 方差），再以 $K=200$ 做 K-means。所有待比较 state 都映射到最近质心。
5. 覆盖率是被访问 cluster 的比例：

$$\mathrm{Cov}(S)=\frac{1}{K}\left|\{c(s):s\in S\}\right|.$$

这是“访问过多少类状态”的广度指标，不计 cluster 被访问频次，也不按 teacher-student gap 或学习信号加权。full-data held-out 在相同预算下达到 100%，所以它是经验参照水平；这个构造并不保证任何设置都能达到 100%。

在 300 步实验中，单 query 达到 71.5% 覆盖，前 100 步已达到 65.9%；之后 200 步只增加 5.6 个百分点。16 个语义不同的 query 达到 98.9%，验证准确率也接近 full-data OPD。固定 query 数为 16 时，来自 16 个语义簇的 query 比来自同一簇的 query 覆盖更广、准确率更高；同簇版本约在 76.8% 覆盖处较早趋平。另一项 off-policy 控制固定 query 和优化预算，只改变保留的初始 rollout 条数，更多不同轨迹也带来更高准确率。这些控制支持“数据多样性通过带来更多 state 起作用”的解释，但覆盖和性能同时变化本身仍不等于逐 state 的因果归因。

作者检查了覆盖率对构造选择的敏感性：$K$ 取 50、100、200、500 时，绝对值会变，但 1/4/16-shot 的排序稳定；把参考集改成各设置 rollout 的合并集，排序也保持；改用最终 token 或绝对位置 2,000/4,000，趋势相同。相对位置采样被选为主分析，因为最终 token 更容易受长度和终止位置影响。另以原始 1536 维空间中 10-NN 平均 cosine distance 检查“最近质心强行分配”掩盖远离参考分布的问题：单 query 的远端尾部更重，但区分其 state 与 held-out full-data state 的 AUROC 为 0.552，接近随机水平。

## State 上的 teacher-student 对齐

论文还分析训练过程中每步访问的 state 上，学生和 teacher 还差多少。距离定义为采样 token 的绝对 log-prob 差的 token 平均：

$$d_t=\frac{1}{|\mathcal T_t|}\sum_{i\in\mathcal T_t}\left|\log\pi_T(y_i|s_i)-\log\pi_{\theta_t}(y_i|s_i)\right|.$$

绝对值避免正负差相消；它不是完整词表 KL，而是访问 token 上的 log-prob gap。吸收率定义为一步消除的相对剩余距离：

$$v_t=\frac{d_t-d_{t+1}}{d_t}.$$

每个 run 都在自己当步访问的 state 上估计 $d_t$。由于不同 query 设置在 step 30 的距离水平不同，作者画 $d_t/d_{30}$ 比较曲线形状；从 step 30 开始是因为此前 gradient clipping 仍生效。相邻步差噪声较大，$v_t$ 用 $[t/1.4,1.4t]$ 的几何窗口估计。1、4、16、full-data query 的 $d_{300}/d_{30}$ 分别为 0.16、0.17、0.18、0.22；从 step 50 到 200，吸收率下降约 4.2 到 6.4 倍。论文据此认为对齐持续但变慢，query 数量没有明显改变减速形状。

辅助对齐指标包括 student 与 teacher 的 top-16 token 集合重叠率、共享 top-k token 上按 student 概率加权的 log-prob 差，以及绝对 entropy gap。它们描述同一 state 上策略分布的不同侧面，不能与 state coverage 混为一谈。把初始 student 的 64 条 rollout 固定复用数百步，仍能持续学习约 200 步；这说明持续出现新 state 不是长训练过程的必要解释。

## 这套分析能说明什么

这篇论文的 state 是“训练目标被评估的上下文”，而非声称找到了模型唯一、真实的内部思维状态。其表示选 teacher hidden state，聚类后衡量相对参考集的语义覆盖；另以分布差异度量 state 上的监督缺口。因此它把两个问题拆开：rollout 是否触及足够多的监督区域，以及优化器每步能吸收多少已有缺口。

覆盖率有清楚的适用边界。它依赖 teacher 表示、full-data 参考集、PCA 和 K-means 粒度；每个 cluster 等权，忽略访问概率、cluster 内差异和监督价值。它只测覆盖 full-data 会访问的区域，不是完整状态空间覆盖，也不天然可用于事前挑数据，因为参考空间来自 full-data rollout。作者明确把无需 full-data 参考的 query 选择指标留作后续问题。吸收率则是观测量：on-policy 下 state 分布本身随学生更新而变，不能单凭该曲线区分参数优化变慢、状态难度变化或分布漂移。

## Rollout 状态分布分析方法排序

如果问题是“不同 rollout policy 会访问哪些状态”，分析对象是每个 policy 在固定 prompt 分布上诱导的前缀分布。精确前缀 $s=(x,y_{<t})$ 是理论上的状态身份，但长前缀的可能取值很多，有限样本下逐前缀计数容易得到近乎全唯一的经验分布。若两批样本的精确前缀集合不相交，经验 TV 就等于 1；即使两批来自同一个 policy，也会发生这种情况。这是经验计数过于稀疏的问题，不能据此断言真实分布不同。精确前缀计数适合查重、排查缓存和验证采样日志，不适合作为主要的语义状态覆盖指标。One-Shot OPD §4.1 也因此选择在表示空间中衡量覆盖广度，而非直接数前缀。[^oneshot]

采用表示 $f$ 后，真正分析的是推前分布 $P_\pi^f=f_\#d_\pi$，而非完整前缀分布 $d_\pi$。不同前缀可能被压到相近或相同表示；因此表示分布接近不推出原始状态分布接近，也不保证两个前缀产生相同的后续推理。下面的排序是面向“访问了哪些可解释区域”的实施建议，不是论文已经证明的方法优劣。若问题改为“来源相对 student 怎样改变前缀概率”，可直接重评分的精确前缀概率比应与第 1 项同时测量；排在后面的只是另行学习密度比的工具。

| 排名 | 方法                         | 主要回答                                         | 对 OPD 的推荐程度                                           |
| ---: | ---------------------------- | ------------------------------------------------ | ----------------------------------------------------------- |
|    1 | 共同表示上的分层占用分布     | 各 policy 在可比较的状态区域分配了多少访问质量？ | 主分析。固定 prompt、深度口径和表示后比较簇占用及逐簇差异。 |
|    2 | MMD 两样本距离               | 两组表示状态的整体分布是否不同？                 | 整体摘要。按 prompt/深度分层，作为占用表的补充。            |
|    3 | Held-out 近邻支持距离        | 某 policy 是否进入参考样本很少覆盖的区域？       | 尾部与支持诊断，补足聚类覆盖盲点。                          |
|    4 | 固定状态上的监督信号加权分析 | 多访问的状态是否有更大的 teacher-student gap？   | 解释访问差异是否对应训练信号，和纯分布结果分开报告。        |
|    5 | Wasserstein / 最优传输       | 两个占用分布之间的质量搬运成本有多大？           | 探索性几何摘要；先验证状态距离与预测/梯度相似度的关系。     |
|    6 | 另行学习 occupancy 密度比    | 无法直接重评分时如何估计分布校正？               | 可直接重评分时优先用精确比值；学习估计器留作重加权扩展。    |

熵、有效状态数和精确前缀去重率可作描述性统计，但不替代上述比较。不同指标回答不同问题，不应合成一个未经校准的“state diversity”总分。

### 1. 共同表示上的分层占用分布

比较开始前先固定共同的 prompt 分布 $\rho(x)$、生成深度口径，以及将前缀映射到共享表示的函数 $z=f(s)$。$f$ 必须对所有 policy 相同并在比较期间冻结；可以用同一个冻结 teacher 在前缀末位置的 hidden state 作主表示，再用固定文本 encoder 或 lexical feature 作稳健性对照。One-Shot OPD 用 teacher 最后一层 hidden vector 构造参考簇；另一项 LLM state-distribution 研究则以 lexical feature 表示 rollout 前缀并计算分布漂移。这些是表示选择的实例，不能说明任一种表示就是普遍正确的状态度量。[^oneshot][^statedist]

在平衡抽取的参考表示上拟合一次聚类器 $c(z)\in\{1,\ldots,K\}$，之后所有 policy 共用同一组簇。对 prompt $x$、深度区间 $b$ 和簇 $k$，定义经验占用概率：

$$\hat p_\pi(k\mid x,b)=\frac{\sum_{n,t}w_{n,t}\,\mathbf{1}\{c(f(s_{n,t}))=k\}}{\sum_{n,t}w_{n,t}},\qquad t\in b.$$

权重 $w_{n,t}$ 决定占用口径。若目标是描述训练目标实际看到的 token 状态，使用每个有效 token 等权；若目标是让每条 rollout 对指标贡献相同，先在每条回答内归一化，再跨回答平均。二者对长回答的权重不同，主结果应匹配训练 loss 的聚合方式，并把另一种口径作为长度敏感性对照。EOS 与 padding 的处理也要固定并注明。

报告中先比较共同 prompt 与深度区间内的簇占用概率，再按目标 prompt 权重聚合。簇级差异可报告每簇的 $\hat p_{\pi_1}(k)-\hat p_{\pi_2}(k)$、JS divergence 或 TV；另报命中簇比例作为支持集覆盖。共同 prompt 很重要，因为若 prompt 组成不同，表示空间距离可能只是在识别题目或领域。默认按 prompt 分层后等权汇总；若目标是复现真实训练流量，则按真实 prompt 分布加权。两种结果回答不同问题。

簇占用概率和 JS/TV 是本文对 One-Shot OPD 二值覆盖的扩展，原论文没有据此验证不同 rollout policy 的优劣。每题必须有足够多的独立 rollout；只有一条回答时，不能靠其中相关的 token 可靠估计该题的状态分布。预算不足时应减少题数或合并预先定义的深度区间，并报告估计误差。参考集与评估集至少按整条 rollout 切分，测试跨题泛化时按 prompt 切分，避免把相邻 token 随机分到两侧。

这项分析依赖表示和聚类粒度，因此要对 $K$、聚类随机种子和参考样本构造做敏感性分析。参考集可由所有待比较 policy 的平衡样本共同构造，再留出独立样本评估；若只用某一个 policy 建簇，会偏向该 policy。共享 teacher 表示度量的是 teacher 的表征几何，不保证距离对应 student 下一 token 分布差或梯度作用。

如果分析目标偏向 OPD 监督作用，可以额外用固定 teacher/student 的下一 token 分布、log-ratio 或梯度投影作功能特征，检验 hidden-state 近邻是否具有相近的局部训练信号。这是待验证的补充设计，不应称为完整推理状态等价：当前动作分布相近不保证未来轨迹相近。DeepMDP 的相关启发在于用奖励与转移预测约束状态表示；其 MDP 结论不能直接转移到任意 LLM hidden state。[^deepmdp]

若任务本身能提供可验证的状态变量，例如工具执行后的环境配置、程序执行状态或规则可重建的棋盘，应优先把这些变量作为语义锚点，并与 hidden-state 分析并列。自由文本数学推理通常没有现成的完整状态解析器；可提取的中间结论、方程或检查结果只是部分标签，不能未经验证就当作充分状态。

### 2. MMD：整体分布差异摘要

最大均值差异（Maximum Mean Discrepancy, MMD）用核函数比较两组表示状态的分布，不必拟合显式密度。对表示 $z$ 和核 $k$：[^mmd]

$$\mathrm{MMD}^2(P,Q)=\mathbb{E}_{z,z'\sim P}[k(z,z')]+\mathbb{E}_{u,u'\sim Q}[k(u,u')]-2\mathbb{E}_{z\sim P,u\sim Q}[k(z,u)].$$

MMD 可在共同 prompt/深度分层内计算，再按目标 prompt 权重汇总。RBF 核带宽会改变检测尺度，建议报告预先规定的多带宽结果；bootstrap 应以 prompt 为首层单位、rollout 为嵌套单位，而不是把同一回答中的 token 当作独立样本。MMD 是总体摘要，需与簇占用差异一起报告，才能定位差异集中在哪里。

LLM state-distribution 预印本在固定 prompt 上采集前缀，用 lexical feature 计算 RBF-MMD，并同时计算 centroid distance、sliced Wasserstein 和 lexical Jaccard。该小规模研究发现相近的 MMD 可伴随不同的遗忘程度，因此分布距离不能独自解释训练结果；它仍提供了直接分析 rollout-state 分布的案例。[^statedist]

### 3. Held-out 近邻支持距离

聚类会把每个状态分到最近质心，即使它远离所有参考样本。对每个 rollout state，计算其到 held-out 参考表示的 $k$ 近邻距离，报告中位数、高分位数及超过 held-out 基线阈值的比例。它回答“哪些 policy 更常访问参考样本难以支持的区域”，与聚类覆盖和 MMD 互补。One-Shot OPD 用原始 teacher 表示中的 10-NN 距离作此类诊断，并检查不同 state 位置和聚类设置下排序的稳定性。[^oneshot]

### 4. 学习信号加权分析

纯访问分布差异不等于训练价值差异。若研究目标是判断某 policy 访问到的状态是否更适合 OPD，可在同一批状态上计算固定 teacher/student 的全词表 KL、teacher-student gap、student entropy，或小步更新后的 gap 下降；再分别报告占用分布与信号分布，或将信号作为权重计算加权占用。不要把加权结果称作纯 state coverage，也不要把 teacher 信号大直接解释为任务上正确或更可学。

与仓库的 [rollout 来源理论](../../theory/rollout-policy/objectives.md) 对齐时，应指定局部目标 $K(s)=\mathrm{KL}(\mu(\cdot\mid s)\Vert\pi_T(\cdot\mid s))$，在固定模型快照上比较 $\mathbb E_{d_q}K(s)$ 及其按簇分解；不把 KL 的方向省略。局部目标的期望变大只说明来源更多访问当前师生分歧大的状态。验证实际学习收益仍需受控更新，并在共同 held-out state bank 与任务评测上比较。

### 5. Wasserstein / 最优传输

Wasserstein 距离把分布差异写成将一组样本质量搬运到另一组所需的最小成本。它不只问两组状态是否不同，也利用状态间距离表达“差了多远”。对于有限经验分布 $\hat P=\sum_i a_i\delta_{z_i}$、$\hat Q=\sum_j b_j\delta_{u_j}$，离散 OT 求解：

$$\min_{\Gamma\ge0}\sum_{i,j}\Gamma_{ij}c(z_i,u_j),\quad \Gamma\mathbf{1}=a,\quad \Gamma^\top\mathbf{1}=b.$$

RL 文献把 policy learning 表示为 occupancy measure 空间的轨迹，并用 OT 比较连续训练阶段的 occupancy，定义 Effort of Sequential Learning 和衡量有效移动方向的 Optimal Movement Ratio。其经验估计使用 rollout policy dataset 近似 occupancy，再用 dataset distance 近似 Wasserstein。[^esl] 另一条离线模仿学习路线 PW-DICE 直接最小化 learner 与 expert 的 state occupancy Wasserstein，并学习对任务有用的对比距离；作者的消融也显示，任意欧氏距离并不一定合适。[^pwdice]

对 LLM 前缀，token 编辑距离、固定 encoder 距离或 hidden-state 距离都能形成 $c(s,s')$，但语义近邻不一定具有相似的下一 token 分布或梯度。建议先把 OT 当作分析量，不直接当训练目标；在小规模样本上验证距离是否能预测 next-token 分布差异或固定 state 上的梯度相似度。高维 pairwise OT 计算昂贵，可用 sliced Wasserstein、Sinkhorn 正则化或分层抽样近似，并报告近似方案。

### 6. Occupancy ratio 与密度比

当已知参考 rollout 分布 $d_\mathrm{ref}$，想知道目标分布 $d_\mathrm{target}$ 相对它在哪些区域增减，可估计比值：

$$w(s)=\frac{d_\mathrm{target}(s)}{d_\mathrm{ref}(s)}.$$

这比单一距离更能定位过采样和欠采样，也可用于重要性加权或构造重采样分布。DICE 系列（如 DualDICE、SMODICE）利用 Bellman/MDP 结构估计 stationary occupancy correction，而不是先估两个高维密度再相除。[^dualdice][^smodice]

仅凭精确前缀出现次数通常难以可靠估计比值，但已知两个自回归 policy 时，可以直接对同一个前缀重新评分。在相同 prompt、固定深度和分母支持非零的条件下，未按存活事件重新归一化的前缀概率比满足：

$$\frac{P_{q_1}(y_{<t}\mid x)}{P_{q_2}(y_{<t}\mid x)}=\prod_{i<t}\frac{q_1(y_i\mid x,y_{<i})}{q_2(y_i\mid x,y_{<i})}.$$

这是自回归分解的直接恒等式，不需要同一个前缀被重复采到。重评分必须使用实际行为分布，包括温度、top-p 和其他截断；如果改为条件于“回答还未结束”的分布，还需考虑存活概率的归一化。长前缀比值可能极端，硬截断也可能造成支持缺口；它适合做概率偏移和重要性权重诊断，不能代替语义覆盖。

若目标是表示空间密度比，可用分类器 odds 或聚类频数估计，但得到的是 $f_\#d$ 之间的比值，不能直接替代精确前缀 IS 权重。要报告权重截断前后的有效样本量、最大权重和支持缺口。DICE 是处理一般离线 MDP 的工具；在可以直接重评分的 LLM 设置里，没有必要仅因前缀计数稀疏就首先引入它。

### 7. 熵、有效状态数与精确前缀诊断

在共同表示的簇占用概率 $p_j$ 上计算熵 $H=-\sum_jp_j\log p_j$ 或有效簇数 $\exp(H)$，可补充访问质量是否集中。它依赖前述表示和离散化，不能独立说明覆盖了哪些有意义状态。精确前缀 hash 只建议用于统计完全相同前缀的重复率、检查日志或缓存行为；不应以 unique-prefix 数或精确前缀 TV 作为策略语义多样性的主结论。

2026 年 ROVER 预印本把 occupancy 覆盖最大化写成熵最大化，并以学习到的 resolvent world model 估计 occupancy、用 sink state 表示未探索区域。[^rover] 这种方法面向具有状态转移和可探索环境的 reward-free RL；对 OPD，熵更适合作为诊断量。单纯最大化状态多样性可能把预算分给无关或低学习价值的前缀。

## 推荐的 OPD 测量组合

首轮比较建议按以下协议执行：

1. 先固定同一组 student/teacher 快照，离线比较不同来源构造，以隔离 rollout policy 本身；各自训练后再比较时，需另报告 learner 改变的影响。固定共同 prompt bank、每题 rollout 数和生成 cap，除所研究的 policy 构造外，其他解码条件保持一致。记录实际 token 数；等 rollout 数和等实际 token 预算是不同实验口径，不要求同时满足。
2. 把状态位置按固定绝对深度分箱作为主口径，并以相对回答进度分箱作敏感性分析；EOS、padding 与 cap 截断分别记录。每个深度同时报告存活比例及条件于存活的状态分布；某策略没有回答到达该深度时，不把缺失的条件分布当成普通零频数。所有策略均较长的幸存回答也可能具有选择偏差。
3. 用同一冻结 teacher 在每个前缀末位置提取 hidden state，构造平衡的 pooled reference 并拟合共享聚类。策略间使用相同 prompt 时，先在 prompt×深度层内计算占用，再按目标 prompt 分布汇总，避免 prompt 组成或长回答主导全局差异。
4. 主结果报告每层 cluster 占用概率、逐簇质量差、JS/TV 和命中簇比例；补充报告多带宽 MMD 与 held-out 近邻距离尾部。以 prompt 和 rollout 为层级做 bootstrap 置信区间，并检查多个 $K$、聚类初始化及第二种冻结表示下的排序是否稳定。
5. 如果还要判断“哪种 policy 访问到更有用的状态”，在共同抽取的状态样本上计算相同 teacher/student 的 KL 或 gap，再报告 gap 加权占用。若比较固定状态上的学习速度，另用一个不随 policy 改变的 held-out state bank 评估每次更新前后的 gap。

这组结果分别回答：各 policy 把访问质量分配到哪些状态区域、整体分布差多少、是否超出参考样本支持，以及访问差异是否对应较大的监督信号。访问分布与固定状态上的可学习性应分开展示。精确前缀去重数可以放在采样审计附表，不作为排序不同 rollout policy 的核心证据。

另把每个 policy 的 rollout 独立分成两份，计算同一 policy 内的距离与覆盖差异，作为有限采样基线；该基线与跨 policy 比较使用相同样本量。每个簇或近邻尾部都抽取代表前缀供人工检查，确认差异反映推理内容，而非仅反映固定措辞、题面或终止 token。二维降维图用于展示，统计距离在原始或统一拟合的表示空间计算。

若只讨论当前固定 prompt bank，bootstrap 保留 prompt，重采各题的 rollout；若要推广到 prompt 总体，再增加外层 prompt 重采样。两种置信区间对应不同估计对象。所有方法的优先级最终取决于问题：定位访问区域以共享占用分布为主，解释来源权重以可重评分的前缀概率比为主，解释训练效果则另需监督与受控更新证据。

## MMD 与近邻诊断的计算细节

以下为面向本项目的测量建议，尚未运行，不是已有实验结果。先固定参考模型 $F$，对所有来源的前缀重编码得到 $z=f_F(x,y_{<t})$。不能直接比较不同模型各自的 hidden vector；即使维度相同，其坐标系也未对齐。可在独立的、按来源平衡的参考集上拟合 PCA，再做 L2 归一化；原始维度的归一化表示作为敏感性对照。所有拟合、核带宽选择和阈值校准均与最终评估回答分离。

### MMD 的理论含义

最大均值差异（Maximum Mean Discrepancy, MMD）是一个两样本分布比较量。它不在精确前缀上逐个配对，也不需要先估计高维密度；它比较两个分布在一类测试函数上的期望是否不同。给定核函数 $k$ 对应的再生核希尔伯特空间（RKHS）及特征映射 $\phi$，分布 $P$ 的核均值嵌入是：

$$\mu_P=\mathbb E_{z\sim P}[\phi(z)].$$

MMD 定义为两个核均值嵌入的距离：

$$\operatorname{MMD}(P,Q)=\|\mu_P-\mu_Q\|_{\mathcal H}.$$

展开内积后得到可计算的形式：

$$\operatorname{MMD}^2(P,Q)=\mathbb E_{z,z'\sim P}k(z,z')+\mathbb E_{u,u'\sim Q}k(u,u')-2\mathbb E_{z\sim P,u\sim Q}k(z,u).$$

直觉是：第一项看 $P$ 内部的相似度，第二项看 $Q$ 内部的相似度，第三项看两组之间的相似度。如果两个分布相同，三项在总体上抵消；如果 $P$ 的样本彼此相似、但与 $Q$ 的样本不相似，MMD 增大。RBF 核 $k_\sigma(z,u)=\exp(-\|z-u\|^2/(2\sigma^2))$ 在欧氏表示空间上是 characteristic kernel，在分布满足相应矩条件时，总体 MMD 为零才推出两个表示分布相同。实际分析的是表示后的推前分布 $f_\#P$，因此结论受表示函数影响。

一个一维手算例子可以说明计算方式。设 $P$ 和 $Q$ 各有两个样本，$P=\{0,1\}$、$Q=\{0,2\}$，取 $\sigma=1$。有 $k(0,1)=e^{-1/2}\approx0.6065$、$k(0,2)=e^{-2}\approx0.1353$、$k(1,2)=e^{-1/2}\approx0.6065$。使用包含对角线的非负经验估计量：

$$\widehat{\operatorname{MMD}}_b^2=\frac14\sum_{i,j}k(z_i,z_j)+\frac14\sum_{i,j}k(u_i,u_j)-\frac{2}{4}\sum_{i,j}k(z_i,u_j)\approx0.1968.$$

它表示这两组样本在当前核尺度下有可见差异。去掉组内对角线的无偏 U 估计量在样本很少时可能为负；负值是有限样本估计误差，不是“负距离”，也不应直接开平方。工程上可报告非负的 biased $\widehat{\operatorname{MMD}}_b^2$，或报告 U 估计量并用整条 rollout 的 bootstrap／置换检验给出不确定性。

RBF 带宽决定“看多大尺度的差异”：$\sigma$ 很小时主要检测局部邻域，$\sigma$ 很大时主要检测较粗的整体位移。不能为每个 policy pair 单独调带宽后比较 MMD 数值；应从独立参考集固定一组带宽，通常报告距离中位数的 $1/2$、$1$、$2$ 倍。MMD 也有 witness function：

$$w(z)=\mathbb E_{u\sim P}k(u,z)-\mathbb E_{v\sim Q}k(v,z).$$

在样本上估计 $w(z)$，可以定位哪些表示区域更像 $P$ 而不像 $Q$；它比一个全局 MMD 标量更适合回看代表前缀，但仍是核和表示依赖的定位工具。MMD 的原始两样本检验框架见 Gretton 等人。[^mmd]

### MMD 的样本、核与估计量

优先固定 prompt $x$ 和绝对生成深度 $t$，每条独立 rollout 在该位置贡献一个状态。令两个来源的表示样本为 $Z=\{z_i\}_{i=1}^n$ 与 $U=\{u_j\}_{j=1}^m$，使用 RBF 核：

$$k_\sigma(z,u)=\exp\left(-\frac{\|z-u\|_2^2}{2\sigma^2}\right).$$

从共享参考集中汇总同题、同深度的成对距离，为每个领域和深度确定一个距离中位数 $\sigma_0$，预先固定 $\{\sigma_0/2,\sigma_0,2\sigma_0\}$ 三个带宽，所有待比较来源使用同一组数值。不要每个配对独立调带宽再横比 MMD 大小。用去掉组内对角线的估计量：[^mmd]

$$\widehat{\mathrm{MMD}}_u^2=\frac{\sum_{i\ne j}k(z_i,z_j)}{n(n-1)}+\frac{\sum_{i\ne j}k(u_i,u_j)}{m(m-1)}-\frac{2\sum_{i,j}k(z_i,u_j)}{nm}.$$

第一、二项是各组内部相似度，第三项是跨组相似度。同分布时，三项在总体上相抵。有限样本无偏估计可以为负，不取平方根，也不把负值解释为负的分布距离。每个 prompt 内计算后按预定题目权重汇总；一个回答贡献多个位置时，位置之间有依赖，重采样要保留整条回答。每个来源独立拆分样本形成同来源基线；其两侧样本数与跨来源比较一致。

终止会改变各深度可用的状态数。主表同时报告存活率和存活条件下的 MMD；在某题某深度任一侧不足预定最小样本数时，条件 MMD 记为不可估计，并报告可评估题目比例。若需要一个包含终止差异、覆盖所有题目的摘要，可另定义终止原子 $\dagger$：

$$\bar P_{q,t}=a_{q,t}P_{q,t}^{\mathrm{alive}}+(1-a_{q,t})\delta_\dagger,\qquad a_{q,t}=\Pr_q(L\ge t).$$

令核满足 $k(\dagger,\dagger)=1$、$k(\dagger,z)=0$，即可在扩展空间计算 MMD。它把终止差异纳入指标；与纯存活状态 MMD 分开标注。生成 cap 造成的是观测删失，不能将 cap 外未知的后续状态当成自然 EOS；重复停止也单独记录为协议触发。

### 有方向的近邻诊断

以来源 $B$ 为参照，按整条 rollout 分成参考集 $R_B$、校准集 $C_B$ 和测试集 $E_B$。对来源 $A$ 的独立测试状态计算：

$$d_k(z,R_B)=\frac1k\sum_{r\in\operatorname{NN}_k(z,R_B)}\left(1-\frac{z^\top r}{\|z\|\|r\|}\right).$$

先限制近邻候选属于同一 prompt 和深度，避免只找到题面相似的其他题。按领域和深度，用 $C_B$ 到 $R_B$ 的距离分布确定 $95\%$ 分位阈值 $\tau_B$，题目等权；再报告 $A$ 的距离分位数及 $\Pr_{E_A}[d_k(z,R_B)>\tau_B]$。阈值不能用 $E_A$ 调整。同来源 $E_B$ 是独立校准检查，有限样本越界率不要求恰好等于 $5\%$。One-Shot OPD 附录 B.1 使用 10-NN 与 held-out 95 分位阈值；这里增加 prompt/深度分层，并按每题参考样本数选 $k$。[^oneshot]

同时计算 $A\to B$ 与 $B\to A$。它们分别测两个方向上参考样本对查询状态的支持，不是对称距离；双向近邻思路也见生成模型的 precision/recall 研究，但这里不是直接复现该论文的流形估计量。[^knn-pr] 每个参照来源的样本量必须一致，不允许把查询状态自身或同一回答的相邻状态放进参考集。参考稀疏也会产生远邻，越界只表示在所选表示和样本预算下难以被参考支持，不能直接宣布发现了新推理能力。

### 2605.22731 中的 MMD：测量内容与结论边界

Nie 的预印本《Post-Training is About States, Not Tokens: A State Distribution View of SFT, RL, and On-Policy Distillation》把 state 定义为完整的自回归前缀 $s_t=(x,y_{<t})$，把 policy 在 prompt 分布上诱导的前缀访问分布记为 $d^\pi$。作者在固定 prompt 集上收集各 post-training 模型的 rollout 和前缀，用轻量 lexical feature 表示 state，再用 RBF-MMD 作为 rollout-state drift 的主标量；同时计算 centroid distance、sliced Wasserstein 和 lexical Jaccard。论文没有把 hidden state 当作其 MMD 表示，也没有在正文给出足以复现的完整特征、带宽、抽样和估计器配置，因此不能把其绝对 MMD 数值直接移植到本项目。[^statedist]

论文主表的结果如下；MMD 是相对 Base rollout 的 drift 读数，Base 行本身没有填 MMD：

| run                 | GSM8K | TruthfulQA |  MMLU |     MMD | retention ratio |
| ------------------- | ----: | ---------: | ----: | ------: | --------------: |
| Mild SFT            | 0.512 |      0.295 | 0.444 | 0.00956 |          1.0008 |
| Stress SFT          | 0.420 |      0.245 | 0.364 | 0.01093 |          0.8258 |
| OPD from mild SFT   | 0.512 |      0.290 | 0.434 | 0.01470 |          0.9810 |
| OPD from stress SFT | 0.466 |      0.275 | 0.430 | 0.01092 |          0.9515 |
| On-policy RL        | 0.472 |      0.290 | 0.442 | 0.01098 |          0.9902 |

这个表最重要的不是哪个 MMD 最大，而是它否定了一个过强的解释：**MMD 越大，遗忘越严重。**Stress SFT 和从 stress SFT teacher 出发的 OPD 的 MMD 几乎相同（0.01093 与 0.01092），但 retention ratio 分别为 0.8258 与 0.9515；on-policy RL 的 MMD 也接近（0.01098），但 retention ratio 为 0.9902。OPD from mild SFT 的 MMD 甚至最高（0.01470），却没有出现相应程度的 retention 损失。

作者因此把 MMD 定位为 state drift 的紧凑代理，而不是训练后果的充分解释。相同的总体 rollout 漂移，可能对应不同的训练 state 来源、局部可恢复性和监督作用。论文还报告 stress SFT teacher 的 OPD 可以超过该 teacher：stress SFT 的 GSM8K 为 0.420，OPD from stress SFT 为 0.466；TruthfulQA 为 0.245 对 0.275，MMLU 为 0.364 对 0.430。作者的解释是 OPD 在 student 自己访问的 state 上查询 teacher 信号，未必复制 teacher 自己的整条轨迹分布。

这对 post-agreement 问题有直接启发：MMD 可以作为 student 与 teacher／initial 的总体状态漂移读数，但不能判断低 KL 是否发生在 teacher 很少访问、难以恢复的前缀。主观测仍应是逐位置 reverse KL、滑动窗口 agreement、teacher 前缀典型性和续写可恢复性；MMD 只作为分层后的背景量，并比较 agreement、post-agreement 与 pre-agreement 三个阶段，而不是用一个全局 MMD 数字替代它们。

## 两类 rollout 状态比较的建议设置

本节给出独立的离线分析设置，不改变现有训练、评测配方，也不启动新 GPU 作业。首先比较各冻结模型自身采样的状态；“用这些 teacher 训练后的 student 访问哪些状态”属于后续问题，需要额外控制训练更新、数据和 checkpoint。

### 训练谱系与 SFT／RL 阶段

把模型身份写成“底座及尺寸 × 训练分支及阶段 × 精确 revision”。以固定 student 起点 $S_0$ 为参照，主比较包括 $S_0$、从它出发的 SFT、direct RL 和 SFT→RL；跨来源模型按自己的底座建立对应阶段链，缺失 checkpoint 留空。SFT→RL 对 SFT 的增量只有在紧邻父权重得到核实时，才可解释为这条路径追加 RL 后的变化；公开分支仍可能混有数据与预算差异。

本仓最完整的同源实例是 MiMo-7B 的 Base、SFT、RL-Zero 和 RL 四份权重，已登记 revision 和共同输入协议，见[模型谱系](../../settings/模型与师生配对.md#mimo)与[阶段实验](../../experiments/teacher-transfer/post-training-stages.md#mimo)。其 SFT／RL-Zero／RL 是三个真实训练目标，并非只改变训练算法的等预算消融。跨来源可用现有 Llama／Tulu 阶段链作另一组描述性比较；Tulu 的 DPO 必须单独列出，不能将 SFT→DPO→RLVR 简写成无中间阶段的 SFT→RL。公开链缺失的阶段不能由同家族其他模型补成严格配对。

所有来源统一用冻结 $S_0$ 作第一观察器，并用一个固定的第二模型重复主要比较；第二观察器可为其中一个 teacher，但选定后对全部来源保持不变。每个观察器分别产生距离矩阵，不能将不同观察器的坐标或距离值拼接。跨 tokenizer 先解码来源的真实前缀，再按观察器协议重编码；保留完整推理文本，记录生成模板、thinking 模式和特殊 token。跨家族按共同观察器 tokenizer 的生成文本长度分箱，选最接近目标深度的真实生成 token 边界并记录偏差，不以各自原生 token 编号直接当作等长。固定相同任务内容、各自原生模板的比较，测量的是各自实际部署协议下的总体差异；额外强制同模板是另一项输入干预。

主要产物为每个领域、深度的两两 MMD 矩阵，以及有方向的近邻越界矩阵；另给出相对 $S_0$ 的深度曲线、逐簇占用差和代表前缀。先在谱系内比较阶段，再跨来源比较；不从跨底座终点的距离排序推断 SFT 或 RL 的普遍因果效果。

### Initial、teacher 与两种 token fusion

固定同一份 $S_0$ 和 teacher $T$，先不训练。四类来源是纯 $S_0$、纯 $T$、逐 token logit fusion 与逐 token probability fusion。令 $p_0,p_T$ 是同一前缀、共同合法 token 支持上的归一化分布，在 temperature1、无 top-p/top-k 截断时：

$$q_{\mathrm{logit},\eta}(a\mid s)=\frac{p_0(a\mid s)^{1-\eta}p_T(a\mid s)^\eta}{\sum_vp_0(v\mid s)^{1-\eta}p_T(v\mid s)^\eta},\qquad q_{\mathrm{prob},\eta}(a\mid s)=(1-\eta)p_0(a\mid s)+\eta p_T(a\mid s).$$

$\eta$ 始终表示 teacher 权重。先比较 $\eta=0.5$，再补 $0.25/0.75$；端点共用纯模型样本。同 tokenizer 和 token-ID 语义是 token fusion 的前提，跨家族状态分析可用共同观察器完成，不代表跨 tokenizer 融合已经可行。仓库数值参考将融合后结果送入共同 temperature/filter 阶段；因此 temperature 不为 1 时，不能直接将最终 probability-fusion 行为写成两个温度化端点的算术平均，见[采样实现](../../../../../verl/experimental/rollout_policy/sampling.py)。停止 token 的实现约定也必须保存。

优先复用 [Qwen 原生来源设置](../../experiments/rollout-policy/native-qwen.md)的 1.7B non-thinking student 与 4B-Instruct-2507 teacher。现有 E66／E67 融合的是 current student 与 teacher；本节离线首轮固定融合 $S_0$，只有训练起点与 current 融合一致。训练过程中比较来源时，必须在同一个保存的 student 快照上重新构造四类来源，不能直接将各训练分支不同终点的 rollout 当作仅来源不同的对照。此时另列 current student，避免把它与 initial 混淆。

两类 token fusion 都不保证状态分布在两端之间，也不保证距离随 $\eta$ 单调。若增加一个成本较低的轨迹混合对照，应先按回答抽选 $S_0$ 或 $T$ 再完整生成；这才是完整轨迹分布的凸组合。固定深度并加入终止原子后，它的状态分布也线性混合；条件于存活后，混合系数会因两端存活率不同而改变。不要把轨迹混合与逐 token probability fusion 混为一谈。

### 首轮预算与共同口径

下表是分析起点，最终样本量由同来源基线、区间宽度和高深度存活数决定，不是经过功效分析的最低预算。

| 项目       | 建议起点                                                          | 解释                                                                                                   |
| ---------- | ----------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| 小规模检查 | 每领域 32 题、每来源每题 16 条回答                                | 检查前缀重编码、停止、长度和表示，不据此做细粒度显著性结论。                                           |
| 正式首轮   | 数学 held-out 128 题、每来源每题 64 条回答，共 8192 答            | 首先完成单域；研究跨域迁移再按领域独立扩展同等预算。预算紧时 32 条可作筛查，需相应标注误差。           |
| 分割       | 每题 32 条 reference、16 条 calibration、16 条 test               | PCA/聚类/带宽只用平衡 reference；MMD 用 test；轮换整条回答的分割作敏感性检查，轮换结果不当作独立重复。 |
| 追加预算   | 有效样本不足时增加到每题 128 条或减少主比较的深度                 | 16 条 test 的条件 MMD 仍有采样误差；高深度终止多时优先补样，不用 token 数冒充独立样本数。              |
| 采样       | temperature1、top-p1、top-k−1                                     | 首轮隔离模型/融合分布；其他解码温度作为独立敏感性实验。                                                |
| 生成长度   | 同 tokenizer 首轮 response cap7168；prompt cap 由共同有效题单固定 | 便于对接既有来源设置；跨 tokenizer 同时记录观察器 token 长度，并做共同文本长度限制的敏感性对照。       |
| 深度       | 256、512、1024、2048、4096、6144 个生成 token                     | 同 tokenizer 用原生深度；跨 tokenizer 用观察器口径；每深度报存活率，8 个相对位置仅作补充。             |
| 近邻       | 每题 32 条参考时先用 $k=5$，补 $k=3/10$                           | 按存活参考样本数检查是否可估计；阈值按领域/深度汇总独立 calibration 确定。                             |
| 统计       | 题目配对、回答成组的 bootstrap，95% 区间                          | 记录逐请求 seed；相同 seed 不保证跨模型轨迹语义配对。                                                  |

跨家族生成时，相同原生 token cap 不代表相同文本预算。主结果注明该截断口径，增加共同观察器 token 长度 cap 的对照，并分别报告 EOS、cap 和重复停止。若 cap 命中率明显，补长窗口测量后才能讨论完整回答分布。历史配置中的重复检测不能静默继承到新测量，必须统一决定启停并记录。

第一组先做 MiMo 四模型，第二组先做固定 Qwen 配对的四来源。第一组扩展第二条谱系、第二组扩展融合系数，分别进行，不直接展开全部 teacher×fusion×系数的大网格。主分析每个深度每条回答等权；若要对接 token-mean 训练目标，另按有效 token 质量计算占用，抽样状态使用相应纳入概率校正，不能将几个等权深度点直接写成训练总体。

## 其他 state 分析技术

这些工作研究的“state”不完全相同。对 OPD 数据效率最直接的是训练 rollout 的状态覆盖；棋类工作提供可验证的外部世界状态；机制解释工作则直接读内部 activation。它们的指标不能直接横比。

| 工作                                                                                                                              | state 如何定义/分析                                                                                                                                             | 与 One-Shot OPD 的关系                                                    |
| --------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| [Rethinking On-Policy Distillation of Large Language Models II: One Training Example](https://arxiv.org/abs/2609.04172)（2026）   | 自回归前缀；teacher 最后一层 hidden vector；PCA + K-means；统计 full-data 参考 cluster 覆盖，并在访问前缀上测 log-prob gap 与吸收率。                           | 本文主对象。                                                              |
| [Emergent World Models and Latent Variable Estimation in Chess-Playing Language Models](https://arxiv.org/abs/2403.15498)（2024） | 用真实棋局训练模型，以线性 probe 从 activation 解码棋盘状态；contrastive activation 和 activation intervention 检验表示是否可因果操纵；还分析玩家水平等潜变量。 | 不是训练数据覆盖研究；方法更强调“内部表征编码了什么”以及干预验证。        |
| [Chess as a Testbed for Language Model State Tracking](https://arxiv.org/abs/2102.13249)（2021）                                  | 棋谱前缀隐含确定棋盘；以合法着法预测、棋子跟踪表现评估模型能否追踪世界状态，也比较给棋盘状态监督和完整历史 attention 的影响。                                   | 用结构化任务和可验证状态作行为级测试，基本不依赖 hidden-state probe。     |
| [Tracking World States with Language Models: State-Based Evaluation Using Chess](https://arxiv.org/abs/2508.19851)（2025）        | 不访问模型内部 activation；从预测的合法着法分布（state affordances）反推模型是否保留正确棋盘语义，评估实际棋盘与预测状态的语义一致性。                          | 适合黑盒评估；它验证状态语义，不衡量 OPD rollout 对训练集参考状态的覆盖。 |

前述方法分析 rollout 的访问分布；以下工作分别研究外部世界状态跟踪和模型内部表征，能补充状态语义验证，但不能替代 occupancy 分析。

1. **外部状态 tracking / 行为 probe。** 在棋盘、对话世界或工具环境中，环境状态可由规则重建；比较模型输出的合法动作、实体位置或状态预测与真实状态。它可黑盒运行，解释较直观，代价是需要结构化任务和可靠状态标签。
2. **内部 activation probe。** 训练线性 probe 或小型分类器，从某层某位置的 activation 解码状态变量；跨时间步或层比较状态信息。必须用控制任务、随机标签/基线和未见组合检验 probe 是否只是读出表面相关性。可解码不代表模型在生成时因果使用该信息。
3. **因果干预与表征编辑。** 对 activation 做 steering、替换或擦除，再看状态相关行为是否按预测改变。它比相关性 probe 更接近因果检验，但干预可能引入分布外 activation，且局部行为改变不等于完整认知机制得到证明。
4. **状态转移动力学。** 将每步 activation、输出的中间状态或外部环境状态组成轨迹，研究转移方向、稳定点、熵/不确定性、循环或收敛。适合问“推理过程怎样演化”，但定义的 state、距离度量和时间对齐通常影响结论。

## 对 OPD 研究的可用启发

若要把这套思路用于自己的 OPD 分析，可以把“是否到过某语义 cluster”升级成带学习价值的覆盖：在固定 held-out 前缀池上同时记录 state 访问概率、全词表 KL、teacher-student gap、更新前后 gap 变化及梯度范数。用相同参考前缀比较训练前后，能避免将“访问了更容易的 state”误读为“同一 state 上吸收变慢”。再以 query/trajectory 多样性做受控干预，测试新增覆盖是否真的带来增量学习，而不是仅与准确率相关。

更稳妥的报告至少并列三类量：几何/语义覆盖（访问了哪些区域）、occupancy（各区域访问多少）、supervision-weighted coverage（访问区域上的师生分歧）。实际可学习性另由受控更新检验。用多个 state 表示、参考集和聚类粒度复算排序，并报告 held-out 距离尾部；不要把单一聚类覆盖率解释为“完整推理空间已覆盖”。

## 参考来源

- Karvonen et al.《Emergent World Models and Latent Variable Estimation in Chess-Playing Language Models》、Noorbakhsh et al.《Chess as a Testbed for Language Model State Tracking》、以及《Tracking World States with Language Models: State-Based Evaluation Using Chess》仍保留在上文的相关技术表中。对应 arXiv ID 依次为 2403.15498、2102.13249、2508.19851。
- 检索方式：读取 2609.04172 arXiv LaTeX 源；通过 paper-search 检索 state coverage、occupancy、分布距离与 state tracking；并核对下列方法论文的 arXiv 原文或摘要。检索结果包含大量标题匹配但对象不同的工作，本文按方法与 OPD 的距离筛选，不声称穷尽所有 state 分析文献。

[^oneshot]: Fu, Zixuan, et al. Rethinking On-Policy Distillation of Large Language Models II: One Training Example. arXiv:2609.04172, 2026. §4.1、§4.2 与 Appendix B.1（Training-State Coverage）. [arxiv.org](https://arxiv.org/abs/2609.04172)

[^statedist]: Nie, Dong. Post-Training is About States, Not Tokens: A State Distribution View of SFT, RL, and On-Policy Distillation. arXiv:2605.22731, 2026. §4 的 State drift、§5.4 与 Limitations. [arxiv.org](https://arxiv.org/abs/2605.22731)

[^esl]: Nkhumise, Reabetswe M., Debabrota Basu, Tony J. Prescott, and Aditya Gilra. How does Your RL Agent Explore? An Optimal Transport Analysis of Occupancy Measure Trajectories. arXiv:2402.09113, 2024. [arxiv.org](https://arxiv.org/abs/2402.09113)

[^pwdice]: Yan, Kai, Alexander G. Schwing, and Yu-Xiong Wang. Offline Imitation from Observation via Primal Wasserstein State Occupancy Matching. ICML 2024. [arxiv.org](https://arxiv.org/abs/2311.01331)

[^dualdice]: Nachum et al. DualDICE: Behavior-Agnostic Estimation of Discounted Stationary Distribution Corrections. NeurIPS 2019. [arxiv.org](https://arxiv.org/abs/1906.04733)

[^smodice]: Ma et al. Versatile Offline Imitation from Observations and Examples via Regularized State-Occupancy Matching. arXiv:2202.02433, 2022. [arxiv.org](https://arxiv.org/abs/2202.02433)

[^rover]: Reward-free Pretraining for Reinforcement Learning via Occupancy Coverage Maximization. arXiv:2606.21271, 2026. [arxiv.org](https://arxiv.org/abs/2606.21271)

[^mmd]: Gretton, Arthur, et al. A Kernel Two-Sample Test. Journal of Machine Learning Research 13, 2012. [jmlr.org](https://www.jmlr.org/papers/v13/gretton12a.html)

[^deepmdp]: Gelada, Carles, et al. DeepMDP: Learning Continuous Latent Space Models for Representation Learning. ICML 2019, PMLR 97:2170–2179. [proceedings.mlr.press](https://proceedings.mlr.press/v97/gelada19a.html)

[^knn-pr]: Kynkäänniemi, Tuomas, et al. Improved Precision and Recall Metric for Assessing Generative Models. NeurIPS 2019. [arxiv.org](https://arxiv.org/abs/1904.06991)
