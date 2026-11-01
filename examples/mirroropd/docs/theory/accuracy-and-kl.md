# 正确率、序列 KL 与 Token KL：从信息论到策略优化

更新时间：2026-10-31。本文沿着一条主线展开：**正确率是输出分布的事件概率；序列 KL 控制事件概率的变化；token KL 通过访问分布加权连接到序列 KL；局部几何与 KL 约束优化进一步解释平方根关系。**

| 核心关系            | 结论                                 | 来源与本文处理                                          |
| ------------------- | ------------------------------------ | ------------------------------------------------------- |
| 正确率与分布信息    | 正确／错误的二元 KL 不超过序列 KL    | 数据处理不等式；本文写出事件划分证明[^erven]            |
| 正确率变化上界      | $\lvert\Delta G\rvert\leq\sqrt{K/2}$ | Pinsker 型界；本文用二元 KL 直接证明[^sason]            |
| 序列与条件 token KL | $K=\sum_t\mathbb E_{d_P^t}k_t$       | KL 链式法则及其 OPD 表述[^dynamics][^rk-theory]         |
| 平滑路径的局部形状  | $G=G_0+m\sqrt K+O(K)$                | Fisher 几何；结合任务收益的一阶展开[^igo][^bao]         |
| 最优小预算收益      | $G_{\max}(K)-G_0=\sqrt{2VK}+O(K)$    | 指数倾斜最优策略；本文推导二元正确性的系数[^dpo][^reps] |

先读第 1–4 节，可以建立指标之间的精确关系；第 5–6 节解释局部直线与最优前沿；第 7–8 节连接 teacher 监督和实际评估。Gold 的研究沿革见[gold score 调研](../literature/teacher-transfer/2026-10-31-Gold-score的来源与奖励过优化研究脉络.md)，OPD 图表语境见[2609.32722 精读](../literature/teacher-transfer/2026-10-31-同家族OPD缩放规律精读与实验启发.md)。

## 1. 定义与统一条件

### 1.1 正确率和序列 KL

固定 prompt 分布 $\mu(x)$。$P(y\mid x)$ 为当前策略，$Q(y\mid x)$ 为参考策略，$c(x,y)\in\{0,1\}$ 判定回答是否正确。令

$$
\bar P(x,y)=\mu(x)P(y\mid x),\qquad
\bar Q(x,y)=\mu(x)Q(y\mid x).
$$

定义期望采样正确率与平均序列 KL：

$$
G(P)=\mathbb E_{\bar P}c(X,Y),\qquad
K(P,Q)=D_{KL}(\bar P\Vert\bar Q)
=\mathbb E_{x\sim\mu}D_{KL}(P(\cdot\mid x)\Vert Q(\cdot\mid x)).
$$

$G\in[0,1]$，百分制分数为 $100G$。所有对数采用自然对数，KL 单位为 nats。$Q=S_0$ 时，$K$ 是相对初始化的位移；$Q=T$ 时，$K$ 是相对 teacher 的残差。

同一策略下，avg@n 是 $G$ 的重复采样估计。独立采样时，pass@n 对应每题至少答对一次，其期望为 $\mathbb E_x[1-(1-G(P\mid x))^n]$；greedy acc 则评价确定性解码结果。这三者分别对应不同的评分过程。

### 1.2 后文采用的条件

| 使用位置     | 条件                                                                        |
| ------------ | --------------------------------------------------------------------------- |
| KL 与上界    | $P\ll Q$ 且 KL 有限；两者共享 prompt 分布与评分规则                         |
| 序列分解     | 相同动作语义、生成时域 $H$ 和终止规则；EOS 后视为吸收状态                   |
| 局部展开     | 有限 prompt／输出空间、共同正支持、足够光滑的参数族；路径的 Fisher 位移非零 |
| 最优前沿     | 在 $Q$ 的支持内自由选择条件分布，直接优化真实正确率，固定 prompt 边际       |
| 无限空间推广 | 相应积分、微分与极限交换成立                                                |

以下先在有限空间证明。退化的二元事件概率用连续极限处理；无限空间中的通用结果可查阅 van Erven–Harremoës 与 Sason–Verdú。[^erven][^sason]

## 2. 从完整回答分布压缩到正确率

数据处理不等式说明：对随机对象做同一个映射后，分布间的 KL 不会增加。把回答映射为“正确／错误”，就得到正确率与序列 KL 的连接。van Erven–Harremoës 的 Theorem 9 包含 KL 作为 Rényi 阶数 1 的情形。[^erven]

### 2.1 按正确与错误划分的精确分解

令 $\mathcal C=\{(x,y):c(x,y)=1\}$，$a=\bar P(\mathcal C)$，$b=\bar Q(\mathcal C)$。当 $0<a,b<1$，

$$
\begin{split}
K(P,Q)={}&\operatorname{kl}(a\Vert b)\\
&+aD_{KL}(\bar P(\cdot\mid\mathcal C)\Vert\bar Q(\cdot\mid\mathcal C))\\
&+(1-a)D_{KL}(\bar P(\cdot\mid\mathcal C^c)\Vert\bar Q(\cdot\mid\mathcal C^c)),
\end{split}
$$

其中

$$
\operatorname{kl}(a\Vert b)=a\log(a/b)+(1-a)\log((1-a)/(1-b)).
$$

**证明。** 在正确集合上分解概率比：

$$
\log\frac{\bar P(x,y)}{\bar Q(x,y)}
=\log\frac ab+
\log\frac{\bar P(x,y\mid\mathcal C)}{\bar Q(x,y\mid\mathcal C)}.
$$

对该集合积分得到第一类贡献，在错误集合上同样展开，两部分相加即得。条件 KL 非负，因此

$$
\boxed{\operatorname{kl}(G(P)\Vert G(Q))\leq K(P,Q).}
$$

序列 KL 包含三部分：正确总概率的差异、正确解法内部的差异、错误输出内部的差异。Accuracy 只保留第一部分的两个概率值。

### 2.2 同分数可以对应很不同的输出分布

考虑两个正确输出 $c_1,c_2$ 和一个错误输出 $w$：

$$
P=(1/4,1/4,1/2),\qquad
Q_\epsilon=(\epsilon,1/2-\epsilon,1/2),\quad0<\epsilon<1/2.
$$

两者正确率始终为 $1/2$，而

$$
D_{KL}(P\Vert Q_\epsilon)
=\frac14\log\frac{1/4}{\epsilon}
+\frac14\log\frac{1/4}{1/2-\epsilon}
\longrightarrow+\infty.
$$

这个构造把差异全部放在正确解法内部，直接说明 accuracy 为什么是更粗的统计量。

## 3. 从序列 KL 得到正确率变化上界

### 3.1 Pinsker 型正确率界

固定 $b\in(0,1)$，令 $f(a)=\operatorname{kl}(a\Vert b)$。有

$$
f(b)=f'(b)=0,\qquad
f''(a)=\frac1{a(1-a)}\geq4.
$$

二阶 Taylor 积分余项给出 $f(a)\geq2(a-b)^2$。结合第 2 节：

$$
\boxed{\lvert G(P)-G(Q)\rvert\leq\sqrt{K(P,Q)/2}.}
$$

这就是正确率场景下的 Pinsker 型界。这里使用
$\operatorname{TV}(P,Q)=\frac12\sum_\omega|P(\omega)-Q(\omega)|$；
部分文献使用未除以 2 的变差记号，比较常数时应同步换算。Sason–Verdú 的引言和相关变差章节系统整理了这类不等式。[^sason]

对任意 $r\in[0,1]$，通过有界函数与 TV 的关系，同样有

$$
\lvert\mathbb E_Pr-\mathbb E_Qr\rvert
\leq\operatorname{TV}(P,Q)\leq\sqrt{K(P,Q)/2}.
$$

这是一个幅度界；增加相同数量的 KL，可以对应收益上升、下降或保持不变。局部斜率的方向在第 5 节由更新方向决定。

### 3.2 一般 gold reward：从指数矩得到上界

Mroueh 将奖励改善与 KL 的关系写成运输不等式，特别分析次高斯奖励及 proxy／gold 之间的误差。[^mroueh]

设

$$
\log\mathbb E_Qe^{\lambda(r-\mathbb E_Qr)}
\leq\frac{\sigma^2\lambda^2}{2}\qquad(\lambda>0).
$$

定义倾斜分布 $Q_\lambda=Qe^{\lambda(r-\mathbb E_Qr)}/Z_\lambda$。由 $D_{KL}(P\Vert Q_\lambda)\geq0$，

$$
\lambda(\mathbb E_Pr-\mathbb E_Qr)
\leq K(P,Q)+\log Z_\lambda.
$$

使用指数矩条件，再对 $\lambda$ 最小化：

$$
\mathbb E_Pr-\mathbb E_Qr
\leq\inf_{\lambda>0}
\left(\frac{K}{\lambda}+\frac{\sigma^2\lambda}{2}\right)
=\sqrt{2\sigma^2K}.
$$

有界 $[0,1]$ 奖励可取 Hoeffding 参数 $\sigma^2=1/4$，恢复正确率界；实值 reward model 则由其尾部条件确定 $\sigma$。这使 accuracy 与一般 gold reward 使用同一套推导，但保留各自的量纲。

## 4. 从 token KL 到序列 KL

自回归模型把完整回答概率分解为条件 token 概率，因此序列 KL 可以沿前缀展开。Piskorz 等的 Proposition C.4 明确给出语言模型中的方向匹配；Liu 等也以该分解定义真实轨迹风险。[^dynamics][^rk-theory]

### 4.1 链式分解

令 $s_t=(x,y_{<t})$，$p_t$、$q_t$ 是两策略在该前缀上的下一 token 分布，定义

$$
k_t(s_t)=D_{KL}(p_t(\cdot\mid s_t)\Vert q_t(\cdot\mid s_t)).
$$

先展开序列概率比，再对下一 token 取条件期望：

$$
K(P,Q)
=\mathbb E_{\bar P}\log\frac{P(Y\mid X)}{Q(Y\mid X)}
=\mathbb E_{\bar P}\sum_{t=1}^{H}\log\frac{p_t(Y_t\mid s_t)}{q_t(Y_t\mid s_t)}
=\sum_{t=1}^{H}\mathbb E_{s_t\sim d_P^t}k_t(s_t).
$$

外层前缀权重是 **$P$ 自己的访问分布**。交换 $P,Q$ 后，得到 reference rollout 上的相反方向条件 KL。EOS 前的比较包含终止动作，EOS 后贡献零。

### 4.2 固定 horizon 平均

定义

$$
K_{\mathrm{tok},H}=\frac1H\sum_{t=1}^{H}\mathbb E_{d_P^t}k_t.
$$

于是

$$
\boxed{K=HK_{\mathrm{tok},H}},\qquad
\boxed{\lvert G(P)-G(Q)\rvert\leq\sqrt{HK_{\mathrm{tok},H}/2}}.
$$

从序列累计换成固定 horizon 的 token 平均，平方根形式保留，系数乘 $\sqrt H$。例如 $H=1024$、token 平均为 0.001 时，这个上界约为 0.716。这个示例说明：很小的 token 平均可能仍对应较宽的正确率范围。

若 $H$ 是截断观察长度，这些公式评价的也是截断序列分布及该输出上的评分函数。

### 4.3 可变长度与三种分母

设一条轨迹的条件 KL 累计为
$C(Y)=\sum_{t\leq L}k_t(s_t)\geq0$。三种量为

$$
K=\mathbb E_P C,\qquad
K_{\mathrm{pool}}=\frac{\mathbb E_PC}{\mathbb E_PL},\qquad
K_{\mathrm{resp}}=\mathbb E_P(C/L).
$$

它们满足

$$
K=(\mathbb E_PL)K_{\mathrm{pool}},
$$

$$
K=(\mathbb E_PL)K_{\mathrm{resp}}
+\operatorname{Cov}_P(L,C/L).
$$

第一种 token 平均按有效 token 加权，第二种先对每条回答归一化再按回答加权。若 $1\leq L\leq H$，还得到 $K\leq HK_{\mathrm{resp}}$。上述恒等式来自随机变量的期望与协方差分解，使用的是非负 full-vocab 条件 KL 累计。

### 4.4 固定银行对应的数学对象

对任意前缀来源 $\rho$，记

$$
R_\rho(P,Q)=\sum_t\mathbb E_{s_t\sim d_\rho^t}
D_{KL}(p_t(\cdot\mid s_t)\Vert q_t(\cdot\mid s_t)).
$$

那么 $R_P(P,Q)=K(P,Q)$；$R_{S_0}(P,Q)$ 则衡量固定初始学生状态上的拟合。这两个量分别服务于行为分布评价与共同输入比较。Liu 等把历史访问分布与候选策略分布的差异作为其理论分析的重要部分。[^rk-theory]

实际评估先声明 $\rho$、评分时的策略分布和终止规则。温度、top-p、动作 mask 或共同词表归一化均属于策略定义的一部分；定义一致后，再选择相应的风险名称与链式关系。[^dynamics]

## 5. 从 Fisher 几何得到局部线性

KL 的局部二阶项由 Fisher 信息矩阵决定，这是信息几何中的标准结果；Ollivier 等 §2.1.2 Eq. (2) 给出多参数表达，van Erven–Harremoës §III-H 讨论参数族的 Taylor 近似。[^igo][^erven]

Bai 等在 RLHF 中观察并解释了奖励与平方根 KL 的近似关系；Bao 等将相同局部思路用于 OPD 的任务收益。[^bai][^bao] 下面将结果写成与本文 $G$ 一致的形式。

### 5.1 两个展开与一个消元

固定 $Q=P_{\theta_0}$，设路径
$\theta(\tau)=\theta_0+\tau h+O(\tau^2)$，$\tau\downarrow0$。定义序列 score、Fisher 矩阵和任务梯度：

$$
u=\left.\nabla_\theta\log P_\theta(Y\mid X)\right|_{\theta_0},
\qquad F=\mathbb E_Q uu^\top,\qquad
g=\nabla_\theta G(\theta_0).
$$

概率归一化使 KL 的一阶项为零，得到

$$
K(\theta(\tau),\theta_0)=\frac{\tau^2}{2}h^\top Fh+O(\tau^3).
$$

正确率期望则满足

$$
G(\theta(\tau))-G_0=\tau g^\top h+O(\tau^2).
$$

当 $h^\top Fh>0$，令 $d=\sqrt K$ 并消去 $\tau$：

$$
\boxed{G(d)=G_0+
\underbrace{\frac{g^\top h}{\sqrt{h^\top Fh/2}}}_{m}
d+O(d^2).}
$$

固定正确性标签在有限输出空间上的期望是概率的线性组合，因此可以随光滑模型参数变化。这里展开的是期望采样正确率；经验分数是它的带噪声估计，greedy acc 对应另一种解码函数。

### 5.2 斜率表达了什么

$m$ 把任务收益方向 $g$ 与更新方向 $h$ 的对齐，除以该更新造成的分布位移：

| $g^\top h$ | 起点附近的任务变化         |
| ---------- | -------------------------- |
| 正         | 随 $\sqrt K$ 一阶提高      |
| 负         | 随 $\sqrt K$ 一阶降低      |
| 零         | 一阶项消失，由更高阶项决定 |

设每题参考正确率 $b_x=G(Q\mid x)$，$V=\mathbb E_xb_x(1-b_x)$。由于 $\mathbb E[u\mid x]=0$，

$$
g^\top h=\mathbb E_Q[(c-b_X)(u^\top h)].
$$

Cauchy–Schwarz 给出

$$
(g^\top h)^2\leq Vh^\top Fh,\qquad
|m_{\mathrm{seq}}|\leq\sqrt{2V}\leq1/\sqrt2.
$$

这把局部斜率与第 3 节的幅度界连接起来。若横轴改为固定 $H$ 的 token KL 平方根，斜率乘 $\sqrt H$。

### 5.3 线性项为零时的曲线

取两个正确输出和一个错误输出：

$$
Q=(1/4,1/4,1/2),\qquad
P_\tau=(1/4+\tau,\ 1/4-\tau-\tau^2,\ 1/2+\tau^2).
$$

小正 $\tau$ 下概率合法，且

$$
G(P_\tau)-G(Q)=-\tau^2,\qquad
K(P_\tau,Q)=4\tau^2+O(\tau^3).
$$

因此

$$
G(P_\tau)=G(Q)-\frac14d^2+O(d^3).
$$

这是零一阶对齐的具体例子。长区间保持近似直线，则要求更新方向、Fisher 几何和任务对齐在该区间相对稳定；这正是有限训练窗口需要由实验检查的性质。[^bao]

## 6. KL 预算下的最优正确率前沿

KL 约束下的指数重加权有成熟的策略优化背景。REPS 从相对熵约束推导策略更新；DPO 的 Eq. (4) 与附录 A.1 给出一般奖励下的条件 Gibbs 最优策略。[^reps][^dpo] 本节把该结构具体化为二元正确性，推导系数 $V$。

### 6.1 指数倾斜的解

固定 $Q$ 和 prompt 边际，允许在参考支持内自由选择回答分布。对 $\eta>0$，定义

$$
P_\eta(y\mid x)=
\frac{Q(y\mid x)e^{\eta c(x,y)}}{Z_x(\eta)},\qquad
Z_x(\eta)=1-b_x+b_xe^\eta.
$$

它把正确回答的概率统一乘 $e^\eta$ 后重新归一化。DPO 使用的 KL 惩罚系数 $\beta$ 与此处的倾斜强度对应为 $\eta=1/\beta$。[^dpo]

对任意共享 prompt 边际的 $P$，

$$
D_{KL}(\bar P\Vert\bar P_\eta)
=K(P,Q)-\eta G(P)+\mathbb E_x\log Z_x(\eta)\geq0.
$$

令 $P=P_\eta$ 时取等号。因此，若 $K(P,Q)\leq K(P_\eta,Q)$，就有
$G(P)\leq G(P_\eta)$：$P_\eta$ 达到该预算下的最优正确率。

### 6.2 小预算展开

令 $V=\mathbb E_x b_x(1-b_x)>0$。倾斜后每题的成功概率为

$$
b_x(\eta)=\frac{b_xe^\eta}{1-b_x+b_xe^\eta}
=b_x+\eta b_x(1-b_x)+O(\eta^2).
$$

对 prompt 求均值并展开 KL：

$$
G(P_\eta)-G(Q)=\eta V+O(\eta^2),
$$

$$
K(P_\eta,Q)
=\eta G(P_\eta)-\mathbb E_x\log Z_x(\eta)
=\frac12\eta^2V+O(\eta^3).
$$

小正 $\eta$ 区间内 KL 严格递增，可以反解预算，得到

$$
\boxed{G_{\max}(K)-G(Q)=\sqrt{2VK}+O(K).}
$$

单题时 $V=b(1-b)$；固定 horizon 的 token 坐标下，系数为 $\sqrt{2VH}$。这个前沿对应真实正确率目标和自由条件分布，实际训练路径则由其目标、模型参数化与优化器共同决定。

### 6.3 上界、路径与前沿分别回答什么

| 对象     | 研究问题                               | 数学量                               |
| -------- | -------------------------------------- | ------------------------------------ |
| 允许范围 | 给定分布距离，成绩最多可能相差多少     | $\lvert\Delta G\rvert\leq\sqrt{K/2}$ |
| 指定路径 | 某个更新方向起初怎样改变成绩           | $G=G_0+m\sqrt K+O(K)$                |
| 最优前沿 | 在 KL 预算内直接优化正确率，可达到什么 | $G_{\max}-G_0=\sqrt{2VK}+O(K)$       |

前沿是可达最优值，路径是具体算法的轨迹。这个区别也解释了为什么 proxy 优化可以在更大位移下回落，而放宽预算后的最优可达值仍然单调不减。

## 7. Teacher 监督的局部任务收益

本节把 RKL 的条件 logit 梯度与正确率梯度连接起来。条件 KL 的梯度形式可参照 Piskorz 等附录 B。[^dynamics]

令 $p=\operatorname{softmax}(z)$，正确性奖励向量为 $r$，teacher 为 $q$：

$$
L_q=D_{KL}(p\Vert q),\qquad
\frac{\partial L_q}{\partial z_i}
=p_i\bigl(\log(p_i/q_i)-L_q\bigr),
$$

$$
(\nabla_zG)_i=p_i(r_i-G).
$$

沿负 RKL 梯度更新时，

$$
\left.\frac{dG}{d\eta}\right|_0
=-\nabla_zG^\top\nabla_zL_q.
$$

这个内积刻画 teacher 给当前 student 的局部任务收益；teacher 自己的正确率 $q^\top r$ 刻画另一件事。

### 三动作示例

设 $c_1,c_2$ 正确，$w$ 错误：

| 分布      | $c_1$ | $c_2$ |  $w$ | 自身正确率 |
| --------- | ----: | ----: | ---: | ---------: |
| Student   |  0.60 |  0.01 | 0.39 |       0.61 |
| Teacher A |  0.01 |  0.89 | 0.10 |       0.90 |
| Teacher B |  0.70 |  0.10 | 0.20 |       0.80 |

A 的主要正确动作是 $c_2$，而 student 很少选择它；A 同时降低了 student 已熟悉的正确动作 $c_1$ 的相对地位。B 更支持 student 已有的正确行为。代入上式：

$$
\left.\frac{dG_A}{d\eta}\right|_0=-0.308164,\qquad
\left.\frac{dG_B}{d\eta}\right|_0=+0.090602.
$$

这个正支持的 softmax 构造展示了：teacher 的自身分数排序与局部教学收益排序可以不同。它只刻画起点的单步方向；训练后的峰值另由后续路径决定。

本文的构造例和有限差分核验保存在[数值结果](data/gold-kl-20261031/numerical-checks.json)与[计算脚本](data/gold-kl-20261031/verify_examples.py)，便于复算代数关系。

## 8. 把关系落实到实验指标

### 8.1 每个指标记录四项身份

| 身份     | 需要记录的内容                         | 作用                            |
| -------- | -------------------------------------- | ------------------------------- |
| 比较对象 | 当前 student／teacher／初始化          | 决定是位移还是拟合残差          |
| 访问分布 | 固定 S0／teacher／current student      | 决定条件风险是否对应目标序列 KL |
| 归一化   | 累计／固定 horizon／有效 token／逐响应 | 决定数值尺度及与序列量的关系    |
| 评分过程 | prompt 分布、解码器、终止和判分规则    | 决定 $G$ 对应的事件与输出分布   |

数学条件集中在这里验收，后续图表即可使用简洁、固定的指标名。固定银行评价共同输入上的拟合；current 银行评价当前行为；task score 评价规定评分过程的结果。

### 8.2 对应文献怎样读

| 文献                      | 阅读位置                     | 在本文中的用途                   |
| ------------------------- | ---------------------------- | -------------------------------- |
| van Erven–Harremoës，2014 | Theorem 9、§III-H            | 数据处理与参数 KL 的局部性质     |
| Sason–Verdú，2016         | 引言 Eq. (1)、变差与 KL 章节 | Pinsker 界及常数约定             |
| Ollivier 等，2017         | §2.1.2，Eq. (2)              | Fisher 二阶展开                  |
| REPS，2010                | Eq. (9)、附录                | 相对熵约束与指数策略更新的背景   |
| DPO，2023                 | Eq. (4)、附录 A.1            | 条件 Gibbs 最优策略              |
| Mroueh，2024              | §3–4                         | 奖励改善的运输不等式             |
| Bai 等，2022              | §4.3                         | 平方根 KL 与奖励的经验／局部解释 |
| Bao 等，2026              | 附录 D                       | OPD 中的局部收益坐标             |
| Piskorz 等，2026          | 附录 B、C.5                  | 条件梯度及 sequence/token 配对   |
| Liu 等，2026              | §2–3                         | 真实轨迹风险与历史访问分布       |

第 2–4 节是经典关系在语言模型评分中的展开；第 5 节结合标准 Fisher 几何与任务收益；第 6 节把 Gibbs 策略具体化为二元正确率的小预算展开；第 7 节是为解释教学方向构造的例子。原始检索与章节核验见[来源清单](sources/accuracy-kl-20261031/reviewed-sources.json)和[完整检索](sources/accuracy-kl-20261031/完整原始检索.md)。

### 参考文献

[^erven]: Tim van Erven, Peter Harremoës. Rényi Divergence and Kullback–Leibler Divergence. IEEE Transactions on Information Theory, 2014；arXiv:1206.2459v2. [Theorem 9,§III-H](https://arxiv.org/html/1206.2459v2)。

[^sason]: Igal Sason, Sergio Verdú. f-Divergence Inequalities. IEEE Transactions on Information Theory, 62(11):5973–6006, 2016；arXiv:1508.00335v7. [引言 Eq. (1) 及变差界](https://arxiv.org/html/1508.00335v7)。本文 TV 采用半个 L1 距离。

[^igo]: Yann Ollivier, Ludovic Arnold, Anne Auger, Nikolaus Hansen. Information-Geometric Optimization Algorithms: A Unifying Picture via Invariance Principles. JMLR 18(18):1–65, 2017. [§2.1.2，PDF 第 7 页，Eq. (2)](https://www.jmlr.org/papers/volume18/14-467/14-467.pdf)。

[^reps]: Jan Peters, Katharina Mülling, Yasemin Altun. Relative Entropy Policy Search. AAAI 2010. [Eq. (9) 与附录](https://ojs.aaai.org/index.php/AAAI/article/download/7727/7588)。该文包含 MDP 一致性约束；本文第 6 节使用固定 prompt 的条件分布问题。

[^dpo]: Rafael Rafailov et al. Direct Preference Optimization: Your Language Model Is Secretly a Reward Model. NeurIPS 2023；核对 arXiv:2305.18290v3. [Eq. (4)、附录 A.1](https://arxiv.org/html/2305.18290v3)。一般奖励下 KL 正则化最优策略。

[^mroueh]: Youssef Mroueh. Information Theoretic Guarantees For Policy Alignment In Large Language Models. arXiv:2406.05883v1, 2024. [§3–4](https://arxiv.org/html/2406.05883v1)。次高斯奖励的运输不等式与 proxy/gold 关系。

[^bai]: Yuntao Bai et al. Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback. arXiv:2204.05862, 2022. [§4.3](https://arxiv.org/html/2204.05862)。平方根 KL 与奖励的经验关系及局部解释。

[^bao]: Bao et al. Scaling Properties of Same-Family On-Policy Distillation. arXiv:2609.32722v1, 2026. [附录 D](https://arxiv.org/html/2609.32722v1)。OPD 的局部几何与经验训练窗口。

[^dynamics]: Piskorz, Berthon, van der Schaar. On-Policy or Off-Policy Learning? A Systematic Study of Distillation Dynamics. arXiv:2609.35259v1, 2026. [附录 B、C.5，Proposition C.4](https://arxiv.org/html/2609.35259v1)。

[^rk-theory]: Liu, Wei, Zimmert. Reverse-KL Policy Distillation: Statistical Guarantees and Limits. arXiv:2610.30844v2, 2026. [§2–3](https://arxiv.org/html/2610.30844v2)。真实轨迹 RKL 与访问分布条件。

<a id="convergence-examples"></a>

## 固定前缀收敛的数值例子

以下例子来自[有限分布验算](../../../../../wandb-logs/MirrorOPD-verl/analysis/teacher-convergence-20261024/mathematical-checks.json)与[验证脚本](../../research/checks/objectives/verify_sequence_kl.py)，用于解释条件风险、序列风险和行为指标的区别。

同一组当前策略轨迹上，token 平均 KL 为 8×10⁻⁴、平均有效长度为 5700 时，平均序列 KL 和为 4.56 nats。这是量纲示例，条件是轨迹、mask 和停止规则一致。

[核心稿的两步树反例](../../../../../MirrorOPD/theory/core/appendices/g_dynamics.tex)中，各前缀的条件 KL 从[4.5032,0,69.0775]下降到[4.3759,0,68.7778]，序列 KL 却从 5.1940 上升到 6.0745。学生对高残差分支的访问增加，抵消了局部拟合收益。

两个 Bernoulli 分布 p=(0.9,0.1)、T=(0.1,0.9) 的熵都为 0.32508 nats，KL 为 1.75778 nats。熵和任务成绩描述特定统计量；完整分布差异由相应概率距离衡量。交叉熵 H(P,T) 需要在学生轨迹上评分，而非使用教师自身的熵。
