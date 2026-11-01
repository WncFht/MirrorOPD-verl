# 镜像下降与 weighted teacher 目标

先在普通对数信号下区分 MD、近端点及两种 NoIS，再讨论一般 κ/η 形状、冻结目标和有限步拟合。两部分共用[符号约定](../符号约定.md)，保留各自假设、完整证明、反例和数值验算；κ=0 的系数换算不直接推广到所有非线性信号。科学比较见[TOP-D 复现与消融](../../experiments/learning-signal/weighted-targets.md)。

<a id="logarithmic"></a>

<a id="logarithmic-镜像下降与离策略四种更新"></a>

## 对数信号：MD、近端点与离策略更新

本篇在固定前缀下，分析 current/frozen 信号与 IS/NoIS 的四种组合加入 $\eta^{-1}\mathrm{KL}(p\Vert q)$ 后的目标、梯度和最优解。Mirror Descent（镜像下降，MD）冻结本轮线性信号；KL proximal-point（KL 近端点）保留当前非线性目标。二者在本问题中都具有几何插值解，但相同 $\eta$ 对应不同插值权重。

基础四格与完整词表计算图分别见 [OPD 目标与梯度](../fixed-prefix.md#objectives-four-cases)和 [Full-vocab 计算图](../fixed-prefix.md#graphs)。本篇的四格按更新开关划分；核心稿的 MDPO、Part、BPO、Mean 按内层拟合损失划分，两者不是一一对应关系。

<a id="logarithmic-1-条件符号与两种-nois-定义"></a>

### 1. 条件、符号与两种 NoIS 定义

固定前缀和有限词表 $\mathcal V$。教师 $T$、学生锚点 $q$、行为分布 $\beta$ 整轮冻结且严格为正；当前学生 $p=p_\theta$ 严格为正并归一化。所有自由分布最优解都指在概率单纯形上的优化；参数化学生未必能够表示这些解。推导不含 clipping、whitening、任务 reward、额外熵项或前缀分布导数。

采用[统一符号](../符号约定.md)，定义

$$
r_a=\log\frac{T_a}{q_a},\qquad
U_a=\log\frac{p_a}{q_a},\qquad
\psi_a=\nabla_\theta\log p_a,
$$

$$
A^{\rm anchor}_a=r_a,\qquad
A^{\rm cur}_a=r_a-U_a=\log\frac{T_a}{p_a}.
$$

用 $D_T(p)=\mathrm{KL}(p\Vert T)$、$D_q(p)=\mathrm{KL}(p\Vert q)$ 表示本篇两个标量目标，避免占用核心稿中冻结教师中心 $K_T$ 和反方向学生中心 $K_S=\mathrm{KL}(q\Vert p)$。令 $c=1/\eta$，$\eta>0$。几何插值解释主要取 $0<\eta\le1$；$\eta$ 不是参数优化器学习率。

在 full-vocab 中，“IS”对应当前分布 $p$ 加权，“NoIS”对应行为分布 $\beta$ 加权；前者可以直接完整求和，无需显式计算 ratio。加入正则后，NoIS 有两个不同定义：

1. 仅 OPD 信号使用 $\beta$ 测度，KL 正则保持精确求导。第 2–5 节使用此定义。
2. 先将 OPD 与 KL 正则合为解析 score 残差，再将整个残差的 $p$ 测度改为 $\beta$。第 6 节单独推导。

这两个定义在离开锚点后一般给出不同梯度。前向数值、被停止梯度的系数和积分测度必须同时写明。

<a id="logarithmic-exact-regularizer"></a>

<a id="logarithmic-2-精确-kl-正则下的四种目标"></a>

### 2. 精确 KL 正则下的四种目标

沿用 OPD 目标篇的未中心化信号，在每个标量目标上加 $cD_q$，得到

| 信号    | 当前分布加权，等价于 IS    | 行为分布加权，等价于 NoIS                          |
| ------- | -------------------------- | -------------------------------------------------- |
| Frozen  | $F_{\rm F,I}=D_T+(c-1)D_q$ | $F_{\rm F,N}=-\mathbb E_\beta[r\log p]+cD_q$       |
| Current | $F_{\rm C,I}=D_T+cD_q$     | $F_{\rm C,N}=\frac12\mathbb E_\beta[(U-r)^2]+cD_q$ |

证明使用基础恒等式

$$
-\mathbb E_p r=D_T-D_q,\qquad
\nabla D_q=\mathbb E_p[U\psi],
$$

以及固定测度下的求导

$$
\nabla\frac12\mathbb E_\beta[(U-r)^2]
=\mathbb E_\beta[(U-r)\psi],\qquad
\nabla[-\mathbb E_\beta(r\log p)]=-\mathbb E_\beta[r\psi].
$$

其中 $\nabla D_q$ 的常数项通过 $\mathbb E_p\psi=0$ 消去。因此四个梯度分别为

$$
\begin{aligned}
\nabla F_{\rm F,I}&=\mathbb E_p[(cU-r)\psi],\\
\nabla F_{\rm C,I}&=\mathbb E_p[((c+1)U-r)\psi],\\
\nabla F_{\rm F,N}&=-\mathbb E_\beta[r\psi]+c\mathbb E_p[U\psi],\\
\nabla F_{\rm C,N}&=\mathbb E_\beta[(U-r)\psi]+c\mathbb E_p[U\psi].
\end{aligned}
$$

NoIS 两格的 OPD 项与正则项使用不同测度；不能把它们自动合并为一个 $\beta$ 加权残差。若用样本 $a\sim\beta$ 估计当前测度项，完整行为权重为 $p_a/\beta_a$。锚点与行为策略不同并不改变正则参考 $q$，只改变采样分母。

<a id="logarithmic-md-solution"></a>

<a id="logarithmic-3-frozen--is标准镜像下降"></a>

### 3. Frozen + IS：标准镜像下降

<a id="logarithmic-31-从-reverse-kl-的线性化得到子问题"></a>

#### 3.1 从 reverse KL 的线性化得到子问题

关于概率坐标，原目标在锚点的梯度为

$$
\left.\frac{\partial D_T}{\partial p_a}\right|_{p=q}
=\log\frac{q_a}{T_a}+1=-r_a+1.
$$

在约束 $\sum_a(p_a-q_a)=0$ 下，线性项中的动作无关常数消去。负熵镜像几何对应 $\mathrm{KL}(p\Vert q)$，因此子问题为

$$
\min_{p\in\Delta}\left\{-\mathbb E_p r+cD_q(p)\right\}
=\min_{p\in\Delta}F_{\rm F,I}(p).
$$

这里整轮冻结 $q,T,r$，并在内层重算当前 $p,U$。这与[核心稿的镜像问题](../../../../../../MirrorOPD/theory/core/sections/01_mirror.tex)在 $\kappa=0$ 时一致。将 $r$ 换为 $C_qr$ 只改变当前测度下目标的常数。

<a id="logarithmic-32-最优分布与-kl-分解"></a>

#### 3.2 最优分布与 KL 分解

引入归一化乘子 $\lambda$，内点的一阶条件为

$$
-r_a+c\left(\log\frac{p_a}{q_a}+1\right)+\lambda=0.
$$

解得 $p_a=q_a\exp(\eta r_a-1-\eta\lambda)$。归一化后

$$
Z_\eta=\sum_aq_ae^{\eta r_a},\qquad
m_a=\frac{q_ae^{\eta r_a}}{Z_\eta}
=\frac{q_a^{1-\eta}T_a^\eta}{Z_\eta}.
$$

由 $\log(p_a/m_a)=U_a-\eta r_a+\log Z_\eta$ 得

$$
F_{\rm F,I}(p)
=\frac1\eta\mathrm{KL}(p\Vert m)-\frac{\log Z_\eta}{\eta}.
$$

$m$ 严格为正，KL 非负且仅在 $p=m$ 时为零，所以 $m$ 是闭单纯形上的唯一最优分布。这也证明存在性与全局最优性，而非只验证驻点。$0<\eta<1$ 时为几何插值，$\eta=1$ 时 $m=T$；$\eta>1$ 的同一公式仍定义正分布，但属于外推。

冻结教师信号不表示总更新系数冻结。该子问题的损失梯度使用残差 $cU-r$；其 reward 形式 $r-cU$ 随当前学生变化。

<a id="logarithmic-proximal-point"></a>

<a id="logarithmic-4-current--iskl-近端点与系数换算"></a>

### 4. Current + IS：KL 近端点与系数换算

Current + IS 保留完整 $D_T$ 后加正则：

$$
F_{\rm C,I}=D_T+cD_q.
$$

这是 KL 近端点子问题。一阶条件为

$$
\log\frac{p_a}{T_a}+1
+c\left(\log\frac{p_a}{q_a}+1\right)+\lambda=0.
$$

令 $\alpha=1/(c+1)=\eta/(1+\eta)$，归一化得到

$$
m^{\rm prox}_a
=\frac{q_ae^{\alpha r_a}}{Z_\alpha}
\propto q_a^{1-\alpha}T_a^\alpha.
$$

直接代入可得

$$
F_{\rm C,I}
=(c+1)\mathrm{KL}(p\Vert m^{\rm prox})
-(c+1)\log Z_\alpha.
$$

因此 $m^{\rm prox}$ 是唯一自由分布最优解。相同数值 $\eta$ 下，MD 的教师指数为 $\eta$，近端点的指数为 $\eta/(1+\eta)$。后者等价于步长为 $\alpha$ 的标准 MD 子问题。两者在 $p=q$ 处具有相同初始参数梯度，但后续内层轨迹和目标教师不同。

同一个 MD 目标也能用 current 信号表达，只需同时换算正则系数：

$$
\boxed{
-\mathbb E_p r+\frac1\eta D_q
=D_T+\left(\frac1\eta-1\right)D_q.
}
$$

对应的 reward 系数恒等式为

$$
r-\frac1\eta U
=A^{\rm cur}-\left(\frac1\eta-1\right)U.
$$

所以 current/frozen 切换且保持 $1/\eta$ 不变会改变目标；切换信号并减去一份正则系数则完全等价。$\eta=1$ 时，冻结信号加一份 KL 等价于无额外正则的 current reverse KL。

例如 $q=(1/2,1/2)$、$T=(0.8,0.2)$、$\eta=1$，标准 MD 目标教师为 $(0.8,0.2)$，current 加相同正则的目标教师为 $(2/3,1/3)$。

<a id="logarithmic-nois-exact-kl"></a>

<a id="logarithmic-5-仅-opd-项-nois-的两个目标"></a>

### 5. 仅 OPD 项 NoIS 的两个目标

<a id="logarithmic-51-current带-kl-正则的对数概率回归"></a>

#### 5.1 Current：带 KL 正则的对数概率回归

目标为

$$
F_{\rm C,N}
=\frac12\sum_a\beta_a\left(\log\frac{p_a}{T_a}\right)^2
+c\sum_ap_a\log\frac{p_a}{q_a}.
$$

关于概率坐标的一阶条件是

$$
\frac{\beta_a}{p_a}\log\frac{p_a}{T_a}
+c\left(\log\frac{p_a}{q_a}+1\right)+\lambda=0.
$$

该方程包含 $\beta_a/p_a$，一般不具有前两节的几何插值解。严格正的 $\beta,T$ 使任何概率坐标趋于零时平方误差趋于正无穷；因此连续目标在某个内点紧子集上取得最小值。这保证存在内点最优解，但本篇不据此断言唯一性或参数空间凸性。

<a id="logarithmic-52-frozen带符号交叉熵仍可无下界"></a>

#### 5.2 Frozen：带符号交叉熵仍可无下界

目标为

$$
F_{\rm F,N}=-\sum_a\beta_ar_a\log p_a+cD_q(p).
$$

若 $q\ne T$，归一化要求至少一个动作满足 $T_a<q_a$，即 $r_a<0$。选定该动作 $a_0$，令 $p_{a_0}=\epsilon\to0$，其余概率按固定正分布分配剩余质量。于是

$$
-\beta_{a_0}r_{a_0}\log\epsilon\to-\infty.
$$

其余交叉熵项有有限极限。由于 $q$ 严格为正，使用 $0\log0=0$ 的连续延拓，$D_q$ 在闭单纯形上有限且有界，因此任意有限 $c>0$ 都不能抵消这一发散。$F_{\rm F,N}$ 在上述条件下无有限下界。

例取 $\beta=q=(1/2,1/2)$、$T=(0.8,0.2)$、$\eta=1$、$p=(1-\epsilon,\epsilon)$：

| $\epsilon$ | $F_{\rm F,N}$ |
| ---------- | ------------: |
| $10^{-2}$  |     -1.470330 |
| $10^{-6}$  |     -5.636380 |
| $10^{-12}$ |    -11.965877 |

这是该自由分布目标的性质，不是有限步神经网络训练必然发散的实验结论。若通过约束强制所有 $p_a$ 保持统一正下界，优化域已改变，不能继续使用本节的边界论证。

<a id="logarithmic-whole-residual"></a>

<a id="logarithmic-6-整个-md-残差一起取消-is"></a>

### 6. 整个 MD 残差一起取消 IS

令 $\delta=0$ 表示 frozen，$\delta=1$ 表示 current，并记 $\gamma=c+\delta>0$。第 2 节两种当前测度梯度统一为

$$
g_{\delta,\rm I}=\mathbb E_p[(\gamma U-r)\psi]
=\mathbb E_\beta\left[\frac p\beta(\gamma U-r)\psi\right].
$$

这里已经利用 $\mathbb E_p\psi=0$ 消去了 KL 求导的常数项。若从这个解析 score 形式取消整个 ratio，新的更新定义为

$$
g^{\rm whole}_{\delta,\rm N}
=\mathbb E_\beta[(\gamma U-r)\psi].
$$

由于 $\nabla U=\psi$、$r,\beta,\gamma$ 固定，它具有普通标量势函数

$$
H_\delta(p)=\frac1{2\gamma}\mathbb E_\beta[(\gamma U-r)^2],\qquad
\nabla H_\delta=g^{\rm whole}_{\delta,\rm N}.
$$

| 信号    | 整个残差 NoIS 的真实目标                                              |
| ------- | --------------------------------------------------------------------- |
| Frozen  | $H_0=\frac1{2\eta}\mathbb E_\beta[(U-\eta r)^2]$                      |
| Current | $H_1=\frac{1+\eta}{2\eta}\mathbb E_\beta[(U-\frac{\eta}{1+\eta}r)^2]$ |

两者均为有下界的回归目标，与第 5 节的精确 KL 正则目标不同。以 frozen 为例，两种 NoIS 定义的梯度之差为

$$
\nabla F_{\rm F,N}-\nabla H_0
=c\left(\mathbb E_p[U\psi]-\mathbb E_\beta[U\psi]\right).
$$

如果取消 ratio 前保留了先前可在 $p$ 测度下消去的常数 $+1$，又会得到额外的 $c\mathbb E_\beta\psi$。因此本节明确规定从化简后的解析残差出发；“NoIS”这个配置名本身不能唯一决定目标。

<a id="logarithmic-normalization"></a>

<a id="logarithmic-7-归一化常数中心化与共同目标教师"></a>

### 7. 归一化常数、中心化与共同目标教师

<a id="logarithmic-71-未归一化残差的偏离"></a>

#### 7.1 未归一化残差的偏离

令 $\alpha=1/\gamma$，$m_\alpha\propto qe^{\alpha r}$。其最优性方程是

$$
U^\star=\alpha r-\log Z_\alpha.
$$

把 $p=m_\alpha$ 代入第 6 节残差，得到 $\gamma U^\star-r=-\gamma\log Z_\alpha$，所以

$$
\left.\nabla H_\delta\right|_{p=m_\alpha}
=-\gamma\log Z_\alpha\,\mathbb E_\beta[\psi]_{p=m_\alpha}.
$$

右侧一般非零，故未归一化回归通常不与 MD/近端点共享最优分布。特殊情形可以重合：例如原始未中心化信号、$\alpha=1$ 时 $Z_1=\sum_aT_a=1$，该残差可以在 $p=T$ 精确归零。不能把“一般偏离”改写成“所有参数下都偏离”。

<a id="logarithmic-72-完整最优性方程回归"></a>

#### 7.2 完整最优性方程回归

将归一化常数补回，定义

$$
H_{\rm Part,\beta}
=\frac\gamma2\mathbb E_\beta[(U-\alpha r+\log Z_\alpha)^2]
=\frac\gamma2\mathbb E_\beta[\log^2(p/m_\alpha)].
$$

该目标非负，在 $p=m_\alpha$ 时为零。因为 $\beta$ 完整正支持，零损失强制每个动作上 $p_a=m_{\alpha,a}$，故自由分布最优解唯一。本篇系数用于与第 6 节梯度尺度对应；核心稿 Part 定义采用另一整体尺度。

<a id="logarithmic-73-中心化残差回归"></a>

#### 7.3 中心化残差回归

定义 $C_\beta h=h-\mathbb E_\beta h$。无需显式计算 $Z_\alpha$ 的目标为

$$
H_{\rm BPO,\beta}
=\frac\gamma2\operatorname{Var}_\beta(U-\alpha r).
$$

令 $R=U-\alpha r$。直接求导得

$$
\nabla H_{\rm BPO,\beta}
=\gamma\mathbb E_\beta[(C_\beta R)(\psi-\mathbb E_\beta\psi)]
=\gamma\mathbb E_\beta[(C_\beta R)\psi],
$$

第二个等号使用 $\mathbb E_\beta C_\beta R=0$。零方差要求 $U_a-\alpha r_a$ 在所有动作上为同一常数；归一化再确定这个常数是 $-\log Z_\alpha$，因此唯一最优分布仍是 $m_\alpha$。

当 frozen、$\beta=q$ 时，$\gamma=1/\eta$，本式成为核心稿的 BPO。Part、BPO 与 MD 的共同最优解结论依赖完整支持与可实现性；共享神经网络参数受到表达能力限制时，不同目标可能产生不同投影。

<a id="logarithmic-74-核心稿-mean-与-baseline-的关系"></a>

#### 7.4 核心稿 Mean 与 baseline 的关系

核心稿冻结优势为 $\mathcal A_k=C_qr$，$G=\eta C_qr$。对 centered 信号执行整个残差 NoIS，且 $\beta=q$，得到 $\eta^{-1}L_{\rm Mean}$，其中

$$
L_{\rm Mean}=\frac12\mathbb E_q[(U-G)^2].
$$

记 $K_S=\mathrm{KL}(q\Vert p)=-\mathbb E_qU$。由于

$$
U-G=C_q(U-\eta r)-K_S,
$$

中心化项的 $q$ 均值为零，平方展开得到

$$
L_{\rm Mean}=\eta L_{\rm BPO}+\frac12K_S^2.
$$

额外的 $K_S^2/2$ 不是常数。对非恒定 $G$，在 $m\propto qe^G$ 处有 $U-G=-\Delta$，其中 $\Delta=\log\mathbb E_qe^G>0$。Mean 的自由分布一阶条件为 $U_a-G_a+\lambda e^{U_a}=0$；代入后要求 $\lambda e^{G_a-\Delta}=\Delta$ 对全部动作成立，非恒定 $G$ 无法满足，故 Mean 最优分布不等于 $m$。

这也解释 baseline 的测度依赖：在 $p$ 测度下，冻结信号加动作无关常数不改变梯度；在 $\beta$ 测度下通常会改变梯度。若选定的目标本来按 $q$ 定义，而实际样本来自 $\beta\ne q$，估计 Part/BPO/Mean 的 $q$ 期望仍需要 $q/\beta$；直接全词表求和则使用 $q$ 权重。

<a id="logarithmic-general-signal"></a>

<a id="logarithmic-8-与一般-mirroropd-信号的连接"></a>

### 8. 与一般 MirrorOPD 信号的连接

本篇 current/frozen 的系数换算使用了 $A^{\rm cur}=r-U$，适用于普通对数 OPD 信号。核心稿的一般冻结信号为 $f=\eta A_{\kappa,\eta}(T/q)$，其标准 MD 目标仍为

$$
L_{\rm MDPO}=\frac1\eta\{D_q-\mathbb E_pf\},\qquad
m_a=\frac{q_ae^{f_a}}{Z},\qquad Z=\mathbb E_qe^f.
$$

同样有

$$
L_{\rm MDPO}=\frac1\eta\mathrm{KL}(p\Vert m)-\frac{\log Z}{\eta},\qquad
\nabla L_{\rm MDPO}=\frac1\eta\mathbb E_p[(U-f)\psi].
$$

从这个残差去掉行为 IS 会得到 $\frac1{2\eta}\mathbb E_\beta[(U-f)^2]$ 的梯度；补 $\log Z$ 或中心化后可分别构造完整方程回归与 BPO。一般非线性信号若将 $q$ 替成 $p$，不再满足普通 log-ratio 的简单差分身份，因此不能直接套用“KL 系数减一”或 $\eta/(1+\eta)$ 的结论。完整定义见[核心稿内层拟合](../../../../../../MirrorOPD/theory/core/sections/04_inner.tex)。

<a id="logarithmic-9-论文关系与验算"></a>

### 9. 论文关系与验算

AsyncOPD 的 current/frozen 和 IS 消融提供普通 OPD 的比较背景；其固定前缀分析没有证明本篇所有正则化变体的任务效果。[^logarithmic-asyncopd] PPO 的冻结 advantage 属于局部策略改进 surrogate；它不使每一种冻结 OPD 信号自动成为原 KL 的梯度。[^logarithmic-ppo] 本篇四格正则目标、近端点换算和 NoIS 分类由明确写出的标量及计算图推导，未将其归为这些论文已经验证的算法。

[verify_offpolicy_objectives.py](../../../research/checks/objectives/verify_offpolicy_objectives.py)在 CPU FP64 下枚举有限词表，核对基础四格、精确正则四格、整个残差 NoIS、几何目标、归一化修正与 baseline 的测度依赖；固定例子展示无下界方向。原始结果保存在[data/offpolicy-objectives.json](../data/offpolicy-objectives.json)。这些是数值身份与反例检查，不替代本篇证明，也不提供语言模型准确率证据。

```bash
CUDA_VISIBLE_DEVICES='' python -m examples.mirroropd.research.checks.objectives.verify_offpolicy_objectives \
  --output examples/mirroropd/docs/theory/data/offpolicy-objectives.json
```

<a id="logarithmic-参考文献"></a>

#### 参考文献

[^logarithmic-asyncopd]: Kang et al. AsyncOPD: How Stale Can On-Policy Distillation Be? arXiv:2606.24143v1，2026，第 3、5 节及附录 F。[论文原文](https://arxiv.org/html/2606.24143v1)。

[^logarithmic-ppo]: Schulman et al. Proximal Policy Optimization Algorithms. arXiv:1707.06347v2，2017。[论文原文](https://arxiv.org/pdf/1707.06347)。

<a id="weighted"></a>

<a id="weighted-形状与插值目标"></a>

## 一般形状：目标教师、κ 与 η

<a id="weighted-1-范围来源与核心选择"></a>

### 1. 范围、来源与核心选择

本篇提取 `MirrorOPD/theory/core` 的 $\eta,\kappa$ 理论，源码基准为 `4be22560a6ca1253b6d92fe9f5e7104a06257b94`。符号遵循本仓[统一约定](../符号约定.md)。训练接口、数据生命周期、计算预算和验收要求见 [统一实现方案](../../infra/numerics/mirror-objectives/reference.md)。

| 核心稿来源                                                                                                                                                    | 本篇使用的内容                                       |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------- |
| [镜像问题与冻结监督](../../../../../../MirrorOPD/theory/core/sections/01_mirror.tex)                                                                          | $f,A,K_T,\mathcal A_k$ 与 MD 子问题                  |
| [目标教师](../../../../../../MirrorOPD/theory/core/sections/02_target.tex)                                                                                    | 归一化幂均值 $m_{\kappa,\eta}$、几何/平方根/算术端点 |
| [教师目标](../../../../../../MirrorOPD/theory/core/sections/03_objective.tex)                                                                                 | $J_{\kappa,\eta}$、凸性、下降与重心 $F_k$            |
| [内层拟合](../../../../../../MirrorOPD/theory/core/sections/04_inner.tex)                                                                                     | MDPO、Part、BPO、Mean 的损失、梯度与冻结语义         |
| [形状性质](../../../../../../MirrorOPD/theory/core/sections/05_parameters.tex)、[有限拟合](../../../../../../MirrorOPD/theory/core/sections/06_finite.tex)    | 覆盖、方差、局部速度与神经网络有限更新的限制         |
| [条件积分](../../../../../../MirrorOPD/theory/core/appendices/e_estimation.tex)、[中心估计](../../../../../../MirrorOPD/theory/core/appendices/f_centers.tex) | IS 的目标测度、支持条件与随机中心偏差                |

如果目标是实现核心稿的固定目标 MirrorOPD，最直接的 full-vocab 方案是：每轮冻结学生锚点 $q_k$，构造目标 teacher $m_k$，在内层优化 $\eta^{-1}\mathrm{KL}(p_{k,n}\Vert m_k)$。全文默认 state 固定；动作积分用完整词表求和，不需要额外 action IS，也不需要 PPO clip。

如果每次更新都把 $A(T/q_k)$ 改成 $A(T/p_{k,n})$，则进入直接优化 $J_{\kappa,\eta}(p;T)$ 的另一条路径。两条路径起点梯度相同，内层一般不同。本文将它们分别称为“固定目标 MDPO”与“current-J”，避免用一个 `current/frozen` 开关掩盖目标变化。

<a id="weighted-2-四种分布与两个参数"></a>

### 2. 四种分布与两个参数

<a id="weighted-symbols"></a>

以下固定前缀 $s$，省略条件记号。词表有限，$T,q,p$ 严格为正且归一化。

| 记号                         | 含义与刷新时机                                                        |
| ---------------------------- | --------------------------------------------------------------------- |
| $T$                          | 原始 teacher，训练期间冻结                                            |
| $q=q_k$                      | 第 $k$ 轮学生锚点，内层期间冻结                                       |
| $p=p_{k,n}=p_{\theta_{k,n}}$ | 当前学生，每次参数更新后重算                                          |
| $\beta$                      | 实际采样行为策略；旧数据、vLLM/FSDP 差异都可能使 $\beta\ne q$         |
| $m=m_k$                      | 由 $q,T,\kappa,\eta$ 定义的本轮目标 teacher，固定目标 MDPO 中整轮冻结 |
| $\widehat d_k$               | 本轮训练使用的经验前缀测度，内层期间固定                              |

核心稿正文取 $0\le\kappa\le1$、$0<\eta<1$；本文另讨论 $\eta=1$ 的连续端点。$\eta=0$ 的目标极限为 $q$，但含 $1/\eta$ 的训练表达式不能直接代入零。$\eta>1$ 属于外推，正形状存在额外正性条件，不列为初始实现范围。

- $\kappa$ 控制目标的**形状**：log 概率插值、平方根插值或概率插值。
- $\eta$ 控制目标的**插值强度**；一般也改变 $J_{\kappa,\eta}$ 本身。
- $\alpha_{\rm opt}$ 是 optimizer 学习率，$N$ 是内层更新次数，二者决定有限更新能走多远。

若需要在同一个 $J_{\kappa,\eta}$ 上独立改变镜像步长，另设 $\tau$，使用 $m^+\propto q\exp[\tau A_{\kappa,\eta}(T/q)]$。核心稿的幂均值闭式对应 $\tau=\eta$，不能一边固定信号中的 $\eta$，一边改变正则系数后仍声称是同一个幂均值目标。

<a id="weighted-3-从冻结信号到目标-teacher"></a>

### 3. 从冻结信号到目标 teacher

<a id="weighted-target"></a>

定义冻结的 teacher/anchor 对数比 $r_a=\log(T_a/q_a)$，以及

$$
f_{\kappa,\eta}(r)=
\begin{cases}
\kappa^{-1}\log(1-\eta+\eta e^{\kappa r}),&\kappa>0,\\
\eta r,&\kappa=0,
\end{cases}
\qquad A_a^q=A_{\kappa,\eta}(T_a/q_a)=f_a/\eta.
$$

冻结教师中心与中心化优势为

$$
K_T=-\mathbb E_q A^q,\qquad
\mathcal A_k=A^q+K_T,\qquad \mathbb E_q\mathcal A_k=0.
$$

核心稿的 MD 子问题是

$$
m=\arg\min_{p\in\Delta}
\left[-\mathbb E_p\mathcal A_k+\frac1\eta\mathrm{KL}(p\Vert q)\right].
$$

因为 $K_T$ 与候选 $p$ 无关，删去常数后等价于最小化

$$
L_{\rm MDPO}(p)=\frac1\eta\left[\mathrm{KL}(p\Vert q)-\mathbb E_p f\right].
$$

给归一化约束加入乘子 $\nu$，内点条件为

$$
\frac{\log(p_a/q_a)+1-f_a}{\eta}+\nu=0.
$$

因此 $p_a\propto q_ae^{f_a}$。定义 $Z=\sum_aq_ae^{f_a}$，得到

$$
\boxed{m_a=\frac{q_ae^{f_a}}Z},\qquad
\boxed{L_{\rm MDPO}(p)=\frac1\eta\mathrm{KL}(p\Vert m)-\frac{\log Z}{\eta}}.
$$

右边第二项整轮固定。$\mathrm{KL}(p\Vert m)\ge0$ 且仅在 $p=m$ 时为零，所以 $m$ 是唯一自由分布最优解；优化非负拟合损失 $L_{\rm fit}=\mathrm{KL}(p\Vert m)/\eta$ 与 MDPO 具有相同梯度。

代入信号后，$\kappa>0$ 时

$$
m_a=\frac{[(1-\eta)q_a^\kappa+\eta T_a^\kappa]^{1/\kappa}}
{\sum_b[(1-\eta)q_b^\kappa+\eta T_b^\kappa]^{1/\kappa}}.
$$

| 形状                        | 目标 teacher                                      |
| --------------------------- | ------------------------------------------------- |
| $\kappa=0$                  | $m_a\propto q_a^{1-\eta}T_a^\eta$                 |
| $\kappa=1/2$                | $m_a\propto[(1-\eta)\sqrt{q_a}+\eta\sqrt{T_a}]^2$ |
| $\kappa=1$                  | $m=(1-\eta)q+\eta T$，此时 $Z=1$                  |
| 任意上述 $\kappa$，$\eta=1$ | $m=T$，回到当前直接 Full-KL                       |

平方根情况的归一化常数是 $(1-\eta)^2+\eta^2+2\eta(1-\eta)\sum_a\sqrt{q_aT_a}$，通常不是 1。将“幂均值分子”当成已归一化概率会算错真实 KL；梯度中能消去一个动作无关常数，不意味着所有中间 normalizer 都可以删除。

<a id="weighted-4-不能混同的-jf-与内层拟合损失"></a>

### 4. 不能混同的 J、F 与内层拟合损失

<a id="weighted-objectives"></a>

<a id="weighted-41-原-teacher-目标-j"></a>

#### 4.1 原 teacher 目标 J

核心稿从信号反推出 $\partial_{p_a}J=-A_{\kappa,\eta}(T_a/p_a)$。令 $x=T_a/p_a$，寻找 $J=\sum_ap_a\omega(x)$，则

$$
\omega-x\omega'=-A,
\qquad (\omega/x)'=A(x)/x^2.
$$

取 $\omega(1)=0$ 得

$$
\omega(x)=x\int_1^x\frac{A(u)}{u^2}\,du,
\qquad J_{\kappa,\eta}(p;T)=\sum_ap_a\omega(T_a/p_a).
$$

又有 $\omega''(x)=x^{\kappa-2}/(1-\eta+\eta x^\kappa)>0$、$\omega'(1)=0$，故该条件目标非负、唯一最优点为原 teacher $T$。两个可读端点为

$$
J_{0,\eta}(p;T)=\mathrm{KL}(p\Vert T),\qquad
J_{1,\eta}(p;T)=\frac{\mathrm{JS}_\eta(p,T)}{\eta(1-\eta)},
$$

其中 $\mathrm{JS}_\eta=(1-\eta)\mathrm{KL}(p\Vert M_p)+\eta\mathrm{KL}(T\Vert M_p)$，$M_p=(1-\eta)p+\eta T$。$\eta\to1$ 的端点按极限理解，不能直接用上式分母为零的形式计算。

<a id="weighted-42-加权重心-f-与内层损失"></a>

#### 4.2 加权重心 F 与内层损失

核心稿另定义

$$
F_k(p)=(1-\eta)D_\kappa(q\Vert p)+\eta D_\kappa(T\Vert p),
$$

$$
D_\kappa(P\Vert Q)=\frac{1-\sum_aP_a^\kappa Q_a^{1-\kappa}}{\kappa(1-\kappa)},
\quad D_0(P\Vert Q)=\mathrm{KL}(Q\Vert P),\quad D_1(P\Vert Q)=\mathrm{KL}(P\Vert Q).
$$

其最优点也是 $m$，但中间形状的超额损失是 $Z^\kappa D_\kappa(m\Vert p)$；两个端点分别是 $\mathrm{KL}(p\Vert m)$ 和 $\mathrm{KL}(m\Vert p)$。因此，直接反传 $F_k$ 与反传 reverse KL 到 $m$ 只有共同自由分布最优解，通常不具有相同有限步梯度。

| 对象                            | 最优点         | 在实现中的用途                                     |
| ------------------------------- | -------------- | -------------------------------------------------- |
| $J_{\kappa,\eta}(p;T)$          | 原 teacher $T$ | 解释外层镜像方向；也可以选择直接优化它             |
| $F_k(p)$                        | 本轮目标 $m_k$ | 重心解释与有限拟合分析；直接优化时是独立 loss      |
| $L_{\rm MDPO}$ 或 $L_{\rm fit}$ | 本轮目标 $m_k$ | 固定目标 MDPO 的内层拟合损失                       |
| Part/BPO/Mean                   | 见第 7 节      | 不同残差拟合方法，不能仅按 current/frozen 标签分类 |

目标 teacher 和拟合损失是两个独立选择。用 $\kappa$ 改目标 teacher，不意味着自动将内层 reverse KL 替换为 $D_\kappa$。

<a id="weighted-5-fixed-advantage-与-current-advantage-的准确含义"></a>

### 5. Fixed advantage 与 current advantage 的准确含义

<a id="weighted-current-frozen"></a>

记 $U_a=\log(p_a/q_a)$、$\psi_a=\nabla_\theta\log p_a$。固定目标 MDPO 的梯度为

$$
\nabla L_{\rm MDPO}
=\frac1\eta\mathbb E_p[(U-f)\psi]
=-\mathbb E_p[B^{\rm MD}\psi],
\qquad B^{\rm MD}=A^q-\frac U\eta.
$$

推导中先对 $p(U-f)$ 求导，得到 $(U-f+1)\nabla p$；$\mathbb E_p\psi=0$ 消去 $+1$。$A^q$ 整轮固定，但当前学生的漂移项 $U$ 必须刷新，因此 **固定教师信号不等于固定整个更新系数**。

同一系数也可用目标 teacher 的当前 gap 表示：

$$
\frac{\log m_a-\log p_a}{\eta}
=A_a^q-\frac{U_a}{\eta}-\frac{\log Z}{\eta}.
$$

最后一项是动作无关常数，在完整 $p$ 积分或正确 action IS 下抵消。因此“对固定 $m_k$ 使用 current log-gap”和“固定原 teacher/anchor 信号并加入正确漂移惩罚”是同一个 MDPO 梯度。

另一种 current 定义是 $A^{\rm cur}_{\kappa,\eta}=A_{\kappa,\eta}(T/p)$。由第 4 节的概率梯度，

$$
\boxed{\nabla_\theta J_{\kappa,\eta}(p;T)
=-\mathbb E_p[A^{\rm cur}_{\kappa,\eta}\psi]}.
$$

这通常不同于固定目标 MDPO。若每次由当前 $p$ 构造 $m(p)\propto p\exp[\eta A(T/p)]$，将 $m(p)$ detach，再反传 $\mathrm{KL}(p\Vert\operatorname{sg}m(p))/\eta$，可以得到同一 current-J 一阶梯度。其前向 KL 数值却通常不是 $J$，应作为局部梯度代理单独命名；若不 detach $m(p)$，又会产生另一条梯度。

| 实现语义                                              | 梯度对应的目标                            |
| ----------------------------------------------------- | ----------------------------------------- |
| Frozen $A(T/q)$ + current $U/\eta$ 惩罚               | 固定目标 MDPO                             |
| Current $\log(m_k/p)/\eta$，$m_k$ 固定                | 同一个 MDPO                               |
| Current 非线性 $A(T/p)$，正确 $p$ 积分                | $J_{\kappa,\eta}(p;T)$                    |
| Current 非线性 $A(T/p)$ 再加 $c\mathrm{KL}(p\Vert q)$ | $J+c\mathrm{KL}(p\Vert q)$，另一近端目标  |
| 只有 frozen $A(T/q)$，没有漂移惩罚                    | 冻结线性目标 $-\mathbb E_pA^q$；不是 MDPO |

对 $\kappa=0$，存在特殊恒等式

$$
A^q=\log(T/q),\quad A^{\rm cur}=\log(T/p)=A^q-U,
$$

$$
\boxed{L_{\rm MDPO}=\mathrm{KL}(p\Vert T)+\left(\frac1\eta-1\right)\mathrm{KL}(p\Vert q)}.
$$

因此可用 current 原始 OPD 信号，再加系数 $1/\eta-1$ 的 anchor reverse KL 表达同一 MDPO。若仍加 $1/\eta$，得到的目标 teacher 是几何插值强度 $\eta/(1+\eta)$，而非 $\eta$。证明与普通 OPD 四格复用[已有 MD 文档](#logarithmic-proximal-point)。$\kappa>0$ 的非线性信号不满足上述差分恒等式，不能照搬“正则减一”。

<a id="weighted-6-固定-state-下的-off-policy-与动作-is"></a>

### 6. 固定 state 下的 Off-policy 与动作 IS

<a id="weighted-offpolicy"></a>

<a id="weighted-61-动作积分的目标测度"></a>

#### 6.1 动作积分的目标测度

目标积分测度为 $\nu$ 时，若 $\beta$ 覆盖所有非零贡献动作，则

$$
\sum_a\nu_ah_a=\mathbb E_{a\sim\beta}\left[\frac{\nu_a}{\beta_a}h_a\right].
$$

| 梯度目标                 | Full-vocab 权重 | 从 $\beta$ 抽动作时的权重 |
| ------------------------ | --------------- | ------------------------- |
| MDPO、current-J          | $p$             | $p/\beta$                 |
| 核心稿 Part/BPO/Mean     | 冻结 $q$        | $q/\beta$                 |
| 有意定义为行为测度的回归 | 冻结 $\beta$    | 1，但目标已经改变         |

**Full-vocab 在固定 state 上直接求和，不需要额外 action IS。** 它已经用目标要求的 $p$ 或 $q$ 加权。再对 full KL 乘一个实际采样 token 的 $p/\beta$，通常会改变目标；也不能因为动作来自旧策略，就把 KL 外部的 $p$ 换成 $\beta$。

Full-vocab 动作积分不依赖 $\beta$ 对整个词表具有支持，只需固定 state 上能计算完整目标分布。支持条件是 sampled IS 的要求：若 top-p 等截断让 $\beta_a=0$，采样估计不能用有限权重恢复该动作的非零贡献。

Sampled MDPO 的一种正确 surrogate 是 $-\mathbb E_\beta[(p/\beta)\operatorname{sg}B^{\rm MD}]$，让 ratio 求导；另一种是 $-\mathbb E_\beta[\operatorname{sg}((p/\beta)B^{\rm MD})\log p]$。两者一阶梯度一致。不要同时让 ratio 和额外的 `log p` 都求导，也不要对两个因素都 detach 后还期待产生梯度。

当 $q\ne\beta$，$p/q$ 只是 anchor ratio；完整行为校正是 $p/\beta=(p/q)(q/\beta)$。这一乘积恒等式不使分别截断后的两个 ratio 自动等价。现有 `advantage_source=rollout` 使用的是 $\log(T/\beta)$，不能在离策略时直接把它叫作核心稿的 $A(T/q)$；非线性 $\kappa>0$ 还需要完整的形状变换。

有限缓存反复拟合时，参数已经依赖缓存，不能把固定参数下的动作期望身份升级为“每个自适应更新都对总体目标条件无偏”。Full-vocab 消除了固定 state 下的动作抽样误差，参数化模型的表示和有限优化误差仍然存在。

本文不讨论 state 访问分布，所有期望和 off-policy 比较均固定相同 state 及其权重。

<a id="weighted-7-noispartbpomean-的区别"></a>

### 7. NoIS、Part、BPO、Mean 的区别

本节比较数学目标。工程支持范围以[统一实现方案](../../infra/numerics/mirror-objectives/reference.md#settings)为准：实现 sampled/full-vocab MDPO 与 full-vocab BPO，不实现 sampled BPO、Part 或 Mean。

<a id="weighted-inner-losses"></a>

冻结 $f,q,T$，令 $R=U-f$、$G=C_qf$、$K_S=-\mathbb E_qU=\mathrm{KL}(q\Vert p)$。核心稿定义如下：

| 内层目标 | Full-vocab 标量                             | 精确梯度                       |
| -------- | ------------------------------------------- | ------------------------------ |
| MDPO     | $(\mathrm{KL}(p\Vert q)-\mathbb E_pf)/\eta$ | $\mathbb E_p[R\psi]/\eta$      |
| Part     | $\frac12\mathbb E_q(R+\log Z)^2$            | $\mathbb E_q[(R+\log Z)\psi]$  |
| BPO      | $\frac1{2\eta}\mathbb E_q(C_qR)^2$          | $\mathbb E_q[(C_qR)\psi]/\eta$ |
| Mean     | $\frac12\mathbb E_q(U-G)^2$                 | $\mathbb E_q[(U-G)\psi]$       |

MDPO 的最优点已由 KL 分解证明。Part 的零损失强制 $\log(p/m)=0$；BPO 的零方差强制 $\log(p/m)$ 为公共常数，再由归一化得到 $p=m$。这些结论要求完整正支持和自由分布可实现性，不保证共享神经网络参数下得到同一个投影。

Mean 满足精确恒等式

$$
L_{\rm Mean}=\eta L_{\rm BPO}+\frac12K_S^2.
$$

因为 $U-G=C_q(U-f)-K_S$，平方后的交叉项按 $q$ 求和为零。额外的 $K_S^2/2$ 随学生变化，非恒定信号时 Mean 一般不再以 $m$ 为最优点。核心稿的完整 KKT 证明见[内层拟合](../../../../../../MirrorOPD/theory/core/sections/04_inner.tex)，数值反例见[验算与复现](#weighted-verification)。

直接从 MDPO 的解析 score 残差删掉 action ratio，会得到

$$
g_{\rm NoIS}=\frac1\eta\mathbb E_\beta[(U-f)\psi]
=\nabla\left[\frac1{2\eta}\mathbb E_\beta(U-f)^2\right].
$$

这不是 MDPO，也不自动等于 BPO。代入 $p=m$ 后，$U-f=-\log Z$，梯度一般为 $-(\log Z/\eta)\mathbb E_\beta\psi\ne0$。算术端点 $\kappa=1$ 有 $Z=1$，这个特例可共享最优点，但有限步梯度仍不同。

若在 $\beta$ 下补回 `log Z` 或用 $C_\beta$ 中心化，可构造以 $m$ 为自由分布最优点的 Part/BPO 型回归；$\beta\ne q$ 时它们并非核心稿的 $q$ 加权目标。若要保留核心稿定义，sampled 估计仍需 $q/\beta$，full-vocab 则直接用 $q$ 权重。

“只把教师信号项取消 IS、anchor KL 仍精确反传”又是另一种定义：$-\mathbb E_\beta[A^q\log p]+\mathrm{KL}(p\Vert q)/\eta$。当存在负信号时，令对应 $p_a\to0$，第一项可趋于负无穷，而有限系数的 anchor reverse KL 在闭单纯形上有界。普通对数信号的详细证明与两种 NoIS 分类见[已有文档](#logarithmic-nois-exact-kl)。不能只写 `NoIS=True` 而不定义改了哪一项。

若对未中心化的 current 非线性信号 $A^{\rm cur}=A_{\kappa,\eta}(T/p)$ 取消 IS，则更新为 $-\mathbb E_\beta[A^{\rm cur}\psi]$。它可以写成另一标量目标的梯度：

$$
H_{\rm cur,\beta}(p)
=\sum_a\beta_a\int_0^{\log(p_a/T_a)}-A_{\kappa,\eta}(e^{-u})\,du,
\qquad \nabla H_{\rm cur,\beta}=-\mathbb E_\beta[A^{\rm cur}\psi].
$$

该等式由积分上限求导与 $\nabla\log(p_a/T_a)=\psi_a$ 得到。$\kappa=0$ 时退化为 $\frac12\mathbb E_\beta\log^2(p/T)$。由于 $A(x)$ 与 $\log x$ 同号，每个积分非负；完整正支持下仅在 $p=T$ 时为零，但它不是 $J$ 的同一梯度。把 current 信号再加动作无关 baseline 后，一般还会增加 $-\mathbb E_\beta[b\psi]$：在 $p$ 测度下可消去的中心或 `log Z`，在 NoIS 的 $\beta$ 测度下不能随意删除。

Full-vocab 的 $K_T,K_S$ 可以精确计算。采样版本不能用同一个动作同时估计信号和它自己的中心后声称无偏；仅有中心数值无偏不够。BPO 等中心化目标需使用正确的辅助积分、独立辅助样本或相应成对估计，细节复用[核心稿附录 F](../../../../../../MirrorOPD/theory/core/appendices/f_centers.tex)。

<a id="weighted-8-clipping-应如何选择"></a>

### 8. Clipping 应如何选择

<a id="weighted-clipping"></a>

| 机制                                    | 与本文精确目标的关系                             | 初始方案                       |
| --------------------------------------- | ------------------------------------------------ | ------------------------------ |
| $\eta^{-1}\mathrm{KL}(p\Vert q)$ 镜像项 | MDPO 定义的一部分                                | 保留或等价吸收到固定 $m$ 中    |
| PPO ratio clip                          | 改变动作梯度，不再等于精确 MDPO/current-J        | Full-vocab 主对照不加          |
| Action IS 截断/自归一化                 | 偏差—方差折中，有限样本不再保持上述精确身份      | 仅 sampled 对照单独消融        |
| Advantage whitening / 按批标准化        | 改变尺度与跨前缀相对权重                         | 主对照不加；若加需标明         |
| Gradient-norm clipping                  | 保留 loss 定义，改变 optimizer 实际步长/方向缩放 | 可沿用现有配置，但记录触发情况 |
| 数值 log-ratio guard                    | 避免极端指数；命中时影响 sampled 梯度            | 与 PPO clip 分开记录           |

对 sampled MDPO，令 reward 系数 $B=(f-U)/\eta$ 在当前反传中 detach，PPO 型 loss 为

$$
L_{\rm clip}=-\mathbb E_\beta\min\left(wB,\operatorname{clip}(w,1-\epsilon,1+\epsilon)B\right),
\qquad w=p/\beta.
$$

Clip 生效时截去一部分 score 更新，因此不再等于 $\nabla L_{\rm MDPO}$。Frozen/current coefficient 的选择和 clipping 是独立轴；clip 也不是对整个词表概率施加硬约束，更不保证停止点就是 $m$。

<a id="weighted-9-ηκ-的理论作用与实验含义"></a>

### 9. η、κ 的理论作用与实验含义

<a id="weighted-parameters"></a>

核心稿的主要结论均有适用层次：

- 对固定形状，$\eta$ 从 0 到 1 将精确目标从 $q$ 推向 $T$；不同 $\kappa$ 下相同 $\eta$ 不保证相同移动距离。
- 对 $0<\kappa\le1$，$m_a\ge(1-\eta)^{1/\kappa}q_a+\eta^{1/\kappa}T_a$；几何端点没有统一正 teacher 覆盖系数。
- 所有形状的精确分布更新满足 $\mathrm{KL}(T\Vert m)\le(1-\eta)\mathrm{KL}(T\Vert q)$。有限神经网络拟合不自动满足这个收缩。
- $0<\kappa\le1/2$ 有统一的小步标准化监督方差控制，几何端点没有该分布无关界；full-vocab 已消除条件动作抽样，不能直接把这条采样方差优势解释为 full-vocab 训练更快或质量更好。
- 在 teacher 附近，$m(T+\delta,T)=T+(1-\eta)\delta+O(\|\delta\|^2)$，一阶局部收缩率不依赖 $\kappa$。形状主要改变远离 teacher 时的方向、覆盖和高阶行为，不能保证平方根对所有实例最好。

当 $\kappa=0$、每轮从 $p=q$ 开始，只做一次梯度更新，并使用 MDPO 的 $1/\eta$ 尺度时，

$$
\nabla L_{\rm MDPO}(q)
=\nabla\mathrm{KL}(q\Vert T)
\quad\text{与 }\eta\text{ 无关}.
$$

因为 anchor KL 在 $p=q$ 处梯度为零。相同 optimizer 状态、数据、精度和随机性下，一次更新的这个 $\eta$ 消融没有改变梯度；raw target KL 的差异则可能只是整体尺度。要研究固定目标的插值效应，应增加内层更新或单独检查拟合终点，而不是默认改 $\eta$ 就改变单步行为。一般 $\kappa>0$ 的 $A_{\kappa,\eta}$ 本身依赖 $\eta$，不能把该特殊结论推广到所有形状。

形状实验可比较 $\kappa=0,1/2,1$，保持 eta、inner update 数与有效 token 预算一致，同时记录 $\mathrm{KL}(m\Vert q)$ 或其他移动距离。固定 MDPO 与 current-J 的目标和 anchor 计算成本不同，需同时报告 optimizer 更新数、生成 token 数和计算量。Sampled 与 full-vocab 的估计器比较应固定相同前缀，再单独改变行为策略、IS 或 clipping。

<a id="weighted-10-数值验算与复现"></a>

### 10. 数值验算与复现

<a id="weighted-verification"></a>

[verify_mirror_targets.py](../../../research/checks/objectives/verify_mirror_targets.py) 在 CPU FP64、7 个动作、120 组随机配置上枚举期望，包括 $\kappa=0,10^{-7},1/4,1/2,1$ 与 $\eta=0.2,0.5,0.8,1$。它检查目标归一化、MDPO 的 KL 分解、正确 action IS、current-J 与 current-NoIS 梯度、Part/BPO 的 $q/\beta$、Mean 恒等式，以及单步几何端点的 $\eta$ 不变性。$J$ 与 current-NoIS 势函数的独立数值参照使用 log-ratio 坐标的 64 点 Gauss–Legendre 积分，未放入生产算子。

验算同时检查 MDPO/BPO 的 logits 行归约与共享 hidden/head VJP，并给出六个反例：未归一化整个残差 NoIS 在 $m$ 处仍有非零梯度，Mean 在 $m$ 处仍有非零梯度，PPO clip 改变精确 MDPO 梯度，刷新 target 却不 detach 会改变 current-J 梯度，BPO 在 $\eta=1$ 时仍不等于 Full-KL，以及先截断归一化 teacher 再进行非线性融合会改变梯度。完整检查数、误差与反例数值见[结果 JSON](../../experiments/teacher-transfer/optimizer-precision/mechanism/data/finite-vocab-checks.json)。这些是有限词表恒等式和计算图检查，不是模型质量或吞吐证据。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 \
  /root/.local/share/mirroropd/venvs/profile-cu128/bin/python \
  -m examples.mirroropd.research.checks.objectives.verify_mirror_targets \
  --output examples/mirroropd/docs/experiments/teacher-transfer/optimizer-precision/mechanism/data/finite-vocab-checks.json
```

普通 $\kappa=0$ 四格与两种 NoIS 的证明和历史验算继续引用 [theory/README](../README.md)，不在本目录重建副本。现有 Full-KL 的代码和数据流见[内部实现说明](../../infra/architecture/full-vocab.md)，支持的训练设置及接入见[实现方案](../../infra/numerics/mirror-objectives/reference.md)。
