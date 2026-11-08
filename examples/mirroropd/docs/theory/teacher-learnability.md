# 优势中心化、跨任务几何与教师可学习性

本文比较 SFT、RL 与 full-vocab OPD 的更新，讨论 baseline 与任务正交性的关系，再给出 RL/SFT 教师可学习性的条件性解释。动机来自 arXiv:2608.03573v2；论文原结论、下述数学身份、局部近似和待验证假说分别标明。公式以 CPU 小词表验算核对，经验假说的验证设计见文末。

主要判断：固定 state 上的 full-vocab RKL 确实可以写成中心化 policy gradient；合法 baseline 在精确求和下不改变更新，不能由选择 baseline 制造正交性。RL 教师可能更容易保留学生已经能够实现的局部修正，但“OPD 必然能学 RL、不能学 SFT”不是现有结果或这些恒等式的推论。

SFT→RL 的阶段组合另见[组合教师的可蒸馏性](2026-11-01-SFT与RL组合教师的可蒸馏性.md)：推导同奖励响应内部的 SFT 偏好保留、只学习奖励重加权时的残差，以及共同局部响应谱下的有限预算上下界。

阅读顺序：[论文条件](#paper)、[三类梯度](#gradients)、[正交性](#orthogonality)、[教师可学习性](#learnability)、[实验](#experiments)、[batch 间迁移](#batch-transfer)、[固定预算吸收预测](#absorption-prediction)。本页采用既有[统一符号](符号约定.md)：$T$ 是教师，$p_\theta$ 是学生，$q_0=p_{\theta_0}$ 是冻结初始学生，$\beta$ 是行为策略；额外用 $\rho$ 表示 SFT 目标分布。$u$ 统一表示理想的参数更新方向，损失梯度的下降方向为 $u=-\nabla L$，不与真实 Adam 位移混用。

<a id="paper"></a>

## 1. 论文的实际研究对象与可借鉴设计

论文研究多任务训练的保持与干扰，比较单任务、混合数据、顺序训练和独立训练后合并。主要使用 R1-Distill-Qwen-1.5B/7B，任务为数学、科学、代码和逻辑。部分参数分析对两类方法都采用 LoRA，rank64、alpha32；SFT 学习率为 1e-5，GRPO 为 3e-6、每题 16 条回答、8K 上限且关闭额外 KL 惩罚。两者训练数据也不完全相同。正文给梯度内积的幅度上界，附录 C.4 在额外残差弱耦合条件下讨论余弦。上述设置和条件均按 v2 正文及附录核对。[^paper]

这些对照适合迁入本项目，但数据、更新幅度、可训练参数和访问分布需要控制。两种方法都使用低秩参数化的观测，不能证明 RL 本质低秩而 SFT 本质高秩。正交更新、参数稀疏、矩阵低秩、任务能力保持也是不同性质，应分别测量。

下文将“能力更新垂直”解释为：在改善任务 i 的同时，任务 j 的相关行为变化较小。参数欧氏正交只能提供一种几何线索；最终需要任务损失或行为读出来确认。本阶段以分布风险为主时，应称为条件分布保持，不直接命名为能力保持。

<a id="gradients"></a>

## 2. 统一比较条件

先固定前缀 $s$，词表有限、支持一致，$p_\theta(a\mid s)>0$，RKL 所用教师 $T(a\mid s)>0$。省略 $s$，定义 score 和当前 log-ratio：

$$
\psi_a=\nabla_\theta\log p_a,
\qquad a_a=\log\frac{T_a}{p_a},
\qquad K=D_{\mathrm{KL}}(p\Vert T).
$$

这里 $a_a$ 的下标表示动作，不是学习率。固定前缀恒等式为

$$
\mathbb E_p\psi_a
=\sum_a p_a\nabla\log p_a
=\sum_a\nabla p_a
=\nabla1=0.
$$

多个前缀时再按共同固定测度 $\mu$ 求平均。现实 SFT 的前缀常来自示范，RL/OPD 的前缀可能来自当前或冻结学生；先在同一测度上比较目标，再单独讨论来源变化。

## 3. Full-vocab OPD 的 baseline 身份

直接求导得

$$
\begin{aligned}
\nabla_\theta K
&=\sum_a \nabla p_a\left(\log\frac{p_a}{T_a}+1\right)\\
&=\mathbb E_p\left[\log\frac{p_a}{T_a}\,\psi_a\right],\\
u_{\mathrm{OPD}}
&=-\nabla K=\mathbb E_p[a_a\psi_a].
\end{aligned}
$$

第二行用到了 score 的零均值。对任意在给定 state 上不依赖动作的标量 $b(s,\theta)$，逐点成立

$$
\mathbb E_p[(a_a-b)\psi_a]
=\mathbb E_p[a_a\psi_a]-b\mathbb E_p\psi_a
=u_{\mathrm{OPD}}.
$$

选择 $b=\mathbb E_p a=-K$，就有中心化信号

$$
\bar a_a=a_a+K,\qquad \mathbb E_p\bar a_a=0,
\qquad u_{\mathrm{OPD}}=\mathbb E_p[\bar a_a\psi_a].
$$

因此 reward 形式是加上 $K$，不是再减去 $K$。若采用 cost $\ell=\log(p/T)$，相应写法才是 $\ell-K$。这与既有[条件梯度与控制变量](estimators.md)一致。

### 3.1 为什么不能靠 baseline 改造 full-vocab 更新

精确求和时，$b=0$、常数或 $-K$ 给出同一个向量。它们也就给出相同的欧氏方向、稀疏性和跨任务内积；不能把一个实现里不存在的 dense 分量通过换 baseline 真正移走。中心化揭示相对权重结构，不是额外算法自由度。

sampled-token 中 baseline 可以改变有限样本方差，但合法条件下不改变期望。若 baseline 由包含当前动作的同一有限样本组估计，需另查依赖与偏差；GRPO 的组内中心化／标准化不是可不加区分地套用的独立 baseline。动作依赖 baseline、未校正离策略测度、clipping 和截断支持都会改变相关身份的条件。

### 3.2 自动微分的正确位置

若用 score surrogate，正确的一次反向计算图为

$$
L_{\mathrm{sur}}
=-\sum_a\operatorname{sg}\{p_a[a_a-b]\}\log p_a.
$$

对系数整体 stop-gradient，便有 $-\nabla L_{\mathrm{sur}}=u_{\mathrm{OPD}}$。每次 forward 仍重新计算当前概率和信号。直接对 $K-b(\theta)$ 求导不是同一操作；例如令 $b=K$ 会使标量恒为零、梯度也为零。当前实现直接对完整 RKL 反传即可，没有为了 baseline 再改目标的必要。

## 4. RL 与 SFT 的对应形式

### 4.1 RL：奖励加权的当前策略 score

在固定 state 的 bandit 比较中，令 $r_a$ 为与参数无关的奖励，则

$$
J_{\mathrm{RL}}=\mathbb E_p r_a,\qquad
u_{\mathrm{RL}}=\nabla J_{\mathrm{RL}}
=\mathbb E_p[(r_a-\mathbb E_pr)\psi_a].
$$

序列任务的精确 policy gradient 需使用回报或适当的 $Q^\pi/V^\pi$；这里不能把任意即时 token 分数叫作终局奖励的精确 advantage。GRPO 的组内经验标准化与真实中心化回报也应分别记录。

OPD 与 RL 共享 score 加权结构，差别在于：OPD 的信号为随当前学生变化的 $\log(T/p)$，RL 信号来自任务奖励或相应价值。这个相似性不要求采用 sampled-token 实现。

### 4.2 SFT：在示范分布上平均 score

固定前缀，令目标动作分布为 $\rho$，hard label 是 $\rho=\delta_y$ 的特例：

$$
L_{\mathrm{SFT}}=-\mathbb E_\rho\log p_a,
\qquad u_{\mathrm{SFT}}=\mathbb E_\rho\psi_a.
$$

在 $\rho$ 的支持被 $p$ 覆盖时，可以等价重写为

$$
\begin{aligned}
u_{\mathrm{SFT}}
&=\mathbb E_p\left[\frac{\rho_a}{p_a}\psi_a\right]\\
&=\mathbb E_p\left[\left(\frac{\rho_a}{p_a}-1\right)\psi_a\right],\\
\mathbb E_p\left[\frac{\rho_a}{p_a}-1\right]&=0.
\end{aligned}
$$

所以“有零均值权重”不是 OPD/RL 独有的属性。这个重写只用于比较数学结构，不建议把稳定的 SFT 实现换成可能有巨大重要性权重的采样估计器。

| 方法    | 固定前缀的理想更新方向              | 权重及目标                           |
| ------- | ----------------------------------- | ------------------------------------ |
| SFT     | $\mathbb E_p[(\rho/p-1)\psi]$       | 线性密度比；示范匹配                 |
| RL      | $\mathbb E_p[(r-\mathbb E_pr)\psi]$ | 任务奖励；实际序列问题需要回报／价值 |
| OPD-RKL | $\mathbb E_p[(\log(T/p)+K)\psi]$    | 对数密度比；教师分布匹配             |

### 4.3 Logit 空间中更清楚的差异

令 $p=\mathrm{softmax}(z)$，则 $\partial\log p_a/\partial z_v=\mathbf1[a=v]-p_v$。代入上述三个方向，得到

$$
u^{z}_{\mathrm{SFT},v}=\rho_v-p_v,
\qquad u^{z}_{\mathrm{RL},v}=p_v(r_v-\mathbb E_pr),
\qquad u^{z}_{\mathrm{OPD},v}=p_v\left(\log\frac{T_v}{p_v}+K\right).
$$

这三个向量都满足各坐标和为零，只代表概率归一化切空间，不代表能力子空间正交。参数空间方向还需乘 logit Jacobian 的转置。

对于固定正的 $T_v$，若学生给某动作的概率 $p_v\to0$，RKL logit 更新中的 $p_v\log(T_v/p_v)\to0$；SFT/soft-CE 的 $\rho_v-p_v$ 则可保持非零。这提示两者对学生极不熟悉动作的响应不同。但该机制不能单独解释本地全部失败：FKL 对照未解除主要平台，且已有 Polaris 分解中大部分剩余差异位于共享 top-k 内部。

### 4.4 邻近目标下，soft-SFT/FKL 与 RKL 一阶相同

令同一个软教师 $T=p+\varepsilon v$，$\sum_av_a=0$，并假设相关坐标上的概率足以保证展开有效，则

$$
\log(T_a/p_a)=\varepsilon v_a/p_a+O(\varepsilon^2),
\qquad K=O(\varepsilon^2).
$$

于是

$$
u^{z}_{\mathrm{OPD},a}=\varepsilon v_a+O(\varepsilon^2),
\qquad u^{z}_{\mathrm{soft\mbox{-}SFT},a}=T_a-p_a=\varepsilon v_a.
$$

因此在相同 state、相同近邻目标下，两种方向并没有普遍不同的一阶几何。该等式不把现实中的 teacher-trajectory SFT 和 student-trajectory OPD 变成同一算法；它只是隔离目标方向后的局部结果。

### 4.5 固定 state OPD 与整个序列目标的区别

若用当前学生访问测度 $d_\theta$ 定义序列条件风险之和，则

$$
\nabla_\theta\sum_s d_\theta(s)K_s(\theta)
=\sum_s d_\theta(s)\nabla K_s(\theta)
+\sum_sK_s(\theta)\nabla d_\theta(s).
$$

常用逐 token、采样前缀 stop-gradient 的 OPD 更新对应第一项。完整词表积分没有自动加入第二项；本地固定 S0 研究更直接固定了 $d$。因此不能仅因更新形似 policy gradient，就宣称 full-vocab OPD 等同于精确序列 RL。

<a id="orthogonality"></a>

## 5. 从中心化到近正交所需的条件

### 5.1 组内分解与均值抵消

一般加权向量 $u=\sum_k w_kS_k$ 若满足 $\sum_kw_k=0$，则对任意共同向量 $c$，

$$
u=\sum_kw_k(S_k-c).
$$

这是论文中 GRPO 组内分解可迁移的代数部分。对于 OPD，在词表上取 $w_a=p_a\bar a_a$ 也有同一身份。但对归一化分布的完整 score，本来就有 $\mathbb E_p\psi=0$；不能据此声称另外发现并删除了一个语义上的“通用能力向量”。

### 5.2 幅度小不推出夹角接近直角

令 $u_i=\epsilon e_1,u_j=\epsilon e_1$，内积是 $\epsilon^2$，却有余弦 1。将第二个向量取负，余弦为 -1。对应奖励分别取 $r$ 与 $r$、或 $r$ 与 $-r$，中心化后仍可产生同向或完全相反的 RL 更新。

对 OPD，定义 $\sigma_a^2(s)=\mathbb E_p\bar a^2$ 和 $F_s=\mathbb E_p[\psi\psi^\top]$。Cauchy–Schwarz 给出

$$
\|u_s\|^2=\|\mathbb E_p[\bar a\psi]\|^2
\leq \mathbb E_p\bar a^2\,\mathbb E_p\|\psi\|^2
=\sigma_a^2(s)\operatorname{tr}F_s.
$$

这个上界约束幅度，没有约束两个任务方向的相关性。on-policy 本身也不保证 $\operatorname{tr}F_s$ 小；优势标准化通常使其经验方差接近 1，不是自动让它接近零。

### 5.3 近正交需要跨任务耦合小

设任务 i、j 的更新分别为 $u_i=\mathbb E_{\mu_i p}[\bar a_i\psi]$，则

$$
u_i^\top u_j
=\mathbb E_{s,a,s',a'}
\left[\bar a_i(s,a)\bar a_j(s',a')
\psi(s,a)^\top\psi(s',a')\right].
$$

中心化限制权重的均值，不能固定交叉 score kernel，也不能决定权重与 kernel 的相关性。要推出余弦小，还需要交叉项足够小且两个更新范数不同时退化。论文附录 C.4 就额外引入了弱残差耦合和范数条件；不能删去这些条件后保留正交结论。[^paper]

### 5.4 三种“干扰”不能混为一谈

对任务损失 $L_j$，在更新 $\Delta\theta_i$ 后有

$$
L_j(\theta+\Delta\theta_i)-L_j(\theta)
=\nabla L_j(\theta)^\top\Delta\theta_i
+\tfrac12\Delta\theta_i^\top H_j(\tilde\theta)\Delta\theta_i.
$$

若 $\Delta\theta_i=-\eta\nabla L_i$，一阶变化为 $-\eta\nabla L_j^\top\nabla L_i$。正的梯度内积在该近似下帮助降低另一任务损失，负的才表示一阶冲突；不能把较大绝对余弦全部叫有害干扰。长训练终点参数差的夹角、同一点的梯度夹角、实际任务保持也不是同一量。真实 Adam 需记录实际位移，不能直接套用 SGD 方向。

同样，跨任务正交与同一 teacher 的跨题共享需要不同测量：前者可能有利于不互相扰动，后者通常需要可迁移的共同更新。若不同题的更新全都正交，反而可能表现为只学当前题、无法改善未见题。我们不能同时以“所有梯度更正交”解释任务保持和跨题泛化。

<a id="learnability"></a>

## 6. OPD 恢复 RL 教师的一个精确条件模型

固定 state，考虑自由概率单纯形上的 KL 正则奖励优化，$q_0$ 为初始学生，$r$ 为固定奖励，$\tau>0$：

$$
\max_T\left\{\mathbb E_T r-\tau D_{\mathrm{KL}}(T\Vert q_0)\right\},
\qquad \sum_aT_a=1.
$$

拉格朗日驻点满足 $r_a-\tau[\log(T_a/q_{0,a})+1]-\lambda=0$，归一化得到

$$
T_a^*=\frac{q_{0,a}e^{r_a/\tau}}{Z},
\qquad Z=\sum_aq_{0,a}e^{r_a/\tau}.
$$

代入 OPD 的 signal，

$$
\log\frac{T_a^*}{p_a}=\frac{r_a}{\tau}-\log Z-\log\frac{p_a}{q_{0,a}}.
$$

与 score 相乘并在 $p$ 下求和，常数项抵消，得到

$$
u_{\mathrm{OPD}}
=\frac1\tau\nabla_\theta\mathbb E_p r
-\nabla_\theta D_{\mathrm{KL}}(p\Vert q_0).
$$

特别地，在 $p=q_0$ 处，第二项梯度为零，OPD 与该奖励的 RL 更新同向。这是“RL 教师把相对初始模型的奖励偏好编码进概率比，OPD 可以恢复这个方向”的一个精确条件解释。

但该推导假设固定 state、自由分布最优解和显式 KL 正则。目标论文的 GRPO 分析关闭了额外 KL 惩罚；真实 Transformer 的优化也不必达到上述最优解。序列 RL 若对应终局奖励，则条件 log-ratio 通常涉及未来价值，不能把任意终局 reward 直接代入单 token 的 $r_a$。此式不是对所有实际 RL teacher 的已验证模型。

另一个必要限制是：任意正教师都能以 $r=\tau\log(T/q_0)$ 写成这种重加权形式。因此，单独展示指数表达式不能识别教师由 RL 训练而来；它有解释力的条件是 $r$ 与实际任务 reward/价值有可核验关系，且实际更新和状态覆盖与之相容。

## 7. 邻近教师的可恢复性不专属于 RL

### 7.1 共同参数化下的一阶拟合

设 teacher 参数为 $\theta_T=\theta_0+\delta$，student 为 $\theta=\theta_0+w$，二者在同一光滑参数化和同一状态测度 $\mu$ 下比较。局部展开给出

$$
\log\frac{T(a\mid s)}{p_\theta(a\mid s)}
=\psi_0(s,a)^\top(\delta-w)
+O(\|\delta\|^2+\|w\|^2).
$$

定义共同初始 Fisher

$$
F_\mu=\mathbb E_{s\sim\mu,a\sim q_0}
[\psi_0(s,a)\psi_0(s,a)^\top].
$$

代入 OPD 方向并忽略高阶项，得到

$$
u_{\mathrm{OPD}}\simeq F_\mu(\delta-w),
\qquad L_\mu\simeq\tfrac12(\delta-w)^\top F_\mu(\delta-w).
$$

这个关系只使用 teacher 的函数和位置，不使用训练标签。只要 SFT teacher 也足够近、模式和状态一致，就有相同的一阶可恢复性。不能从公式中导出“RL 都能学、SFT 都学不好”。

### 7.2 有限时间选择的是方向，不是训练标签

在固定 Fisher 的普通梯度流近似下，$\dot w=F_\mu(\delta-w)$、$w(0)=0$，因此

$$
w(t)=(I-e^{-F_\mu t})\delta.
$$

若 $F_\mu v_i=\lambda_i v_i$，$\delta=\sum_i c_i v_i$，则

$$
L_\mu(t)\simeq\frac12\sum_i\lambda_i c_i^2e^{-2\lambda_i t}.
$$

较大响应特征值方向下降快，较小者慢。该谱来自同一 student、同一测度，换 teacher 主要改变 $c_i$；已有 teacher-conditioned loss-gradient Gram 不能直接充当这里的 $F_\mu$。真实 Adam 的局部固定预条件器还会改变有效谱，非线性特征演化使固定 Fisher 只在局部有效。此类近源谱解释已有先例，不作为本页的新定理。[^vanishing]

### 7.3 一个直接反例与更合理的假说

取 $F=\mathrm{diag}(1,0.01)$，两个一维目标分别为 $\delta_1=(1,0)$、$\delta_2=(0,10)$。它们的初始二次风险都为 0.5，但在 $t=10$ 时，残差分别约为 $1.03\times10^{-9}$ 与 0.409。两者都只占一个方向，却有很不同的学习速度。低秩与同样的初始 KL 都不足以排序可学习性。

更可检验的假说是：某些直接从学生出发的 RL teacher 只要求在学生已访问的状态上进行较近、较可达且可跨题共享的修正；一些 SFT teacher 则使用外部示范、不同格式或状态，形成更远或较弱可达的修正。这里分别涉及距离、状态覆盖、方向和模型非线性，RL/SFT 标签只是可能相关的因素。

SFT teacher 也可能是近邻，RL teacher 也可能较远或产生相互冲突的修正。跨尺寸 teacher 没有共享权重坐标，不能直接相减参数；可以在共同前缀上测函数残差在学生局部响应空间中的可达性，并单独验证非线性变化能否突破局部限制。

## 8. 与本地已有证据的关系

| 本地结果                                                        | 对假说的作用                           | 不应作的外推                            |
| --------------------------------------------------------------- | -------------------------------------- | --------------------------------------- |
| Qwen e3/GRPO-KL 的参数距离小、64 步数学留出下降约 94.65%/89.98% | 与近邻可恢复解释一致                   | 不能独立证明由 RL 算法或低秩造成        |
| Samsung SFT 的同一面板 64 步下降 72.23%，320 步下降 89.94%      | SFT 目标可以明显学入                   | 不能说 OPD 排斥 SFT；其模式归属仍需核验 |
| 同构早分叉 teacher-copy 提供更低风险参数点                      | 剩余损失不自动是表达下界               | copy 不是梯度优化可达性的证明           |
| 普通 Qwen 4B/8B 64 步训练银行约下降一半，数学留出约 2.97%/1.59% | 共同修正推广不足；有必要研究跨状态响应 | 不能只看小梯度或参数规模归因            |
| 小教大也有训练／留出分离                                        | 简单压缩容量解释不足                   | 不证明任意大架构严格包含小架构函数类    |
| 同源近邻的 Code/IF 风险也明显下降                               | 同源更新并非对所有域绝对正交           | 跨域条件分布改善不是已测任务能力变化    |
| RKL 可记忆随机映射；FKL 未解除 Polaris 平台                     | 严格的信息筛选和单一 KL 方向解释均不足 | 不否定有限时间的方向偏好                |

数值与状态以[最新谱系总览](../experiments/teacher-transfer/fixed-state-results/README.md)、[能力筛选](../experiments/teacher-transfer/mechanisms/capability-filter.md)及[实际更新诊断](../experiments/teacher-transfer/cross-batch-transfer.md#batches)为准。不同系列的题池、预算、模式和支持口径不合并为独立重复。现有 Qwen 模式不匹配 SFT 结果只适合作描述性材料。

<a id="experiments"></a>

## 9. 可证伪的实验顺序

### 9.1 已完成：baseline 与反向路径的代数验算

在 CPU、FP64、小词表上比较直接 RKL 梯度、多个合法 baseline 的精确 score 梯度和 detached surrogate；核对 SFT 密度比重写与奖励倾斜教师身份。另验证中心化更新完全反向的反例、近邻 soft-SFT/RKL 的一阶接近、非线性学生下 $F_\mu\delta$ 的局部误差及等初始风险的慢模态例子。

身份误差最大约 $6.7\times10^{-16}$；中心化相反奖励的余弦为 -1。非线性教师位移从 0.1 缩小到 0.0001 时，Fisher 预测相对误差从约 0.00957 缩小到 0.00000957。它们验证公式与反例，不验证真实 LLM 的 RL/SFT 差异。原始数据、脚本和论文源见[验算结果](../../../../../wandb-logs/MirrorOPD-verl/analysis/opd-baseline-geometry-20261030/check-results.json)、[CPU 脚本](../../../../../wandb-logs/MirrorOPD-verl/analysis/opd-baseline-geometry-20261030/check.py)和[固定论文源](../../../../../wandb-logs/MirrorOPD-verl/analysis/opd-baseline-geometry-20261030/2608.03573v2.tex)。

### 9.2 优先一：建立同初始化、匹配距离的 RL/SFT teacher 对照

使用同一原始 student，先只选择数学与代码两个域。每域建立 SFT 与 GRPO 分支，父权重、prompt 池、输入模式、可训练参数集合固定。SFT 使用可核验的示范，RL 使用相同题目的奖励，记录不可避免的监督来源差异与全部 token 成本。预算对照与距离对照分别报告，不能声称同时严格匹配所有因素。

优先沿用本地全参数设置；若先复现论文的 LoRA 分析，则两种方法都用相同 rank，结果独立成块。不能用 full-SFT 对 LoRA-RL，然后把秩差归因于算法。

保存 teacher 中间 checkpoint，在独立开发面板上匹配 $D_{\rm KL}(q_0\Vert T)$ 的若干重叠水平。之后所有 teacher 都由同一个原始 student、同一个 S0 状态银行和同一个 OPD 配方拟合。报告残差、相对下降、旧批保持及独立题目风险，不以 teacher accuracy 筛选。

预测与否证：若距离和模式匹配后 SFT teacher 与 RL teacher 同样易学，就拒绝算法标签本身决定可蒸馏性的解释；若差距仍在，再检查目标方向和状态覆盖。现成 e3/GRPO-KL/Samsung、Llama GT/算术 SFT 可先描述性分析，但其训练数据、模式和配方不同，不能替代此受控比较。

### 9.3 优先二：复用论文的跨任务矩阵，加入 OPD

从共同起点分别执行数学、代码的 SFT/RL/OPD；按固定处理预算比较，再补以初始 student KL 为尺度的匹配位移比较。后者避免把更新较小误认为天然更正交。测量真实 FP32 参数位移、各层谱、阈值敏感的坐标稀疏率和有符号余弦，不用单一绝对内积总结。

建立响应矩阵 $\Delta R_{ij}=R_j(\theta+\Delta\theta_i)-R_j(\theta)$，其中 $R_j$ 在独立固定的 j 域前缀上测量。并列报告一阶预测 $\nabla R_j^\top\Delta\theta_i$ 与真实有限变化。先以 RKL/JS 研究分布保持，若要声称能力垂直更新，再另做任务功能读出。

借鉴论文的混合、顺序和并行设计，对代表设置比较 Math→Code、Code→Math、交替混合；参数差合并仅用于严格同构且同一初始权重的模型。若近正交但顺序风险仍恶化，检查曲率、状态迁移及优化器历史；若正余弦同时带来跨域改善，则它是共享收益，不能归入有害冲突。

### 9.4 优先三：把谱偏好与 teacher 标签分开

固定 student、状态和度量，比较不同 teacher 的函数残差与同一个学生响应算子的关系。已有 loss-gradient Gram 可作描述，但不能直接代替共同 Fisher/输出核；可用矩阵向量积测少量代表模态，先检验局部预测能否解释真实短步变化。

对于同构 teacher，比较其参数增量在共同 Fisher 快／慢方向上的分量，并测对应风险下降。构造控制时匹配初始函数风险与增量幅度，改变方向或有效秩；重新核验实际模型函数，不能认为权重 SVD 截断只改变秩。正预测是慢方向残差下降更慢；反例是低秩弱响应目标也难学、高秩强响应目标也可快速下降。

跨尺寸教师只测共同输入上的函数残差及学生可达响应。若局部预测失效，应研究表示演化或优化器，而不把局部零空间叫作全局容量界限。

### 9.5 与现有 state 研究衔接

对一个近源正对照和一个跨尺寸困难对照，补同题新轨迹、异题同来源轨迹的固定读出，保持 teacher 不变。用相同数学工具区分“域间低干扰”和“域内跨题可迁移”：前者需要非目标域保持，后者需要未训练题持续改善。

若只有状态覆盖改变就明显恢复谱投影和留出下降，支持状态条件解释；若模式与覆盖匹配后仍有同构目标的大残差，再研究可达方向与非线性路径。当前阶段不因中心化身份而新增 baseline 调参，也不把 Delta-OPD 作为验证这些假说的必要方法。

<a id="batch-transfer"></a>

## 10. Batch 间迁移：从中心化信号到共享更新

本节将任务间分析改写为同一 teacher 的固定 batch 风险。目标是解释“当前批拟合而未见批改善有限”，不把不同 batch 的起点 loss 差当作遗忘。以下向量均在同一个参数点测量，state 银行与 mask 固定。

### 10.1 一步更新的跨批收益

定义 $R_A(\theta)=\mathbb E_{s\sim\mu_A}K_s(\theta)$，$u_A=-\nabla R_A$；B 同理。在 A 上做一步理想 SGD，$\theta'=\theta+\eta u_A$。定义正值表示改善的收益

$$
G_{A\to B}=R_B(\theta)-R_B(\theta').
$$

Taylor 展开给出

$$
G_{A\to B}
=\eta u_B^\top u_A
-\tfrac12\eta^2u_A^\top H_B(\tilde\theta)u_A,
\qquad
G_{A\to A}=\eta\|u_A\|^2+O(\eta^2).
$$

在小步长且局部收益非零时，

$$
\frac{G_{A\to B}}{G_{A\to A}}
\simeq\frac{u_B^\top u_A}{\|u_A\|^2}
=\frac{\|u_B\|}{\|u_A\|}\cos(u_A,u_B).
$$

自内积控制局部一阶收益，交叉内积控制另一批的一阶收益；只看 cosine 会遗漏剩余误差的梯度幅度。正交表示缺少一阶迁移，负内积表示局部冲突，正内积表示正迁移。二阶曲率和有限步幅仍可能改变真实收益。

### 10.2 中心化不消除批间共享信号

由第 3 节的精确身份，

$$
u_A=\mathbb E_{s\sim\mu_A,a\sim p}[\bar a(s,a)\psi(s,a)].
$$

所以

$$
u_B^\top u_A
=\mathbb E_{s,a,t,b}
[\bar a(s,a)\bar a(t,b)\,\psi(s,a)^\top\psi(t,b)].
$$

需要共同满足教师信号有对应关系、学生跨状态响应可共享。仅有 token overlap 或优势方差不足以决定该内积。

这里的零均值是每个 state 内对动作的中心化。它不等于 batch 梯度的总体均值为零，也不等于将每个 batch 的向量减去跨 batch 平均向量。后者会直接移除我们想保留的共享学习方向，改变优化算法。不同任务低干扰与同一 teacher 跨题正迁移，应分别测量。

### 10.3 均值—异质性分解

在固定参数点、相同来源分布上，假设 batch 由 n 个独立 prompt/轨迹簇组成，单簇更新随机向量为 $U$。令

$$
m=\mathbb E[U],\qquad \Sigma=\operatorname{Cov}(U),\qquad
u_A=m+\epsilon_A,\quad u_B=m+\epsilon_B.
$$

对独立的新 batch A、B，有 $\mathbb E\epsilon_A=\mathbb E\epsilon_B=0$、$\operatorname{Cov}(u_A)=\Sigma/n$。展开即可得到

$$
\begin{aligned}
\mathbb E[u_A^\top u_B]
&=\|m\|^2,\\
\mathbb E\|u_A\|^2
&=\|m\|^2+\operatorname{tr}(\Sigma)/n.
\end{aligned}
$$

因此一阶收益的期望之比为

$$
\rho_n\simeq
\frac{\mathbb E G_{A\to B}}{\mathbb E G_{A\to A}}
=\frac{\|m\|^2}{\|m\|^2+\operatorname{tr}(\Sigma)/n}.
$$

这不是逐批比值的期望，也不是无条件成立的因果泛化比例。若 A 已被反复优化、参数依赖 A，或两个集合共享题目、采用有限总体无放回抽样，上述独立性和均值条件需要重新检查。token 不是独立簇，不能用 token 数直接代入 n。

$\epsilon$ 包含有限样本误差和真实题目差异，不等于无用噪声。小 $\rho_n$ 表示当前批方向的能量主要不与另一独立批共享：当前批仍能下降，未见批收益却弱。更大 batch 可降低该差异项，但相同步长下的一阶总体收益仍为 $\eta\|m\|^2$；增大 batch 不会凭空制造消失的总体方向，也不保证等 token 成本更优。

若训练与评估 state 分布不同，改写为 $u_A=m_A+\epsilon_A$、$u_B=m_B+\epsilon_B$，独立时

$$
\mathbb E[u_A^\top u_B]=m_A^\top m_B,
\qquad
\rho_{A\to B}\simeq
\frac{m_A^\top m_B}{\|m_A\|^2+\operatorname{tr}(\Sigma_A)/n}.
$$

这个分布不同的版本允许负迁移，适合分开分析 student/teacher state 来源、同题异轨迹及跨领域面板。

### 10.4 同构近源教师下的条件解释

沿第 7 节的局部近似，令当前尚未恢复的共同参数修正为 $e=\delta-w$，则 $u_A\simeq F_Ae$、$u_B\simeq F_Be$。记 $F=\mathbb E F_A$，有

$$
m\simeq Fe,\qquad
\mathbb E\|u_A-m\|^2
\simeq\mathbb E\|(F_A-F)e\|^2.
$$

若同一修正 e 在不同题目的学生响应中都能稳定实现，共享项相对大，跨批收益好；若响应随 state 变化很大，即使存在可行 teacher 参数，也可能局部学习强、跨批学习弱。$F_A,F_B$ 都为正半定不保证每一对 $e^\top F_BF_Ae$ 为正，不能把共同参数修正本身写成逐批正迁移保证。

这个解释不区分 RL/SFT 标签。在线性近似内，仅把 e 乘同一个系数会同时缩放两类能量，不能单靠更小参数距离保证更高迁移比；方向、状态覆盖及离开线性区间的程度仍重要。

### 10.5 实际 Adam 与整块更新

对任何实际更新 $\Delta\theta_A$，有效的一阶预测都是

$$
G_{A\to B}\simeq u_B^\top\Delta\theta_A.
$$

若只是固定正定预条件器 P 的无动量更新，则 $\Delta\theta_A=\eta Pu_A$，将自／交叉内积改成 P 度量即可。真实 Adam 的矩、预条件器、裁剪和衰减都可能随当前批改变，因此使用完整参数位移；不能把经验 Gram 比值直接解释成 Adam 收益。

对一个 16 步块，应保留初始与最终的真实风险差，并可在块内逐步累加 $u_B(\theta_t)^\top\Delta\theta_t$。这比仅用块首梯度乘总位移更能检查非线性误差；所有 B 的前缀需保持不变。共同参数点的 raw Gram 是对称的，有限训练的 $A\to B$ 与 $B\to A$ 不必对称。

### 10.6 已有 32 批 Gram 的只读复算

复用 N 系列四个固定 checkpoint 的全参数梯度。每个案例有 32 个 B16 梯度，另有独立 Held64；各自 warm128 是不同训练终点。记 $G_{ij}=u_i^\top u_j$，计算

$$
\widehat L=\frac1m\sum_iG_{ii},\qquad
\widehat C=\frac1{m(m-1)}\sum_{i\ne j}G_{ij},\qquad m=32.
$$

| Teacher / 阶段    | 平均自内积 $\widehat L$ | 平均跨批内积 $\widehat C$ | $\widehat C/\widehat L$ |
| ----------------- | ----------------------: | ------------------------: | ----------------------: |
| JustRL / 初始     |                2.589183 |                  2.283864 |                  0.8821 |
| JustRL / warm128  |                0.132185 |                  0.056612 |                  0.4283 |
| Polaris / 初始    |                2.169155 |                  1.526813 |                  0.7039 |
| Polaris / warm128 |                0.424837 |                  0.032129 |                  0.0756 |

Polaris 后期的自内积仍明显非零，但与其他批共享的比例很低。跨批内积仍为正，数据更支持正迁移减弱，而非所有批次互相冲突。这里的 32 批来自有限固定池且部分与训练历史有关，因此表是描述性统计，不直接当作总体 $\|m\|^2$ 的无偏估计或真实迁移率。

独立 Held64 与 32 批的平均内积，再除以平均自内积，JustRL 初始／后期为 0.9341／0.5196，Polaris 为 0.8180／0.1072，方向与批内交叉统计一致。这些结果不能和 L 单遍实验的整块迁移比 0.197→0.00584 混成同一数值，因为起点、训练路径、测量单位和优化器作用不同。

原始输入、完整数字和定义见[Gram 复算](../../../../../wandb-logs/MirrorOPD-verl/analysis/batch-transfer-geometry-20261030/gram-reanalysis.json)，[复算与验算脚本](../../../../../wandb-logs/MirrorOPD-verl/analysis/batch-transfer-geometry-20261030/check.py)同时核对独立抽样的均值—方差身份和有限步二次目标的 Taylor 展开，[验算结果](../../../../../wandb-logs/MirrorOPD-verl/analysis/batch-transfer-geometry-20261030/identity-check.json)保留数值。

### 10.7 下一项机制验证

优先在初始、早期及平台 checkpoint，用相同的少量独立诊断批测 raw Gram、实际更新和固定风险。分开报告局部更新能量、共享方向、真实迁移和剩余风险，不用一个小 cosine 归因。

在不使用最终测试梯度选方向的前提下，比较旧题原轨迹、同题新轨迹和新题轨迹。如果新轨迹恢复跨批内积和真实收益，则支持状态覆盖解释；若 raw 共享方向仍在但 Adam 的迁移差，进一步用同一状态的优化器控制；若两者都低，再讨论表示和目标响应。即使总体共享梯度很小，也需分清接近教师、局部驻点、状态测度不足和函数不可表达，不能直接宣布容量硬界限。

<a id="absorption-prediction"></a>

## 11. 用什么预测固定预算的 OPD 吸收

本节将第 7 节的局部谱模型转成可测预测量，并补充 RL/SFT 一步更新的条件性界。新增推导以固定前缀、完整动作支持和局部近似为范围；它不把教师来源标签当作一般收敛定理。实际 E 系列采用 AdamW、按批刷新与有限 reuse，最后必须用真实更新读出验证近似。

### 11.1 三个散度都不能单独决定吸收比例

以 $p_0$ 为初始学生、$T$ 为目标教师。$\mathrm{KL}(p_0\Vert T)$ 是本项目的初始分布风险，$\mathrm{KL}(T\Vert p_0)$ 描述相反方向，$\chi^2(T\Vert p_0)$ 对低学生概率处的教师质量更敏感。三个量都依赖概率，却不包含学生参数对各处概率的响应强度，所以不能单独决定规定优化器和预算下的风险下降。

一个严格反例使用两个等权状态、两个动作。学生在状态 $i$ 的正动作概率为 $\sigma(a_i\theta_i)$，初始化 $\theta=0$，取 $a=(\sqrt8,\sqrt{0.08})$。两种 teacher 分别只将状态 1 或状态 2 的 logit 从 0 改为 0.1。两个 teacher 都属于完全相同的学生函数族，只是目标偏移发生在不同状态。对共享的等权前缀测度，它们的两个方向 KL、Pearson 散度和中心化 log-ratio 方差完全相同；初始化 Fisher 分别对两个参数方向给出特征值 1 和 0.01。相同 SGD、LR0.1、64 次更新后，两者的剩余风险相差很大，见本节验算。

这个反例不依赖 teacher 不可表示、缺少动作支持或前缀采样失败。它直接排除“任何一个初始散度标量都能唯一决定吸收比例”。对局部扰动，两个方向 KL 和半个 Pearson 散度还具有相同的二阶项，换一个散度名字不会补上缺失的响应信息。

### 11.2 中心化教师信号与学生响应算子

固定前缀测度 $\mu$，定义 Hilbert 内积和中心化信号：

$$
\langle h_1,h_2\rangle_\mu
=\mathbb E_{s\sim\mu,a\sim p_0}[h_1(s,a)h_2(s,a)],
\qquad
f=C_{p_0}\log\frac{T}{p_0}.
$$

中心化对每个状态分别进行。令 $\psi_0=\nabla\log p_{\theta_0}$，定义线性算子 $S\delta=\psi_0^\top\delta$。其伴随满足 $S^*h=\mathbb E_{\mu p_0}[\psi_0h]$，参数 Fisher 为 $F=S^*S$，函数响应算子为 $\mathcal K=SS^*$。固定正定预条件器 $M$ 时，函数算子改为 $\mathcal K_M=SMS^*$；以下默认 $M=I$。

初始 OPD 下降方向有精确身份：

$$
u_0=-\nabla L_\mu(\theta_0)=S^*f.
$$

设教师与学生沿所考虑轨迹的 log-ratio 扰动逐点足够小、所需三阶导数有界、测度固定。局部指数模型将学生中心化 log-ratio 近似为 $S\delta$，KL 展开给出

$$
L_\mu(\theta_0+\delta)
=\frac12\|f-S\delta\|_\mu^2+O(\varepsilon^3),
\qquad L_\mu(\theta_0)=\frac12\|f\|_\mu^2+O(\varepsilon^3).
$$

这里 $\varepsilon$ 控制目标与学生局部扰动。以下动力学对该二次代理精确，对非线性模型仅是局部近似，不能将省略项长期视为零。普通梯度流的参数位移与函数残差满足

$$
\dot\delta=S^*(f-S\delta),\qquad
e=f-S\delta,\qquad \dot e=-\mathcal K e.
$$

在有限诊断银行和有限参数下，对 $\mathcal K$ 做谱分解。设 $\mathcal K\phi_i=\lambda_i\phi_i$、$\lambda_i>0$，将 $f$ 分解成

$$
f=f_\perp+\sum_i c_i\phi_i,
\qquad f_\perp\in\ker S^*.
$$

解为 $e(t)=f_\perp+\sum_i c_i e^{-\lambda_it}\phi_i$，所以二次代理的剩余比例为

$$
\boxed{
R^{(2)}(t)=
\frac{\|f_\perp\|^2+\sum_i c_i^2e^{-2\lambda_i t}}
{\|f\|^2}.
}
$$

对固定 SGD 步长 $\eta$，$0<\eta<2/\lambda_{\max}$，做 $k$ 次更新则将指数因子替换为 $(1-\eta\lambda_i)^{2k}$。预测的吸收比例为 $1-R^{(2)}$。该表达式同时包含目标能量、学生快慢方向和优化预算，比单个散度多出的正是这些方向信息。

定义低响应能量占比

$$
m_\tau=
\frac{\|f_\perp\|^2+\sum_{\lambda_i\le\tau}c_i^2}{\|f\|^2}.
$$

对 $\lambda_i\le\tau$，指数因子至少为 $e^{-2\tau t}$；对其余方向，该因子至多为此值。分别求和得到

$$
e^{-2\tau t}m_\tau\le R^{(2)}(t)
\le m_\tau+e^{-2\tau t}(1-m_\tau).
$$

这使“固定预算内还会剩多少”对应到目标在慢方向上的能量。$f_\perp$ 只是当前固定特征下的局部不可达项；非线性表示变化可以改变它。同架构 teacher-copy 仍是全模型的可行点，不能用局部残差宣称永久不可表示。

### 11.3 可以便宜测量的初始速率

二次代理的初始化下降率为

$$
-\dot L^{(2)}(0)=\|S^*f\|^2=\|u_0\|^2.
$$

因此初始相对速率由

$$
\nu_0=\frac{\|u_0\|^2}{\|f\|^2}
\simeq\frac{\|u_0\|^2}{2L_\mu(\theta_0)}
=\frac{\sum_i\lambda_i c_i^2}{\|f\|^2}
$$

决定，$-\dot L/L\simeq2\nu_0$。固定预条件器下将分子改为 $u_0^\top M u_0$。它测量单位剩余差异能产生多少真实下降方向，不能用未经归一化的小梯度区分“接近 teacher”和“仍远但信号弱”。

初始速率也不唯一决定终点：在同一个算子的特征值 0.01、1、1.99 上，令一个信号的全部能量位于 1，另一个将能量均分到 0.01 与 1.99。两者的初始风险与 $\nu_0=1$ 都相同，$t=10$ 的剩余比例却分别为 $e^{-20}$ 和约 0.4094。单步速率是低成本预测特征，完整预算预测仍需要谱形状或多个早期读出。

相同测度下，完整局部谱预测还可写成参数空间的矩阵函数：

$$
1-R_k^{(2)}
=\frac{u_0^\top F^+\left[I-(I-\eta F)^{2k}\right]u_0}{\|f\|^2}.
$$

证明由 $S$ 的奇异值分解得到：在特征值 $\lambda_i$ 上，$u_0$ 的系数平方是 $\lambda_i c_i^2$，$F^+$ 恰好消去这个因子。零特征值对应 $u_0$ 分量为零。这个形式可用 Fisher-vector product 和 Lanczos 近似，无需显式存储完整 Fisher。它是待验证的测量方案，不表示本项目已经完成 LLM 谱估计；teacher-conditioned loss-gradient Gram 也不能直接代替该 Fisher。

### 11.4 留出预测必须包含跨题传播

若训练与评估前缀分布不同，记其下降方向为 $u_{\rm tr}$、$u_{\rm val}$。一次固定预条件更新 $\Delta\theta=\eta M u_{\rm tr}$ 的 Taylor 展开为

$$
L_{\rm val}(\theta+\Delta\theta)
=L_{\rm val}(\theta)-\eta u_{\rm val}^\top M u_{\rm tr}
+O(\eta^2\|M u_{\rm tr}\|^2).
$$

因此最直接的留出预测特征是

$$
\Gamma_{\rm tr\to val}
=\frac{u_{\rm val}^\top M u_{\rm tr}}{L_{\rm val}(\theta_0)}.
$$

这比只测训练梯度范数更接近“未见题能吸收多少”；它允许负迁移。两个前缀分布的 Fisher 都是正半定，也不保证同一目标位移产生正的跨域内积，见 CPU 反例。

用独立诊断批估计时，交叉内积避免把自内积中的采样方差算成总体有效信号。要求冻结参数与 $M$，并区分 teacher 候选采样、提示采样和前缀来源。真实 AdamW、动量、clipping 和 weight decay 下，优先测 $u_{\rm val}^\top\Delta\theta_{\rm actual}$，或在 4–16 个探测更新后直接测固定开发 bank 的相对风险下降。用开发 bank 选择预测量，永久测试 bank 只用于最终验证；不能让测试梯度参与更新。

### 11.5 RL 一步更新为何可能产生较少慢方向能量

现在把“RL 更容易学”限制到一个可证明模型：共同初始化、共同固定测度、有限二阶矩的真实优势 $A$，先做一小步普通精确策略梯度。Teacher 参数位移为

$$
\delta_R=\gamma S^*A.
$$

对 teacher 函数变化做一阶展开，得到 $f_R\simeq S\delta_R=\gamma\mathcal K A$。记 $A=\sum_i a_i\phi_i+A_\perp$，则 $c_i=\gamma\lambda_i a_i$。弱响应方向多乘了一个 $\lambda_i$，所以二次代理中有

$$
L_R^{(2)}(t)=\frac{\gamma^2}{2}\sum_i\lambda_i^2a_i^2e^{-2\lambda_it}.
$$

对 $\lambda\ge0$，函数 $\lambda^2e^{-2\lambda t}$ 在 $\lambda=1/t$ 取最大值 $1/(e^2t^2)$。逐项放缩得

$$
\boxed{L_R^{(2)}(t)\le\frac{\gamma^2\|A\|^2}{2e^2t^2},\qquad t>0.}
$$

慢方向初始能量也满足 $\frac12\sum_{\lambda_i\le\tau}c_i^2\le\frac12\gamma^2\tau^2\|A\|^2$。这是局部 teacher 一步更新和固定二次 OPD 模型下的绝对风险界；高阶 teacher 展开误差没有被这个公式控制。

它不保证相对剩余比例小。若 $A$ 全在唯一慢方向 $\lambda=0.01$，则 teacher 初始能量很小，剩余比例仍是 $e^{-0.02t}$，在 $t=10$ 时约 81.87%。长程 RL、自然梯度、Adam、状态改变和大 teacher 位移也不自动保留这个简单滤波形式；若两阶段使用同一个固定预条件器，可以在变换后的算子 $\mathcal K_M$ 中重做推导。

同测度的一小步 SFT 同样可写成 $\delta_S=\gamma S^*h$，其中 $h=\rho/p_0-1$，所以同一个二次模型也给出

$$
L_S^{(2)}(t)\le\frac{\gamma^2\|h\|^2}{2e^2t^2},
\qquad
\|h\|^2=\mathbb E_\mu\chi^2(\rho\Vert p_0).
$$

Hard label 的二阶矩可以很大，这使 SFT 的界变弱。差别是有界真实优势与可能巨大的示范密度比，以及两者是否共用状态测度；不是 SFT 不存在中心化，也不是它一定写入不可学方向。这里的 $\rho$ 是构造 teacher 时的示范目标，不是已经训练完成的 teacher 分布 $T$。两种 Pearson 散度不能混用。

若 SFT 前缀来自外部示范测度 $\mu_D$，在评估测度上看到的变化是 $\gamma S_{\rm eval}S_D^*h_D$，一般不再是同一正半定算子作用于同一优势。这个跨分布算子是状态覆盖与跨题共享机制进入推导的位置。

### 11.6 理想 RL 最优解只保证目标对齐

另一种模型在自由分布上优化真实奖励与参考 KL，得到 $T\propto p_0e^{\beta A}$。这和上一节的“一小步参数梯度 teacher”是不同构造。固定状态下精确有 $f=\beta C_{p_0}A$、$u_0=\beta S^*A$，即初始 OPD 更新与真实 RL 方向对齐；序列最优解及终局奖励的条件展开见[组合教师的对应推导](2026-11-01-SFT与RL组合教师的可蒸馏性.md#signal)。

此处 $f$ 没有额外的 $\mathcal K$ 滤波。若优势就在慢方向，仍然会学得慢。梯度对齐不能单独推出条件数良好或固定预算完全吸收。既有局部恢复论文也明确说明其证明不使用 teacher 的训练方式，条件是同参数空间、足够接近与相应正则性和谱尾控制。[^vanishing]

### 11.7“不像 RL”的部分应拆成任务对齐与可学习性

固定实际任务优势 $A$ 后，才可以检验 teacher 与 RL 的相似性。任意正 teacher 都可人为设置伪奖励 $\log(T/p_0)$，所以不固定真实 reward 时，“像 RL teacher”没有区分力。

函数空间投影可写成 $f=\alpha A+\xi$、$\langle A,\xi\rangle=0$。但 $\langle A,\xi\rangle=0$ 不推出 $(S^*A)^\top(S^*\xi)=0$，因为后者含有算子 $\mathcal K$。若想解释真实更新的一阶任务收益，应改用参数更新度量。令 $u_A=S^*A$，$u_D=S^*f$，固定 $M$，定义

$$
b=\frac{u_A^\top M u_D}{u_A^\top M u_A},
\qquad u_\perp=u_D-bu_A.
$$

分母非零时，$u_A^\top M u_\perp=0$。这个残差在固定测度任务目标上没有一阶收益，却仍可被 OPD 快速拟合；例如任务梯度在慢方向、teacher 残差在另一个快方向。它也可能携带其他任务的有用知识，不能自动称作无用风格。

因此分别报告：真实任务对齐程度；teacher 差异的快慢谱；独立提示上的更新传播。只有观察到非任务分量集中于慢方向且留出传播弱，才支持“这部分在给定预算下难吸收”的机制。把一次任务投影的残差直接当成不可学习比例，没有上述理论依据；函数投影与谱投影一般也不交换，不能把两条独立拟合曲线随意相加。

### 11.8 建议的预测指标与验算

主评估继续用永久留出 $\mathrm{KL}(p_k\Vert T)$ 和任务增量恢复，预测特征按成本分层：

| 成本                   | 预测特征                                                              | 适用解释                                     |
| ---------------------- | --------------------------------------------------------------------- | -------------------------------------------- |
| 已有 forward           | 初始双向 KL、$\chi^2(T\Vert p_0)$、中心化 log-ratio 方差、尾部与长度  | 距离和密度比特征；单独不决定吸收             |
| 少量 backward          | 归一化跨题下降 $\Gamma_{\rm tr\to val}$、独立批交叉内积、真实任务对齐 | 初始留出速率与共享更新方向                   |
| 4–16 个实际更新        | 相同 optimizer 下两个固定开发 bank 的相对下降、后续斜率               | 实际初期可吸收性；不假定长期单指数           |
| Fisher-vector products | 目标加权谱预测 $R_k^{(2)}$ 或慢方向质量 $m_\tau$                      | 在局部、固定测度和指定预算下预测剩余分布风险 |

如果只允许一个低成本指标，优先使用相同 optimizer 下的固定开发 bank 相对下降；若要求训练前测量，使用归一化跨题梯度内积。理论上更完整的是预算条件化的谱预测，而非一个 teacher 固有的 KL 数值。所有预测都需明确预算、optimizer、训练来源和评估 bank；分布吸收与任务能力恢复分别建模。

CPU/FP64脚本 [absorption_predictor_checks.py](../../research/studies/fixed_transfer/absorption_predictor_checks.py)验证同散度反例、谱递推、RL 一步绝对风险界、相对慢收敛反例，以及 PSD 曲率下的跨分布负迁移。命令在仓库根目录执行，输出目录必须尚不存在：

```bash
OPENBLAS_NUM_THREADS=1 python -m examples.mirroropd.research.studies.fixed_transfer.absorption_predictor_checks --output runs/absorption-predictor-recheck
```

两个非线性二动作 teacher 的实测值如下；所有指标在同一固定前缀测度上计算，训练使用完整动作 RKL，不含采样噪声。

| Teacher    |    初始 RKL |    初始 FKL | Pearson 散度 | 归一化初始梯度能量 | 64 步剩余比例 |  局部谱预测 |
| ---------- | ----------: | ----------: | -----------: | -----------------: | ------------: | ----------: |
| 快响应状态 | 0.000624740 | 0.000624220 |  0.001247920 |           1.000417 |   0.000001426 | 0.000001390 |
| 慢响应状态 | 0.000624740 | 0.000624220 |  0.001247920 |           0.010004 |      0.879748 |    0.879797 |

谱递推与闭式解的最大误差为 $1.7\times10^{-16}$。RL 一步绝对风险界在所测时间点成立；唯一慢方向的 RL 例子在 $t=10$ 仍剩 81.87%，而一个与指定任务梯度正交的快方向 teacher 信号已经降至 $2.1\times10^{-9}$。跨分布例子的预测一阶收益为−1，实测单位步长收益为−1.00005，验证 PSD 曲率不保证跨分布正迁移。

[完整数值与逐步曲线](data/absorption-predictor-20261108/checks.json)、[运行时源码](data/absorption-predictor-20261108/source.py)及[来源和 hash](data/absorption-predictor-20261108/provenance.json)一起冻结。这不是 LLM 训练、真实 RL/SFT 分类结果或未来 320 步吸收率的已验证预测器。

<a id="razor-and-ood"></a>

### 11.9 RL’s Razor 的预测对象与 KL 方向

该文与数据构造最直接的关系是二值奖励的概率投影。令 $C=\{y:R(y)=1\}$、$z=p(C)>0$，则

$$
q_C(y)=p(y\mid C)
=\arg\min_{q:q(C)=1}\mathrm{KL}(q\Vert p),
\qquad
\mathrm{KL}(q\Vert p)=\mathrm{KL}(q\Vert q_C)-\log z.
$$

最后的分解直接证明最优解保留 $p$ 在正确集合内的相对偏好。对当前策略冻结采样、记 $q_t=p_tR/Z_t$ 时，

$$
\nabla_\theta\mathbb E_{p_\theta}R\big|_{\theta_t}
=Z_t\,\mathbb E_{q_t}\nabla_\theta\log p_\theta\big|_{\theta_t}.
$$

因此一次 REINFORCE 梯度可写成“从自身分布挑正确回答，再做似然上升”，两边相差奖励通过率尺度 $Z_t$；若将成功条件分布的 SFT 梯度直接归一化为均值，必须处理这个步长尺度。这个身份解释我们为什么研究 Base 候选、奖励重加权和直接奖励训练，而不是把 SFT 名称本身当作失败原因。CCS 将硬接受放宽为有界软重加权，ERT 则保持直接 on-policy 更新。

该身份先针对固定 prompt。多 prompt 时，$Z_t(x)$ 随题变化；每题固定抽一条正确答案的 SFT 与原始 reward-weighted policy gradient 还差逐题的 $Z_t(x)$ 权重，不能只用一个全局学习率补偿。有限候选的全失败题也没有可抽取的正确条件分布，这正是需要记录覆盖率、保留 uniform 对照的原因。

论文接着把条件化视为 I-projection、把拟合 $q_t$ 视为 M-projection。其最小 KL 极限的正式条件包含 e-flat 模型族、可实现最优集合，以及精确拟合或可求和的拟合误差；作者也明确承认神经网络策略族一般不满足 e-flat 条件。一次有限 Adam 更新只沿投影目标的梯度走一步，不等于已经完成那个分布投影。若正确概率必须达到 1，还需允许模型族的边界极限；严格全支持策略在存在错误回答时不能有限步给错误集合零质量。因此不能将该理想解释转述成任意 GRPO/PPO/Adam 都保证找到离初始模型最近的最优策略，更不能由此直接断言其 teacher 能被另一轮 OPD 迅速恢复。[^razor]

RL’s Razor 的 empirical forgetting law[^razor] 使用新任务输入分布 $\nu_{\rm new}$ 上的

$$
D_{\rm Razor}=\mathbb E_{x\sim\nu_{\rm new}}
\mathrm{KL}\bigl(B(\cdot\mid x)\Vert T(\cdot\mid x)\bigr)
$$

解释旧任务能力下降。附录 Alternative Hypothesis 明确把 $B\Vert T$ 称为 forward KL，把 $T\Vert B$ 称为 reverse KL。表 1 的二次回归 $R^2=0.96\pm0.01$ 与 $0.93\pm0.01$ 来自 MNIST 设置；正文 LLM 结果的二次拟合为 $R^2=0.71$。这些数值不能移作 LLM 的 OPD 吸收预测精度。

在本项目符号下，$D_{\rm Razor}$ 的条件概率方向恰好与初始化 OPD 风险 $\mathrm{KL}(p_0\Vert T)$ 相同，其中 $p_0=B$。名字不同是参照对象不同。LLM 实验还需说明它测的是整条回答的 sequence KL，还是指定前缀 bank 上的条件 token KL；两者只有在对应 rollout 测度与长度求和一致时才由链式法则相等。固定长度截取、逐题 token 均值或换成 teacher 前缀后都不能直接沿用 sequence 等式。

**投影方向与遗忘预测方向也不同。**上面的最小化使用 $q\Vert B$，经验规律使用 $B\Vert T$。硬条件化的 $q_C$ 满足 $\mathrm{KL}(q_C\Vert B)=-\log z$、$\chi^2(q_C\Vert B)=1/z-1$；当 $0<z<1$ 时，$\mathrm{KL}(B\Vert q_C)=\infty$。因此“正确集合内 KL 最小”不能自动解释为两个方向都小。CCS 的总体生成边缘若满足 $\epsilon\le Q/B\le w_{\max}$ 和 Pearson 预算 $\rho$，则有

$$
\mathrm{KL}(Q\Vert B)\le\log(1+\rho),
\qquad
\mathrm{KL}(B\Vert Q)\le-\log\epsilon.
$$

第二个界来自逐点 $\log(B/Q)\le-\log\epsilon$ 后取 $B$ 期望。它解释保留失败候选和正权重下限的作用；这些界仍只约束总体生成分布，不约束有限经验样本或训练后的神经 teacher。

论文对我们的帮助有两层。一层是**应该保留 $B\Vert T$ 为强经验基线**：分布距离虽不能普遍决定学习速度，在受限模型族和训练轨迹上仍可能有预测力。另一层是将“优化器名称”改为“产生了什么数据分布”的可检验假设：论文用 RL 模型生成监督数据，SFT 可以逼近其能力—遗忘权衡。我们的 RAW、Base 候选重加权与直接奖励训练因此要共用初始化，并同时报告能力、距离和二次吸收。

可以用共同局部位移 $d=\theta_T-\theta_B$ 理解两者的联系。以 $B$ 为软目标的旧域交叉熵增加恰好是旧域 $\mathrm{KL}(B\Vert T)$；局部展开为

$$
D_{\rm new}=\tfrac12d^\top F_{\rm new}d+O(\|d\|^3),
\qquad
D_{\rm old}=\tfrac12d^\top F_{\rm old}d+O(\|d\|^3).
$$

若一组实际训练位移集中在两个二次型近似成比例的方向上，新域 KL 可以强烈预测旧域分布变化；这是一种解释其经验规律的附加几何条件，不是该论文已经证明的任意 OOD 定理。旧任务准确率还取决于分类 margin，不能直接等同于上述交叉熵代理。OPD 又增加了随更新预算变化的残差算子，因此即使初始 KL 能预测遗忘，也仍需检验它能否预测经过优化后的剩余比例。

它没有建立“离 Base 越近，OPD 相对吸收比例必然越高”的定理。Razor 的因变量是旧任务遗忘，本节的因变量是指定优化预算后学生到 teacher 的剩余风险；前者还包含旧任务决策边界，后者包含学生响应谱和蒸馏 prompt 覆盖。表 1 也不是对任意 teacher 留一法或跨架构预测器的保证。实验应分别预测旧能力保持、绝对剩余 RKL、相对剩余 RKL 和能力增量恢复，不能用其中一个的相关性替另一个背书。

### 11.10 IID、有限样本与 OOD 的不同命题

前文固定 $\mu$ 可以是总体分布，不必是训练题的经验分布。因此它可以讨论同分布新题的总体风险；若实际只在有限训练 bank 上估计 $S$ 和 $F$，则还缺少从经验算子到总体算子的泛化控制。**前文谱公式本身没有提供任意 OOD 保证**。Razor 反而已经在经验上研究了新任务变化与旧任务遗忘的关系，我们需要补的是蒸馏训练到各个评估域的响应关系。

将测度写成 $\mu_d(x,s)=\nu_d(x)\kappa_d(s\mid x)$：$\nu_d$ 是 prompt 分布，$\kappa_d$ 是该 prompt 下的前缀分布。新数学题主要检验同域 prompt 泛化；代码、科学与指令题检验跨域 prompt 迁移；Base bank 换为 teacher bank 检验前缀来源迁移。真实在线 OPD 的 $\kappa$ 还随 actor 更新，以下固定测度分析只描述单个局部窗口，不能将它当作完整在线轨迹的精确解。

给训练域和评估域分别定义 $S_{\rm tr}$、$S_d$、$f_{\rm tr}$、$f_d$，都以同一个 $B$ 的 score 为参数化起点。训练二次代理为

$$
\widehat L_{\rm tr}(\delta)=\tfrac12\|f_{\rm tr}-S_{\rm tr}\delta\|^2,
\qquad F_{\rm tr}=S_{\rm tr}^*S_{\rm tr}.
$$

从 $\delta_0=0$ 做普通 SGD，令 $A=I-\eta F_{\rm tr}$，递推 $\delta_{k+1}=A\delta_k+\eta S_{\rm tr}^*f_{\rm tr}$，逐次代入得

$$
\delta_k=\eta\sum_{j=0}^{k-1}A^jS_{\rm tr}^*f_{\rm tr},
\qquad
e_{d,k}=f_d-\eta\sum_{j=0}^{k-1}S_dA^jS_{\rm tr}^*f_{\rm tr}.
$$

因此 OOD 剩余风险为 $\widehat L_d(k)=\tfrac12\|e_{d,k}\|^2$。控制评估残差的是**训练—评估交叉响应** $S_dA^jS_{\rm tr}^*$，单独的训练域特征值和训练 KL 没有包含它。初始化一步的评估风险变化满足

$$
\widehat L_d(\eta u_{\rm tr})
=\widehat L_d(0)-\eta\,u_d^\top u_{\rm tr}
+\tfrac{\eta^2}{2}u_{\rm tr}^\top F_du_{\rm tr},
\qquad u_d=S_d^*f_d.
$$

交叉内积 $u_d^\top u_{\rm tr}$ 可以为负；此时在足够小的正步长下，训练风险下降而 OOD 风险上升。实际 Adam 的开发集读出应使用 $u_d^\top\Delta\theta$；若报告归一化量，则除以该域初始风险，并在初始风险接近零时直接报告绝对变化。

在更强的**共同局部 teacher 位移**假设下，存在同一个 $d$ 使所有域满足 $f_d=S_dd$。此时 $S_{\rm tr}^*f_{\rm tr}=F_{\rm tr}d$，几何级数不要求 $F_{\rm tr}$ 可逆，得到

$$
\delta_k=(I-A^k)d,
\qquad
\boxed{\widehat L_d(k)=\tfrac12d^\top(A^k)^\top F_dA^kd.}
$$

这里 $F_d$ 一般不与 $F_{\rm tr}$ 交换，不能简单把训练域的 $c_i^2$ 换成另一个域的数值后照搬独立指数衰减。一个完全可实现的例子是 $F_{\rm tr}=\operatorname{diag}(1,0)$、$F_d=\operatorname{diag}(0,1)$、$d=(1,1)$：梯度流的训练风险为 $\tfrac12e^{-2t}$，评估风险始终为 $\tfrac12$。相同参数族、有训练域收敛，都不补上未覆盖方向。

一个可验证的充分覆盖条件是 $F_d\preceq cF_{\rm tr}$。对 $v=A^kd$ 使用二次型序即可得到

$$
\widehat L_d(k)\le c\widehat L_{\rm tr}(k),
\qquad
\frac{\widehat L_d(k)}{\widehat L_d(0)}
\le c\frac{\widehat L_{\rm tr}(0)}{\widehat L_d(0)}
\frac{\widehat L_{\rm tr}(k)}{\widehat L_{\rm tr}(0)}.
$$

相对形式要求两个初始风险为正。任意跨域不存在统一的小 $c$，有限训练 bank 的 Fisher 还可能在某些 OOD 方向恰好为零。这个充分条件给出应测的“覆盖”含义，不是默认成立的假设。

RL 一步的过滤解释也要标注域：若 teacher 位移为 $d=\gamma S_{\rm RL}^*A_{\rm reward}$，只有在 RL 与蒸馏采用相同响应算子时，才可直接使用前文的共同谱推导；换成蒸馏域 $S_{\rm tr}$ 和 OOD 域 $S_d$ 后，要分别处理 $S_{\rm tr}S_{\rm RL}^*$ 与 $S_dS_{\rm RL}^*$。训练奖励域上的有界优势不能独自保证这些交叉算子传递所有 teacher 变化。SFT 在共同局部模型下也可获得条件性可学结论，区别落在实际信号大小、方向、覆盖和非线性轨迹，而不是方法标签。

预测实验据此把输入特征分成初始距离、训练到各域的开发集下降、前缀来源差异。每域单独预测最终 RKL，再报告域宏平均；用开发 bank 选择特征，用永久测试 bank 评估预测。首轮单 seed 四条构造路径只适合检验曲线和明显反例，无法可靠估计跨 teacher 泛化的回归 $R^2$。

<a id="sequence-distance"></a>

### 11.11 固定前缀风险与真实序列距离

设 $P$ 为当前学生、$Q$ 为固定教师；固定题目 $x$，共同截断长度为 $H$，EOS 后进入共同吸收态。采样与评分使用同一温度、支持集和条件分布。设两者在有效轨迹上互相绝对连续，条件 KL 可积。截断轨迹概率分解为

$$
P^H(y\mid x)=\prod_{t=1}^H P_t(y_t\mid x,y_{<t}),
\qquad Q^H(y\mid x)=\prod_{t=1}^H Q_t(y_t\mid x,y_{<t}).
$$

因此 log-ratio 是逐步 log-ratio 的和。对 $P^H$ 取期望，并对每个前缀先积分当前动作，得到

$$
\begin{aligned}
D_{\mathrm{KL}}(P^H\Vert Q^H)
&=\mathbb E_{Y\sim P^H}\sum_{t=1}^H
\log\frac{P_t(Y_t\mid x,Y_{<t})}{Q_t(Y_t\mid x,Y_{<t})}\\
&=\sum_{t=1}^H\mathbb E_{Y_{<t}\sim P}
D_{\mathrm{KL}}\!\left(P_t(\cdot\mid x,Y_{<t})\Vert Q_t(\cdot\mid x,Y_{<t})\right).
\end{aligned}
$$

交换 $P,Q$ 得到另一方向：

$$
D_{\mathrm{KL}}(Q^H\Vert P^H)
=\sum_{t=1}^H\mathbb E_{Y_{<t}\sim Q}
D_{\mathrm{KL}}\!\left(Q_t(\cdot\mid x,Y_{<t})\Vert P_t(\cdot\mid x,Y_{<t})\right).
$$

前缀来源和条件 KL 的第一参数必须对应。Teacher 轨迹上的 $P_t\Vert Q_t$ 只定义了另一种固定测度风险；它没有上述序列 RKL 或 FKL 的等式。Base 轨迹的优点是各 checkpoint、各 teacher 共用状态，适合测量同状态拟合与行为保持；其覆盖范围由 Base 的状态访问决定。把两种固定 bank 的 RKL 平均，也不能得到两种序列 KL 的平均。

对题分布 $\mu$，主行为读出分别取 $\mathbb E_{x\sim\mu}D_{\mathrm{KL}}(P^H\Vert Q^H)$ 与 $\mathbb E_{x\sim\mu}D_{\mathrm{KL}}(Q^H\Vert P^H)$。前者逐 checkpoint 更新学生轨迹，后者可复用固定教师轨迹。固定 $H$ 下，二者都从非负条件 KL 求和估计；不需要对整条路径密度比做重要性加权。非负是精确条件 KL 的性质，有限精度实现中的微小负值另行审计。按实际长度先除后平均会改变目标：

$$
\mathbb E\!\left[\frac1{L(Y)}\sum_t k_t(Y_{<t})\right]
\ne\frac{\mathbb E\sum_t k_t(Y_{<t})}{\mathbb E L(Y)}
\ne\mathbb E\sum_t k_t(Y_{<t}).
$$

其中 $L$ 是截至 EOS（含 EOS）的有效长度，末项是序列量；只有指定固定长度等附加条件时，前两项才可简化为共同归一化。实际保存逐题条件 KL 总和、有效长度和停止原因，再分别报告 nats/sequence 与 pooled nats/token。各域先各自估计，macro 另外声明权重。公共混合前缀可用于受控定位状态覆盖，行为主读出仍使用两个链式法则方向。

这些等式也要求明确测哪一个策略：若用 top-p0.95、温度变换或重复停止生成，再用未经相同变换的模型 logits 评分，前缀测度与被积策略通常不同。此时可报告该固定测度下的模型风险，不能直接标为原模型 sequence KL。重复停止还改变存活到后续状态的概率；若要测该执行策略，采样和评分都必须表示共同的停止规则与状态。HF/vLLM、不同 head 精度与 batch/context 数值漂移属于等式落到实现时要核验的分布偏差。

序列距离很小也不等于所有任务增量相同，更不意味着未来固定预算的拟合必然快。前者需用同协议的任务正确率、teacher 相对 Base 增益与学生恢复率评价；后者还取决于 §11.2–11.4 的响应谱和训练到目标域的梯度传递。初始距离、固定状态诊断、当前策略距离与任务增益各回答一个不同问题。

### 参考文献

[^paper]: Kejian Zhu et al. SFT Conflicts, RL Coexists: A Theoretical and Empirical Analysis of Multi-Task Learning for LLMs. arXiv:2608.03573v2, 2026. [版本入口](https://arxiv.org/abs/2608.03573v2)、[正文与附录](https://arxiv.org/html/2608.03573v2)。

[^vanishing]: Lei Zhao et al. Why On-Policy Distillation Sometimes Fails: Vanishing Learning Signals. arXiv:2610.11247v1, 2026. [局部恢复与谱分析](https://arxiv.org/html/2610.11247v1)。

[^razor]: Idan Shenfeld, Jyothish Pari, Pulkit Agrawal. RL’s Razor: Why Online Reinforcement Learning Forgets Less. arXiv:2509.04259v1, 2025. [正文与 Alternative Hypothesis 附录](https://arxiv.org/html/2509.04259v1)。
