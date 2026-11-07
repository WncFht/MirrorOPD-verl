# Rollout 来源、最优融合与等效训练权重

更新日期：2026-11-06。本文统一讨论三个步骤：选择训练前缀的来源，求解该来源分布，以及将其改写为 fresh student rollout 上的加权更新。Initial、teacher、逐 token 融合、完整序列中心和 IW-OPD 都按这三个步骤比较。

全文使用 $\eta$ 表示 teacher 权重。局部 RKL 学习规则固定时，更新起点的等效 advantage 为 $A_t=W_t\Delta_t$：$W_t$ 改变前缀的训练权重，$\Delta_t$ 保留 teacher 对 student 的局部监督。这个关系是期望局部梯度的恒等式；是否改善最终任务表现，需要实验验证。

| 阅读目的                               | 位置                          |
| -------------------------------------- | ----------------------------- |
| 确定概率空间、来源与 learner loss      | [第 1 节](#notation)          |
| 逐步推导等效 advantage                 | [第 2 节](#change-of-measure) |
| 从不同优化目标求融合分布               | [第 3 节](#centers)           |
| 比较 initial、fusion 与 IW-OPD 的权重  | [第 4 节](#source-weights)    |
| 判断单条 rollout 能算哪些权重          | [第 5 节](#computability)     |
| 求解系数、检查误差与比较 sum / minimax | [第 6 节](#solver)            |
| 核对执行条件、公开实现和数值证据       | [第 7 节](#evidence)          |

<a id="notation"></a>

## 1. 来源分布与学习目标

固定 prompt $x$。$\mu_0$ 是 initial student，$\mu=\mu_{\bar\theta}$ 是刚生成 fresh rollout 的 student 快照，$\mu_\theta$ 是正在更新的 learner，$\pi$ 是冻结 teacher。每批更新固定来源和采样快照；同批重复更新后，$\mu_\theta$ 可以不同于 $\mu$。

| 符号                                                               | 统一含义                                                     |
| ------------------------------------------------------------------ | ------------------------------------------------------------ |
| $h_t=(x,y_{<t})$                                                   | 位置 $t$ 的前缀                                              |
| $q(a\mid h)$                                                       | 来源策略的下一 token 分布                                    |
| $P_q(y\mid x)=\prod_tq(y_t\mid h_t)$                               | 逐步生成得到的完整回答分布                                   |
| $Q(y\mid x)$                                                       | 直接在完整回答空间定义的来源分布                             |
| $d_{q,t}(h)$                                                       | 来源在位置 $t$ 访问前缀 $h$ 的概率                           |
| $r_t=\pi(y_t\mid h_t)/\mu(y_t\mid h_t)$                            | 当前动作的 teacher/student 概率比                            |
| $R_t=P_\pi(y_{<t})/P_\mu(y_{<t})=\prod_{k<t}r_k$                   | 前缀概率比                                                   |
| $\Delta_{\theta,t}=\log[\pi(y_t\mid h_t)/\mu_\theta(y_t\mid h_t)]$ | Learner 的局部 RKL advantage；更新起点为 $\Delta_t=\log r_t$ |
| $W_t=d_{q,t}/d_{\mu,t}$                                            | 相对 fresh student 的前缀密度比                              |
| $\eta\in[0,1]$                                                     | Teacher 权重；$\rho$ 仅表示 KL 预算                          |
| $\beta,\tau,\kappa$                                                | 分别为 divergence 阶数、平滑参数、sum/max 折中参数           |

IW-OPD 论文的 $\alpha$ 对应本文 $\eta$；旧版本文和部分冻结核验数据使用 student 权重，对应 $1-\eta$。后文公式统一采用 teacher 权重，不沿用两种相反的 $\alpha$ 约定。

来源决定在哪些前缀上学习，learner loss 决定在这些前缀上如何学习。主线固定条件 reverse KL（RKL）：

$$
K_\theta(h)=\mathrm{KL}(\mu_\theta(\cdot\mid h)\Vert\pi(\cdot\mid h)),\qquad
L_q(\theta)=\sum_{t=1}^{T}\mathbb E_{h\sim d_{q,t}}K_\theta(h).
$$

取有限最大长度 $T$，EOS 后吸收态的损失为零。各来源共享 prompt 分布、动作语义和停止协议。以下在共同合法支持上假设概率严格为正；换测度至少要求目标前缀被 proposal 覆盖，动作校正还要求 $\mu_\theta\ll\mu$。

更新时只对 $K_\theta$ 求导，不对来源或离散采样反传。即使来源由当前师生求解，也先冻结该快照。Full-vocab 表示固定前缀上的词表求和完整计算，不表示已经对整个序列状态分布求了梯度。来源优化与 learner divergence 可以分别改变。

<a id="change-of-measure"></a>

## 2. 从任意来源到 fresh rollout 的统一证明

### 2.1 固定前缀上的局部梯度

固定 $h$，简记 $p_a=\mu_\theta(a\mid h)$。对条件 RKL 直接微分：

$$
\begin{aligned}
\nabla_\theta K_\theta(h)
&=\sum_a\nabla_\theta p_a\left(\log\frac{p_a}{\pi_a}+1\right)\\
&=\sum_a p_a\log\frac{p_a}{\pi_a}\nabla_\theta\log p_a.
\end{aligned}
$$

第二步使用 $\sum_a\nabla p_a=\nabla1=0$。令 $s_\theta(h,a)=\nabla_\theta\log\mu_\theta(a\mid h)$，得到

$$
-\nabla_\theta K_\theta(h)
=\mathbb E_{a\sim\mu_\theta(\cdot\mid h)}[\Delta_\theta(h,a)s_\theta(h,a)].
$$

Full-vocab 计算这个动作期望；sampled policy gradient 用 detached 的 $\Delta_\theta$ 乘 score 估计它。仅对一个已采样的 log-ratio 直接反传，不会产生上述梯度。

### 2.2 换前缀测度与当前动作测度

对任意前缀函数 $F$，展开离散求和：

$$
\begin{aligned}
\mathbb E_{d_{q,t}}F(h)
&=\sum_hd_{q,t}(h)F(h)\\
&=\sum_hd_{\mu,t}(h)\frac{d_{q,t}(h)}{d_{\mu,t}(h)}F(h)
=\mathbb E_{d_{\mu,t}}[W_t(h)F(h)].
\end{aligned}
$$

将 $F=-\nabla K_\theta$ 代入，再换动作分布。fresh rollout 的 token 来自快照 $\mu$，重复更新时需要 $U_t=\mu_\theta(y_t\mid h_t)/\mu(y_t\mid h_t)$：

$$
\boxed{
-\nabla_\theta L_q
=\mathbb E_{y\sim P_\mu}\sum_t
\underbrace{W_tU_t\Delta_{\theta,t}}_{A_t^{q\leftarrow\mu}}
s_\theta(h_t,y_t).
}
$$

实现 sampled surrogate 时，对整个 advantage 系数应用 stop-gradient。更新起点 $\theta=\bar\theta$ 有 $U_t=1$；full-vocab 路径已完成动作积分，任意更新步均无需动作比：

$$
\boxed{
A_t^{q\leftarrow\mu}=W_t\Delta_t\quad(\theta=\bar\theta),\qquad
L_q(\theta)=\mathbb E_{P_\mu}\sum_t\operatorname{sg}[W_t]K_\theta(h_t).
}
$$

这就是本文的“等效 reward”：固定来源的期望局部更新可以改写成 fresh 样本上的加权更新。前缀权重取到 $t-1$，因为当前动作积分仍属于 learner。若直接用 $q$ 采出的 token 估计 sampled 梯度，则另需 $\mu_\theta(y_t\mid h_t)/q(y_t\mid h_t)$；直接计算 full-vocab KL 则无需这一步。

证明也适用于逐位置单独指定的目标前缀分布 $\nu_t$，只需令 $W_t=\nu_t/d_{\mu,t}$。这些 $\nu_t$ 不一定是同一个序列分布的边缘；这正好容纳 IW-OPD 的理论前缀 surrogate。

### 2.3 来源固定、归约与 learner loss

主证明针对有效位置求和。确定性的位置系数可以并入 $K_\theta$；population token-mean 还需除以目标来源的期望有效 token 数。有限 batch 的随机分母会产生额外估计差异。

回答平均的 $1/L(y)$ 依赖后续长度。令 $c_{q,t}(h)=\mathbb E_q[1/L(y)\mid h_t=h]$，换到 fresh 前缀后的精确系数是 $W_t(h)c_{q,t}(h)$；直接使用 fresh 回答的长度一般不同。也可对完整回答函数使用序列 IS 比，随后再按条件期望处理局部动作梯度。

若 learner 改用 $D_\varphi(\mu_\theta\Vert\pi)=\sum_a\pi_a\varphi(\mu_{\theta,a}/\pi_a)$，同样直接微分得

$$
-\nabla_\theta D_\varphi
=\mathbb E_{a\sim\mu_\theta}
\left[-\varphi'(\mu_\theta(a)/\pi(a))s_\theta(h,a)\right].
$$

所以来源权重 $W_t$ 保留，局部信号替换为 $-\varphi'$；允许去掉只依赖前缀的 baseline。RKL 给出 $\log(\pi/\mu_\theta)$，forward KL 给出 $\pi/\mu_\theta$，归一化平方 Hellinger 给出 $\tfrac12\sqrt{\pi/\mu_\theta}$。使用 Hellinger 来源中心与使用 Hellinger learner loss 是不同改动。

固定来源梯度不等于对 $q_\theta$ 的完整复合目标求导。后者还包含状态分布、归一化或求解系数的导数。即使期望梯度相同，IS 方差、AdamW、梯度裁剪、PPO clipping 和有限步更新仍可能产生不同训练轨迹。

<a id="centers"></a>

## 3. 从最优化目标求来源分布

本节先在固定前缀上优化整个下一 token 分布 $q\in\Delta_V$，不优化网络参数。相同分布公式也能用于完整回答空间，但其归一化和条件生成方式需按第 4 节计算。

### 3.1 距离与目标聚合

令 $a(q)$ 是到 student 的距离，$b(q)$ 是到 teacher 的距离。严格同时 on-policy 要求 $q=\mu=\pi$；师生不同就需要多目标取舍。

| 聚合方式   | 目标                                  | 选取原则                          |
| ---------- | ------------------------------------- | --------------------------------- |
| 等权 sum   | $S=(a+b)/2$                           | 最小化平均偏离                    |
| 加权 sum   | $(1-\eta)a+\eta b$                    | 预先指定双方重要性                |
| Minimax    | $M=\max(a,b)$                         | 最小化最差侧偏离                  |
| $L_p$      | $[(a^p+b^p)/2]^{1/p}$，$p\ge1$        | 从平均值连续过渡到 max            |
| 平滑最大值 | $\tau\log[(e^{a/\tau}+e^{b/\tau})/2]$ | 用 $\tau>0$ 控制较差侧权重        |
| 单边预算   | $\min_q b(q)$，$a(q)\le\rho$          | 在 student 允许范围内接近 teacher |

等权 sum 与 minimax 都对师生交换对称，但 $M=S+|a-b|/2$，后者多惩罚两侧差距。候选距离 $(0.05,0.65)$ 与 $(0.40,0.40)$ 分别受 sum 和 minimax 偏好；$(0.60,0.60)$ 被第二个候选 Pareto 支配。平均值、最差侧和预算约束表达不同需求，优化理论本身不能替需求选择。

$L_p$ 在 $p=1$ 时为平均值，在 $p\to\infty$ 时为 max；$p=2$ 满足 $(a^2+b^2)/2=S^2+(a-b)^2/4$。平滑最大值在 $\tau\to0$ 时趋于 max，在 $\tau\to\infty$ 时趋于平均值，其偏导数是两侧距离的 softmax 权重。

共同有限支持上，两方向 KL、平方 Hellinger、TV 和 Pearson $\chi^2$ 具有相关凸性。凸且坐标单调的聚合及凸距离预算保留凸性；满足有限性和约束资格时可用 KKT 判断全局最优。

单边 student KL 预算是 TRB 和 IW-OPD 理论投影所属的目标类型；前者讨论受控 behavior blending 并将预算退火到零，后者先在完整回答空间投影再构造加权更新。[^trb2026][^iwopd] 同时给两侧过紧预算可能无解。严格先把 student KL 优化到零会固定 $q=\mu$；需要第二目标的自由度时，应显式允许第一目标退让。按任务成功率或预测评分学习权重又是另一个目标，不能只凭师生概率判断 teacher 是否正确或状态是否可恢复。[^heskes][^stacking]

### 3.2 几何中心及 teacher 权重

先取 $a=\mathrm{KL}(q\Vert\mu)$、$b=\mathrm{KL}(q\Vert\pi)$。设可微聚合 $\Phi(a,b)$ 在最优点的偏导为 $c_S,c_T\ge0$，总和为正。归一化乘子为 $\nu$，驻点条件是

$$
c_S\left(\log\frac{q_v}{\mu_v}+1\right)
+c_T\left(\log\frac{q_v}{\pi_v}+1\right)+\nu=0.
$$

合并 $\log q_v$、整理并归一化：

$$
\boxed{
q_\eta(v)=\frac{\mu_v^{1-\eta}\pi_v^\eta}{Z_h(\eta)},\qquad
\eta=\frac{c_T}{c_S+c_T},\qquad
Z_h(\eta)=\sum_v\mu_v^{1-\eta}\pi_v^\eta.
}
$$

| 聚合         | Teacher 权重的条件                 |
| ------------ | ---------------------------------- |
| 等权 sum     | $\eta=1/2$                         |
| 固定加权 sum | 预设的 $\eta$                      |
| $L_p$，$p>1$ | $\eta=b^{p-1}/(a^{p-1}+b^{p-1})$   |
| 平滑最大值   | $\eta=\sigma((b-a)/\tau)$          |
| Minimax      | 最优对偶权重；非退化时两侧 KL 相等 |

这里的 $a,b$ 在最优 $q$ 上计算，后两种平滑目标是自洽条件，不能把原始师生间的 KL 代入一次就完成。师生相同时直接返回该分布；额外熵项或约束需要重新推导。

为求 minimax，写成 $\min_{q,r}r$，约束为 $a\le r,b\le r$。两乘子满足 $\lambda_S+\lambda_T=1$，故 $\eta=\lambda_T$。对 $q$ 最小化后对偶目标为 $-\psi_h(\eta)$，其中 $\psi_h=\log Z_h$。强对偶给出

$$
\eta^*=\arg\min_{0\le\eta\le1}\psi_h(\eta),\qquad
r^*=-\psi_h(\eta^*).
$$

定义 $\ell_v=\log(\pi_v/\mu_v)$，直接微分配分函数：

$$
g_h(\eta)=\psi_h'(\eta)=\mathbb E_{q_\eta}\ell,\qquad
v_h(\eta)=\psi_h''(\eta)=\operatorname{Var}_{q_\eta}\ell.
$$

两侧 KL 因而是

$$
a(\eta)=\eta g_h(\eta)-\psi_h(\eta),\qquad
b(\eta)=(\eta-1)g_h(\eta)-\psi_h(\eta).
$$

严格正且不同的分布使 $v_h>0$。又有 $g_h(0)=-\mathrm{KL}(\mu\Vert\pi)<0$、$g_h(1)=\mathrm{KL}(\pi\Vert\mu)>0$，所以唯一零点 $g_h(\eta^*)=0$ 恰好平衡两侧 KL。这是 Chernoff 中心的已有结果。[^nielsen2013][^nielsen2022]

单边预算则使用拉格朗日函数 $b+\lambda(a-\rho)$。同一驻点计算给出 $\eta=1/(1+\lambda)$；预算活跃时，$a(\eta)=\rho$，即

$$
\boxed{
\rho=\eta\psi_h'(\eta)-\psi_h(\eta),\qquad
\frac{d\rho}{d\eta}=\eta\operatorname{Var}_{q_\eta}\ell\ge0.
}
$$

因此 $0<\rho<\mathrm{KL}(\pi\Vert\mu)$ 对应唯一 $\eta\in(0,1)$；预算零取 student，teacher 可行后取 $\eta=1$，不必用满预算。小预算下，若 $\operatorname{Var}_\mu\ell>0$，有 $\eta\simeq\sqrt{2\rho/\operatorname{Var}_\mu\ell}$。固定融合权重不等于固定 KL 预算，同一个 $\eta$ 对不同前缀通常给出不同 $\rho$。这些代数关系在完整回答空间也成立；IW-OPD 的理论投影使用该序列版本。[^iwopd]

### 3.3 其他距离给出的中心

| 距离与聚合                                                 | 最优下一 token 分布                          | 行为与条件              |
| ---------------------------------------------------------- | -------------------------------------------- | ----------------------- |
| 等权 $\mathrm{KL}(q\Vert\mu),\mathrm{KL}(q\Vert\pi)$ sum   | $q\propto\sqrt{\mu\pi}$                      | 倾向共同支持            |
| 同方向 KL minimax                                          | $q\propto\mu^{1-\eta^*}\pi^{\eta^*}$         | 平衡最差侧 KL           |
| 等权 $\mathrm{KL}(\mu\Vert q),\mathrm{KL}(\pi\Vert q)$ sum | $q=(\mu+\pi)/2$                              | 覆盖两侧支持            |
| Endpoint-first KL minimax                                  | $q=(1-\eta^*)\mu+\eta^*\pi$                  | 通过对偶权重平衡两侧 KL |
| 两分布等权平方 Hellinger sum / minimax                     | $q\propto(\sqrt\mu+\sqrt\pi)^2$              | 两目标共享闭式解        |
| Endpoint-first $\chi^2$ minimax                            | $q\propto\sqrt{(1-\eta^*)\mu^2+\eta^*\pi^2}$ | 控制重要性权重二阶矩    |
| 两分布 TV minimax                                          | 算术中点是最优解之一                         | 可能不唯一              |

不同距离及方向决定不同中心，不能用最优半径的数值直接比较距离优劣。均值和中心的概率融合研究已有系统文献。[^amari2007][^candan2020][^fusion2022]

Endpoint-first KL 的加权和展开为 $-\sum_v[(1-\eta)\mu_v+\eta\pi_v]\log q_v+$ 常数。驻点条件为 $-[(1-\eta)\mu_v+\eta\pi_v]/q_v+\nu=0$，归一化给出 $\nu=1$，故最优解是算术混合。

归一化平方 Hellinger 定义为 $H^2(p,q)=1-\sum_v\sqrt{p_vq_v}$。令 $u=\sqrt q,s=\sqrt\mu,t=\sqrt\pi$，三者为非负单位向量，则

$$
\min(u^Ts,u^Tt)\le\tfrac12u^T(s+t)\le\tfrac12\lVert s+t\rVert_2.
$$

取 $u=(s+t)/\lVert s+t\rVert_2$ 时两次取等，也最大化平均内积。令 $B=\sum_v\sqrt{\mu_v\pi_v}$，得到

$$
q_H(v)=\frac{(\sqrt{\mu_v}+\sqrt{\pi_v})^2}{2+2B},\qquad
H^2(q_H,\mu)=H^2(q_H,\pi)=1-\sqrt{(1+B)/2}.
$$

这是平方根球面及 categorical Fisher–Rao 测地线中点；等权 sum/max 等价限于两个分布，不能直接推广到多分布。[^barycenter2025] 若 $q_A=(\mu+\pi)/2$、$q_G=\sqrt{\mu\pi}/B$，则 $q_H=(q_A+Bq_G)/(1+B)$。它满足 $q_H\ge(\mu+\pi)/4$，因此单步两端概率比不超过 4；轨迹比相乘后没有同样的长度无关常数界。

对 $\chi^2$，固定对偶权重后的目标为 $\sum_v[(1-\eta)\mu_v^2+\eta\pi_v^2]/q_v-1$。对 $q_v$ 求导得 $-[(1-\eta)\mu_v^2+\eta\pi_v^2]/q_v^2+\nu=0$，归一化得到表中平方根中心。Minimax 还需求最优对偶权重。TV 的三角不等式给出最大距离至少为 $\mathrm{TV}(\mu,\pi)/2$，算术中点达到该下界。

这些均值也可用 divergence 阶数 $\beta$ 统一。采用 $D_\beta(p\Vert q)=[1-\sum_vp_v^\beta q_v^{1-\beta}]/[\beta(1-\beta)]$，$0<\beta<1$，加权和的驻点满足

$$
-\frac{(1-\eta)\mu_v^\beta+\eta\pi_v^\beta}{\beta q_v^\beta}+\nu=0,\qquad
q_v\propto[(1-\eta)\mu_v^\beta+\eta\pi_v^\beta]^{1/\beta}.
$$

$\beta\to0$ 对应 $\mathrm{KL}(q\Vert p)$ 和几何融合，$\beta\to1$ 对应 $\mathrm{KL}(p\Vert q)$ 和算术融合，$\beta=1/2$ 为平方 Hellinger 的常数倍。固定权重解是平均中心，minimax 需另外选择对偶权重；不同文献的阶数约定应按公式核对。[^candan2020]

<a id="source-weights"></a>

## 4. 各类来源的统一权重

### 4.1 从单步来源到前缀权重

局部中心若能写成 $q(v\mid h)=\mu(v\mid h)f_h(r(v\mid h))/C_h$，其中 $C_h=\mathbb E_{\mu(\cdot\mid h)}f_h(r)$，则

$$
\boxed{
W_t=\prod_{k<t}\frac{q(y_k\mid h_k)}{\mu(y_k\mid h_k)}
=\prod_{k<t}\frac{f_{h_k}(r_k)}{C_{h_k}},\qquad
A_t=W_t\Delta_t\quad(\theta=\bar\theta).
}
$$

第 3 节的中心直接给出 $f$：几何中心为 $r^\eta$，算术中心为 $(1-\eta)+\eta r$，等权 Hellinger 为 $(1+\sqrt r)^2$，$\chi^2$ 中心为 $\sqrt{(1-\eta)+\eta r^2}$。改变 sum、minimax 或预算，主要改变对应的系数规则；所有归一化因子都保留在乘积中。

| 来源或加权规则                        | 更新起点的权重                                | 与统一框架的关系                              |
| ------------------------------------- | --------------------------------------------- | --------------------------------------------- |
| Fresh student                         | $W_t=1$                                       | 来源 $q=\mu$                                  |
| Initial student                       | $W_t=P_{\mu_0}(y_{<t})/P_\mu(y_{<t})$         | 来源 $q=\mu_0$                                |
| Teacher                               | $W_t=R_t$                                     | 来源 $q=\pi$                                  |
| 逐 token 几何融合                     | $W_t=\prod_{k<t}r_k^{\eta_k}/Z_{h_k}(\eta_k)$ | 单步解归一化后生成                            |
| 逐 token 算术融合                     | $W_t=\prod_{k<t}[(1-\eta_k)+\eta_kr_k]$       | 每步混合条件概率                              |
| 整轨迹混合 $(1-\eta)P_\mu+\eta P_\pi$ | $W_t=(1-\eta)+\eta R_t$                       | 一次选择来源后生成完整回答                    |
| 完整序列几何中心，含 IW-OPD 理论投影  | $W_t=R_t^\eta H_\eta(h_t)/Z_{\mathrm{seq}}$   | 对序列密度求真实前缀边缘                      |
| IW-OPD 理论前缀 surrogate             | $W_t=R_t^\eta/Z_t^{\mathrm{pref}}$            | 逐位置定义 $\nu_t$，一般不是上一行的边缘      |
| IW-OPD 实验权重                       | $w_t(y)=1+\gamma(1-D_t/D_L)$                  | 在原 OPD 信号上做稳定化加权，非精确来源 IS 比 |

表中 $L=L(y)$ 为该回答的有效长度，$D_t=\sum_{k<t}|\Delta_k|$、$D_L=\sum_{k<L}|\Delta_k|$，$\gamma\ge0$；这里按论文约定，分母累计到最后一个 token 之前。IW-OPD 将理想权重依次作 log scaling、绝对差异累计、回答内归一化与 OPD 混合，最终使用 $\operatorname{sg}[w_t(y)]\Delta_t$，默认 $\gamma=0.5$。$w_t$ 的分母依赖整个回答，不能直接当作只依赖前缀的密度比；全零差异还需实现约定处理退化归一化。[^iwopd]

Initial 的原目标是 $L_0=\sum_t\mathbb E_{d_{\mu_0,t}}K_\theta(h_t)$。它保留 initial 的状态分布，监督仍来自 teacher。其额外权重不含 teacher；初始化时为 1，之后衡量 initial 相对 fresh 的前缀概率。它没有添加靠近 initial 的惩罚，也一般不属于某个固定 teacher 权重：要有 $P_{\mu_0}\propto P_\mu^{1-\eta}P_\pi^\eta$，必须对所有回答同时满足 $\log(P_{\mu_0}/P_\mu)=\eta\log(P_\pi/P_\mu)-\log Z$。

整轨迹混合使用“乘积的和”，逐 token 算术融合使用“和的乘积”。固定两侧样本配额并按配额比例聚合，也得到相应混合期望。空前缀上，所有精确来源比均为 1。

### 4.2 序列中心的边缘与局部融合

对任意完整回答来源 $Q$，记 $\omega(y)=Q(y)/P_\mu(y)$。把后缀 $u$ 求和可得

$$
W_t(h)
=\frac{\sum_uQ(hu)}{P_\mu(h)}
=\sum_uP_\mu(u\mid h)\omega(hu)
=\mathbb E_{P_\mu}[\omega(y)\mid h_t=h].
$$

因此前缀权重是完整 IS 权重的条件期望。对固定来源的 full-vocab 梯度，也可直接使用完整回答权重：

$$
-\nabla L_Q
=\mathbb E_{P_\mu}\left[\omega(y)\sum_t-\nabla K_\theta(h_t)\right]
=\mathbb E_{P_\mu}\sum_tW_t(h_t)[-\nabla K_\theta(h_t)].
$$

第一种估计只需完整密度比，无需显式计算每个前缀的条件期望；第二种对后缀作了条件积分。若用 sampled 的 $\Delta_{\theta,t}s_\theta$ 替代 full-vocab 梯度，直接乘 $\omega(y)$ 会同时把当前动作变成 $Q(\cdot\mid h_t)$ 分布，还需乘 $\mu_\theta(y_t\mid h_t)/Q(y_t\mid h_t)$ 才恢复 learner 的动作期望。第 2 节的前缀 IS 写法保留 fresh 的动作分布，对应的动作比则是 $U_t$。

以 $Q_\eta(y)=P_\mu(y)^{1-\eta}P_\pi(y)^\eta/Z_{\mathrm{seq}}$ 为例，定义

$$
Z_{\mathrm{seq}}=\sum_yP_\mu(y)^{1-\eta}P_\pi(y)^\eta,\qquad
H_\eta(h)=\sum_uP_\mu(u\mid h)^{1-\eta}P_\pi(u\mid h)^\eta.
$$

代入上一式即得 $W_t=R_t^\eta H_\eta(h_t)/Z_{\mathrm{seq}}$。真实下一 token 条件分布还含后续重叠量：

$$
Q_\eta(a\mid h)
=\mu(a\mid h)^{1-\eta}\pi(a\mid h)^\eta\frac{H_\eta(ha)}{H_\eta(h)}.
$$

逐 token 几何融合只用局部 $Z_h$。固定 $\eta$ 时，将它的条件概率相乘得到 $P_q(y)=P_\mu(y)^{1-\eta}P_\pi(y)^\eta/\prod_tZ_{h_t}$；分母随路径变化，通常不同于共享 $Z_{\mathrm{seq}}$ 的序列中心。

前缀 surrogate 则直接定义 $\nu_t(h)=P_\mu(h)^{1-\eta}P_\pi(h)^\eta/Z_t^{\mathrm{pref}}$，其中 $Z_t^{\mathrm{pref}}=\mathbb E_{d_{\mu,t}}R_t^\eta$。它精确描述逐位置的几何重加权，但通常既不是 $Q_\eta$ 的前缀边缘，也不是局部 fusion 的前缀分布；各位置的 $\nu_t$ 还可能不构成一致边缘族。IW-OPD 从完整序列 IS 转向该前缀成分时，附录将其标为 token surrogate。[^iwopd]

KL 链式法则能联系局部与序列距离，但不消除这种区别。在共同 prompt、环境和停止协议下，$\mathrm{KL}(P_q\Vert P_\mu)=\mathbb E_q\sum_ta_t$，teacher 侧同理；于是

$$
\max\left\{\mathbb E_q\sum_ta_t,\mathbb E_q\sum_tb_t\right\}
\le\mathbb E_q\sum_t\max(a_t,b_t).
$$

局部最差侧可以随位置改变：两步距离 $(10,0),(0,10)$ 的局部 max 之和为 20，序列两侧之和的 max 为 10。Sum 虽能与期望交换，当前动作仍会改变未来状态，所以局部贪心 sum 也不自动解决序列优化。

任意 $Q$ 还满足 $-\mathrm{KL}(Q\Vert P_\pi)=\mathbb E_Q\log(P_\pi/P_\mu)-\mathrm{KL}(Q\Vert P_\mu)$。活跃固定预算才把最后一项变成常数 $-\rho$。该目标值变换、精确 IS 和固定来源 semi-gradient 是不同步骤，不能据分布中心最优性直接宣称获得了原序列目标的完整梯度。

<a id="computability"></a>

## 5. 单条 rollout 与归一化的计算

单条完整回答就能计算沿途的 log-prob 和概率比。是否需要同一 prompt 多条回答，取决于所用分布的归一化是否已知。

| 要计算的权重              | 单条回答提供的信息                                       | 还需什么                             |
| ------------------------- | -------------------------------------------------------- | ------------------------------------ |
| Initial / teacher 前缀比  | 沿相同前缀评分即可得到精确比                             | 对应版本的模型概率，无需额外生成     |
| 逐 token fusion 的前缀比  | 可沿已访问路径累积                                       | 每步完整词表分布以精确计算局部归一化 |
| 序列中心的 $Q_\eta/P_\mu$ | 可得到 $R_{L+1}^{\eta}=\exp(\eta\sum_{t=1}^{L}\Delta_t)$ | 完整回答空间的 $Z_{\mathrm{seq}}(x)$ |
| 序列中心的真实前缀比      | 可得到前缀概率比                                         | 还需 $H_\eta(h)$，不是截去后缀即可   |
| 理论前缀 surrogate        | 可得到 $R_t^\eta$                                        | 位置级 $Z_t^{\mathrm{pref}}(x)$      |
| 实验中的回答内位置权重    | 单条回答可算累计差异与归一化                             | 不需要估计上述配分函数               |

若 $Z_{\mathrm{seq}}(x)$ 已知，$\omega(y)F(y)$ 的单个样本就是 $\mathbb E_{Q_\eta}F$ 的无偏 IS 估计；多个 rollout 不是恒等式的前提，只影响估计精度。困难在于一般 LLM 无法枚举所有回答来精确求 $Z$。

若改用同一 prompt 的 $N$ 条独立回答 $y_i\sim P_\mu$ 估计未知配分函数，令 $u_i=[P_\pi(y_i)/P_\mu(y_i)]^\eta$，则

$$
\widehat Z=\frac1N\sum_i u_i,\qquad
\widehat{\mathbb E}_{Q_\eta}F
=\frac{\sum_i u_iF(y_i)}{\sum_i u_i}.
$$

这是自归一化 IS，有限 $N$ 通常有偏；在常规可积条件下随样本增多一致。当 $N=1$ 时，归一化权重 $u_1/\widehat Z=1$，估计退化为 $F(y_1)$，完全失去这项重加权。同一回答里的多个 token 不是该序列分布的多次独立采样。跨不同 prompt 混合估计分母，也一般不能替代各自的 $Z(x)$。

在固定 prompt、冻结权重的线性梯度中省略 $Z(x)$ 只会改变整体尺度，但跨 prompt 聚合时会改变 prompt 的相对权重；不能据此宣称仍是同一个精确目标。改用批量估计、独立估计或显式未归一化目标，都需要写明估计方法及偏差。

论文实验每个 prompt 训练采样 1 条回答，使用第 4 节的回答内稳定化权重，而非用单条回答精确估计序列或位置配分函数。[^iwopd] 单条可算这个算法权重，与单条足以估出严格 $Q_\eta/P_\mu$，是两个不同判断。

<a id="solver"></a>

## 6. 系数求解与 sum / minimax 的取舍

### 6.1 求根、误差与停止条件

第 3.2 节中 $g_h$ 严格递增，二分可求 minimax 的零点；Newton 候选为 $\eta-g_h/v_h$，越界或方差过小时退回二分。只使用已有两份分布，不重新运行模型。预算版本则求 $a(\eta)-\rho=0$。

设 $F(\eta)=\max(a(\eta),b(\eta))$。任意系数都有对偶下界 $-\psi_h(\eta)\le r^*\le F(\eta)$，所以

$$
0\le F(\eta)-r^*\le\operatorname{gap}(\eta)
=\max\{\eta g_h(\eta),(\eta-1)g_h(\eta)\}.
$$

用 $\mathrm{gap}\le\mathrm{atol}+\mathrm{rtol}F$ 停止，能直接限制最差 KL 的目标误差；仍需迭代上限和未收敛记录。该界基于精确算术，FP32 容差应对照 FP64 验证。

求根使用归一化 log-prob：

$$
\ell=(z_T-z_S)-\operatorname{LSE}(z_T)+\operatorname{LSE}(z_S).
$$

仅用原始 logit 差会使解依赖任意的整体 logit 平移。$g_h>0$ 表示 student 侧更远，应减小 teacher 权重；$g_h<0$ 则增大。师生相同时直接返回共同分布，权重可固定为 $1/2$。

### 6.2 用中心变化判断是否值得求 minimax

固定 $\mu=(0.8,0.19,0.01)$、$\pi=(0.1,0.2,0.7)$，CPU FP64 结果如下，距离单位为 nats。冻结数据记录的是 student 系数；此表已转换成 teacher 权重。

| 规则     | Teacher 权重 $\eta$ | Student KL | Teacher KL | 平均 KL | 最大 KL |
| -------- | ------------------: | ---------: | ---------: | ------: | ------: |
| 等权 sum |              0.5000 |     0.3789 |     0.7756 |  0.5772 |  0.7756 |
| Minimax  |              0.5764 |     0.5929 |     0.5929 |  0.5929 |  0.5929 |

Minimax 降低最差距离并增加平均距离。令 $q_S=q_{1/2}$、$q_M=q_{\eta^*}$、$B_0=-\psi_h(1/2)$、$C_0=-\psi_h(\eta^*)$。展开 log-density 有

$$
S(q)=\mathrm{KL}(q\Vert q_S)+B_0,\qquad
S(q_M)=C_0,
$$

因此

$$
\mathrm{KL}(q_M\Vert q_S)=C_0-B_0,\qquad
0\le M(q_S)-C_0\le\tfrac12|a(q_S)-b(q_S)|.
$$

这能衡量改用 minimax 的分布变化和潜在半径改善。一般非负距离还满足 $M(q_S)\le2S(q_S)\le2S(q_M)\le2M(q_M)$，这是目标值的 2-近似，不是任务性能保证。

若用 $F_\kappa=S+\kappa|a-b|/2$，$0\le\kappa\le1$，其对偶权重被限制在 $[(1-\kappa)/2,(1+\kappa)/2]$。由于 $-\psi_h$ 为凹函数，最优系数是把 $\eta^*$ 截断到该区间。不能改成同时最小化自由权重和加权和，否则取 $\eta=0,q=\mu$ 就退化为零；minimax 对权重做最大化。

### 6.3 共同低概率事件的反例

考虑 $\mu=(0.999,10^{-12},0.001-10^{-12})$、$\pi=(10^{-12},0.999,0.001-10^{-12})$。交换对称性使几何 sum 和 minimax 都取 $\eta=1/2$：

| 中心               |    事件 1 |    事件 2 |    事件 3 |
| ------------------ | --------: | --------: | --------: |
| 几何 sum / minimax | 0.0009975 | 0.0009975 | 0.9980050 |
| Hellinger          | 0.4990010 | 0.4990010 | 0.0019980 |
| 算术中点           |    0.4995 |    0.4995 |  约 0.001 |

双方仅给约 0.1% 概率的共同事件被几何中心放大到约 99.8%，共同 KL 半径约为 6.906 nats。这来自距离的共同支持倾向，并非 minimax 独有。最优中心仍可能离两端都远，也不保证正确性。

<a id="evidence"></a>

## 7. 执行条件、文献与核验

### 7.1 概率合同与 IS 方差

可解释的初始对照使用温度 1，无 top-p/top-k 截断和额外 repetition penalty。若使用公共温度 $T_{\mathrm{dec}}$，先把两端定义为 $\mathrm{softmax}(z_S/T_{\mathrm{dec}}),\mathrm{softmax}(z_T/T_{\mathrm{dec}})$ 再求中心；求解后过滤或改温度会改变分布。

双方必须共享动作语义。Raw-token 融合与把多个 EOS ID 归为 STOP 事件是不同定义；仅把两个 ID 放入停止列表不能修复融合概率。不同硬支持、空支持、NaN、零概率和 smoothing 均需显式处理。每批固定来源权重版本，每个前缀独立求局部系数，不能平均成 batch 标量后仍声称逐前缀最优。

精确前缀比满足 $\mathbb E_{d_{\mu,t}}W_t=1$，但均值为 1 不足以证明它对应指定来源。裁剪、幂次压缩、绝对差异累计或加权下限都会改变更新。IS 无法补出 batch 中缺失的罕见前缀；比值随长度相乘可能有很高方差。Full-vocab 的动作求和消除了条件动作采样误差，并不消除前缀覆盖问题。

二阶矩 $\mathbb E_\mu[(\pi/\mu)^2]=1+\chi^2(\pi\Vert\mu)$ 在两端固定时不含来源 $q$。实际从 $q$ 估计 teacher 期望，应分析 $\mathbb E_q[(\pi/q)^2]$；小 KL 不保证小 $\chi^2$。VESPO 的旧 behavior 样本重塑定义隐式分布，并约束特定权重矩；它不同于直接按任意 $q$ 生成，也不直接给出 full-vocab 梯度方差。[^vespo2026]

批量 full-vocab 求系数约需 $O(N_{\mathrm{active}}VK)$ 个标量操作，$K$ 是迭代数。避免逐请求 Python 循环与每轮 CPU 同步；TP 分片词表需要全局归一化和期望归约。推荐记录实际两侧 KL、gap、迭代数、中心相对中点的变化、生成时间和显存。没有额外模型 forward 不代表没有可测延迟。

### 7.2 文献所支持的结论

| 工作                            | 已核对的结论与边界                                                                                                                                           |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Amari；Candan；概率融合综述     | 不同 divergence 导出不同均值与中心，平均中心和 minimax 需分别求解。[^amari2007][^candan2020][^fusion2022]                                                    |
| Nielsen 2013 / 2022             | 给出 Chernoff 点、等距条件和数值求解，不保证 student 学习优于 sum。[^nielsen2013][^nielsen2022]                                                              |
| Akopyan 等 2025                 | Endpoint-first KL 的 JS 与最优最大值 $R$ 满足 $(e\ln2/2)R\le JS\le R$，系数约 0.94208；不是反方向 KL 的界，也不是 sum 中心的 1.061 倍保证。[^minmax2025]     |
| Barycenter Policy Design        | 给出 KL、Hellinger、χ² behavior 平均中心并讨论 max；100 臂、1000 个 policy 的合成实验支持适当聚类改善 regret，未直接比较 max 与 mean 中心。[^barycenter2025] |
| IW-OPD                          | 序列约束投影导出几何中心，精确序列 IS 后使用前缀 surrogate 和稳定化权重；实验每 prompt 训练采样 1 条。[^iwopd]                                               |
| RMOD                            | 最差 reward 鲁棒解码中，HH 最坏侧胜率 59.1%，CD-Uniform 57.6%，均为 $B=16,K=16$；师生 KL 不能自动替代任务 reward。[^rmod2025]                                |
| Heskes；Stacking                | 可按预测评分选择组合权重；不能据此证明几何中心对蒸馏最优。[^heskes][^stacking]                                                                               |
| TRB；VESPO；Divergence Decoding | 分别讨论预算控制的 behavior、隐式重塑权重和 JS 门控路由，目标与执行位置不同。[^trb2026][^vespo2026][^divdec2026]                                             |

多策略论文早期 arXiv:2503.02735 标题为 Clustered KL-barycenter design for policy evaluation，主要聚焦 KL；更一般的 KL/Hellinger/χ² 结论来自后续作者稿，版本不能混引。[^clustered2025] 新近 reverse-KL 理论区分 full-distribution 与 sampled-action feedback，但没有证明本文中心来源最优。[^rkl2026]

2026-10-28 的检索未核实到精确两模型 Chernoff 中心直接用于 LLM 蒸馏逐 token rollout 的公开工作，不构成首创性证明。研究可比较来源对学习的因果影响、与 full-vocab loss 的交互、局部与序列目标差距及解码成本。

### 7.3 公开实现与 CPU 证据

pyBregMan 的 `ChernoffInformation.chernoff_point` 可用 `MultinomialManifold(k=V,n=1)` 求 categorical 中心。固定 commit 为 `6f375e4d38e8ff798801bf5d34572ed52b1fadbb`，默认 THETA 坐标返回 point_1 的指数权重，以系数区间宽度停止；调用时需将端点顺序映射到本文 teacher 权重。[^pybregman2024][^py-code]

40 对五类随机分布核验中，原始 helper 与独立 log-space 求根的最大系数差约 $3.16\times10^{-12}$。同版本 `dissimilarity` 的 Burbea–Rao 标量带有不同尺度：三 token 例返回 2.18751246，而标准 Chernoff 半径为 0.59285868，不能把求系数 helper 与半径接口混为一谈。

Frank Nielsen 的 Bernoulli Java 示例 $p=0.1,q=0.7$ 已编译运行，原端点约定的系数为 0.46607633，中心概率为 0.36084881，两侧 KL 约为 0.24432138；Gaussian 专用实现只阅读了源码。[^java-code][^java-gaussian] 这些实现不提供双模型 KV、GPU 批量解码或训练集成。

| 证据与复算                                                                                                                                    | 核验范围                                                                                                        |
| --------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| [中心结果](../../literature/rollout-policy/data/numerical-audit.json)与[脚本](../../../research/checks/rollout_policy/audit_centers.py)       | 40 对合成分布；Hellinger 半径相对 SLSQP 最大误差约 $4.39\times10^{-13}$                                         |
| [聚合结果](../../literature/rollout-policy/data/aggregation-audit.json)与[脚本](../../../research/checks/rollout_policy/audit_aggregation.py) | 100 对分布；$\mathrm{KL}(q_M\Vert q_S)=C_0-B_0$ 最大误差约 $2.98\times10^{-16}$；折中目标在几何族上对照稠密网格 |
| [来源与 IS 核验脚本](../../../research/checks/rollout_policy/audit_source_weights.py)                                                         | 二元三步树、16 种来源；精确枚举局部梯度、重复更新动作比、序列中心和错误权重反例，并核对单样本归一化             |
| [Java 输出](../../literature/rollout-policy/data/java-audit.txt)                                                                              | 作者 Bernoulli 示例，源码与启动器见理论目录 `scripts/`                                                          |

来源脚本 seed20261106 的直接梯度与精确 IS 梯度最大误差小于 $1.12\times10^{-16}$，有限差分误差约 $1.28\times10^{-11}$。误用前缀 surrogate 替代局部 fusion 时差为 $6.73\times10^{-4}$，误用 fresh 回答长度时差为 $8.31\times10^{-2}$。这是同一合成树上的恒等式与反例，不是 LLM 学习效果或 GPU 性能结果。

从仓库根目录复算；中心脚本首次需安装其声明的固定依赖：

```bash
python -m examples.mirroropd.research.checks.rollout_policy.audit_centers --out /tmp/centers-recheck.json
python -m examples.mirroropd.research.checks.rollout_policy.audit_aggregation --out /tmp/aggregation-recheck.json
python -m examples.mirroropd.research.checks.rollout_policy.audit_source_weights \
  --out runs/rollout-policy-is-20261106/verification.json
```

实际研究保持 learner loss、初始化、prompt、动作空间和 token / 更新预算可比，以纯 student 为参照，按问题选择几何中点、minimax、Hellinger 或覆盖方向对照。先在真实同前缀分布上检查中心变化与 EOS 差异，再以纯 learner accuracy、轨迹质量、token 与墙钟成本评价，不按各自 rollout loss 排序。已运行的协议与结果见[实验入口](../../experiments/rollout-policy/README.md)。

完整检索记录见[理论目录](README.md#完整检索记录)：三轮分别为 76、59、58 条，未跨轮去重，保留无关命中；第一轮和聚合检索各有一条 OpenAlex 记录缺少标题，目标来源已按作者 PDF 核实。后续判断以核对原文为准，原始报告和冻结核验数据不随本文符号重构改写。

### 参考文献

[^iwopd]: Yan Xie et al. On the Position Bias of On-Policy Distillation. arXiv:2606.22600，§3–4 与附录的 constrained projection、change of measure 和 token-local semi-gradient 推导，2026-11-06 核对。[arXiv 正文](https://arxiv.org/html/2606.22600)。本文第 4 节区分理论序列投影、前缀 surrogate 与稳定化实验权重；每 prompt 采样数见论文附录训练参数表。

[^amari2007]: Shun-ichi Amari. Integration of Stochastic Models by Minimizing α-Divergence. Neural Computation 19(10):2780–2796, 2007. DOI 10.1162/neco.2007.19.10.2780. [作者机构 PDF](https://bsi-ni.brain.riken.jp/database/file/284/290.pdf)。

[^nielsen2013]: Frank Nielsen. An Information-Geometric Characterization of Chernoff Information. IEEE Signal Processing Letters 20(3):269–272, 2013. DOI 10.1109/LSP.2013.2243726. [作者 PDF](https://www.lix.polytechnique.fr/Labo/Frank.Nielsen/J2013-SPL-ChernoffInformationIG.pdf)。

[^nielsen2022]: Frank Nielsen. Revisiting Chernoff Information with Likelihood Ratio Exponential Families. Entropy 24(10):1400, 2022. [arXiv:2207.03745v4](https://arxiv.org/html/2207.03745v4)。

[^candan2020]: Çağatay Candan. Chebyshev Center Computation on Probability Simplex With α-Divergence Measure. IEEE Signal Processing Letters 27:1515–1519, 2020. DOI 10.1109/LSP.2020.3018661. [作者 PDF](https://users.metu.edu.tr/home103/ccandan/wwwhome/pub_dir/Cheb_Center_IEEE_SPL_2021.pdf)。

[^fusion2022]: Günther Koliander, Yousef El-Laham, Petar M. Djurić, Franz Hlawatsch. Fusion of Probability Density Functions. 2022. [arXiv:2202.11633](https://arxiv.org/abs/2202.11633)。此处仅用于概率融合的文献定位。

[^minmax2025]: Arseniy Akopyan, Herbert Edelsbrunner, Žiga Virk, Hubert Wagner. Tight Bounds Between the Jensen–Shannon Divergence and the Minmax Divergence. Entropy 27:854, 2025, Theorem 1. [作者 PDF](https://pub.ista.ac.at/~edels/Papers/2025-10-MinmaxDivergence.pdf)。

[^barycenter2025]: Simon Weissmann, Till Freihaut, Claire Vernade, Giorgia Ramponi, Leif Döring. Barycenter Policy Design for Multiple Policy Evaluation. EWRL 2025 公开稿及作者公开稿。本文分析使用 34 页作者稿，公式 (3)、表 1、§5、附录 B.4/B.5。[作者稿](https://tfreihaut.github.io/assets/publications/2025_is/is_paper.pdf)、[EWRL 稿](https://openreview.net/pdf/a041a53d80087e57b06466c0cd09fa5a4f20c5e4.pdf)。

[^clustered2025]: Simon Weissmann et al. Clustered KL-barycenter design for policy evaluation. 2025. [arXiv:2503.02735v1](https://arxiv.org/html/2503.02735v1)。

[^pybregman2024]: Frank Nielsen, Alexander Soen. pyBregMan: A Python library for Bregman Manifolds. 2024. [arXiv:2408.04175](https://arxiv.org/html/2408.04175)。

[^py-code]: Alexander Soen, Frank Nielsen. pyBregMan，commit `6f375e4d38e8ff798801bf5d34572ed52b1fadbb`，2026-10-28 核对。[ChernoffInformation](https://github.com/alexandersoen/pyBregMan/blob/6f375e4d38e8ff798801bf5d34572ed52b1fadbb/bregman/dissimilarity/bregman.py#L249)。

[^java-code]: Frank Nielsen. BernoulliExpFam.java，源码注明 November 2024，commit `e4c782ff4fb52e0c2b9d7285f47e41a1324ef900`。[源码](https://github.com/FrankNielsen/FrankNielsen.github.io/blob/e4c782ff4fb52e0c2b9d7285f47e41a1324ef900/P5/CategoricalChernoffInformationPDE/BernoulliExpFam.java)。

[^java-gaussian]: Frank Nielsen. DemoChernoffInformatioMultivariateGaussian.java，同上 commit。[源码](https://github.com/FrankNielsen/FrankNielsen.github.io/blob/e4c782ff4fb52e0c2b9d7285f47e41a1324ef900/ChernoffInformation/DemoChernoffInformatioMultivariateGaussian.java)。仅源码核对。

[^trb2026]: Daniil Plyusov et al. Trust-Region Behavior Blending for On-Policy Distillation. 2026. [arXiv:2605.31159v1](https://arxiv.org/html/2605.31159v1)。

[^vespo2026]: Guobin Shen et al. VESPO: Variational Sequence-level Soft Policy Optimization for Off-Policy LLM Training. 2026. [arXiv:2602.10693v3](https://arxiv.org/html/2602.10693v3)。

[^rmod2025]: Seongho Son et al. Robust Multi-Objective Controlled Decoding of Large Language Models. 2025，核对 v2（2026-02-16），§6.2、Table 1。[arXiv:2503.08796v2](https://arxiv.org/html/2503.08796v2)。

[^divdec2026]: Divergence Decoding: Training-Free Capability Fusion. 2026. [arXiv:2607.27248](https://arxiv.org/html/2607.27248)。

[^rkl2026]: Haolin Liu, Chen-Yu Wei, Julian Zimmert. Reverse-KL Policy Distillation: Statistical Guarantees and Limits. 2026，v2（2026-10-25）。[arXiv:2610.30844v2](https://arxiv.org/abs/2610.30844v2)。仅核对摘要中的 feedback 区分，未审计完整证明。

[^heskes]: Tom Heskes. Selecting Weighting Factors in Logarithmic Opinion Pools. NIPS 1997。[官方页面](https://papers.nips.cc/paper_files/paper/1997/hash/59f51fd6937412b7e56ded1ea2470c25-Abstract.html)。

[^stacking]: Yuling Yao, Aki Vehtari, Daniel Simpson, Andrew Gelman. Using stacking to average Bayesian predictive distributions. arXiv 2017，Bayesian Analysis 2018，DOI 10.1214/17-BA1091。[arXiv:1704.02030](https://arxiv.org/abs/1704.02030)。核对摘要与方法定位，未重跑其预测实验。
