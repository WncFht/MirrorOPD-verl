# 固定前缀上的目标、梯度与计算图

固定前缀集合及其权重；先区分 current/frozen 信号与 IS/NoIS 的目标，再核对 sampled 与完整词表的反向图。完整推导、支持条件、clipping 与有限缓存边界均保留。符号由[统一约定](符号约定.md)维护；[镜像下降](mirror-descent/objectives.md)扩展到正则和目标教师。

<a id="objectives"></a>

<a id="objectives-固定前缀下的-opd-目标与梯度"></a>

## 目标函数、信号与离策略梯度

本文在固定前缀上区分目标函数、梯度、抽样估计器与反向计算图，给出 current/frozen 信号与 IS/NoIS 的四种对应关系。记号遵循[符号约定](符号约定.md)。前缀集合及其权重固定；全文不校正前缀访问分布，也不对采样轨迹求导。

推导从原[固定前缀文档](../literature/learning-signal/2026-10-21-固定前缀下的优势估计与重要性采样.md)迁入；原文保留论文实验、代码刷新时机与历史验算。镜像正则的完整分析见[镜像下降与离策略四种更新](mirror-descent/objectives.md#logarithmic)。

<a id="objectives-1-符号与比较条件"></a>

### 1. 符号与比较条件

固定前缀 $s$，词表为有限集合 $\mathcal V$。为简化书写，省略条件 $s$：

| 符号                               | 含义                                      | 是否随本批内部更新变化 |
| ---------------------------------- | ----------------------------------------- | ---------------------- |
| $\beta(a)$                         | 真实采样策略，例如 student vLLM           | 否                     |
| $q(a)$                             | 冻结的更新参照，例如 FSDP 重算的 old 策略 | 否                     |
| $p_\theta(a)$                      | 当前 student                              | 是                     |
| $T(a)$                             | 固定 teacher                              | 否                     |
| $w_\theta(a)=p_\theta(a)/\beta(a)$ | 从采样分布到当前 student 的 IS 比         | 是                     |

理想同步初始化为 $p_{\theta_0}=q=\beta$。实际代码中的 `old_log_probs` 可以对应训练后端的 $q$，而论文中的 old 常指真实行为策略 $\beta$。相同模型版本不保证两个推理后端逐位相同。

以下精确等式假设 $p_\theta,T,q$ 在讨论的支持集上为正，且 $p_\theta(a)>0$ 时有 $\beta(a)>0$。temperature/top-p 处理后的真实采样概率应进入 IS 分母；如果采样截断使某些目标动作概率为零，不能依靠有限权重恢复缺失支持。本文概率指进入目标和采样器的实际归一化分布；温度、截断与支持条件必须一致。

比较时还固定 mask、token 聚合方式和 teacher；不混入 task reward、advantage whitening、额外 entropy bonus 或其他正则。PPO clipping 和 IS 截断会在后文单独引入。

本页的 $A^{\rm cur}=\log(T/p)$、$A^{\rm anchor}=\log(T/q)$ 都是未中心化 OPD 信号。后者是核心稿 $\kappa=0$ 的 $A=f/\eta$；核心稿的冻结优势为 $\mathcal A_k=C_qA=A+K_T$，不应与本页代码所称的 advantage 混同。$J$ 在此专指核心稿的 $J_{0,\eta}(p;T)=\mathrm{KL}(p\Vert T)$。

<a id="objectives-2-monte-carlo-与三个-student-概率"></a>

### 2. Monte Carlo 与三个 student 概率

若动作 $a_i\sim\beta$，直接平均一个函数估计的是 $\beta$ 下的期望：

$$
\widehat F=\frac1M\sum_{i=1}^M h(a_i),
\qquad
\mathbb E[\widehat F]=\sum_a\beta(a)h(a).
$$

即使程序没有读取保存的 `rollout_log_probs`，样本频率也已经体现了 $\beta$。当要估计当前 student 下的期望时：

$$
\sum_a p_\theta(a)h_\theta(a)
=\sum_a\beta(a)\frac{p_\theta(a)}{\beta(a)}h_\theta(a)
=\mathbb E_\beta[w_\theta h_\theta].
$$

本仓 PPO ratio 是 $w^{\rm PPO}=p_\theta/q$。如果实际采样来自 $\beta$，则未截断时还需要行为修正 $w^{\rm corr}=q/\beta$：

$$
w^{\rm corr} w^{\rm PPO}=\frac q\beta\frac{p_\theta}q=\frac{p_\theta}\beta.
$$

在 $q\ne\beta$ 时，单用 $p_\theta/q$ 并不完成上述期望变换。工程字段对应关系见[统一符号](符号约定.md)与原文的实现审计。

<a id="objectives-3-当前-advantage-与-reverse-kl"></a>

### 3. 当前 advantage 与 reverse KL

<a id="objectives-31-目标梯度"></a>

#### 3.1 目标梯度

目标为：

$$
J(\theta)=D_{\mathrm{KL}}(p_\theta\Vert T)
=\sum_a p_\theta(a)\log\frac{p_\theta(a)}{T(a)}.
$$

由于 teacher 固定，逐项求导：

$$
\begin{aligned}
\nabla_\theta J
&=\sum_a\nabla_\theta p_\theta(a)
\left(\log\frac{p_\theta(a)}{T(a)}+1\right)\\
&=\sum_a p_\theta(a)
\left(\log\frac{p_\theta(a)}{T(a)}+1\right)
\nabla_\theta\log p_\theta(a).
\end{aligned}
$$

概率归一化给出：

$$
\sum_a p_\theta(a)\nabla_\theta\log p_\theta(a)
=\sum_a\nabla_\theta p_\theta(a)=\nabla_\theta1=0.
$$

定义：

$$
A^{\rm cur}(a)=\log T(a)-\log p_\theta(a).
$$

消去上一式中的常数项后：

$$
\nabla_\theta J
=-\mathbb E_{p_\theta}
\left[A^{\rm cur}(a)\nabla_\theta\log p_\theta(a)\right]
=-\mathbb E_\beta
\left[w_\theta(a)A^{\rm cur}(a)\nabla_\theta\log p_\theta(a)\right].
$$

这也是 AsyncOPD 固定前缀分析的核心身份关系。[^objectives-asyncopd] 此处的推导不依赖异步调度。

<a id="objectives-32-两种正确的反向计算图"></a>

#### 3.2 两种正确的反向计算图

下面两种逐样本 surrogate 具有相同梯度：

$$
 L^{\rm sur}_1=-w_\theta\,\operatorname{sg}(A^{\rm cur}),
\qquad
 L^{\rm sur}_2=-\operatorname{sg}(w_\theta A^{\rm cur})\log p_\theta.
$$

其中 $\operatorname{sg}$ 表示 stop-gradient：

$$
\nabla L^{\rm sur}_1
=-\operatorname{sg}(A^{\rm cur})\nabla w_\theta
=-w_\theta A^{\rm cur}\nabla\log p_\theta
=\nabla L^{\rm sur}_2.
$$

第一种写法让梯度经过 ratio，第二种让梯度经过 log-prob。不能把两者拼成未 detach 的 $-w_\theta A^{\rm cur}\log p_\theta$，那会引入额外导数项。

两种 surrogate 的数值一般不同。第一种的前向数值为 $w_\theta\log(p_\theta/T)$，所以在正确、未截断 IS 下，其总体期望就是 $J$；第二种不具有这个数值身份关系。相同梯度不意味着相同 loss 数值。比较不同实现时应统一真实 KL 或同一个估计器，不能直接比较它们各自记录的 surrogate loss。

`detach()` 只冻结本次反向中的系数，不表示下一次 forward 仍使用旧数值。每次训练 forward 已经得到当前 log-prob，重新形成 $A^{\rm cur}$ 只需要与缓存 teacher log-prob 相减，不需要额外运行一次模型。

<a id="objectives-33-普通标量目标与控制变量"></a>

#### 3.3 普通标量目标与控制变量

直接对 $w_\theta\log(p_\theta/T)$ 自动微分，每个样本会多出 $w_\theta\nabla\log p_\theta$。这一项在完整期望下为零，但有限样本时不必为零。省略它是一种零均值项的去除，不是忽略一个非零的真实目标梯度。

还可以构造无需 stop-gradient 的逐样本标量：

$$
 L^{\rm corr}(a)
=\frac{
p_\theta(a)\log\frac{p_\theta(a)}{T(a)}
-p_\theta(a)+T(a)
}{\beta(a)}.
$$

其期望与梯度分别为：

$$
\mathbb E_\beta[ L^{\rm corr}]
=J-\sum_a p_\theta(a)+\sum_aT(a)=J,
$$

$$
\nabla L^{\rm corr}
=\frac{\nabla p_\theta(a)}{\beta(a)}
\log\frac{p_\theta(a)}{T(a)}
=-w_\theta A^{\rm cur}\nabla\log p_\theta.
$$

$-p_\theta$ 的导数逐样本抵消额外的 $+1$。Dr. OPD 附录给出相应的带权形式；这里取动作权重为 1。[^objectives-dropd] CPU 验算同时核对了这个形式与前面的 surrogate。

<a id="objectives-34-固定-token-不要求固定-kl-信号"></a>

#### 3.4 固定 token 不要求固定 KL 信号

$$
J(\theta)=\mathbb E_{p_\theta}[-\log T(a)]-\mathcal H(p_\theta).
$$

固定前缀和 token 后，teacher 的 $\log T(a)$ 固定，但 student 熵项仍随当前策略变化。KL 学习信号因此不同于一个与策略参数无关的外部正确率 reward。

<a id="objectives-four-cases"></a>

<a id="objectives-4-四种更新及其标量目标"></a>

### 4. 四种更新及其标量目标

<a id="objectives-41-固定信号与-is"></a>

#### 4.1 固定信号与 IS

定义冻结信号：

$$
A^{\rm anchor}(a)=\log T(a)-\log q(a).
$$

使用正确 IS，但不 clipping：

$$
\begin{aligned}
L_{\mathrm{fixed}}(\theta)
&=-\mathbb E_\beta[w_\theta A^{\rm anchor}]\\
&=\sum_a p_\theta(a)\log\frac{q(a)}{T(a)}\\
&=\sum_a p_\theta(a)\log\frac{p_\theta(a)}{T(a)}
-\sum_a p_\theta(a)\log\frac{p_\theta(a)}{q(a)}\\
&=J(\theta)-D_{\mathrm{KL}}(p_\theta\Vert q).
\end{aligned}
$$

因此：

$$
\nabla L_{\mathrm{fixed}}
=\nabla J-\nabla D_{\mathrm{KL}}(p_\theta\Vert q).
$$

在 $p_\theta=q$ 处，第二项梯度为零。固定与重算 advantage 的初始梯度相同，之后差异由 student 偏离 $q$ 的程度决定。IS 正确变换了期望，却不会把冻结的信号自动更新成当前信号。

<a id="objectives-42-重算-advantage-但不做-is"></a>

#### 4.2 重算 advantage 但不做 IS

固定采样分布下的更新：

$$
-\operatorname{sg}(A^{\rm cur}(a))\log p_\theta(a)
$$

具有梯度：

$$
\mathbb E_\beta[
(\log p_\theta-\log T)\nabla\log p_\theta]
=\nabla_\theta\frac12
\mathbb E_\beta[(\log p_\theta-\log T)^2].
$$

这是固定采样权重下的 log-prob 平方误差梯度，不是当前分布权重下的 reverse-KL 梯度。Semi-OPD 的固定回答与重算 student 信号提供了相关算法证据，但不能据其效果宣称正确 IS 在数学上多余。[^objectives-semiopd]

<a id="objectives-43-固定信号与-nois"></a>

#### 4.3 固定信号与 NoIS

保持未中心化信号 $r=\log(T/q)$，NoIS 的 score 更新为 $-\mathbb E_\beta[r\psi]$。因为 $\beta,r$ 冻结，它正是以下目标的梯度：

$$
L_{\rm frozen,NoIS}(p)=-\mathbb E_\beta[r\log p],\qquad
\nabla L_{\rm frozen,NoIS}=-\mathbb E_\beta[r\psi].
$$

该目标是带符号权重的交叉熵。完整支持且 $q\ne T$ 时，存在 $r_a<0$；令 $p_a\to0$，相应损失趋于负无穷。冻结信号加 IS 的目标则是概率上的线性函数，其在闭单纯形上的最优分布仅支持 $r$ 最大的动作；若要求 $p$ 严格为正，一般只能趋近这个边界下确界。它通常不以 $T$ 为最优解。

<a id="objectives-44-统一对照"></a>

#### 4.4 统一对照

下表固定 $\beta,q,T$，所有信号均未中心化，不含 clipping、whitening 或额外正则。$\psi=\nabla\log p$。表中标量是产生对应期望梯度的真实目标，通常不等于程序中 detached surrogate 的前向值。

| 信号与测度     | 损失梯度                            | 对应标量目标                                  |
| -------------- | ----------------------------------- | --------------------------------------------- |
| Current + IS   | $-\mathbb E_p[A^{\rm cur}\psi]$     | $\mathrm{KL}(p\Vert T)$                       |
| Frozen + IS    | $-\mathbb E_p[r\psi]$               | $\mathrm{KL}(p\Vert T)-\mathrm{KL}(p\Vert q)$ |
| Current + NoIS | $-\mathbb E_\beta[A^{\rm cur}\psi]$ | $\frac12\mathbb E_\beta[\log^2(p/T)]$         |
| Frozen + NoIS  | $-\mathbb E_\beta[r\psi]$           | $-\mathbb E_\beta[r\log p]$                   |

IS 改变积分测度，current/frozen 改变信号。两者是独立选择。在完整支持、自由分布可实现时，current 两格均以 $p=T$ 为唯一全局最优分布，但梯度、参数化受限时的投影和有限样本行为不同。若 $\beta=q$ 且 $p=q$，四格初始梯度相同；内层离开锚点后才产生差异。固定正则对应的目标与最优解由镜像下降一篇统一推导。

<a id="objectives-baseline"></a>

<a id="objectives-45-中心化改变-nois-的目标"></a>

#### 4.5 中心化改变 NoIS 的目标

对动作无关、在当前反向中停止梯度的标量 $b$，将 $A$ 改成 $A+b$，梯度变化为

$$
\Delta g=-b\,\mathbb E_\nu\psi.
$$

当前测度 $\nu=p$ 时该项为零；行为测度 $\nu=\beta$ 时一般非零。若 $b$ 整轮冻结，NoIS 目标额外增加 $-b\mathbb E_\beta\log p$；若 $b=b(p)$ 每次刷新但在反向中 detach，不能直接把同一个表达式按普通全导数处理。本文四格表采用未中心化信号，其他 baseline 必须单独声明。方差最优 baseline 见条件梯度估计一篇。

<a id="objectives-sampling-boundary"></a>

<a id="objectives-5-clipping-与有限缓存"></a>

### 5. Clipping 与有限缓存

<a id="objectives-51-clipping-与信号刷新分别比较"></a>

#### 5.1 Clipping 与信号刷新分别比较

令 $c_\epsilon(w)=\operatorname{clip}(w,1-\epsilon,1+\epsilon)$：

$$
 L^{\rm clip}(A^{\rm sig})
=-\min\{w_\theta\operatorname{sg}(A^{\rm sig}),
c_\epsilon(w_\theta)\operatorname{sg}(A^{\rm sig})\}.
$$

固定或重算 $A^{\rm sig}$，以及有无 clipping，是两个独立开关。Clipping 改变梯度，前面的无偏性不再直接成立；它既可能限制固定信号的过度更新，也可能截掉当前 KL 仍需要的修正。

当 $p_\theta=T$ 时，当前 advantage 对每个 token 都为零。固定 advantage 通常仍非零，是否停止取决于 surrogate 与约束。PPO clipping 不是概率的硬投影，也不保证停止点恰好等于 teacher。

其中 $A^{\rm sig}$ 选择 $A^{\rm cur}$ 或 $A^{\rm anchor}$；两者在反向中均停止系数梯度。

<a id="objectives-52-有限缓存的统计边界"></a>

#### 5.2 有限缓存的统计边界

“无偏”是对动作抽样的期望而言。固定一份有限缓存、反复拟合后，参数依赖这份缓存，不能再不加条件地声称每次自适应更新均是对总体目标的条件无偏估计。一个 prefix 只缓存一个 token，也不会因重复更新而获得 teacher 对其他 token 的分数。

若在同一 prefix 下独立有放回采样 $M$ 个 next-token 候选：

$$
\widehat g_M=\frac1M\sum_{i=1}^M
-w_\theta(a_i)A^{\rm cur}(a_i)\nabla\log p_\theta(a_i),
$$

固定 $\theta,s$ 且样本独立时有：

$$
\operatorname{Cov}(\widehat g_M\mid s)
=\frac1M\operatorname{Cov}(\widehat g_1\mid s).
$$

这里增加的是同一前缀的局部候选，不要求展开 $M$ 条完整回答。重复候选应保留其抽样次数；不带相应权重地去重或改为确定性 top-k，会改变估计器。实际方差还取决于 $w_\theta$ 的长尾，不能只看样本数。

<a id="objectives-6-数值验算与论文证据"></a>

### 6. 数值验算与论文证据

[原有 CPU 验算](../../research/checks/objectives/verify_fixed_prefix.py)与[保存结果](../literature/learning-signal/data/gradient-audit.json)覆盖 current IS、冻结信号加 KL、修正标量和 NoIS 平方误差。四种更新及镜像正则的新增枚举验算见[verify_offpolicy_objectives.py](../../research/checks/objectives/verify_offpolicy_objectives.py)及[结果](data/offpolicy-objectives.json)。论文实验与固定版本实现证据继续在[原审计文档](../literature/learning-signal/2026-10-21-固定前缀下的优势估计与重要性采样.md#6-论文与实现证据)维护。

<a id="objectives-参考文献"></a>

#### 参考文献

[^objectives-asyncopd]: Kang et al. AsyncOPD: How Stale Can On-Policy Distillation Be? arXiv:2606.24143v1，2026，第 3/5/6 节、表 1。[论文](https://arxiv.org/html/2606.24143v1)。

[^objectives-dropd]: Zhenyu Wang, Tianze Wang, Linjun Zhang, Yifan Hu. Dr. OPD: Learning What to Follow for Optimal On-Policy Distillation of Large Language Models. arXiv:2609.38025v1，2026，附录 C.2。[论文](https://arxiv.org/html/2609.38025v1)。

[^objectives-semiopd]: Zhao et al. When Do We Need On-Policy Distillation? Distilling on Offline Student Rollouts Is Often Better. arXiv:2610.11291v1，2026，第 2 节。[论文](https://arxiv.org/html/2610.11291v1)。

<a id="graphs"></a>

<a id="graphs-full-vocab-计算图与四种更新"></a>

## 完整词表计算图与四种更新

固定前缀，教师与学生共享 token-ID 语义。采用[OPD 目标篇](#objectives)的 $p,q,T,\beta,\psi$；记 $\ell_a=\log(p_a/T_a)$、$K=\mathrm{KL}(p\Vert T)$。本篇区分精确 KL 数值、精确一阶梯度和 sampled-token 的四种更新。

<a id="graphs-graphs"></a>

<a id="graphs-1-full-vocab-kl-的计算图"></a>

### 1. Full-vocab KL 的计算图

<a id="graphs-11-只计算数值作为诊断或-baseline"></a>

#### 1.1 只计算数值，作为诊断或 baseline

给定 student 与 teacher 的全部分布，$K=\sum_a p_a\ell_a$ 是精确有限求和，已经没有 next-token Monte Carlo 抽样。它仍只针对给定前缀，不是对所有可能轨迹的枚举。

在诊断或 baseline 路径，使用 `no_grad` / detach 得到 scalar 即可。用于 sampled-token 时，reward advantage 是

$$
A^{\rm cur}(a)+K=-\ell_a+K,
$$

其系数停止梯度；这是 sampled-token + full KL baseline。**算出了精确 KL baseline，不等于使用精确 full-vocab 梯度**：最终更新仍依赖 sampled action。

该数值可用 hidden states + LM head 流式计算，只保留每位置 $K,\log Z_p,\log Z_T$，不用保存完整词表的 autograd 图。原配方已有 teacher 和 old-log-prob 阶段，可考虑在无梯度 scoring 阶段算好 baseline，再进入 actor backward；得到的是按该快照计算的 baseline，多次更新后不再是当前精确 KL。

<a id="graphs-12-作为训练-loss直接自动微分"></a>

#### 1.2 作为训练 loss，直接自动微分

最简单的正确参照是：

```python
log_p = student_logits.float().log_softmax(-1)
log_T = teacher_logits.detach().float().log_softmax(-1)
p = log_p.exp()
kl_per_token = (p * (log_p - log_T)).sum(-1)
loss = aggregate_with_global_response_mask(kl_per_token)
loss.backward()
```

这里 `p` 和 `log_p` 都保留 student 梯度，teacher 固定。对 student logits $z$，

$$
\delta_a:=\frac{\partial K}{\partial z_a}
=p_a(\ell_a-K),\qquad \sum_a\delta_a=0.
$$

这条路径前向输出真实 KL，反向输出真实条件 KL 梯度。它适合作为小规模 reference，直接物化大词表中间量的显存成本则较高。

<a id="graphs-13-只传递精确梯度可以使用代理或自定义-backward"></a>

#### 1.3 只传递精确梯度，可以使用代理或自定义 backward

以下代理具有相同的一阶参数梯度：

$$
L_{\rm score}=\sum_a\operatorname{sg}(p_a\ell_a)\log p_a,
\qquad
L_{\rm logit}=\sum_a\operatorname{sg}(\delta_a)z_a.
$$

但二者的前向数值通常不是 $K$。应分别记录真实 KL 与梯度代理；或返回

$$
L=\operatorname{sg}(K)+L_{\rm logit}-\operatorname{sg}(L_{\rm logit}),
$$

使前向值为真实 KL、一阶 backward 为所需梯度。不能因为这种代理的一阶梯度正确，就把它的二阶导数也当作真实 KL Hessian。

还有一个特殊的 full-vocab 形式：

$$
L_{\rm gap}=\sum_a p_a\operatorname{sg}(\ell_a).
$$

它前向数值也等于 $K$，一阶梯度也正确，因为被省略的常数项为 $\sum_a p_a\psi_a=0$。但若只对真子集求和，这个消去不再自动成立；这正是普通 truncated KL、stopped-coefficient head 和 normalized KL 不同的原因。

高效 exact full-vocab 实现可以在 forward 分块累计归一化量与 $K$，在 backward 重建每个词表块并计算 $\delta_a$，然后累加

$$
\nabla_{h_S}K=\sum_a\delta_aW_{S,a},\qquad
\nabla_{W_{S,a}}K=\delta_a h_S^\top.
$$

不需要把全部 logits 留到 backward；student transformer 的 backward 仍只执行一次。详细 online reduction 及数值预算复用工程文档。Nitrobrew 是可参考的 exact chunked 实现；SparseOPD 则进一步稀疏化 $\delta$，后者一般是近似梯度，不能因为它先计算 full KL 就称最终更新完全精确。[^graphs-nitro][^graphs-sparse]

<a id="graphs-14-两个容易写错的-detach"></a>

#### 1.4 两个容易写错的 detach

```python
# 错误：数值仍是 KL，但对完整词表的 student 梯度几乎为零。
wrong = (p.detach() * (log_p - log_T)).sum(-1)
```

它的梯度是 $\sum_a p_a\psi_a=0$。另一个错误是直接对固定 sampled action 的 $\ell_a$ 求导，再声称这就是 reverse-KL 梯度；它仅产生 $\psi_a$，在当前分布下期望同样为零。

反过来，**数值无偏与梯度无偏可以同时实现，但要写对计算图**。对 $a\sim\beta$，$w_a=p_a/\beta_a$：

$$
\widehat K_{\rm IS}=w_a\ell_a,\qquad
\mathbb E_\beta\widehat K_{\rm IS}=K.
$$

直接求导得到 $w_a(\ell_a+1)\psi_a$，其期望也是 $\nabla K$，但比常用 stopped-coefficient 梯度多一个零均值随机项。构造

$$
\widehat K_{\rm corr}=w_a(\ell_a-1)+\frac{T_a}{\beta_a}
$$

后，数值期望仍为 $K$，逐样本梯度变为 $w_a\ell_a\psi_a$。这里 $T,\beta$ 都冻结、权重未截断。这个例子说明应同时写清“估计什么数值”和“让哪些量参与 backward”，不能只凭记录的 loss 名称判断。

<a id="graphs-four-cases"></a>

<a id="graphs-2-四种-score-更新的完整词表形式"></a>

### 2. 四种 score 更新的完整词表形式

对任意信号 $A$ 和积分权重 $\nu$，定义

$$
g_{\nu,A}=-\sum_a\nu_aA_a\psi_a,\qquad
L^{\rm sur}_{\nu,A}=-\sum_a\operatorname{sg}(\nu_aA_a)\log p_a.
$$

则普通反向满足 $\nabla L^{\rm sur}_{\nu,A}=g_{\nu,A}$。取 $\nu=p$ 对应 sampled-token 的精确行为 IS，因为 $\beta_a(p_a/\beta_a)=p_a$；取 $\nu=\beta$ 对应 NoIS。信号取 $A^{\rm cur}=\log(T/p)$ 或 $r=\log(T/q)$，就得到 OPD 目标篇的四格。完整词表直接计算四个真实标量目标，或按本式计算一阶梯度代理，都能实现这些更新。

Full-vocab 下无需显式保存 $p/\beta$ 来计算 $p$ 加权求和；这里 IS/NoIS 是与采样算法的对应名称，实际开关是当前分布加权与行为分布加权。精确积分消除动作抽样方差，但保留四格之间的目标差异。若选择行为分布加权，则需要整个词表上的 $\beta$；只缓存 rollout token 的行为概率不足以重建该权重。

在每个固定前缀上，真实 KL 的梯度满足

$$
\nabla K=\mathbb E_{a\sim\beta}\left[-\frac{p_a}{\beta_a}A^{\rm cur}_a\psi_a\right].
$$

full-vocab backward 已经使用当前信号。由于 $\mathbb E_pA^{\rm cur}=-K$，第 1 节的 logit 梯度也可写成

$$
\frac{\partial K}{\partial z_a}=-p_a(A^{\rm cur}_a+K).
$$

其中的中心化来自 softmax 求导，不需要另建 advantage estimator。精确求和时，任意动作无关 scalar baseline 都由 $\sum_a p_a\psi_a=0$ 抵消；full-vocab 加 scalar baseline 不会再降低已为零的条件动作方差。它与 sampled-token 使用 full-vocab KL baseline 是两种不同计算。

<a id="graphs-literal-substitution"></a>

<a id="graphs-3-直接替换-kl-概率得到的目标"></a>

### 3. 直接替换 KL 概率得到的目标

若从标量 $\sum_a p_a\log(p_a/T_a)$ 出发，替换外部权重和内部对数中的概率，并对保留的当前 $p$ 普通求导，则得到：

| 外部权重 | 当前对数差 $\log(p/T)$                                | 冻结对数差 $\log(q/T)$                        |
| -------- | ----------------------------------------------------- | --------------------------------------------- |
| $p$      | $\mathrm{KL}(p\Vert T)$                               | $\mathrm{KL}(p\Vert T)-\mathrm{KL}(p\Vert q)$ |
| $\beta$  | $\mathrm{KL}(\beta\Vert T)-\mathrm{KL}(\beta\Vert p)$ | 常数 $\mathbb E_\beta\log(q/T)$               |

下排不是四种 score 更新中的 NoIS 两格。其当前列的梯度是 $\mathbb E_\beta\psi$，教师只贡献常数；冻结列的梯度为零。要复现 NoIS score 更新，应使用 OPD 目标篇中的平方误差或带符号交叉熵，而非直接替换 KL 的权重。

<a id="graphs-4-固定前缀与实现范围"></a>

### 4. 固定前缀与实现范围

在固定经验前缀测度 $\widehat d$ 上，完整训练目标为 $\mathbb E_{s\sim\widehat d}K(s;\theta)$。旧轨迹决定了 $\widehat d$，直接 backward 对这个目标是精确的。full-vocab 仅消除当前动作积分误差，不恢复缺失前缀，也不包含轨迹采样分布的导数。固定序列求条件散度梯度与 GKD 的计算约定一致；MiniLLM 的整序列推导还包含未来影响，超出本篇范围。[^graphs-gkd][^graphs-minillm]

本仓 head-loss 路径使用当前 student hidden/head 和冻结 teacher hidden/head，反向同时更新学生 head 与 backbone。配置、token-mean 聚合及数值验收见[Full-KL 实现与验收](../infra/numerics/full-vocab/validation.md)。词表分块与重计算是实现选择，不改变本篇的目标。

<a id="graphs-5-验算"></a>

### 5. 验算

[既有 full-KL 审计](../../research/checks/objectives/verify_fullkl_and_support.py)检查数值、logit 梯度与支持集；[统一四格验算](../../research/checks/objectives/verify_offpolicy_objectives.py)检查当前分布加权、行为分布加权与直接替换概率之间的差别。验算固定词表并完整枚举，不加载语言模型。

<a id="graphs-参考文献"></a>

#### 参考文献

[^graphs-nitro]: Tilde Research. Nitrobrew，`17f6f16b7451929209ac495c6b0ea0fcc4d0cedc`。[forward 与自定义 backward](https://github.com/tilde-research/nitrobrew-release/blob/17f6f16b7451929209ac495c6b0ea0fcc4d0cedc/nitrobrew.py)。

[^graphs-sparse]: Miao et al. Look Before You Select: Rethinking Vocabulary Sparsification in On-Policy Distillation. arXiv:2609.34386v1, 2026，第 2–3 节。[论文](https://arxiv.org/html/2609.34386v1)。

[^graphs-gkd]: Agarwal et al. On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes. arXiv:2306.13649v3，ICLR 2024。[论文](https://arxiv.org/html/2306.13649v3)。

[^graphs-minillm]: Gu et al. MiniLLM: On-Policy Distillation of Large Language Models. arXiv:2306.08543v6，第 2.2 节。[论文](https://arxiv.org/html/2306.08543v6)。
