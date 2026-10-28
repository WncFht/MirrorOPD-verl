# Full-vocab 计算图与四种更新

固定前缀，教师与学生共享 token-ID 语义。采用[OPD 目标篇](2026-10-22-固定前缀下的OPD目标与梯度.md)的 $p,q,T,\beta,\psi$；记 $\ell_a=\log(p_a/T_a)$、$K=\mathrm{KL}(p\Vert T)$。本篇区分精确 KL 数值、精确一阶梯度和 sampled-token 的四种更新。

<a id="graphs"></a>

## 1. Full-vocab KL 的计算图

### 1.1 只计算数值，作为诊断或 baseline

给定 student 与 teacher 的全部分布，$K=\sum_a p_a\ell_a$ 是精确有限求和，已经没有 next-token Monte Carlo 抽样。它仍只针对给定前缀，不是对所有可能轨迹的枚举。

在诊断或 baseline 路径，使用 `no_grad` / detach 得到 scalar 即可。用于 sampled-token 时，reward advantage 是

$$
A^{\rm cur}(a)+K=-\ell_a+K,
$$

其系数停止梯度；这是 sampled-token + full KL baseline。**算出了精确 KL baseline，不等于使用精确 full-vocab 梯度**：最终更新仍依赖 sampled action。

该数值可用 hidden states + LM head 流式计算，只保留每位置 $K,\log Z_p,\log Z_T$，不用保存完整词表的 autograd 图。原配方已有 teacher 和 old-log-prob 阶段，可考虑在无梯度 scoring 阶段算好 baseline，再进入 actor backward；得到的是按该快照计算的 baseline，多次更新后不再是当前精确 KL。

### 1.2 作为训练 loss，直接自动微分

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

### 1.3 只传递精确梯度，可以使用代理或自定义 backward

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

不需要把全部 logits 留到 backward；student transformer 的 backward 仍只执行一次。详细 online reduction 及数值预算复用工程文档。Nitrobrew 是可参考的 exact chunked 实现；SparseOPD 则进一步稀疏化 $\delta$，后者一般是近似梯度，不能因为它先计算 full KL 就称最终更新完全精确。[^nitro][^sparse]

### 1.4 两个容易写错的 detach

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

<a id="four-cases"></a>

## 2. 四种 score 更新的完整词表形式

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

<a id="literal-substitution"></a>

## 3. 直接替换 KL 概率得到的目标

若从标量 $\sum_a p_a\log(p_a/T_a)$ 出发，替换外部权重和内部对数中的概率，并对保留的当前 $p$ 普通求导，则得到：

| 外部权重 | 当前对数差 $\log(p/T)$ | 冻结对数差 $\log(q/T)$ |
| --- | --- | --- |
| $p$ | $\mathrm{KL}(p\Vert T)$ | $\mathrm{KL}(p\Vert T)-\mathrm{KL}(p\Vert q)$ |
| $\beta$ | $\mathrm{KL}(\beta\Vert T)-\mathrm{KL}(\beta\Vert p)$ | 常数 $\mathbb E_\beta\log(q/T)$ |

下排不是四种 score 更新中的 NoIS 两格。其当前列的梯度是 $\mathbb E_\beta\psi$，教师只贡献常数；冻结列的梯度为零。要复现 NoIS score 更新，应使用 OPD 目标篇中的平方误差或带符号交叉熵，而非直接替换 KL 的权重。

## 4. 固定前缀与实现范围

在固定经验前缀测度 $\widehat d$ 上，完整训练目标为 $\mathbb E_{s\sim\widehat d}K(s;\theta)$。旧轨迹决定了 $\widehat d$，直接 backward 对这个目标是精确的。full-vocab 仅消除当前动作积分误差，不恢复缺失前缀，也不包含轨迹采样分布的导数。固定序列求条件散度梯度与 GKD 的计算约定一致；MiniLLM 的整序列推导还包含未来影响，超出本篇范围。[^gkd][^minillm]

本仓 head-loss 路径使用当前 student hidden/head 和冻结 teacher hidden/head，反向同时更新学生 head 与 backbone。配置、token-mean 聚合及数值验收见[Full-KL 实现与验收](../infra/numerics/full-vocab/validation.md)。词表分块与重计算是实现选择，不改变本篇的目标。

## 5. 验算

[既有 full-KL 审计](../../research/objectives/verify_fullkl_and_support.py)检查数值、logit 梯度与支持集；[统一四格验算](../../research/objectives/verify_offpolicy_objectives.py)检查当前分布加权、行为分布加权与直接替换概率之间的差别。验算固定词表并完整枚举，不加载语言模型。

### 参考文献

[^nitro]: Tilde Research. Nitrobrew，`17f6f16b7451929209ac495c6b0ea0fcc4d0cedc`。[forward 与自定义 backward](https://github.com/tilde-research/nitrobrew-release/blob/17f6f16b7451929209ac495c6b0ea0fcc4d0cedc/nitrobrew.py)。

[^sparse]: Miao et al. Look Before You Select: Rethinking Vocabulary Sparsification in On-Policy Distillation. arXiv:2609.34386v1, 2026，第 2–3 节。[论文](https://arxiv.org/html/2609.34386v1)。

[^gkd]: Agarwal et al. On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes. arXiv:2306.13649v3，ICLR 2024。[论文](https://arxiv.org/html/2306.13649v3)。

[^minillm]: Gu et al. MiniLLM: On-Policy Distillation of Large Language Models. arXiv:2306.08543v6，第 2.2 节。[论文](https://arxiv.org/html/2306.08543v6)。
