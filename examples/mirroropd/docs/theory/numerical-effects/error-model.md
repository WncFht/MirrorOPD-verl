# KL 数值误差模型

> 本文保留误差推导及假设。原数值实验和接口变更已归入 infra。当前生产计算和诊断接口见[Full-vocab](../../infra/architecture/full-vocab.md)及[指标系统](../../infra/architecture/metrics.md)。

日期：2026-10-24。目标：给出带假设的精度量级估计、可复算的小实验，并审查和简化 Full-KL 位置诊断。本文区分**目标分布被量化改变**、**KL 数值计算误差**和**优化未收敛**。dtype 本身不能决定一个统一的正 KL 下限；当前 E16/E17 使用的同架构、同支持集模型存在分布完全相同的可表示解。

## 1. 从 Logit 扰动推导 KL

### 1.1 精确表达式、局部近似与上界

本篇局部记号用 p 表示教师、q 表示受扰动后的学生，与全局约定中的 T、p 分别对应；以下公式保持原推导记号。固定前缀，令教师 $p=\operatorname{softmax}(z)$，受扰动后的学生 $q=\operatorname{softmax}(z+\delta)$。这里 $\delta$ 可以表示模型本身的差异，也可以表示两种精度的计算误差差。定义：

$$
A(t)=\log\sum_v p_v\exp(t\delta_v),\qquad
p_t(v)=\frac{p_v\exp(t\delta_v)}{\sum_u p_u\exp(t\delta_u)}.
$$

则 $A(0)=0$、$A'(t)=\mathbb E_{p_t}\delta$、$A''(t)=\operatorname{Var}_{p_t}(\delta)$，且 $q=p_1$。将 $\log(q_v/p_v)=\delta_v-A(1)$ 代入 KL，积分分部得到：

$$
\begin{aligned}
K&=D_{\mathrm{KL}}(q\Vert p)=A'(1)-A(1)\\
 &=A'(1)-\int_0^1 A'(t)\,dt
 =\int_0^1t\,\operatorname{Var}_{p_t}(\delta)\,dt.
\end{aligned}
$$

如果扰动幅度足够小，$p_t$ 与 $p$ 接近：

$$
K\approx\frac12\operatorname{Var}_{p}(\delta)
=\frac12\delta^\top F(p)\delta,
\qquad F(p)=\operatorname{diag}(p)-pp^\top.
$$

令 $R=\max_v\delta_v-\min_v\delta_v$。任意支撑在长度为 $R$ 的区间上的随机变量都满足方差不超过 $R^2/4$，因此精确有：

$$
0\leq K\leq\frac{R^2}{8}.
$$

此外 $A'''(t)=\mathbb E_{p_t}[(\delta-\mathbb E_{p_t}\delta)^3]$，绝对值不超过 $R^3$，所以：

$$
\left|K-\frac12\operatorname{Var}_{p}(\delta)\right|
\leq\int_0^1t(tR^3)\,dt=\frac{R^3}{3}.
$$

这是保守余项界；实际小扰动下通常更紧。共同 logit 平移被方差消去：$\delta_v$ 对所有词元相同，KL 精确为零。有效误差是概率加权、去掉共同平移后的误差，不是原始 logit 最大绝对值。

### 1.2 由随机误差预测期望 KL

假设理想师生已匹配，两边实现误差之差为 $\delta=\epsilon_S-\epsilon_T$，均值 $b$、协方差 $\Sigma$。由二次型期望：

$$
\mathbb E K\approx\frac12\left[b^\top F(p)b+\operatorname{tr}(F(p)\Sigma)\right].
$$

若进一步假设误差零均值、不同词元坐标独立且同方差 $\sigma_\delta^2$：

$$
\mathbb E K\approx\frac{\sigma_\delta^2}{2}
\left(1-\sum_v p_v^2\right).
$$

记 $C(p)=1-\sum_vp_v^2$。它表示本地分布对独立 logit 误差的敏感程度：高置信、接近单点的分布通常 $C$ 较小。该结论依赖误差协方差假设，不表示低熵对所有结构化扰动都稳健。

如果还存在未学完的确定性差异 $d$，把 $\delta$ 换成 $d+\epsilon$ 即可。零均值噪声下，二阶期望由 $\frac12d^\top Fd$ 与 $\frac12\operatorname{tr}(F\Sigma)$ 相加。只有假定这份随机误差在优化中持续存在且无法被学习或抵消，后者才具有“预期噪声底”的解释。

## 2. BF16 能对应多大的分布误差

### 2.1 均匀舍入近似与误差相关性

BF16 有 7 位显式尾数。round-to-nearest 的 unit roundoff 为 $u_b=2^{-8}$；FP32 对应 $u_f=2^{-24}$。它们是相对误差量级，不是 KL 的绝对下限；`torch.finfo.eps` 则分别是 $2^{-7}$、$2^{-23}$，两者相差一倍。[^dtype]

在一个固定指数区间内，设有效量化步长为 $\Delta$，单边舍入误差近似均匀分布在 $[-\Delta/2,\Delta/2]$，方差 $\Delta^2/12$。两边同方差，相关系数为 $\rho$，不同词元误差独立，则：

$$
\operatorname{Var}(\epsilon_S-\epsilon_T)
=\frac{\Delta^2}{6}(1-\rho),
\qquad
\boxed{\mathbb E K\approx\frac{\Delta^2}{12}(1-\rho)C(p)}.
$$

只有一边有量化误差、另一边是理想参考时，系数改为 $\Delta^2C(p)/24$。两边误差完全相同则 $\rho=1$，KL 为零，而不是一个固定正数。

若**直接把 logits 舍入为 BF16**，指数区间给出的示例为：

| Logit 绝对值区间 | 步长 $\Delta$ | 两边独立误差、$C\approx1$ 时的期望 KL |
| ---------------- | ------------: | ------------------------------------: |
| [1,2)            |     0.0078125 |                   $5.09\times10^{-6}$ |
| [4,8)            |       0.03125 |                   $8.14\times10^{-5}$ |
| [8,16)           |        0.0625 |                   $3.26\times10^{-4}$ |
| [16,32)          |         0.125 |                   $1.30\times10^{-3}$ |

**这张表不能直接套到我们的 loss 上。** 当前 CUDA 输出头保留 FP32 logits，不做这一步 BF16 logit 舍入。表格只是展示：在指定误差模型下，精度的影响如何随尺度和相关性变化。

### 2.2 当前 BF16 输入、FP32 输出的 Head

对于 $z=Wh$，将权重和 hidden 的量化误差记为 $E_W,e_h$：

$$
\widehat z-z=W e_h+E_W h+E_W e_h+e_{\rm GEMM}.
$$

保留一阶项，并假设权重误差逐元素独立、hidden 误差逐坐标独立、二者互相独立且零均值。设它们方差为 $s^2_{W,vj}$、$s^2_{h,j}$，则：

$$
\Sigma_z=W\operatorname{diag}(s_h^2)W^\top+
\operatorname{diag}\left(\sum_jh_j^2s^2_{W,vj}\right).
$$

代入 $\frac12\operatorname{tr}(F\Sigma_z)$，得到无需构建词表平方协方差矩阵的计算式：

$$
\begin{aligned}
\mathbb E K\approx\frac12\Bigg[
&\sum_j s_{h,j}^2\left(\sum_vp_vW_{vj}^2-
\left(\sum_vp_vW_{vj}\right)^2\right)\\
+&\sum_v p_v(1-p_v)\sum_jh_j^2s^2_{W,vj}
\Bigg].
\end{aligned}
$$

这里保留了共享 hidden 误差造成的词元间相关性，不能把全部 logit 误差简单当作独立噪声。小实验使用各输入所在指数区间的实际 BF16 步长估计 $s^2=\Delta^2/12$。

真实 Transformer 还包含前面各层的误差传播。若各层误差为 $e_\ell$，一阶近似为 $\delta z\approx\sum_\ell J_\ell e_\ell$；协方差需要 Jacobian 及跨层相关项。只有额外假设“映射到输出的噪声独立、每层贡献相同”，才可把期望 KL 近似相加；不能拿 head 的误差直接乘 28 当作 1.5B 模型的误差预测。

### 2.3 精度噪声模型不等于训练收敛定理

真实舍入通常是参数和输入的确定性函数，学习可能让两边误差变得相关，也可能放大差异。E16/E17 的 student/teacher 架构相同、支持集相同；如果参数及执行路径相同，BF16 条件分布也可以相同。因此不存在仅由“BF16”三个字决定的不可突破正 KL 下限。

额外词表支持则是另一类真实下界。若学生只能在 $S$ 上分配概率，教师在 $S$ 外仍有质量：

$$
D_{\mathrm{KL}}(q\Vert p)=D_{\mathrm{KL}}(q\Vert p(\cdot\mid S))-\log p(S)
\geq-\log p(S).
$$

这属于支持集限制，不属于精度限制；现有额外教师词表行修正正是在保留这项量。

## 3. FP32 KL 计算的分辨能力

### 3.1 为什么残差小于中间项时容易受误差影响

当前公式为 $K=\sum_vq_v(\log q_v-\log p_v)$。在接近匹配时，正负项相互抵消，最终 KL 是二阶小量。设 log-softmax 误差分别为 $a_v,b_v$，满足绝对值不超过 $\varepsilon$；暂时忽略 exp 和求和的舍入，则计算相当于：

$$
\widehat K=\sum_v q_v e^{a_v}(r_v+a_v-b_v),
\qquad r_v=\log(q_v/p_v).
$$

直接减去 $K$ 并用三角不等式可得：

$$
|\widehat K-K|
\leq(e^\varepsilon-1)\mathbb E_q|r|
+2\varepsilon e^\varepsilon+\eta,
$$

其中 $\eta$ 汇总实际 exp、乘法和归约的额外误差。这个界很保守，但显示了问题：KL 变小时，绝对误差项未必同比例下降。GPU 归约顺序、输入尺度和后端会影响误差，FP32 标量本身的相邻数间隔不足以预测整个计算。[^numerical]

对完全相同的两份计算，$a=b$ 且 $r=0$，直接相减仍可以精确得到零。因此也不存在 FP32 必然输出某个正 loss 的结论。

### 3.2 如何把数值实验转为工作阈值

在指定输入和实现下，以高精度参考测得误差包络 $B$。如果希望 KL 的相对计算误差不超过 $\tau$，可把 $K\gtrsim B/\tau$ 作为**经验诊断阈值**。

本次合成输入的普通 FP32 公式，最大逐位置误差约 $10^{-7}$–$10^{-6}$。若要求约 1% 误差，需要把 $10^{-5}$–$10^{-4}$ 视为需要关注的量级；在 $10^{-6}$ 附近，部分 token 的正负号已经不可靠。这个包络来自有限输入，不是对所有 logits 的严格误差保证。跨 token 的平均可能抵消部分误差，也可能保留归一化偏差；不能默认除以 $\sqrt N$。

当前 E16/E17 的 $4\times10^{-4}$–$1.5\times10^{-3}$ 总体 KL 高于该量级，尚无证据说明被**损失算术**卡住。但某些位置桶可能已显著更低，且这个判断不涵盖完整模型的 BF16 前向误差。

### 3.3 一个能进一步改善数值测量的恒等式

在相同完整支持、两份分布严格归一化时，$\mathbb E_qe^{-r}=1$，所以：

$$
K=\mathbb E_q\left[e^{-r}-1+r\right].
$$

括号内非负，且在零附近：

$$
e^{-r}-1+r=\frac{r^2}{2}-\frac{r^3}{6}+\frac{r^4}{24}-\cdots.
$$

实验在 $|r|<0.01$ 使用到六阶的多项式，其余用 `expm1(-r)+r`，显著减少一阶项相消带来的误差。它仍无法恢复 BF16 前向已损失的信息。未经处理的极端 log-ratio 可能使指数溢出，支持集不一致时也需单独保留归一化修正。

该公式当前仅用于数值审计，**没有替换训练 kernel 的 loss 或 VJP**。生产替换还需验证长尾、不同后端及 loss/梯度的一致性。

## 4. 小实验的结果

该部分统一见[所属正文](../../infra/numerics/precision/2026-10-24-误差实验与接口验收.md#model-audit-4-小实验的结果)。

## 5. 代码走读与结构改进

该部分统一见[所属正文](../../infra/numerics/precision/2026-10-24-误差实验与接口验收.md#model-audit-5-代码走读与结构改进)。

## 6. 验证与交付

该部分统一见[所属正文](../../infra/numerics/precision/2026-10-24-误差实验与接口验收.md#model-audit-6-验证与交付)。

[^dtype]: PyTorch 2.11。Type Info，浮点 epsilon 与数值范围。[官方文档](https://docs.pytorch.org/docs/2.11/type_info.html)。

[^numerical]: PyTorch 2.11。Numerical accuracy，归约次序与低精度矩阵乘法的影响。[官方文档](https://docs.pytorch.org/docs/2.11/notes/numerical_accuracy.html)。
