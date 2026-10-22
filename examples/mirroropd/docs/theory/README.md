# 固定前缀下的 OPD 与镜像下降理论

本目录集中维护固定前缀上的蒸馏目标、离策略梯度与镜像下降推导。前缀集合及权重固定；完整词表求和仅消除条件动作积分误差。本目录不处理前缀分布 IS 或整序列 KL 的完整梯度。

## 结论速览

以下固定教师 $T$、锚点 $q$ 和行为策略 $\beta$，当前学生为 $p$。采用普通对数 OPD 信号（核心稿的 $\kappa=0$），取完整正支持、未中心化信号、无 clipping；定义

$$
r=\log(T/q),\quad U=\log(p/q),\quad
A^{\rm cur}=r-U,\quad A^{\rm anchor}=r,
$$

$$
D_T=\mathrm{KL}(p\Vert T),\quad D_q=\mathrm{KL}(p\Vert q),\quad c=1/\eta>0.
$$

IS 修正积分测度，current/frozen 决定学习信号。两者分别变化时，对应的目标如下。正则列始终对 $cD_q$ 精确求导，NoIS 只作用于 OPD 项。

| 信号 | 积分测度 | 普通 OPD 目标 | 加精确 $cD_q$ 后的目标 |
| --- | --- | --- | --- |
| Current + IS | $p$ | $D_T$ | $D_T+cD_q$ |
| Frozen + IS | $p$ | $D_T-D_q$ | $D_T+(c-1)D_q$ |
| Current + NoIS | $\beta$ | $\frac12\mathbb E_\beta[(U-r)^2]$ | $\frac12\mathbb E_\beta[(U-r)^2]+cD_q$ |
| Frozen + NoIS | $\beta$ | $-\mathbb E_\beta[r\log p]$ | $-\mathbb E_\beta[r\log p]+cD_q$ |

这些是真实标量目标，不是 detached surrogate 的前向值。完整支持且 $q\ne T$ 时，最后一行的目标无下界，有限系数的 $D_q$ 正则无法阻止负权重动作的概率趋于零。逐格证明见[OPD 四格](2026-10-22-固定前缀下的OPD目标与梯度.md#four-cases)及[精确正则四格](2026-10-22-镜像下降与离策略四种更新.md#exact-regularizer)。

### Full-vocab 的含义

| 计算方式 | 固定前缀下的含义 |
| --- | --- |
| 对 $D_T$ 直接 backward | 等于 current + 精确 IS 的期望梯度；当前动作已完整求和，无需额外 action IS 或 advantage 模块 |
| 完整求和 $p$ 加权的 score | 与 sampled-token 的 IS 分支对应，省去 ratio 的显式计算 |
| 完整求和 $\beta$ 加权的 score | 与 NoIS 分支对应，需要全词表行为分布；目标见上表 |
| 直接把 KL 外部的 $p$ 换成 $\beta$ | 得到 $\mathbb E_\beta\log(p/T)$，不等于上表的 NoIS score 目标 |

完整词表消除动作抽样方差，仍保留不同目标之间的差异；精确 KL baseline 用于 sampled-token 时仍有动作噪声。计算图与 detach 位置见[Full-vocab 篇](2026-10-22-Full-vocab计算图与四种更新.md#graphs)。

### Mirror Descent 的目标教师

自由分布空间中，在当前分布 $p$ 下精确积分（对应 IS）时，frozen 信号加 $1/\eta$ KL 得到标准 MD，current 信号加相同系数得到 KL 近端点：

$$
m_{\rm MD}\propto q^{1-\eta}T^\eta,
\qquad
m_{\rm prox}\propto q^{1/(1+\eta)}T^{\eta/(1+\eta)}.
$$

相同 $\eta$ 下两者通常不同。要用 current 信号表达同一个 MD 子问题，KL 系数应从 $1/\eta$ 改为 $1/\eta-1$。$\eta=1$ 时，frozen + KL 等价于原始 $D_T$，current + 同样 KL 则仍是近端点目标。证明见[MD 与近端点换算](2026-10-22-镜像下降与离策略四种更新.md#proximal-point)。

若把正则也并入解析 score 残差，再一起取消 IS，目标改为

$$
H_\delta=\frac1{2\gamma}\mathbb E_\beta[(\gamma U-r)^2],
\qquad \gamma=1/\eta+\delta,
\quad \delta=0\ \text{为 frozen},\ \delta=1\ \text{为 current}.
$$

这与上表 NoIS 两格不同。补入归一化常数或中心化残差，可以得到与相应目标教师共享自由分布最优解的 Part/BPO；未修正回归一般会偏离。见[两种 NoIS](2026-10-22-镜像下降与离策略四种更新.md#whole-residual)和[归一化修正](2026-10-22-镜像下降与离策略四种更新.md#normalization)。动作无关 baseline 在 $p$ 测度下抵消，在 NoIS 的 $\beta$ 测度下一般改变更新。

## 阅读顺序

| 文档 | 内容 |
| --- | --- |
| [统一符号](../符号约定.md) | 教师 $T$、锚点 $q$、当前学生 $p$、行为策略 $\beta$，以及冻结中心与当前中心的区别 |
| [固定前缀下的 OPD 目标与梯度](2026-10-22-固定前缀下的OPD目标与梯度.md) | Current/frozen × IS/NoIS 四种目标；正确计算图；clipping、支持和有限缓存边界 |
| [Full-vocab 计算图与四种更新](2026-10-22-Full-vocab计算图与四种更新.md) | KL 数值、一阶梯度代理和直接 backward；四格的完整求和；直接替换概率的不同含义 |
| [镜像下降与离策略四种更新](2026-10-22-镜像下降与离策略四种更新.md) | $1/\eta$ KL 正则；MD 与 KL 近端点；两种 NoIS 定义；归一化、中心化及 Part/BPO/Mean |
| [条件梯度估计与控制变量](2026-10-22-条件梯度估计与控制变量.md) | KL baseline、最优标量 baseline、top-k、TA、HETS、ResOPD、SC-TA 与离策略残差 |

前三篇构成普通 OPD 到 Mirror Descent 的理论主线；第四篇是估计器扩展。各篇文末保留实际引用的论文及版本，实验结果和代码审计则引用原资料。

## 主要论文与资料位置

| 来源 | 在本组推导中的作用 | 本地证据入口 |
| --- | --- | --- |
| AsyncOPD | 固定前缀下的 current advantage、IS、clipping 及有限缓存 | [原论文与实现审计](../opd-objective/2026-10-21-固定前缀下的优势估计与重要性采样.md#6-论文与实现证据) |
| Semi-OPD、Lightning OPD | 固定学生轨迹、信号刷新与离线 surrogate 的实例 | [陈旧数据研究第 5–6 节](../research/2026-10-21-陈旧数据训练的指标与实验对照.md#5-lightning-opd年龄大也可能漂移有限) |
| GKD、MiniLLM | 条件分布拟合与单步词表积分；局部梯度的适用边界 | [Full-vocab 篇](2026-10-22-Full-vocab计算图与四种更新.md#4-固定前缀与实现范围) |
| PPO | 冻结 advantage 的局部策略改进背景 | [MD 篇](2026-10-22-镜像下降与离策略四种更新.md#9-论文关系与验算) |
| vOPD、TA-OPD、ResOPD、TT-OPD | baseline、支持集与尾部控制变量 | [条件估计篇](2026-10-22-条件梯度估计与控制变量.md) |
| MirrorOPD 核心稿 | 冻结信号、一般 $f_{\kappa,\eta}$、MDPO/Part/BPO/Mean | [镜像问题](../../../../../MirrorOPD/theory/core/sections/01_mirror.tex)、[内层拟合](../../../../../MirrorOPD/theory/core/sections/04_inner.tex) |

MD 篇的四格分类与两种 NoIS 定义由本目录的标量目标独立推导，不将其记作上述论文已经完成的端到端效果验证。核心稿链接指向相邻仓库；本目录已写出本组推导所需定义，单独阅读时不依赖该仓库存在。

## 验算

| 范围 | 脚本与证据 |
| --- | --- |
| 基础四格、两种正则 NoIS、几何目标与归一化修正 | [verify_offpolicy_objectives.py](scripts/verify_offpolicy_objectives.py)、[FP64 结果](data/offpolicy-objectives.json) |
| 原 sampled-token 与 clipping 对照 | [verify_fixed_prefix.py](../opd-objective/scripts/verify_fixed_prefix.py)、[历史结果](../opd-objective/data/gradient-audit.json) |
| Full-vocab 计算图与支持集 | [verify_fullkl_and_support.py](../opd-objective/scripts/verify_fullkl_and_support.py)、[历史结果](../opd-objective/data/support-selection-20261021/audit.json) |
| Baseline 与稀疏估计器 | [verify_reward_estimators.py](../opd-objective/scripts/verify_reward_estimators.py)、[历史结果](../opd-objective/data/reward-research-20261021/estimator-audit.json) |

新增脚本只使用 CPU 张量，完整枚举小词表，不加载语言模型。证明给出解析条件，脚本检查公式与反向计算图的一致性。

## 原文职责与迁移

2026-10-22 将原固定前缀文档的第 1–5 节、支持集文档的第 2 节、估计器文档的第 2–5 节中的数学内容集中到本目录。原路径及章节锚点通过旧文末尾的索引保留，以便已有链接跳转；重复证明已替换为新章节链接。

- [OPD 实验与审计](../opd-objective/README.md)继续维护论文实测、作者源码、实验预设和历史验算。
- [Full-vocab 工程](../full-vocab/README.md)继续维护算子、hidden-state 数据通路、预算和训练验收。
- [陈旧数据研究](../research/README.md)继续维护论文设置、原图、指标和实验解释。

理论更新直接修改本目录；旧文只更新实现、实验信息与引用，避免重新复制完整推导。
