# RL 前后模型的 KL：文献与本地配对测量

更新：2026-11-08。对现有同尺寸 RL 师生配对，实际 token 平均的 `KL(RL前 ∥ RL后)` 为 **0.0209–0.3181 nats/token**；MiMo Base→RL-Zero 的 initial bank 题均值为 **0.0700**。这是从已经完成的原始全词表 forward 输出重新聚合的测量，没有新增模型推理或训练。

## 问题与测量定义

这里把原始 student 记为 $\pi_0$，公开 RL teacher 记为 $\pi_R$。Student 后续 OPD 更新不进入 RL 前后比较，只读取每个 run 的 `step=0`。对固定前缀 $s$，计算两个方向：

$$
k_{0\to R}(s)=\sum_v\pi_0(v\mid s)\log\frac{\pi_0(v\mid s)}{\pi_R(v\mid s)},\qquad
k_{R\to0}(s)=\sum_v\pi_R(v\mid s)\log\frac{\pi_R(v\mid s)}{\pi_0(v\mid s)}.
$$

所有对数为自然对数。对 $N$ 道题、每题 $L_i$ 个有效 response token，主表采用 token 加权平均：

$$
\bar k_{0\to R}=\frac{\sum_{i=1}^{N}\sum_{t=1}^{L_i}k_{0\to R}(s_{it})}{\sum_{i=1}^{N}L_i}.
$$

项目原面板另用 $NH$ 作分母，$H=1024$。两种归约的关系是：

$$
K_H=\bar k_{0\to R}\times\frac{\sum_i L_i}{NH}.
$$

短回答使 $K_H$ 低于实际 token 均值。逐题 P10–P90 则先计算每题的 token 平均，再计算题之间的分位数；它们不表示逐 token 的稀疏性。

### 实际条件与配对身份

- 主比较每组读取相同的 Math64、Code32、IF32，合计 128 道永久留出题。前缀由各自原始 student 产生，temperature=1、H1024、EOS 后停止；同一 student 内各 teacher 使用同一冻结 bank。输入模式、精度和评分协议见[固定状态协议](../protocols/fixed-state.md)。各 student 的前缀不同，跨 family 比较的是各自状态上的读数。
- R1 的 DeepScaleR、DeepCoder、JustRL、Nemotron v1 均从 R1-Distill-1.5B 继续 RL。R1 的起点已包含蒸馏 SFT；这里度量的是之后的 RL 增量。BroRL 从 ProRLv2 近终点继续 RL，因此表中 R1→BroRL 是累计 RL 距离，不是 BroRL 最后阶段的直接前后差。
- e3、GRPO-KL 的已披露父型号是官方后训练 Qwen3-1.7B；父模型不称为纯 SFT，也不替换成 Qwen3-1.7B-Base。Llama 的原始 student 是 Unsloth BF16 镜像，teacher 声明的父型号为 Llama-3.2-3B-Instruct。精确父 revision 未披露、Unsloth/Meta 字节相等未确认的身份缺口沿用[师生谱系](../../../settings/模型与师生配对.md#training-rl)。
- 12 个主配对均核对本地固定 revision 的 head vocabulary size 和 token→ID 映射。R1 原始输出保存了 `fkl_shared`；相同完整支持下，它等于反方向的全词表 KL。Qwen3、Llama 原始输出没有该列，保留缺失值。

## 文献已经研究到哪里

### 直接报告 RL/SFT 前后 token KL 的研究

Huan 等人的 _Does Math Reasoning Improve General LLM Capabilities? Understanding Transferability of LLM Reasoning_（2025，arXiv:2507.00432v2，§4/Fig.4）直接比较后训练模型与 backbone 的 token 分布，报告以下数值。[^huan]

| 同一 Qwen3-14B backbone 的分支 | MATH-500 KL | IFEval KL |
| ------------------------------ | ----------: | --------: |
| UniReason RL                   |       0.084 |     0.019 |
| UniReason SFT-no-think         |       0.372 |     0.283 |

这是与当前问题最直接的一篇：RL 在这两个任务上的 KL 比该 SFT 对照小约 4.4 倍和 14.9 倍。该文报告 token 分布 KL，但没有在该段完整明确 KL 方向和归约实现，所以保留作者原数值，不把它当作当前协议的复现。它的结论与我们 Qwen3、Llama 的 RL/SFT 比较一致；R1 对照存在重叠，见下文。

### 分布变化集中在少数 token 决策

Qwen Pilot Team 的 _Sparse but Critical: A Token-Level Analysis of Distributional Shifts in RLVR Fine-Tuning of LLMs_（ICLR 2026，arXiv:2603.22446，§2–§3）在 RL 生成的相同前缀上比较 base/RL 条件分布，主要量是 JS，而非 KL。Qwen2.5-32B 的 DAPO 超过 83% 位置、SimpleRL 超过 98% 位置呈近零 JS；主要图采用 top-p 截断，附录另检查 full-distribution 估计的稳健性。[^sparse]

其 cross-sampling 实验把少量 RL token 决策插入 base 生成，或反过来替换 RL 生成中的关键决策。例如 SimpleRL/Qwen2.5-32B 在 AIME2024 上少于 4% 的有效 RL token 替换可以恢复 RL 水平表现。它支持“平均分布变化小，同时少数决策变化对任务表现很重要”，没有把小平均 KL 等同于小能力变化。

### RLHF 已经把相对初始模型的 KL 当作训练距离

Gao、Schulman、Hilton 的 _Scaling Laws for Reward Model Overoptimization_（2022 预印本，ICML 2023）系统研究相对初始 SFT policy 的 KL 与真实/proxy reward 的关系，图中以 $\sqrt{\mathrm{KL}}$ 为优化距离，观察到真实 reward 先升后降。[^gao]

这篇支持用“相对 RL 起点的累计 KL”观察训练进程，其 response-level 距离需结合回答长度阅读。KL 正则系数、相邻更新的 KL、相对最初模型的 KL 也分别是不同量。

## 本地实测

### 9 个 RL 配对：整体留出

下表 KL 均以 nats 为单位；前三个数值列分别是每个有效 token 上的两个方向和原面板的固定 H 读数。“同一 pre 前缀”表示第二个方向仍在 $\pi_0$ 生成的状态上计算，没有换成 RL rollout。

| RL 前 student         | RL 后 teacher    | KL(pre ∥ post) | KL(post ∥ pre)，同一 pre 前缀 | 原 H1024 均值 |  逐题 P10–P90 |
| --------------------- | ---------------- | -------------: | ----------------------------: | ------------: | ------------: |
| R1-Distill-1.5B       | DeepScaleR       |         0.0357 |                        0.0377 |        0.0330 | 0.0162–0.0514 |
| R1-Distill-1.5B       | DeepCoder        |         0.0959 |                        0.0700 |        0.0886 | 0.0301–0.1670 |
| R1-Distill-1.5B       | JustRL           |         0.1277 |                        0.1268 |        0.1178 | 0.0902–0.1718 |
| R1-Distill-1.5B       | Nemotron v1      |         0.2169 |                        0.2396 |        0.2002 | 0.1180–0.3597 |
| R1-Distill-1.5B       | BroRL（累计 RL） |         0.3181 |                        0.4848 |        0.2936 | 0.1879–0.4852 |
| Qwen3-1.7B            | GRPO-KL          |         0.0209 |                             — |        0.0196 | 0.0134–0.0293 |
| Qwen3-1.7B            | e3               |         0.0360 |                             — |        0.0338 | 0.0201–0.0493 |
| Llama-3.2-3B-Instruct | GT-MATH          |         0.0444 |                             — |        0.0229 | 0.0097–0.1094 |
| Llama-3.2-3B-Instruct | Co-rewarding-I   |         0.0423 |                             — |        0.0218 | 0.0088–0.1140 |

R1、Qwen3、Llama 的实际 token 覆盖率分别为 **92.31%、93.79%、51.62%**。因此 Llama GT 的原面板值 0.02293 对应实际 token 均值 0.04443，接近两倍。按原 H1024 口径把 Llama 排成“变化最小”，会混入回答长度差异。

### 各 domain 的变化

以下全部使用实际 token 均值；整体留出按 token 合并，不是 Math/Code/IF 三列的算术平均。

| student               | RL teacher       |   Math |   Code |     IF |
| --------------------- | ---------------- | -----: | -----: | -----: |
| R1-Distill-1.5B       | DeepScaleR       | 0.0401 | 0.0384 | 0.0192 |
| R1-Distill-1.5B       | DeepCoder        | 0.0884 | 0.1538 | 0.0341 |
| R1-Distill-1.5B       | JustRL           | 0.1193 | 0.1434 | 0.1291 |
| R1-Distill-1.5B       | Nemotron v1      | 0.1636 | 0.2426 | 0.3339 |
| R1-Distill-1.5B       | BroRL（累计 RL） | 0.2533 | 0.3626 | 0.4410 |
| Qwen3-1.7B            | GRPO-KL          | 0.0208 | 0.0247 | 0.0161 |
| Qwen3-1.7B            | e3               | 0.0346 | 0.0429 | 0.0308 |
| Llama-3.2-3B-Instruct | GT-MATH          | 0.0590 | 0.0144 | 0.0149 |
| Llama-3.2-3B-Instruct | Co-rewarding-I   | 0.0569 | 0.0124 | 0.0125 |

DeepCoder 在 Code 上变化最大（0.1538），符合它的代码 RL 条件。JustRL 虽然用数学 reward，Code/IF 变化为 0.1434/0.1291，与 Math 的 0.1193 接近；任务数据只在数学域，不代表参数更新只改变数学域。Nemotron v1 的 IF 变化 0.3339 大于 Math 的 0.1636，多领域 RL 终点的分布移动也不是各域均匀的。

### SFT 对照与“距离解释”

以下 SFT 对照使用对应 student 的同一个初始 bank。

| student               | SFT teacher         | KL(pre ∥ post) |
| --------------------- | ------------------- | -------------: |
| R1-Distill-1.5B       | MiniMath SFT        |         0.1008 |
| Qwen3-1.7B            | Samsung SFT         |         0.2653 |
| Llama-3.2-3B-Instruct | arithmetic-full SFT |         0.4250 |

Qwen3 的 Samsung SFT 比 e3/GRPO-KL 大约 7.4/12.7 倍；Llama 的 arithmetic-full SFT 比 GT/Co-rewarding-I 大约 9.6/10.0 倍。但 **R1 的 MiniMath SFT（0.1008）比 JustRL（0.1277）、Nemotron v1（0.2169）、BroRL（0.3181）都近**。因此“RL 更容易吸收，因为它一开始更近”不足以解释我们的全部观察；需进一步比较同一初始距离下的迁移效率、收敛残差与变化位置。

### MiMo Base→RL-Zero：两个状态分布

E72 的 `optimizer_updates=0` 已有全词表 `KL(MiMo-Base ∥ MiMo-RL-Zero)`。这里沿用 MiMo 的“题内 token 均值→题均值”归约，每个 domain 128 题，共 512 题，与上面 token 加权均值分开报告。

| bank 的生成模型       |    Math |    Code | Science |      IF |     Avg |
| --------------------- | ------: | ------: | ------: | ------: | ------: |
| initial：MiMo Base    | 0.07037 | 0.11044 | 0.07671 | 0.02267 | 0.07005 |
| teacher：MiMo RL-Zero | 0.09216 | 0.08221 | 0.04195 | 0.04266 | 0.06475 |

两个 bank 的 **KL 方向完全相同**，只改变前缀分布。整体值接近，但 IF 从 0.02267 到 0.04266，约 1.88 倍；Science 则从 0.07671 降到 0.04195。不能只看整体差异就判断状态选择没有影响。MiMo Base→MiMo-RL 同时跨过 SFT 和 RL，不作为纯 RL 增量；SFT→RL 需要对应的初始评分。原始身份和双 bank 协议见[MiMo 条件](../cases/mimo-position-kl.md)及[吸收分析](absorption-diagnostics.md#mimoinitial-与-teacher-双-held-out-bank)。

## 对当前吸收问题的含义

1. **RL 增量通常是百分之几到十分之几 nats/token，但不是恒定小量。** 该主比较最小为 GRPO-KL 的 0.0209，最大为累计 BroRL 的 0.3181，跨度 15.2 倍；同一 R1 parent 下跨度 8.9 倍。这只概括当前公开终点和 bank，不代表所有 RL 配方的固定区间。
2. **RL/SFT 的可吸收性不能用一个初始平均 KL 排序解释。** R1 提供了距离重叠的对照；Qwen3/Llama 则提供 SFT 明显更远的对照。后续比较应同时报告初始距离与剩余比例，避免把 teacher 起点差异算成迁移机制差异。
3. **反方向能揭示均值没有显示的非对称性。** BroRL 的 `KL(post ∥ pre)` 为 0.4848，是正方向 0.3181 的 1.52 倍；DeepCoder 则为 0.0700，低于正方向 0.0959。小扰动下 KL 两方向的二阶项相同，实际偏离表明尾部与有限幅度的概率重排已经影响读数；具体由哪些 token 引起，需要逐 token 分布才能定位。
4. **题间异质性和 token 级稀疏性需要分别测。** Llama GT 的逐题中位数为 0.0301、P90 为 0.1094，DeepCoder 的逐题中位数为 0.0660、P90 为 0.1670。当前分位数可以描述题间差异，不能验证文献里的“98% token 近零”。当前原始快照只保留每题及位置段汇总，完整逐 token 分位数不能从这些和重建。

### 条件 KL 与整条生成序列 KL

令 $P_0,P_R$ 是相同 prompt 和停止规则下，两个模型完整生成序列的分布，链式法则给出：

$$
D_{\mathrm{KL}}(P_0\Vert P_R)=\mathbb E_{Y\sim P_0}\left[\sum_{t\le L(Y)} k_{0\to R}(X,Y_{<t})\right].
$$

反方向则需要在 $Y\sim P_R$ 下累加 $k_{R\to0}$。本页第二方向保持 pre 前缀，不能直接解释为 $D_{\mathrm{KL}}(P_R\Vert P_0)$。只有采样 policy、前缀分布与停止/截断定义一致时，条件 KL 的序列累和才对应这个完整分布 KL；把一个 token 均值机械乘 1024 不能消除这些条件。也因此，单个 token KL 为 0.03 不表示整条长回答只改变了 3% 的概率或 3% 的 token。

## 数据与复算

- [主表、各 domain、early/middle/late 及训练池汇总 CSV](figures/rl-parent-kl-20261108/parent-kl.csv)：12 个主配对、240 行 panel/band 结果。位置段沿用原始绝对分段 early=0–255、middle=256–511、late=512–1023；短回答在后段可能没有有效位置。训练池列表示各原 run 已评分的题，DeepCoder 与其他 run 的训练池覆盖不同，主比较只使用共同永久留出。
- [1536 行留出逐题 CSV](figures/rl-parent-kl-20261108/questions.csv)：实际 token 数、KL 和、可用的反方向 KL 和及题均值。
- [MiMo 双 bank 初始值](figures/rl-parent-kl-20261108/mimo-parent-kl.csv)：10 行，含每个 domain 和 Avg。
- [来源、模型 revision、token ID 支持和 SHA256](figures/rl-parent-kl-20261108/provenance.json)；原始 forward 文件保留在各来源目录，不覆盖冻结数据。
- [三篇直接相关论文的完整检索报告](figures/rl-parent-kl-20261108/literature-search.txt)保留原始检索表和摘要；上文数值另经论文正文核对。
- [复算脚本](../../../../research/studies/fixed_transfer/measure_rl_parent_kl.py)。核验 S0、题组完整性、同 student 的 held-out 身份元数据、完整 token 支持、位置段重建及原端点一致性。

在仓库根目录执行：

```bash
python -m examples.mirroropd.research.studies.fixed_transfer.measure_rl_parent_kl
```

### 参考文献

[^huan]: M. Huan et al. _Does Math Reasoning Improve General LLM Capabilities? Understanding Transferability of LLM Reasoning_. 2025. [arXiv:2507.00432v2](https://arxiv.org/abs/2507.00432v2)，§4、Fig.4。

[^sparse]: Qwen Pilot Team. _Sparse but Critical: A Token-Level Analysis of Distributional Shifts in RLVR Fine-Tuning of LLMs_. ICLR 2026. [arXiv:2603.22446](https://arxiv.org/abs/2603.22446)，§2–§3、Appendix A.5。

[^gao]: L. Gao, J. Schulman, J. Hilton. _Scaling Laws for Reward Model Overoptimization_. ICML 2023. PMLR 202（本地文件 `https://proceedings.mlr.press/v202/gao23h.html`），§2–§3。
