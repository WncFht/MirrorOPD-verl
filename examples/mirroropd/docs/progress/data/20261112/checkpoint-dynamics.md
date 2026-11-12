# OPD checkpoint 的谱与 frame 动力学

本报告独立维护 E94–E104 的 checkpoint 分析，结果、曲线、图与复算输入均保存在本目录。研究重点是：输出吸收较好与较差的 pair，在更新谱形状、更新子空间和参数残差上有哪些差异。后续 pair 使用相同协议接入；已完成的测量按实际覆盖更新。

## 实验设置

| 项目          | 实际设置                                                                                                                    |
| ------------- | --------------------------------------------------------------------------------------------------------------------------- |
| 训练          | current student rollout；DAPO 每 outer 64 prompts，reuse 16；40 outer、640 次更新；学习率 $10^{-6}$；seed 42；不测 accuracy |
| 长度          | prompt 上限 4096；response 上限 7168；固定 bank 使用完整有效响应                                                            |
| Held-out bank | initial 与 teacher 各生成一套固定 bank；每套 Math 128、Code 64、IF 64；每 outer 评分共 512 行                               |
| RKL 聚合      | 每题有效 token 平均，再域内平均；all 为 Math、Code、IF 三域等权平均                                                         |
| Checkpoint    | 更新步 16、32、48、64、128、192、320、480、640；FP32 导出；原训练快照经过 native reload RKL 检查                            |
| 几何覆盖      | 每模型 28 层，每层 q/k/v/o 与 gate/up/down 七类矩阵，共 196；十一组共 2156 个矩阵任务                                       |
| 分解          | 读取 FP32 权重，FP64 精确 SVD；完整奇异值；frame 使用右子空间，k=16/32/64/128/256                                           |
| Teacher 比较  | E94–E101、E103–E104 使用对应坐标的同尺寸矩阵；E102 为跨尺寸，只比较 student 自身训练动态                                    |
| 统计单位      | 每个矩阵先计算指标，再汇总；阴影为矩阵间四分位区间，不是多 seed 置信区间                                                    |

初始 student 记作 $W_0$，teacher 记作 $W_T$，checkpoint 记作 $W_t$。累计更新为 $\Delta W_t=W_t-W_0$。它的奇异值 $\sigma(\Delta W_t)$ 与完整权重奇异值之差 $\sigma(W_t)-\sigma(W_0)$ 是两个不同对象。

## Sigma variation 的定义

采用 SoSD 论文[^sosd]的归一化方式：对降序奇异值向量除以其元素和。

$$
p(W)=\frac{\sigma(W)}{\sum_i\sigma_i(W)},\qquad
D_\Sigma(a,b)=\Vert p(W_a)-p(W_b)\Vert_2.
$$

整体缩放矩阵不改变 $D_\Sigma$，因此同时保留核范数与 Frobenius 范数。完整权重分析包括初始 step 0；累计更新从 step 16 开始，因为零更新没有归一化谱。

| 测量                          | 解释                                                                             |
| ----------------------------- | -------------------------------------------------------------------------------- |
| $D_\Sigma(0,T)$               | 初始 student 与 teacher 的谱形状差距                                             |
| $D_\Sigma(t,T)$               | 训练过程中与 teacher 的谱距离；不等于能力差距                                    |
| $D_\Sigma(t_a,t_b)$           | 相邻或任意两个 checkpoint 的净谱变化                                             |
| $D_\Sigma(t_a,t_b)/(t_b-t_a)$ | 区间平均净变化速度；区间内部往返运动无法恢复                                     |
| $D_\Sigma(t,640)$             | 谱形状何时接近本次训练终点                                                       |
| $D_\Sigma^\Delta(t_a,t_b)$    | 累计更新自身的谱形状变化                                                         |
| Hellinger 距离                | $\Vert\sqrt{p_a}-\sqrt{p_b}\Vert_2/\sqrt{2}$；用于矩阵奇异值个数不同时的补充汇总 |

初始 teacher 距离大于 $10^{-12}$ 时保留剩余比例 $D_\Sigma(t,T)/D_\Sigma(0,T)$；基线更小时存为 null，并展示绝对距离。原始 SD 距离随谱维度及分布形状改变，Attention 与 MLP 的中位数曲线是描述统计；模块间比较同时查看 Hellinger、逐层结果与矩阵幅度。

完整权重 ISO 指标继续保留：$\delta_\Sigma=\Vert\sigma(W_t)-\sigma(W_0)\Vert_2/\Vert W_0\Vert_F$，$\rho_\Sigma$ 将分母改为 $\Vert W_t-W_0\Vert_F$；$\kappa_{\rm spec}$ 度量更新在初始奇异向量配对方向上的集中度。它们与归一化的 SD variation 分别回答幅度、更新构成和谱形状问题。

## 结果与图

<!-- results:start -->

数据更新于 2026-11-12 18:30:16（北京时间）。矩阵完成后包含全部九个 checkpoint；完整层要求七类投影齐全。

| Pair | Student → teacher                                                 | 已测矩阵 | 完整层 | 全模型几何覆盖 |
| ---- | ----------------------------------------------------------------- | -------: | -----: | -------------- |
| E94  | DeepSeek-R1-Distill-Qwen-1.5B → deepscaler                        |  196/196 |     28 | 完成           |
| E95  | DeepSeek-R1-Distill-Qwen-1.5B → JustRL-DeepSeek-1.5B              |  196/196 |     28 | 完成           |
| E96  | DeepSeek-R1-Distill-Qwen-1.5B → minimath                          |  196/196 |     28 | 完成           |
| E97  | DeepSeek-R1-Distill-Qwen-1.5B → nemotron                          |  196/196 |     28 | 完成           |
| E98  | DeepSeek-R1-Distill-Qwen-1.5B → brorl                             |  196/196 |     28 | 完成           |
| E99  | Qwen3-1.7B → Qwen3-1.7B-GRPO-KL-math-reasoning                    |   53/196 |      7 | 部分           |
| E100 | Qwen3-1.7B → e3-1.7B                                              |    0/196 |      0 | 部分           |
| E101 | Qwen3-1.7B → Qwen3-1.7B-Math                                      |    0/196 |      0 | 部分           |
| E102 | Qwen3-1.7B → Qwen3-4B-Instruct-2507                               |    0/196 |      0 | 部分           |
| E103 | Llama-3.2-3B-Instruct → Co-rewarding-I-Llama-3.2-3B-Instruct-MATH |   90/196 |     12 | 部分           |
| E104 | Llama-3.2-3B-Instruct → Llama-3.2-3B-Instruct-arithmetic-full     |    0/196 |      0 | 部分           |

所有 pair 的 held-out RKL 均来自完成的 640 更新训练。initial 和 teacher 两种 bank 分开报告；括号为终点相对初始的剩余百分比。

| Pair | initial bank：平均 RKL | teacher bank：平均 RKL | initial bank：Math RKL | teacher bank：Math RKL |
| ---- | ---------------------: | ---------------------: | ---------------------: | ---------------------: |
| E94  |   0.000484536（1.54%） |    0.00046542（1.59%） |   0.000257067（0.60%） |   0.000221113（0.57%） |
| E95  |    0.0361471（25.73%） |    0.0833731（44.30%） |    0.00211075（1.92%） |     0.0018512（2.01%） |
| E96  |    0.0269825（25.22%） |    0.0302335（21.05%） |    0.00637784（9.66%） |    0.00820156（6.25%） |
| E97  |    0.0622798（22.07%） |    0.0523476（23.78%） |    0.00356046（1.45%） |    0.00178394（0.97%） |
| E98  |    0.0910948（26.57%） |     0.121539（38.05%） |    0.00478825（1.85%） |    0.00468845（2.02%） |
| E99  |   0.000629341（2.62%） |   0.000674672（2.37%） |   0.000331525（1.07%） |   0.000342272（0.81%） |
| E100 |   0.000958098（2.30%） |   0.000863318（2.37%） |   0.000337136（0.72%） |   0.000343142（0.83%） |
| E101 |     0.0179311（8.58%） |     0.0421213（7.31%） |     0.0170987（9.97%） |    0.00910418（1.39%） |
| E102 |     0.846753（90.71%） |    0.990329（104.56%） |     0.252773（60.15%） |     0.274474（65.28%） |
| E103 |   0.00766375（12.95%） |    0.00423648（9.07%） |    0.00972591（8.62%） |   0.000378029（0.52%） |
| E104 |     0.245294（45.46%） |     0.250099（19.63%） |     0.135539（30.11%） |    0.0877396（12.50%） |

以下几何统计使用每个 pair 当前完整层的相同矩阵集合，均为逐矩阵中位数。不同 pair 尚未测齐时不据此排序。

| Pair | 纳入矩阵 | 完整权重：0→640 SD 距离 | 累计更新：16→640 SD 距离 | k64:16→640 调整后 overlap | OPD/teacher 更新幅度 | 参数残差能量比 |
| ---- | -------: | ----------------------: | -----------------------: | ------------------------: | -------------------: | -------------: |
| E94  |      196 |               7.521e-07 |                 0.005734 |                    0.5680 |               0.9374 |          1.167 |
| E95  |      196 |               1.429e-06 |                 0.007744 |                    0.4552 |               0.4576 |         0.9741 |
| E96  |      196 |               1.754e-06 |                 0.008998 |                    0.3800 |               0.0996 |         0.9963 |
| E97  |      196 |               1.462e-06 |                  0.00897 |                    0.4224 |               0.4794 |          1.001 |
| E98  |      196 |               1.801e-06 |                 0.009847 |                    0.3951 |               0.4983 |         0.9421 |
| E99  |       49 |               5.285e-07 |                 0.009025 |                    0.5066 |                1.443 |          2.466 |
| E103 |       84 |               4.475e-07 |                  0.01647 |                    0.4446 |               0.9577 |          1.438 |

### 跨 pair 总览

![overview-geometry-rkl](figures/overview-geometry-rkl.png)

![overview-rkl-endpoints](figures/overview-rkl-endpoints.png)

![overview-rkl-trajectories](figures/overview-rkl-trajectories.png)

![overview-shared-student-E94](figures/overview-shared-student-E94.png)

完整分类图集见[图集](图集.md)。下列四组图优先展示更新维度、方向和后续学习。

### E94: DeepSeek-R1-Distill-Qwen-1.5B → deepscaler

196/196 个纳入矩阵，28 个完整层；全模型覆盖。

![E94：update-dimension](figures/E94-update-dimension.png)

![E94：frame-rank-time](figures/E94-frame-rank-time.png)

![E94：frame-increments](figures/E94-frame-increments.png)

![E94：singular-energy-rank](figures/E94-singular-energy-rank.png)

### E95: DeepSeek-R1-Distill-Qwen-1.5B → JustRL-DeepSeek-1.5B

196/196 个纳入矩阵，28 个完整层；全模型覆盖。

![E95：update-dimension](figures/E95-update-dimension.png)

![E95：frame-rank-time](figures/E95-frame-rank-time.png)

![E95：frame-increments](figures/E95-frame-increments.png)

![E95：singular-energy-rank](figures/E95-singular-energy-rank.png)

### E96: DeepSeek-R1-Distill-Qwen-1.5B → minimath

196/196 个纳入矩阵，28 个完整层；全模型覆盖。

![E96：update-dimension](figures/E96-update-dimension.png)

![E96：frame-rank-time](figures/E96-frame-rank-time.png)

![E96：frame-increments](figures/E96-frame-increments.png)

![E96：singular-energy-rank](figures/E96-singular-energy-rank.png)

### E97: DeepSeek-R1-Distill-Qwen-1.5B → nemotron

196/196 个纳入矩阵，28 个完整层；全模型覆盖。

![E97：update-dimension](figures/E97-update-dimension.png)

![E97：frame-rank-time](figures/E97-frame-rank-time.png)

![E97：frame-increments](figures/E97-frame-increments.png)

![E97：singular-energy-rank](figures/E97-singular-energy-rank.png)

### E98: DeepSeek-R1-Distill-Qwen-1.5B → brorl

196/196 个纳入矩阵，28 个完整层；全模型覆盖。

![E98：update-dimension](figures/E98-update-dimension.png)

![E98：frame-rank-time](figures/E98-frame-rank-time.png)

![E98：frame-increments](figures/E98-frame-increments.png)

![E98：singular-energy-rank](figures/E98-singular-energy-rank.png)

### E99: Qwen3-1.7B → Qwen3-1.7B-GRPO-KL-math-reasoning

49/196 个纳入矩阵，7 个完整层；部分层预览。

![E99：update-dimension](figures/E99-update-dimension.png)

![E99：frame-rank-time](figures/E99-frame-rank-time.png)

![E99：frame-increments](figures/E99-frame-increments.png)

![E99：singular-energy-rank](figures/E99-singular-energy-rank.png)

### E103: Llama-3.2-3B-Instruct → Co-rewarding-I-Llama-3.2-3B-Instruct-MATH

84/196 个纳入矩阵，12 个完整层；部分层预览。

![E103：update-dimension](figures/E103-update-dimension.png)

![E103：frame-rank-time](figures/E103-frame-rank-time.png)

![E103：frame-increments](figures/E103-frame-increments.png)

![E103：singular-energy-rank](figures/E103-singular-energy-rank.png)

<!-- results:end -->

## 已测结果的解读

E94（DeepScaleR）已经覆盖全部 28 层、196 个矩阵，三域平均 RKL 在两套 bank 上都降到约 $5\times10^{-4}$，剩余约 1.5%。但完整权重的归一化谱没有明显向教师靠近：终点相对初始 teacher 距离的逐矩阵中位数为 1.022。输出匹配得很好，并不要求完整谱形状逐项恢复到教师。

E94 从初始到终点的完整权重 SD 距离中位数为 $7.52\times10^{-7}$；累计更新从第 16 步到第 640 步的谱距离为 0.00573。两者使用不同的参照对象，不能解释为同一种运动的幅度倍数。它们共同表明：完整权重谱基本保留预训练形状，而 OPD 累计更新的内部结构仍在变化。

E94 第 16 步的 top-64 更新右子空间，与最终 top-64 子空间的调整后 overlap 中位数为 0.568，但能捕获最终更新的能量达到最终最优 top-64 捕获量的 87.0%。早期空间保留了大量重要更新方向，同时也存在后续子空间演化。“重要方向较早形成”比“所有方向早早锁定”更符合这组结果。

E94 的 OPD/teacher 更新幅度比中位数为 0.937，参数残差能量比仍为 1.167。这里输出分布吸收很好，却没有恢复 teacher 参数。这个现象已经有全层证据支持。

E95（JustRL）在 Math 上的终点 RKL 已很低，但 teacher bank 的 IF RKL 仍为 0.2437，三域平均为 0.08337。其 28 层、196 个矩阵现已测齐，可以与 E94 做同一 student、不同 teacher 的全层对照：E94 的输出吸收更全面，E95 的主要缺口在非 Math 域。两组更新空间是否不同，需同时看不同 rank 的主角分布、能量捕获和逐模块结果。仅凭两组的全模型中位数，还不能把 frame 稳定性写成吸收好坏的通用判据。

### 同一 student 的 E94 / E95 对照

以下均为 28 层、196 个同坐标矩阵的逐矩阵中位数。参照空间为第 16 步累计更新的右子空间，目标为第 640 步累计更新。

| 指标                                      | E94：DeepScaleR | E95：JustRL |
| ----------------------------------------- | --------------: | ----------: |
| k16 调整后 overlap                        |           0.661 |       0.568 |
| k64 调整后 overlap                        |           0.568 |       0.455 |
| k256 调整后 overlap                       |           0.432 |       0.318 |
| k64 主角中位数（度）                      |           35.35 |       45.01 |
| k64 相对最优能量捕获                      |          87.05% |      76.28% |
| step16 top64 空间外：16→32 增量能量比例   |          29.78% |      36.19% |
| step16 top64 空间外：480→640 增量能量比例 |          77.41% |      77.88% |

在这个同 student 对照中，跨域输出吸收更好的 E94，早期更新空间与最终方向更接近，覆盖最终重要能量的比例也更高；这个差异同时出现在 k16、k64 和 k256。放大 rank 后，两组的 overlap 都下降，表明更宽子空间纳入了更多后续改变的方向。

但两组后期新增更新都有约 78% 的能量位于 step16 的 top64 空间之外。**累计终点能量被早期空间较好覆盖，与后期增量继续探索新方向，同时发生。** 早期大更新可以主导累计终点，后期较小增量仍可持续改变方向。该对照支持“早期空间质量有差异”，不支持“吸收好的模型后续几乎不再离开早期空间”。后期区间长度不同，绝对增量大小需结合内外能量图阅读。

## 分析方法与判断

优先按两套固定 bank 的绝对 RKL 和剩余比例评价吸收，Math 与三域平均分别展示。当前实验测量输出分布匹配，不将其直接写成任务 accuracy 或能力完全恢复。

完整权重谱稳定与累计更新谱稳定分开判断。前者受预训练权重占主导的影响；后者直接描述 OPD 更新的组织方式。重点检查 RKL 快速下降与变缓的阶段，谱变化速度、frame 与终点的相似度以及后续更新的能量捕获是否同步改变。

谱形状相似不要求 frame 相同，frame 相似也不要求参数幅度相同。若完整权重谱几乎不变但 RKL 明显下降，意味着这段输出适配不需要大幅重塑完整谱；是否由子空间变化解释，要结合 frame 和能量捕获，不能由小谱距离单独推出。

参数残差按同坐标矩阵计算。令 $a=\Vert\Delta W_t\Vert_F/\Vert W_T-W_0\Vert_F$，$c$ 为两种更新的余弦，则

$$
\frac{\Vert W_T-W_t\Vert_F^2}{\Vert W_T-W_0\Vert_F^2}=1+a^2-2ac.
$$

因此更新幅度较大时，残差仍可接近或大于 1；幅度、方向和输出 RKL 必须并排看。

SoSD 论文的主体实验是预训练，其理论针对带假设的简化 Transformer。这里检验 OPD 的谱与输出动态关联；观察到谱稳定不能直接认定是吸收失败的原因。Checkpoint 间隔不等，曲线的每步速度是区间净变化速度，不是逐更新测得的波动。

## 图表索引与指标定义

[完整图集](图集.md)展示已经生成的图片。本节列出 **28 类单 pair 图和 4 类跨 pair 总览**，逐项说明实际计算的指标。单 pair 文件名为 `E编号-下表中的后缀.png`，另有同名 PDF。每类图通常包含多个面板；图的类别数不等于图片文件数。各 pair 按测量覆盖发布，不能把目录中的图类清单当成全部模型已经测齐。

### 比较对象与统计单位

| 记号               | 实际对象                                                           |
| ------------------ | ------------------------------------------------------------------ |
| $W_0,W_t,W_T$      | 初始 student、更新步 t 的 OPD checkpoint、teacher 的同坐标权重矩阵 |
| $\Delta_t=W_t-W_0$ | 从初始 student 出发的累计更新；参照时刻从 step16 开始              |
| $I_{a,b}=W_b-W_a$  | 相邻保存时刻之间的净增量，等于 $\Delta_b-\Delta_a$                 |
| $V_{t,k}$          | $\Delta_t$ 的前 k 个右奇异向量构成的正交基，位于矩阵输入空间       |
| $V^W_{t,k}$        | 完整权重 $W_t$ 的前 k 个右奇异向量，与更新 frame 分开              |
| $n,q$              | 矩阵输入维度 n；完整奇异值个数 $q=\min(m,n)$；实际 k 按 q 截断     |

除特别标明外，几何折线先对每个矩阵计算指标，再画矩阵间中位数和四分位区间。Attention 包含 q/k/v/o，MLP 包含 gate/up/down。只有七类矩阵都已测齐的层进入该 pair 的图。层 × 模块热图每格对应一个矩阵；主角分布图先计算单个矩阵内部的角度分布，再跨矩阵汇总。阴影体现不同矩阵的差异，不是多 seed 置信区间。

### A. 输出与 teacher 对齐：5 类

| 文件后缀                 | 实际测量指标                                                                                                                 | 横轴、分组与读法                                                                                |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| `sigma-teacher`          | 上排为 initial/teacher 两套固定 bank 的绝对 RKL：all、Math、Code、IF；下排为 $H(W_t,W_T)$ 与 $H(\Delta_t,\Delta_T)$          | 横轴为 optimizer 更新步；谱距离按 Attention/MLP 汇总。跨尺寸 pair 的 teacher 谱面板标为不可比较 |
| `sigma-teacher-distance` | 完整权重 $D_\Sigma(W_t,W_T)$、相对初始剩余距离 $D_\Sigma(W_t,W_T)/D_\Sigma(W_0,W_T)$、累计更新 $D_\Sigma(\Delta_t,\Delta_T)$ | 横轴为更新步；相对距离跳过近零分母。需要 teacher 与 student 同坐标                              |
| `frame-teacher-final`    | $V_{t,k}$ 与 teacher 更新 frame 的调整后 overlap，以及与自身终点 $V_{640,k}$ 的调整后 overlap                                | 横轴为更新步；k=16/64/256 三面板。区分“趋近自己的终点”和“趋近教师方向”                          |
| `parameter-residual`     | OPD/teacher 更新范数比 a、更新余弦 c、参数残差能量比 R                                                                       | 横轴为更新步；三面板分别展示幅度、方向和剩余参数差距                                            |
| `residual-layer-modules` | 终点的 a、c、R                                                                                                               | 横轴为七类模块，纵轴为层；定位参数差异集中在哪些矩阵，各面板独立色标                            |

RKL 先对每题有效响应 token 求平均，再在域内对题目平均；all 为三个域等权平均。它测量固定 bank 上的输出分布匹配。上表中的 H 是 Hellinger 距离，定义见 E 类；a、c、R 的公式见本节末尾。

### B. Frame 的方向与时间结构：8 类

| 文件后缀               | 实际测量指标                                                                                                                | 横轴、分组与读法                                                                                |
| ---------------------- | --------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| `early-final`          | 调整后 overlap $\widetilde O(V_{t,k},V_{640,k})$；早期 frame 捕获最终累计更新能量相对于最优 rank-k 的比例 $C^*_{t\to640,k}$ | 横轴为 anchor 更新步；曲线为 k=16/64/256                                                        |
| `frame-angles`         | 各 anchor 到终点的完整主角分布：33 点分位曲线，以及每 5 度精确统计的 ECDF                                                   | 上排横轴为分位点、纵轴为角度；下排横轴为角度、纵轴为累计比例。anchor=16/64/192/480，k=16/64/256 |
| `frame-angle-time`     | 到终点主角的 median、P90、最大值；各值先在单个矩阵内部计算                                                                  | 横轴为更新步，纵轴为角度（度）；k=16/64/256。区分典型方向和转动最大的方向                       |
| `frame-pairwise`       | 任意两个保存 checkpoint 累计更新 frame 的调整后 overlap                                                                     | 横纵轴均为更新步，颜色为矩阵间中位数；k=16/64/256，三面板共同色标                               |
| `frame-rank-time`      | 到终点的调整后 overlap、原始能量捕获 C、相对最优捕获 $C^*$                                                                  | 横轴为 anchor 更新步，纵轴为五档 k=16/32/64/128/256；三种量分别成图                             |
| `frame-modules`        | 七类模块各自到终点的调整后 overlap                                                                                          | 横轴为更新步，曲线为 k=16/64/256；每个模块单独一个面板，跨完整层汇总                            |
| `frame-layer-modules`  | anchor=16/64/192、k64 到终点的调整后 overlap                                                                                | 横轴为模块、纵轴为层；三个 anchor 共用色标，每格一个矩阵                                        |
| `weight-update-frames` | 完整权重 $V^W_{0,k}\leftrightarrow V^W_{t,k}$ 的主角中位数，以及累计更新 $V_{16,k}\leftrightarrow V_{t,k}$ 的主角中位数     | 横轴为更新步，k=16/64/256。两条线参照对象不同，完整权重参照 W0，更新参照 step16                 |

设 A、B 为同一输入空间内实际 k 维正交基，$c_i=\sigma_i(A^\top B)$，则主角和 overlap 定义为

$$
\theta_i=\arccos(c_i),\qquad
O(A,B)=\frac1k\sum_i c_i^2,\qquad
\widetilde O(A,B)=\frac{O(A,B)-k/n}{1-k/n},\qquad
d_{\mathrm{chord}}(A,B)=\sqrt{1-O(A,B)}.
$$

角度以度展示。调整后 overlap 扣除了同维随机子空间的期望 $k/n$；0 表示达到该基线，1 表示空间重合，可以略小于 0。当 k=n 时校正分母为零，调整值不定义。这里的 chordal distance 除以了 $\sqrt{k}$，取值在 0–1；与使用未归一化距离的论文对照时需换算。

捕获指标定义为

$$
C_{a\to b,k}=\frac{\Vert\Delta_bV_{a,k}\Vert_F^2}{\Vert\Delta_b\Vert_F^2},\qquad
F_{b,k}=\frac{\sum_{i=1}^{k}\sigma_i(\Delta_b)^2}{\Vert\Delta_b\Vert_F^2},\qquad
C^*_{a\to b,k}=\frac{C_{a\to b,k}}{F_{b,k}}.
$$

C 表示最终总能量的多少被捕获；$C^*$ 表示达到同 rank 最优捕获量的多少。Overlap 不按目标能量加权，因此与两种捕获量回答不同问题。

### C. 早期空间对后续增量的覆盖：2 类

| 文件后缀               | 实际测量指标                                                                                                                                                                 | 横轴、分组与读法                                                                                |
| ---------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| `frame-increments`     | 将目标 $\Delta_b$ 换成区间增量 $I_{a,b}$ 后的相对最优捕获：$\Vert I_{a,b}V_{h,k}\Vert_F^2/\sum_{i\le k}\sigma_i(I_{a,b})^2$                                                  | 横轴为相邻 checkpoint 区间，纵轴为历史 anchor h；k=16/64/256。只计算 $h\le a$，未来 anchor 留白 |
| `frame-outside-energy` | step16 frame 内绝对能量 $E_{\rm in}=\Vert I_{a,b}V_{16,k}\Vert_F^2$、外部能量 $E_{\rm out}=\Vert I_{a,b}\Vert_F^2-E_{\rm in}$、外部比例 $E_{\rm out}/\Vert I_{a,b}\Vert_F^2$ | 横轴为区间终点；上排为对数绝对能量，下排为外部比例，k=16/64/256                                 |

增量是区间净更新，区间内的往返变化未保存。区间长度不等，绝对能量曲线没有按更新步数归一化；因此不能把这些区间当作等预算训练。

### D. Singular spectrum 与更新维度：7 类

| 文件后缀                   | 实际测量指标                                                                                               | 横轴、分组与读法                                                                                                            |
| -------------------------- | ---------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| `update-dimension`         | $\Vert\Delta_t\Vert_F$、$\Vert\Delta_t\Vert_F/\Vert W_0\Vert_F$、stable rank、rank90、rank95、能量熵有效秩 | 横轴为更新步，六面板；Attention/MLP 分组                                                                                    |
| `singular-energy-rank`     | 完整权重、累计更新、区间增量各自的前 r 个奇异值平方累计比例 $\sum_{i\le r}s_i^2/\sum_i s_i^2$              | 横轴为保留 rank/完整奇异维度，纵轴为能量比例；完整权重和累计更新选 step16/64/192/640，增量选 16→32、48→64、128→192、480→640 |
| `singular-profiles`        | 完整权重的实际奇异值 $\sigma_i(W_t)$，以及同尺寸 teacher 的完整谱                                          | 横轴为奇异值序号，纵轴为奇异值，双对数；step0/16/64/640；七类模块分别展示                                                   |
| `singular-update-profiles` | 累计更新的实际奇异值 $\sigma_i(\Delta_t)$，以及同尺寸 teacher 更新谱                                       | 双对数；step16/64/192/640；零奇异值从对数坐标中略去                                                                         |
| `singular-rank-time`       | 完整权重逐 rank 的有符号变化 $[\sigma_i(W_t)-\sigma_i(W_0)]/\sigma_1(W_0)$                                 | 横轴为保存时刻，纵轴为线性抽取的 80 个精确 rank；七类模块分别画热图，各面板独立对称色标                                     |
| `singular-scale-shape`     | 最佳整体谱缩放 $\alpha$、去除该缩放后的形状残差、最大逐 rank 扰动 $\max_i\lvert s_i-r_i\rvert/r_1$         | 横轴为更新步；完整权重参照 step0，累计更新参照 step16；Attention/MLP 分组                                                   |
| `singular-layer-modules`   | step16 与 step640 的累计更新 stable rank、rank95、相对 W0 的更新范数                                       | 横轴为模块，纵轴为层；每格一个矩阵，各面板独立色标                                                                          |

原始奇异值曲线和 rank–时间热图确定性地选择当前完整层集合中的中间层，七类模块全部展示，没有按效果挑层。能量–rank 汇总将每个矩阵的 r/q 插值到公共网格，增量使用已导出的 65 点能量 CDF。

对奇异值向量 s，三种维度统计分别为

$$
\mathrm{stable\ rank}=\frac{\sum_i s_i^2}{s_1^2},\qquad
r_p=\min\left\{r:\frac{\sum_{i=1}^{r}s_i^2}{\sum_i s_i^2}\ge p\right\},\qquad
r_{\mathrm{entropy}}=\exp\left(-\sum_i q_i\log q_i\right),\quad q_i=\frac{s_i^2}{\sum_j s_j^2}.
$$

rank90/95 分别取 p=0.90/0.95。Stable rank 受最大奇异值影响，rank90/95 描述累计能量覆盖，熵有效秩描述全谱能量分散程度。它们均测更新维度，不直接衡量输出能力。

谱缩放和形状分解令 s 为当前、r 为参照奇异值向量：

$$
\alpha=\frac{r^\top s}{r^\top r},\qquad
e_{\mathrm{shape}}=\frac{\Vert s-\alpha r\Vert_2}{\Vert s\Vert_2}.
$$

### E. 谱形状变化及其与 frame 的关系：6 类

| 文件后缀               | 实际测量指标                                                                                       | 横轴、分组与读法                                                                                |
| ---------------------- | -------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| `weight-spectrum`      | 完整权重相对 W0 的 ISO 三指标：$\delta_\Sigma$、$\rho_\Sigma$、$\kappa_{\rm spec}$                 | 横轴为更新步，三面板；矩阵间中位数和四分位区间                                                  |
| `sigma-dynamics`       | 相邻 checkpoint 的 SD 距离、SD 距离/区间更新步数、到最终 checkpoint 的 SD 距离                     | 横轴为区间终点或 checkpoint 更新步；完整权重、累计更新两排，Attention/MLP 分组                  |
| `sigma-pairwise`       | 完整权重和累计更新各自的任意 checkpoint 两两 SD 距离                                               | 横纵轴均为保存时刻，颜色为逐矩阵距离的中位数；两面板独立色标                                    |
| `sigma-layers`         | 每层相邻区间 Hellinger 距离/区间更新步数                                                           | 横轴为区间，纵轴为层；先计算每个矩阵的距离，再对该层七类矩阵取中位数；完整权重和累计更新分开    |
| `singular-gap-angles`  | 起点累计更新的边界相对谱隙 $g_k=(s_k-s_{k+1})/s_k$，与相邻累计更新 frame 的归一化 chordal distance | 横轴为 g（对数），纵轴为转角距离；每点一个矩阵，k=16/64/256；选 16→32、64→128、192→320、480→640 |
| `singular-frame-phase` | 相邻区间的完整权重/累计更新 SD 距离，与该区间累计更新 frame 的 k64 chordal distance                | 横轴为谱距离（对数），纵轴为 frame 距离；每点一个矩阵；选 16→32、48→64、128→192、480→640        |

SD 使用奇异值和归一化 $p_i=s_i/\sum_j s_j$，Hellinger 使用同一 p：

$$
D_\Sigma(A,B)=\Vert p(A)-p(B)\Vert_2,\qquad
H(A,B)=\frac{\Vert\sqrt{p(A)}-\sqrt{p(B)}\Vert_2}{\sqrt2}.
$$

这与 D 类能量熵采用的 $q_i\propto s_i^2$ 不同。谱速度都是区间净距离除以更新步数，不是实际路径长度。

ISO 三指标在本实现中为

$$
\delta_\Sigma=\frac{\Vert\sigma(W_t)-\sigma(W_0)\Vert_2}{\Vert W_0\Vert_F},\qquad
\rho_\Sigma=\frac{\Vert\sigma(W_t)-\sigma(W_0)\Vert_2}{\Vert\Delta_t\Vert_F},\qquad
\kappa_{\rm spec}=\frac{mn}{q}\frac{\sum_{i=1}^{q}(u_{0,i}^{\top}\Delta_t v_{0,i})^2}{\Vert\Delta_t\Vert_F^2}.
$$

$u_{0,i},v_{0,i}$ 来自 W0 的配对奇异向量，q 为完整奇异维度。$\kappa_{\rm spec}$ 是相对随机方向基线缩放后的配对方向能量集中度，可以大于 1；它不是奇异值最大值/最小值的条件数，也不是 0–1 的能量比例。

### F. 跨 pair 总览：4 类

| 文件名                          | 实际测量指标                                                                                                                                                          | 纳入条件与读法                                                                                          |
| ------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| `overview-rkl-endpoints`        | step640 的绝对 RKL（颜色为 log10）和终点/初始 RKL 比例；all、Math、Code、IF，initial/teacher 两 bank 分开                                                             | 所有已纳入报告的完整训练 pair，不要求几何测齐；横轴为域，纵轴为 pair；不同面板色标可能不同              |
| `overview-rkl-trajectories`     | 两套 bank 的 all 与 Math 绝对 RKL 随更新步变化                                                                                                                        | 每个完整训练 pair 一个面板，纵轴为对数；各面板独立纵轴，比较绝对值需读刻度                              |
| `overview-geometry-rkl`         | 横轴六指标：k64 step16→640 调整后 overlap、相对最优捕获、终点 stable rank、相对更新范数、完整权重 0→640 SD 距离、终点参数残差比；纵轴统一为 teacher-bank 终点平均 RKL | 只纳入几何全层测齐的 pair；每点代表一个 pair 的逐矩阵中位数，不拟合因果关系；跨尺寸 pair 略去参数残差点 |
| `overview-shared-student-E编号` | 同一初始 student、不同 teacher 的 k64 到终点 overlap、累计更新 stable rank、相对更新范数                                                                              | 只纳入全层测齐的 pair，并匹配相同 tensor 坐标；横轴为更新步，曲线为不同 pair                            |

### 参数残差三指标

A 类图中的 teacher 更新为 $\Delta_T=W_T-W_0$，三个指标为

$$
a=\frac{\Vert\Delta_t\Vert_F}{\Vert\Delta_T\Vert_F},\qquad
c=\frac{\langle\Delta_t,\Delta_T\rangle_F}{\Vert\Delta_t\Vert_F\Vert\Delta_T\Vert_F},\qquad
R=\frac{\Vert W_T-W_t\Vert_F^2}{\Vert W_T-W_0\Vert_F^2}=1+a^2-2ac.
$$

a 比较更新范数，R 比较残差能量，两者不是同一个量。R=1 表示参数残差能量等于初始值，R>1 表示更大；它衡量同坐标参数距离。三项都需要同尺寸、同坐标的 teacher，跨尺寸 E102 仅展示 student 自身动态和输出 RKL。

## 尚未绘制的内容

| 内容                                                       | 当前缺口与后续处理                                                                       |
| ---------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| 尚未测齐 pair 的全模型对照                                 | 几何测量仍在进行；只发布已齐的完整层预览，全模型关联图只纳入已完整测量的 pair            |
| Teacher 与 OPD 更新相互捕获的能量                          | 已有更新 frame 主角，但缺少双向、能量加权的投影量；需要新增投影测量，不能由 overlap 代算 |
| 按相同能量选择不同 rank 的 frame 比较                      | 已有固定五档 rank；可计算 rank90/95，但对应可变 rank 的主角需要补测                      |
| 每个区间增量的完整奇异值曲线                               | 导出数据保留 65 点能量 CDF；可画能量–rank，完整增量谱需从原始测量迁入                    |
| PNT 的坐标 mask 与 GRRR 的 diagonal/off-diagonal/null 通道 | 当前任务未测这些分解；需新测量和明确定义，不能从整体谱图推导                             |
| 谱尾 Hill 指数及尾部拟合稳定性                             | 尚未实现；需要先定义拟合区间并做区间敏感性检查，避免单条拟合线制造结论                   |
| 早期子空间投影后继续训练                                   | 属于新干预实验；按当前约定暂不进行                                                       |

现有 checkpoint 只能恢复九个保存时刻及相邻区间的净变化。逐更新瞬时角速度、区间内往返轨迹以及多 seed 误差条均不在这些数据中。

## 数据与复算

[完整 readout](data/readout.json.gz) 保存每个矩阵的归一化谱向量、核范数、Frobenius 范数、SD/Hellinger 距离、所有 checkpoint 对、ISO 指标、角度摘要、能量捕获、参数残差、输入 tensor 哈希及全部 RKL 事件。[覆盖表](data/coverage.json) 标明哪些 pair、哪些层已齐。图以 PNG 与 PDF 同时输出，并附审计结果。

代码将测量、数据后处理、绘图和报告更新分开：GPU 测量保留现有 `spectral/trajectory.py`；CPU 后处理入口为 `absorption_dynamics/geometry_readouts.py`；纯指标在 `sigma_variation.py`，绘图按 `frame_figures.py`、`singular_figures.py`、`overview_figures.py` 及原有 `geometry_figures.py`、`sigma_figures.py` 分工；公共布局在 `geometry_plotting.py`，图目录在 `figure_catalog.py`，表格更新在 `geometry_report.py`。

在仓库根目录运行，`FIGURE_VIZ_MPL` 指向 figure-viz 的 `assets/mpl`。以下命令只读取本目录数据即可复画全部图并更新结果表：

```bash
python -m examples.mirroropd.research.studies.absorption_dynamics.geometry_readouts \
  --readout examples/mirroropd/docs/experiments/teacher-transfer/checkpoint-dynamics/data/readout.json.gz \
  --figures examples/mirroropd/docs/experiments/teacher-transfer/checkpoint-dynamics/figures \
  --report examples/mirroropd/docs/experiments/teacher-transfer/checkpoint-dynamics
```

测量尚未结束时使用 `--source` 指向任务输出目录、`--data` 指向本报告的数据目录，并加 `--watch`。多组并行任务用 `--extra-source` 追加目录；各目录须包含互不重叠的 pair，重复 pair 或目录直接报错，防止混入重复测量。默认每分钟检查新矩阵，可用 `--interval` 调整刷新间隔。它在 CPU 上计算指标、刷新图表；已发布且内容未变的 pair 跳过重绘；测量结束时再读取 exit 并完成最后一次发布。这个进程不创建巡检 hook，也不申请 GPU。测量失败会保留错误数量与 exit 信息，已有成功矩阵仍可复算。

### 参考文献

[^sosd]: The Stability of Singular Distribution: A Spectral Perspective on the Two-Phase Dynamics of Language Model Pre-training. arXiv, 2026. [2605.26489](https://arxiv.org/abs/2605.26489).
