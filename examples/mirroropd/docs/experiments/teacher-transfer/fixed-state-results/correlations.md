# 训练拟合与跨领域迁移的 RKL 相关性

固定状态实验中，每个 teacher–student run 作为一个点，比较训练池拟合与永久留出迁移，以及不同留出领域之间的共同变化。47 个输入条件按 student／输入模式分成七组；64 与 320 更新的已有单遍终点在同图展示，每个 run 只出现一次。完整配对、训练设置与位置诊断见[结果总览](README.md)。

逐训练快照的变化另见[RL 与 SFT 的吸收轨迹](absorption-trajectories.md)；该页每个点是同一 run 的一个检查点，与本页每个点为独立配对 run 的统计单位不同。

## 指标与点的含义

对训练池或留出领域 $D$，记初始 RKL 为 $K_D(0)$，该 run 的已有终点为 $K_D(T)$。两种下降指标分别为：

$$
\Delta_D = K_D(0)-K_D(T), \qquad r_D = 100\left(1-\frac{K_D(T)}{K_D(0)}\right)\%.
$$

绝对下降量 $\Delta_D$ 表示消除的 RKL；相对下降比例 $r_D$ 表示消除初始差距的百分比。所有 RKL 均为累计 KL 除以题数×1024（fixed-H），不使用按实际 token 平均的另一组端点列。负下降量保留，表示 RKL 增加。

Avg 按 Math64／Code32／IF32 的题数合并风险，即 $K_{\mathrm{Avg}}=0.5K_{\mathrm{Math}}+0.25K_{\mathrm{Code}}+0.25K_{\mathrm{IF}}$，再计算相对下降；三个领域的百分比不能直接平均。

| 编码         | 含义                                   |
| ------------ | -------------------------------------- |
| 一个点       | 一个 run 在已有单遍终点的初始→终点变化 |
| 颜色         | 同组内的 teacher 身份；各面板一致      |
| 三角形／圆形 | 64／320 次更新                         |
| 实心／空心   | 同尺寸／跨尺寸 teacher–student 配对    |
| 灰色虚线     | 两轴下降相等，即 $y=x$                 |

Training → held-out 图含 Math、Code、IF、Avg 四个面板，横轴是训练池的下降，纵轴是对应留出的下降；绝对和相对指标各一张图。固定训练池是冻结训练题池，64 更新时可能含尚未参与优化的后续批题目，因此这里衡量训练题池与永久留出的关系，不是当前 mini-batch loss 的相关性。Math 对应同领域不同题目；Code／IF 同时涉及跨领域迁移。

Cross-domain 图含 Math→Code、Math→IF、Code→IF 三个面板。箭头仅指定横纵轴，不表示因果方向。Avg 是三个领域的合成量，保留在 Training → held-out 图，不作为独立跨领域证据。

面板给出 Pearson $r$、Spearman $\rho$ 和 run 数 $n$；相关系数使用图中全部点，混合 64／320 预算及同尺寸／跨尺寸条件。它描述这些实际端点的共同变化，不能解释为控制预算、尺寸、初始距离或 teacher 训练前史后的因果效应。各点也不是同一设置的独立 seed 重复。少于三个点或任一轴恒定时不计算系数。

## 对数吸收倍数与最终残差

对数吸收倍数定义为：

$$
A_D=\log_{10}\frac{K_D(0)}{K_D(T)}.
$$

$A_D=1,2,3$ 分别表示初始 RKL 被压低 10、100、1000 倍，对应 90%、99%、99.9% 的相对下降；$A_D<0$ 表示训练后 RKL 增加。它将相对下降在接近 100% 处挤在一起的点展开，体现压低误差的数量级。每个 student／输入模式新增一张四面板图：横轴是训练池 $A$，纵轴是 Math／Code／IF／Avg 的 $A$。灰色虚线表示训练和留出压低相同的数量级，$A=0$ 表示没有改善。

最终残差图使用双对数坐标：横轴为最终训练池 RKL，纵轴为最终 Math／Code／IF／Avg RKL。越靠左下，训练与留出残差越小；低训练残差与高留出残差的组合体现训练拟合和留出吸收之间的差距。灰色虚线表示两边最终 RKL 相等。最终残差包含初始距离的影响，应与对数吸收倍数一起读取。

最终残差图的 Pearson 系数在 $\log_{10}K(T)$ 上计算，与图中对数坐标对应；Spearman 系数也基于该坐标计算，正数上的对数变换保持排序。其余图的相关系数均在图中显示的数值上计算。CSV 的 `correlation_space` 列记录所用空间。所有 235 个 run×panel 的初始和最终 RKL 均为正数；脚本直接计算对数，不添加 epsilon 或截断小残差。Avg 先合并领域风险，再取对数，不能平均三个领域的 $A$。

## Training → Avg 的相关系数

下表与图中 Avg 面板相同，绝对下降与相对下降分别计算，未合并不同 student 或输入模式。完整四个留出面板及三个跨领域配对见[相关系数 CSV](figures/correlations/correlations.csv)。

| Student／输入             | run 数（64／320） | 相对下降 Pearson／Spearman | 绝对下降 Pearson／Spearman |
| ------------------------- | ----------------- | -------------------------- | -------------------------- |
| R1-1.5B                   | 9（4／5）         | 0.980／0.950               | 0.953／0.917               |
| R1-7B                     | 3（1／2）         | 0.437／0.500               | 0.995／1.000               |
| Qwen3-1.7B / thinking     | 13（1／12）       | 0.950／0.929               | 0.964／0.918               |
| Qwen3-1.7B / non-thinking | 10（0／10）       | 0.988／0.976               | 0.984／0.976               |
| Llama-3.2-3B-Instruct     | 10（0／10）       | 0.980／0.927               | 0.954／0.939               |

相对下降与绝对下降回答不同问题。绝对下降随初始距离变化的空间更大；解释相关性时应结合[逐点数据](figures/correlations/points.csv)中的 initial 和 final，而不能仅按一个相关系数给 teacher 排学习难度。

## R1-1.5B

共 9 个 run，64 更新 4 个，320 更新 5 个。

### Training → held-out：对数吸收倍数

![R1-1.5B，对数吸收倍数](figures/correlations/r1-cross-batch-log_absorption.png)

[矢量 PDF](figures/correlations/r1-cross-batch-log_absorption.pdf)

### Training → held-out：最终残差（双对数）

![R1-1.5B，最终残差（双对数）](figures/correlations/r1-cross-batch-final.png)

[矢量 PDF](figures/correlations/r1-cross-batch-final.pdf)

### Training → held-out：相对下降

![R1-1.5B，Training → held-out，相对下降](figures/correlations/r1-cross-batch-relative.png)

[矢量 PDF](figures/correlations/r1-cross-batch-relative.pdf)

### Training → held-out：绝对下降

![R1-1.5B，Training → held-out，绝对下降](figures/correlations/r1-cross-batch-absolute.png)

[矢量 PDF](figures/correlations/r1-cross-batch-absolute.pdf)

### Cross-domain：相对下降

![R1-1.5B，Cross-domain，相对下降](figures/correlations/r1-cross-domain-relative.png)

[矢量 PDF](figures/correlations/r1-cross-domain-relative.pdf)

### Cross-domain：绝对下降

![R1-1.5B，Cross-domain，绝对下降](figures/correlations/r1-cross-domain-absolute.png)

[矢量 PDF](figures/correlations/r1-cross-domain-absolute.pdf)

## R1-7B

共 3 个 run，64 更新 1 个，320 更新 2 个。

### Training → held-out：对数吸收倍数

![R1-7B，对数吸收倍数](figures/correlations/r7-cross-batch-log_absorption.png)

[矢量 PDF](figures/correlations/r7-cross-batch-log_absorption.pdf)

### Training → held-out：最终残差（双对数）

![R1-7B，最终残差（双对数）](figures/correlations/r7-cross-batch-final.png)

[矢量 PDF](figures/correlations/r7-cross-batch-final.pdf)

### Training → held-out：相对下降

![R1-7B，Training → held-out，相对下降](figures/correlations/r7-cross-batch-relative.png)

[矢量 PDF](figures/correlations/r7-cross-batch-relative.pdf)

### Training → held-out：绝对下降

![R1-7B，Training → held-out，绝对下降](figures/correlations/r7-cross-batch-absolute.png)

[矢量 PDF](figures/correlations/r7-cross-batch-absolute.pdf)

### Cross-domain：相对下降

![R1-7B，Cross-domain，相对下降](figures/correlations/r7-cross-domain-relative.png)

[矢量 PDF](figures/correlations/r7-cross-domain-relative.pdf)

### Cross-domain：绝对下降

![R1-7B，Cross-domain，绝对下降](figures/correlations/r7-cross-domain-absolute.png)

[矢量 PDF](figures/correlations/r7-cross-domain-absolute.pdf)

## Qwen3-1.7B / thinking

共 13 个 run，64 更新 1 个，320 更新 12 个。

### Training → held-out：对数吸收倍数

![Qwen3-1.7B / thinking，对数吸收倍数](figures/correlations/q3-1.7-cross-batch-log_absorption.png)

[矢量 PDF](figures/correlations/q3-1.7-cross-batch-log_absorption.pdf)

### Training → held-out：最终残差（双对数）

![Qwen3-1.7B / thinking，最终残差（双对数）](figures/correlations/q3-1.7-cross-batch-final.png)

[矢量 PDF](figures/correlations/q3-1.7-cross-batch-final.pdf)

### Training → held-out：相对下降

![Qwen3-1.7B / thinking，Training → held-out，相对下降](figures/correlations/q3-1.7-cross-batch-relative.png)

[矢量 PDF](figures/correlations/q3-1.7-cross-batch-relative.pdf)

### Training → held-out：绝对下降

![Qwen3-1.7B / thinking，Training → held-out，绝对下降](figures/correlations/q3-1.7-cross-batch-absolute.png)

[矢量 PDF](figures/correlations/q3-1.7-cross-batch-absolute.pdf)

### Cross-domain：相对下降

![Qwen3-1.7B / thinking，Cross-domain，相对下降](figures/correlations/q3-1.7-cross-domain-relative.png)

[矢量 PDF](figures/correlations/q3-1.7-cross-domain-relative.pdf)

### Cross-domain：绝对下降

![Qwen3-1.7B / thinking，Cross-domain，绝对下降](figures/correlations/q3-1.7-cross-domain-absolute.png)

[矢量 PDF](figures/correlations/q3-1.7-cross-domain-absolute.pdf)

## Qwen3-1.7B / non-thinking

共 10 个 run，64 更新 0 个，320 更新 10 个。

### Training → held-out：对数吸收倍数

![Qwen3-1.7B / non-thinking，对数吸收倍数](figures/correlations/q3-1.7-nonthinking-cross-batch-log_absorption.png)

[矢量 PDF](figures/correlations/q3-1.7-nonthinking-cross-batch-log_absorption.pdf)

### Training → held-out：最终残差（双对数）

![Qwen3-1.7B / non-thinking，最终残差（双对数）](figures/correlations/q3-1.7-nonthinking-cross-batch-final.png)

[矢量 PDF](figures/correlations/q3-1.7-nonthinking-cross-batch-final.pdf)

### Training → held-out：相对下降

![Qwen3-1.7B / non-thinking，Training → held-out，相对下降](figures/correlations/q3-1.7-nonthinking-cross-batch-relative.png)

[矢量 PDF](figures/correlations/q3-1.7-nonthinking-cross-batch-relative.pdf)

### Training → held-out：绝对下降

![Qwen3-1.7B / non-thinking，Training → held-out，绝对下降](figures/correlations/q3-1.7-nonthinking-cross-batch-absolute.png)

[矢量 PDF](figures/correlations/q3-1.7-nonthinking-cross-batch-absolute.pdf)

### Cross-domain：相对下降

![Qwen3-1.7B / non-thinking，Cross-domain，相对下降](figures/correlations/q3-1.7-nonthinking-cross-domain-relative.png)

[矢量 PDF](figures/correlations/q3-1.7-nonthinking-cross-domain-relative.pdf)

### Cross-domain：绝对下降

![Qwen3-1.7B / non-thinking，Cross-domain，绝对下降](figures/correlations/q3-1.7-nonthinking-cross-domain-absolute.png)

[矢量 PDF](figures/correlations/q3-1.7-nonthinking-cross-domain-absolute.pdf)

## Llama-3.2-3B-Instruct

共 10 个 run，64 更新 0 个，320 更新 10 个。

### Training → held-out：对数吸收倍数

![Llama-3.2-3B-Instruct，对数吸收倍数](figures/correlations/llama3-3b-off-cross-batch-log_absorption.png)

[矢量 PDF](figures/correlations/llama3-3b-off-cross-batch-log_absorption.pdf)

### Training → held-out：最终残差（双对数）

![Llama-3.2-3B-Instruct，最终残差（双对数）](figures/correlations/llama3-3b-off-cross-batch-final.png)

[矢量 PDF](figures/correlations/llama3-3b-off-cross-batch-final.pdf)

### Training → held-out：相对下降

![Llama-3.2-3B-Instruct，Training → held-out，相对下降](figures/correlations/llama3-3b-off-cross-batch-relative.png)

[矢量 PDF](figures/correlations/llama3-3b-off-cross-batch-relative.pdf)

### Training → held-out：绝对下降

![Llama-3.2-3B-Instruct，Training → held-out，绝对下降](figures/correlations/llama3-3b-off-cross-batch-absolute.png)

[矢量 PDF](figures/correlations/llama3-3b-off-cross-batch-absolute.pdf)

### Cross-domain：相对下降

![Llama-3.2-3B-Instruct，Cross-domain，相对下降](figures/correlations/llama3-3b-off-cross-domain-relative.png)

[矢量 PDF](figures/correlations/llama3-3b-off-cross-domain-relative.pdf)

### Cross-domain：绝对下降

![Llama-3.2-3B-Instruct，Cross-domain，绝对下降](figures/correlations/llama3-3b-off-cross-domain-absolute.png)

[矢量 PDF](figures/correlations/llama3-3b-off-cross-domain-absolute.pdf)

## Qwen3-1.7B / native Thinking-2507

共 1 个 run，64 更新 0 个，320 更新 1 个。原生模板独立分组；单点仅展示位置，相关系数不可估计。

### Training → held-out：对数吸收倍数

![Qwen3-1.7B / native Thinking-2507，对数吸收倍数](figures/correlations/q3-1.7-native-thinking2507-cross-batch-log_absorption.png)

[矢量 PDF](figures/correlations/q3-1.7-native-thinking2507-cross-batch-log_absorption.pdf)

### Training → held-out：最终残差（双对数）

![Qwen3-1.7B / native Thinking-2507，最终残差（双对数）](figures/correlations/q3-1.7-native-thinking2507-cross-batch-final.png)

[矢量 PDF](figures/correlations/q3-1.7-native-thinking2507-cross-batch-final.pdf)

### Training → held-out：相对下降

![Qwen3-1.7B / native Thinking-2507，Training → held-out，相对下降](figures/correlations/q3-1.7-native-thinking2507-cross-batch-relative.png)

[矢量 PDF](figures/correlations/q3-1.7-native-thinking2507-cross-batch-relative.pdf)

### Training → held-out：绝对下降

![Qwen3-1.7B / native Thinking-2507，Training → held-out，绝对下降](figures/correlations/q3-1.7-native-thinking2507-cross-batch-absolute.png)

[矢量 PDF](figures/correlations/q3-1.7-native-thinking2507-cross-batch-absolute.pdf)

### Cross-domain：相对下降

![Qwen3-1.7B / native Thinking-2507，Cross-domain，相对下降](figures/correlations/q3-1.7-native-thinking2507-cross-domain-relative.png)

[矢量 PDF](figures/correlations/q3-1.7-native-thinking2507-cross-domain-relative.pdf)

### Cross-domain：绝对下降

![Qwen3-1.7B / native Thinking-2507，Cross-domain，绝对下降](figures/correlations/q3-1.7-native-thinking2507-cross-domain-absolute.png)

[矢量 PDF](figures/correlations/q3-1.7-native-thinking2507-cross-domain-absolute.pdf)

## Qwen3-1.7B / native Instruct-2507

共 1 个 run，64 更新 0 个，320 更新 1 个。原生模板独立分组；单点仅展示位置，相关系数不可估计。

### Training → held-out：对数吸收倍数

![Qwen3-1.7B / native Instruct-2507，对数吸收倍数](figures/correlations/q3-1.7-native-instruct2507-cross-batch-log_absorption.png)

[矢量 PDF](figures/correlations/q3-1.7-native-instruct2507-cross-batch-log_absorption.pdf)

### Training → held-out：最终残差（双对数）

![Qwen3-1.7B / native Instruct-2507，最终残差（双对数）](figures/correlations/q3-1.7-native-instruct2507-cross-batch-final.png)

[矢量 PDF](figures/correlations/q3-1.7-native-instruct2507-cross-batch-final.pdf)

### Training → held-out：相对下降

![Qwen3-1.7B / native Instruct-2507，Training → held-out，相对下降](figures/correlations/q3-1.7-native-instruct2507-cross-batch-relative.png)

[矢量 PDF](figures/correlations/q3-1.7-native-instruct2507-cross-batch-relative.pdf)

### Training → held-out：绝对下降

![Qwen3-1.7B / native Instruct-2507，Training → held-out，绝对下降](figures/correlations/q3-1.7-native-instruct2507-cross-batch-absolute.png)

[矢量 PDF](figures/correlations/q3-1.7-native-instruct2507-cross-batch-absolute.pdf)

### Cross-domain：相对下降

![Qwen3-1.7B / native Instruct-2507，Cross-domain，相对下降](figures/correlations/q3-1.7-native-instruct2507-cross-domain-relative.png)

[矢量 PDF](figures/correlations/q3-1.7-native-instruct2507-cross-domain-relative.pdf)

### Cross-domain：绝对下降

![Qwen3-1.7B / native Instruct-2507，Cross-domain，绝对下降](figures/correlations/q3-1.7-native-instruct2507-cross-domain-absolute.png)

[矢量 PDF](figures/correlations/q3-1.7-native-instruct2507-cross-domain-absolute.pdf)

## 数据与复算

图使用[单遍诊断端点表](../figures/fixed-state-clean-20261107/endpoints.csv)中全部 `band=all` 的条件，沿用该图集排除 Polaris 与附录条件的来源选择。输入哈希、run 数与口径见[来源记录](figures/correlations/provenance.json)；每张图旁保留 `.audit.json`。绘图脚本验证 run 身份、五个面板完整性及 Avg 的按题数合并恒等式。

在仓库根目录运行以下命令，只读取现有端点，不加载模型或启动训练：

```bash
export FIGURE_VIZ_MPL=/mnt/tidalfs-bdsz01/dataset/llm_dataset/zkn_data/fht_data/dotfiles/.agents/skills/figure-viz/assets/mpl
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /root/.local/share/mirroropd/venvs/plotting/bin/python -m examples.mirroropd.research.studies.fixed_transfer.plot_correlations
```

可编辑源码为 [plot_correlations.py](../../../../research/studies/fixed_transfer/plot_correlations.py)，输出 PNG、PDF、逐点数据、相关系数表及输入哈希。原始日志到端点的流程见[单遍诊断复算](../reproduction.md#clean-diagnostics)。
