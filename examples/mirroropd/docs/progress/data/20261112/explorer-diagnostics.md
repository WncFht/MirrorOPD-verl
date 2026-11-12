# 吸收量、迁移效率与后期偏离

交互页（本地文件 `data/explorer.html`）的“视图”新增四项：终点吸收量、累计迁移效率、早期—后期迁移效率、累计吸收差 Δ_D 随 step。四项固定使用下述定义，原有指标选择在这些视图中禁用；student、尺寸、阶段、teacher 显隐及更新上限仍生效。这里分析的是固定 S0 前缀上的分布距离，不等同于任务成绩或教师知识的实际吸收比例。

## 指标与窗口

训练评分池记为 $T$，留出域记为 $D$，RKL 为 $K$。对数吸收量的单位为 dex：

$$
A_D(t)=\log_{10}\frac{K_D(0)}{K_D(t)},\qquad
\eta_D(t)=\frac{A_D(t)}{A_T(t)},\qquad
\Delta_D(t)=A_T(t)-A_D(t).
$$

$A_D=1$ 表示风险降低十倍。$\eta_D=1$ 表示训练与留出消除的风险倍数相同；它是两个对数改善量的比值，可以大于 1 或为负，不是限制在 0 到 1 的知识吸收率。$\Delta_D>0$ 表示留出改善落后，$\Delta_D<0$ 表示留出改善领先。三者同时保留训练拟合程度、相对迁移和累计差异。

原始双对数 RKL 图上的对角线表示 $K_D(t)=K_T(t)$，不能直接表示吸收同步。原始曲线到对角线的有符号距离还含初始差异：

$$
\log_{10}\frac{K_D(t)}{K_T(t)}
=\log_{10}\frac{K_D(0)}{K_T(0)}+\Delta_D(t).
$$

新增偏离图减去了初始偏移，所有 run 从零出发。$\Delta_D$ 增长表示相对于起点的训练—留出差距扩大；绝对正负仍按累计吸收量解释。它衡量的是固定训练池与留出池的改善差异，训练池早期包含未来批，不能直接当作经典已见训练集的过拟合指标。

早窗固定为 step0→64。设更新上限内每条 run 的实际终点为 $e$，末窗取 $e-64\to e$，要求 $e\ge128$，保证两窗不重叠：

$$
\eta_D^{\rm early}=\frac{A_D(64)}{A_T(64)},\qquad
\eta_D^{\rm late}(e)=\frac{A_D(e)-A_D(e-64)}{A_T(e)-A_T(e-64)}.
$$

比值仅在对应训练改善量大于 0.01 dex 时显示。这是约 2.3% 风险下降的分母稳定性保护，不是吸收成功或失败的门槛。训练改善非正、改善过小、预算不足时，记录比值为缺失，并保留两个原始增量与原因。HTML 图卡可展开查看，CSV 也保留这些行。缺失点不压到坐标边界，不替换成零，不截断很大的有效比值。

64-step run 在终点吸收量与累计效率中保留实际终点，但不进入早晚窗口散点。滑块会改变各 run 的实际终点与末窗；滑块为 64 时全部末窗比值缺失。静态图使用各 run 的实际完整预算，比较时须查看图例中的 64／320 标记。所有曲线均按实际检查点连接，不插值。

| 视图         | 横轴                 | 纵轴                   | 每个点的含义           |
| ------------ | -------------------- | ---------------------- | ---------------------- |
| 终点吸收量   | $A_T(e)$             | $A_D(e)$               | 一个 run 的实际终点    |
| 累计迁移效率 | Teacher              | $\eta_D(e)$            | 同一实际终点的累计比值 |
| 早期—后期    | $\eta_D^{\rm early}$ | $\eta_D^{\rm late}(e)$ | 两个不重叠窗口的比值   |
| 偏离随 step  | 学生更新数           | $\Delta_D(t)$          | 一个 run 的全部快照    |

圆点为同尺寸、菱形为跨尺寸，颜色标识 teacher；时间曲线统一实线。四个面板为 Math、Code、IF、Avg，各图内同类纵轴范围一致。终点吸收量的横纵范围相同，参考线为 $A_D=A_T$；累计效率参考线为 1；窗口图参考线为早／晚效率等于 1；偏离图参考线为 0。Avg 先合并 Math64／Code32／IF32 风险，再计算指标，并非各域指标的算术平均。

## 数据支持的读法

Qwen3-1.7B thinking 学习官方 4B／8B teacher 时，训练吸收量分别为 0.339／0.335 dex，Avg 为 0.0064／0.0011 dex，累计迁移效率约 0.019／0.0033。更准确的描述是“训练池可以拟合、留出迁移接近停滞”。这是当前预算和冻结状态上的观察，不能推出跨尺寸教师永远不能蒸馏。

同尺寸 e3／GRPO-KL 的 Avg 累计效率约 1.186／1.287；两条训练池曲线的最低点分别在 step192／208，末窗训练吸收增量为负，但 Avg 留出增量为正。窗口图将它们标为训练反弹，不画误导性的负效率点。偏离曲线继续向负方向移动：留出仍在改善，不能把所有“偏离”都解释成留出恶化。

DeepMath SFT 的 early Math 改善与训练池接近：0.543／0.564 dex；early Avg 则是 0.358／0.564 dex，不能把 Math 的数值当成 Avg。其最后 64 步训练改善 0.0688 dex，Avg 仅 0.00452 dex，末窗效率约 0.066。该条件支持“后期训练拟合继续、跨域改善更慢”；其 thinking 输入与公开训练模式存在不匹配，不能只归因于 SFT。

Qwen non-thinking 的三条同尺寸 Base-RL 分支，Avg 累计效率约 0.542–0.546，明显低于 thinking 的 e3／GRPO-KL；Samsung SFT 在 non-thinking 的 Avg 吸收量约 1.003 dex，高于这三条 RL 分支的 0.468–0.478 dex。RL／SFT 标签不能代替父模型、初始化、输入模式和初始距离的分析。两模式使用不同冻结银行，这也不是只切换一个输入开关的因果对照。

连续检查点不是独立重复。各点没有跨 seed 置信区间，窗口比值是描述统计；训练单遍持续接触新批，末窗不是固定训练集的纯收敛阶段。“RL 更容易整体吸收”是部分已配对条件的观察，不能概括成“所有同尺寸 RL 都会沿对角线到底”。

## 三个视图的联合解释

三个量并非三份独立证据。训练吸收量为正时，有恒等式：

$$
\Delta_D=A_T-A_D=A_T(1-\eta_D).
$$

因此需要同时看总进展和相对迁移。较大的 $\Delta_D$ 既可能来自迁移效率低，也可能来自训练池已经消除了很多误差。较小的 $\Delta_D$ 也可能只是两边都几乎没动。窗口效率与累计效率同样不能互换：前者衡量指定区间的增量比，后者保留整条路径的历史。

下表均为实际 320 步终点的 Avg，训练评分池为 320 题。模型名中的 SFT step80／640 是 teacher 版本，不是学生更新预算。

| Student／输入模式 | Teacher        | $A_T$ | $A_D$ | $\Delta_D$ | $\eta_D$ | 最终 $K_D$ |
| ----------------- | -------------- | ----: | ----: | ---------: | -------: | ---------: |
| Qwen thinking     | e3             | 1.194 | 1.416 |     −0.222 |    1.186 |   0.001298 |
| Qwen thinking     | GRPO-KL        | 0.969 | 1.247 |     −0.278 |    1.287 |   0.001109 |
| Qwen thinking     | Samsung SFT    | 1.042 | 0.857 |      0.185 |    0.822 |   0.034588 |
| Qwen thinking     | DeepMath SFT   | 1.050 | 0.506 |      0.544 |    0.482 |   0.191977 |
| Qwen thinking     | 4B SFT step80  | 0.387 | 0.128 |      0.259 |    0.331 |   0.287476 |
| Qwen thinking     | 4B SFT step640 | 0.311 | 0.053 |      0.258 |    0.171 |   0.370862 |
| Qwen thinking     | 官方 4B        | 0.339 | 0.006 |      0.332 |    0.019 |   0.295592 |
| Qwen thinking     | 官方 8B        | 0.335 | 0.001 |      0.333 |    0.003 |   0.312846 |
| R1-1.5B           | DeepScaleR     | 1.506 | 1.664 |     −0.157 |    1.104 |   0.000716 |
| R1-1.5B           | JustRL         | 1.241 | 0.822 |      0.419 |    0.662 |   0.017768 |
| R1-1.5B           | BroRL          | 1.241 | 0.605 |      0.636 |    0.488 |   0.072881 |
| R1-1.5B           | Nemotron       | 1.328 | 0.646 |      0.682 |    0.486 |   0.045246 |
| R1-1.5B           | MiniMath SFT   | 0.968 | 0.511 |      0.457 |    0.528 |   0.028692 |
| Llama-3.2-3B      | GT-RL          | 1.327 | 1.628 |     −0.301 |    1.227 |   0.000540 |
| Llama-3.2-3B      | arithmetic SFT | 0.833 | 0.487 |      0.346 |    0.585 |   0.071417 |

### Δ：用相对残差解释差距，并区分增长方向

定义相对残差 $r_D=K_D(t)/K_D(0)$，则：

$$
10^{\Delta_D}=\frac{r_D}{r_T}.
$$

$\Delta_D=0.3$ 表示留出保留的初始误差比例约为训练池的两倍；它不直接比较两池的绝对 RKL。R1 的 Nemotron 终点 $\Delta_D=0.682$，相对残差比约 4.81 倍。Qwen 的 e3 为−0.222，对应约 0.60 倍，留出相对残差反而更少。

Qwen 官方 4B／8B 的 $\Delta_D$ 几乎相同，分别 0.332／0.333，并且两个吸收坐标也接近：它们在当前状态与预算下都主要改善训练池。仅改变官方 teacher 尺寸 4B→8B 并未带来更多留出吸收。另一方面，SFT step80／640 的 $\Delta_D$ 也接近，但后者训练与留出吸收同时更少，累计效率下降约一半。相近的吸收差可以对应不同的问题。

还应区分 $\Delta_D$ 的值与变化方向：正值且继续上升，表示累计训练领先且差距扩大；正值但下降，表示留出在追赶；负值表示累计留出领先。末段 $\Delta_D$ 下降也可能有训练池反弹的贡献，需一起查看 $K_T$ 与 $K_D$。e3、GRPO-KL 和 Llama 同尺寸 RL 的末段包含这种情况，不能一律解读为泛化恶化。

Avg 会掩盖域内差异。e3 与 GRPO-KL 的 Avg $\Delta_D$ 为负，但 IF 仍为正，分别约 0.199／0.039；IF 效率约 0.834／0.960。DeepScaleR 的 Avg 效率约 1.104，但 IF 仅 0.739、IF 吸收差约 0.394。部分同尺寸 RL 能在 Math／Code 取得超出训练池的相对改善，同时在 IF 仍然落后；“整体吸收好”不能替代逐域结论。

### η：比较相对于训练改善的迁移，不能代替吸收总量

对正训练吸收量，有：

$$
r_D=r_T^{\eta_D}.
$$

例如 $\eta_D=0.5$ 且训练残差降到初始的 0.1 时，留出残差为初始的约 0.316，下降约 68.4%；“效率 0.5”不表示吸收一半知识或留出下降 50%。它衡量相对收缩关系，不识别教师信息的语义组成。

Qwen thinking 的 GRPO-KL 比 e3 的累计效率更高，但 e3 的留出吸收量更大；前者是相对于训练进展更有效，后者是累计消除更多倍数的初始误差。两者初始距离不同，最终 RKL 排序又与吸收量排序相反，因此还需要保留 $K_D(0)$ 和 $K_D(e)$。

R1 同尺寸 RL 内部的效率从 BroRL／Nemotron 的约 0.49，到 JustRL 的 0.66、DeepScaleR 的 1.10，变化较大。MiniMath SFT 约 0.53，高于前两条 RL，但低于 JustRL／DeepScaleR。当前数据支持部分同尺寸 RL 具有很强迁移，不能支持“同尺寸 RL 普遍完全吸收、SFT 普遍不能吸收”的二分结论。

### 终点吸收量：分清拟合不足和迁移不足

终点图的横坐标是训练拟合进展，纵坐标是留出吸收进展。左下方既可能是初始差距已很小，也可能是当前预算无法拟合；右下方则表示已经明显拟合训练池，留出改善落后。坐标是各自初始距离的倍数改善，不能独立判断残差是否已经足够小。

JustRL 与 BroRL 的 $A_T$ 几乎相同，约 1.241，但 $A_D$ 分别 0.822 与 0.605；训练池都压低约 17.4 倍，Avg 分别压低约 6.63 与 4.03 倍。这个比较表明，仅解释训练拟合速度不足以解释留出吸收差异。Samsung 与 DeepMath SFT 同样具有接近的 $A_T$，却有明显不同的 $A_D$；追加阶段标签内部也存在迁移差异。

官方 Qwen4B／8B 的 $A_T$ 约 0.34，训练残差压低约 2.2 倍，留出残差几乎没变；这支持当前冻结状态下的迁移瓶颈。跨尺寸的 4B Base-GRPO 却有 $A_D=0.297$、$\eta_D=0.440$，高于官方 4B。因此尺寸本身不足以决定蒸馏结果，训练历史和输入关系也影响曲线。

共同高吸收的例子是 Qwen e3／GRPO-KL、Llama GT-RL 和 R1 DeepScaleR；这些配对同时具有较高留出吸收量和较小最终残差。它们仍然不能证明到达最终收敛极限，也不能证明吸收的是可单独相加的“RL 分量”。

### 用速度差区分后期漂移

只要曲线可微，累计吸收差满足：

$$
\frac{d\Delta_D}{dt}=\frac{dA_T}{dt}-\frac{dA_D}{dt}.
$$

在 $dA_T/dt>0$ 的区间，局部迁移效率为 $(dA_D/dt)/(dA_T/dt)$；小于 1 时差距扩大，大于 1 时差距缩小。训练吸收增量非正时，用速度差仍有定义，效率比值则不适合解释。现有窗口图同时保留两个增量，便于区分“训练改善而留出平台”和“训练触底而留出仍改善”。当前每 16 步一个快照，窗口为描述统计，不据此定位唯一的机制转折点。

## 静态图集

每组四张图均含四个留出面板。只想探索少量 teacher 时，交互页的筛选更方便。

### R1-1.5B

![R1-1.5B：终点吸收量](figures/absorption-diagnostics/r1-absorption.png)

[矢量 PDF](figures/absorption-diagnostics/r1-absorption.pdf)

![R1-1.5B：累计迁移效率](figures/absorption-diagnostics/r1-efficiency.png)

[矢量 PDF](figures/absorption-diagnostics/r1-efficiency.pdf)

![R1-1.5B：早期—后期迁移效率](figures/absorption-diagnostics/r1-windows.png)

[矢量 PDF](figures/absorption-diagnostics/r1-windows.pdf)

![R1-1.5B：累计吸收差 Δ_D 随 step](figures/absorption-diagnostics/r1-drift.png)

[矢量 PDF](figures/absorption-diagnostics/r1-drift.pdf)

### R1-7B

![R1-7B：终点吸收量](figures/absorption-diagnostics/r7-absorption.png)

[矢量 PDF](figures/absorption-diagnostics/r7-absorption.pdf)

![R1-7B：累计迁移效率](figures/absorption-diagnostics/r7-efficiency.png)

[矢量 PDF](figures/absorption-diagnostics/r7-efficiency.pdf)

![R1-7B：早期—后期迁移效率](figures/absorption-diagnostics/r7-windows.png)

[矢量 PDF](figures/absorption-diagnostics/r7-windows.pdf)

![R1-7B：累计吸收差 Δ_D 随 step](figures/absorption-diagnostics/r7-drift.png)

[矢量 PDF](figures/absorption-diagnostics/r7-drift.pdf)

### Qwen3-1.7B / thinking

![Qwen3-1.7B / thinking：终点吸收量](figures/absorption-diagnostics/q3-1.7-absorption.png)

[矢量 PDF](figures/absorption-diagnostics/q3-1.7-absorption.pdf)

![Qwen3-1.7B / thinking：累计迁移效率](figures/absorption-diagnostics/q3-1.7-efficiency.png)

[矢量 PDF](figures/absorption-diagnostics/q3-1.7-efficiency.pdf)

![Qwen3-1.7B / thinking：早期—后期迁移效率](figures/absorption-diagnostics/q3-1.7-windows.png)

[矢量 PDF](figures/absorption-diagnostics/q3-1.7-windows.pdf)

![Qwen3-1.7B / thinking：累计吸收差 Δ_D 随 step](figures/absorption-diagnostics/q3-1.7-drift.png)

[矢量 PDF](figures/absorption-diagnostics/q3-1.7-drift.pdf)

### Qwen3-1.7B / non-thinking

![Qwen3-1.7B / non-thinking：终点吸收量](figures/absorption-diagnostics/q3-1.7-nonthinking-absorption.png)

[矢量 PDF](figures/absorption-diagnostics/q3-1.7-nonthinking-absorption.pdf)

![Qwen3-1.7B / non-thinking：累计迁移效率](figures/absorption-diagnostics/q3-1.7-nonthinking-efficiency.png)

[矢量 PDF](figures/absorption-diagnostics/q3-1.7-nonthinking-efficiency.pdf)

![Qwen3-1.7B / non-thinking：早期—后期迁移效率](figures/absorption-diagnostics/q3-1.7-nonthinking-windows.png)

[矢量 PDF](figures/absorption-diagnostics/q3-1.7-nonthinking-windows.pdf)

![Qwen3-1.7B / non-thinking：累计吸收差 Δ_D 随 step](figures/absorption-diagnostics/q3-1.7-nonthinking-drift.png)

[矢量 PDF](figures/absorption-diagnostics/q3-1.7-nonthinking-drift.pdf)

### Llama-3.2-3B-Instruct

![Llama-3.2-3B-Instruct：终点吸收量](figures/absorption-diagnostics/llama3-3b-off-absorption.png)

[矢量 PDF](figures/absorption-diagnostics/llama3-3b-off-absorption.pdf)

![Llama-3.2-3B-Instruct：累计迁移效率](figures/absorption-diagnostics/llama3-3b-off-efficiency.png)

[矢量 PDF](figures/absorption-diagnostics/llama3-3b-off-efficiency.pdf)

![Llama-3.2-3B-Instruct：早期—后期迁移效率](figures/absorption-diagnostics/llama3-3b-off-windows.png)

[矢量 PDF](figures/absorption-diagnostics/llama3-3b-off-windows.pdf)

![Llama-3.2-3B-Instruct：累计吸收差 Δ_D 随 step](figures/absorption-diagnostics/llama3-3b-off-drift.png)

[矢量 PDF](figures/absorption-diagnostics/llama3-3b-off-drift.pdf)

### Qwen3-1.7B / native Thinking-2507

![Qwen3-1.7B / native Thinking-2507：终点吸收量](figures/absorption-diagnostics/q3-1.7-native-thinking2507-absorption.png)

[矢量 PDF](figures/absorption-diagnostics/q3-1.7-native-thinking2507-absorption.pdf)

![Qwen3-1.7B / native Thinking-2507：累计迁移效率](figures/absorption-diagnostics/q3-1.7-native-thinking2507-efficiency.png)

[矢量 PDF](figures/absorption-diagnostics/q3-1.7-native-thinking2507-efficiency.pdf)

![Qwen3-1.7B / native Thinking-2507：早期—后期迁移效率](figures/absorption-diagnostics/q3-1.7-native-thinking2507-windows.png)

[矢量 PDF](figures/absorption-diagnostics/q3-1.7-native-thinking2507-windows.pdf)

![Qwen3-1.7B / native Thinking-2507：累计吸收差 Δ_D 随 step](figures/absorption-diagnostics/q3-1.7-native-thinking2507-drift.png)

[矢量 PDF](figures/absorption-diagnostics/q3-1.7-native-thinking2507-drift.pdf)

### Qwen3-1.7B / native Instruct-2507

![Qwen3-1.7B / native Instruct-2507：终点吸收量](figures/absorption-diagnostics/q3-1.7-native-instruct2507-absorption.png)

[矢量 PDF](figures/absorption-diagnostics/q3-1.7-native-instruct2507-absorption.pdf)

![Qwen3-1.7B / native Instruct-2507：累计迁移效率](figures/absorption-diagnostics/q3-1.7-native-instruct2507-efficiency.png)

[矢量 PDF](figures/absorption-diagnostics/q3-1.7-native-instruct2507-efficiency.pdf)

![Qwen3-1.7B / native Instruct-2507：早期—后期迁移效率](figures/absorption-diagnostics/q3-1.7-native-instruct2507-windows.png)

[矢量 PDF](figures/absorption-diagnostics/q3-1.7-native-instruct2507-windows.pdf)

![Qwen3-1.7B / native Instruct-2507：累计吸收差 Δ_D 随 step](figures/absorption-diagnostics/q3-1.7-native-instruct2507-drift.png)

[矢量 PDF](figures/absorption-diagnostics/q3-1.7-native-instruct2507-drift.pdf)

## 符号与导出字段

界面、悬停和静态诊断图统一使用以下符号，CSV 字段保留已有名称。$R$ 表示相对下降百分比，$B$ 表示原始风险的绝对相减，避免将绝对下降与累计吸收差 $\Delta_D$ 混用。

| 符号                                    | 含义               | 单位       | CSV 字段                              |
| --------------------------------------- | ------------------ | ---------- | ------------------------------------- |
| $K_T,K_D$                               | 当前训练／留出 RKL | nats/H1024 | `train_rkl`, `held_rkl`               |
| $K_T(0),K_D(0)$                         | 两池初始 RKL       | nats/H1024 | `train_initial`, `held_initial`       |
| $A_T,A_D$                               | 对数吸收量         | dex        | `train_absorption`, `held_absorption` |
| $\Delta_D$                              | $A_T-A_D$          | dex        | `gap`                                 |
| $\eta_D$                                | $A_D/A_T$          | 无量纲     | `efficiency`                          |
| $\Delta A_T[0,64],\Delta A_D[0,64]$     | 早窗吸收增量       | dex        | `early_train`, `early_held`           |
| $\Delta A_T[e-64,e],\Delta A_D[e-64,e]$ | 末窗吸收增量       | dex        | `late_train`, `late_held`             |
| $\eta_D[0,64],\eta_D[e-64,e]$           | 两窗增量比         | 无量纲     | `early_efficiency`, `late_efficiency` |

`efficiency_status`、`early_status`和`late_status`分别解释三个比值：`eligible`为可计算，`insufficient_budget`为窗口缺失，`train_rebound`为相应训练吸收增量非正，`small_train_progress`为正增量未超过 0.01 dex。初始检查点的累计比值因训练吸收量为 0 留空。新加的状态和初始 RKL 列用于解释读数，既有指标值及定义保持一致。

## MiMo：initial 与 teacher 双 held-out bank

MiMo-7B-Base 的 E71／E72／E73 正式 OPD 运行分别以 MiMo-7B-RL、MiMo-7B-RL-Zero、MiMo-7B-SFT 为 teacher。这里把 `public_initial`（初始 student 生成的固定轨迹）和 `public_teacher`（teacher 生成的固定轨迹）作为两套独立的 held-out 状态集，各自计算吸收量。MiMo 交互面板（本地文件 `figures/mimo-absorption-20261107/data/explorer.html`）可切换 bank、domain、teacher、指标与训练—留出相图，默认叠加三个 teacher。这里只使用正式训练数据，不纳入连续 batch 短训。

### 实际设置与指标

三组均为 full-vocabulary RKL、fresh AdamW、LR `1e-6`、weight decay `0.01`，batch128、rollout n1、response cap7168。E71 reuse32、25 outer／800 updates；E72/E73 reuse16、20 outer／320 updates。[原实验条件](../cases/mimo-position-kl.md#实验设置)保留完整来源。比较 teacher 阶段时优先看共同的 update320，E71 的后续 800 终点另看。

每套固定 bank 有 Math、Code、Science、IF 各 128 题。这里的 Avg 是四域 RKL 的等权均值，先合并 RKL 再计算相对下降或对数吸收量；它与主固定状态实验的 Math／Code／IF 加权 Avg 不同。MiMo 的 RKL 按题内有效 token 平均后再按题平均，单位 nat；主实验使用题数×1024 分母。两套 bank 在 OPD 过程中保持前缀固定，“held-out”在此指用于独立评分的状态集，bank 名称本身不证明与训练 prompt 严格无重叠。Teacher bank 的前缀还随 teacher 改变，因此跨 teacher 比较同时包含轨迹来源差异。

对每个 run、bank 与 domain，以自己的初始值计算：

$$
A_{b,D}(t)=\log_{10}\frac{K_{b,D}(0)}{K_{b,D}(t)},\qquad
R_{b,D}(t)=100\left[1-\frac{K_{b,D}(t)}{K_{b,D}(0)}\right],\qquad
B_{b,D}(t)=K_{b,D}(0)-K_{b,D}(t).
$$

训练侧使用首个更新前的 rollout RKL 作初值，在每个已评分 outer 取最后一次更新前的 RKL，记其对数变化为 $\widetilde A_T(t)$。Bank 读出对应参数 $\theta_t$，训练读出对应 $\theta_{t-1}$；CSV 同时保留 bank step、训练日志 update 与训练实际参数 step。相图按 outer 近似配对，不插值，不补出缺失的最终训练后读数。

$$
\widetilde\Delta_{b,D}(t)=\widetilde A_T(t)-A_{b,D}(t),\qquad
\widetilde\eta_{b,D}(t)=\frac{A_{b,D}(t)}{\widetilde A_T(t)}.
$$

仅当 $\widetilde A_T(t)>0.01$ dex 时报告比值。训练 rollout 状态随 outer 改变，且训练 loss 按 token 归约；这两个量是 rollout 参照的描述指标，不能与固定训练池定义的 $\Delta_D,\eta_D$ 混合排名。交互面板和 CSV 将它们标为 `proxy`。原始 RKL 随时间视图还叠加训练端读数，训练点放在其真实参数 step 上。

### 共同 update320 的结果

| Teacher | Bank    | Avg 初始 RKL | Avg 当前 RKL | Avg 相对下降 | Avg 吸收量 dex |
| ------- | ------- | -----------: | -----------: | -----------: | -------------: |
| RL      | Initial |     2.117732 |     0.357253 |       83.13% |          0.773 |
| RL-Zero | Initial |     0.070049 |     0.004287 |       93.88% |          1.213 |
| SFT     | Initial |     1.967694 |     0.408868 |       79.22% |          0.682 |
| RL      | Teacher |     0.406126 |     0.294315 |       27.53% |          0.140 |
| RL-Zero | Teacher |     0.064745 |     0.002795 |       95.68% |          1.365 |
| SFT     | Teacher |     0.463225 |     0.337572 |       27.13% |          0.137 |

**RL-Zero 的固定状态吸收在两种轨迹来源上均较充分；RL 与 SFT 的表现对 bank 来源敏感。** 同为 320 步，RL/SFT 在 initial bank 的 Avg 下降约 79–83%，在 teacher bank 仅下降约 27%；RL-Zero 则在两套 bank 都下降约 94–96%。因此“RL 都容易吸收”并不符合这组三个 teacher 的区分：带 SFT 历史的 RL 与 SFT 更接近，RL-Zero 与它们差异更大。三组的初始 RKL 与 teacher 前缀分布也不同，这个结果支持阶段相关的观察，尚不能单独归因于 SFT。

Math 上这个差异更清楚：RL/SFT 的 initial-bank 吸收量分别为 1.043／0.914 dex，teacher-bank 为 0.225／0.175 dex；RL-Zero 为 1.725／1.876 dex。只看 initial bank 会得到“RL/SFT 已消除约 88–91% 差异”的印象，但 teacher bank 上仍保留约 60–67% 的初始差异。这里量化的是对固定前缀的分布拟合，不能等同于相应比例的能力已经迁移。

**累计差与效率也会因 bank 改变而反转。** Math 在 320 步时，RL/SFT 的 initial-bank $\widetilde\Delta$ 为−0.249／−0.304 dex、$\widetilde\eta$ 为 1.314／1.499；teacher-bank 的对应值为 0.569／0.435 dex 和 0.283／0.286。Initial bank 的效率超过 1，表示相对自身初值下降得比所选 rollout 参照更多，不能解释为超过 100% 的知识迁移。RL-Zero 的 teacher Math-bank 效率约 0.886，但 Avg 约 0.645；Math 的拟合不能代表其他域。

### Initial-student bank 图

![MiMo initial bank 各域 RKL 随训练变化](figures/mimo-absorption-20261107/public_initial-rkl.png)

![MiMo initial bank 各域吸收量](figures/mimo-absorption-20261107/public_initial-log_absorption.png)

![MiMo initial bank 训练 rollout 与固定留出相图](figures/mimo-absorption-20261107/public_initial-phase.png)

![MiMo initial bank rollout 参照累计吸收差](figures/mimo-absorption-20261107/public_initial-gap_proxy.png)

![MiMo initial bank rollout 参照累计效率](figures/mimo-absorption-20261107/public_initial-efficiency_proxy.png)

### Teacher bank 图

![MiMo teacher bank 各域 RKL 随训练变化](figures/mimo-absorption-20261107/public_teacher-rkl.png)

![MiMo teacher bank 各域吸收量](figures/mimo-absorption-20261107/public_teacher-log_absorption.png)

![MiMo teacher bank 训练 rollout 与固定留出相图](figures/mimo-absorption-20261107/public_teacher-phase.png)

![MiMo teacher bank rollout 参照累计吸收差](figures/mimo-absorption-20261107/public_teacher-gap_proxy.png)

![MiMo teacher bank rollout 参照累计效率](figures/mimo-absorption-20261107/public_teacher-efficiency_proxy.png)

每张图同时提供同名 PDF；[完整指标 CSV](figures/mimo-absorption-20261107/metrics.csv)含 220 行，覆盖三个 run、两套 bank、五个 domain 汇总和全部已保存检查点。[来源回执](figures/mimo-absorption-20261107/provenance.json)保留输入与生成器哈希。生成器为 [mimo_absorption.py](../../../../research/studies/fixed_transfer/mimo_absorption.py)，在仓库根目录执行：

```bash
export FIGURE_VIZ_MPL=/mnt/tidalfs-bdsz01/dataset/llm_dataset/zkn_data/fht_data/dotfiles/.agents/skills/figure-viz/assets/mpl
uv run --no-project --with matplotlib --with numpy --with plotly python -m examples.mirroropd.research.studies.fixed_transfer.mimo_absorption
```

## 数据与复算

来源为[已验证逐快照数据](figures/student-paths/snapshots.csv)及[run 清单](figures/student-paths/inventory.csv)，覆盖 47 条 run。派生[指标 CSV](figures/absorption-diagnostics/metrics.csv)包含 3564 行：891 个检查点×四个留出面板；[来源回执](figures/absorption-diagnostics/provenance.json)记录输入哈希、窗口与分母阈值。每个检查点保留原始增量、残差、评分题数与缺失原因。静态图与 HTML 调用同一个指标函数。

源码为[quantify_absorption.py](../../../../research/studies/fixed_transfer/quantify_absorption.py)，在仓库根目录运行：

```bash
export FIGURE_VIZ_MPL=/mnt/tidalfs-bdsz01/dataset/llm_dataset/zkn_data/fht_data/dotfiles/.agents/skills/figure-viz/assets/mpl
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /root/.local/share/mirroropd/venvs/plotting/bin/python -m examples.mirroropd.research.studies.fixed_transfer.quantify_absorption
uv run --no-project --with plotly==7.1.0 python -m examples.mirroropd.research.studies.fixed_transfer.build_explorer
```

HTML 筛选后导出的 CSV：前三种新视图按各 run 实际终点导出四个域；偏离视图导出上限内全部检查点。缺失比值导出为空，状态列保留原因。原有视图继续导出原始五题池快照。
