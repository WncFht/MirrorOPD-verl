# Teacher-difference diagnostics

在固定 token bank 上重测 checkpoint 的全参数 PAC、跨 batch Gram、有限梯度空间覆盖，以及逐 prefix 的 live / frozen-request 响应。轨迹测量复用生产 `precompute_kl_rows` 与 `precompute_rows`，逐位置算子继续复用 `full_reverse_kl_result`；两者保留 teacher 原生词表的归一化；轨迹 PAC 与逐 prefix PAC 分别输出。

## 长响应机制探测协议

E99/E101 新测量使用同一包，配置中显式设置：

```json
{
  "defaults": {
    "cap": 7168,
    "group_size": 8,
    "checkpointing": true,
    "reduction": "valid_tokens",
    "source_policy": "independent"
  }
}
```

该片段合入下方完整 plan；模型、bank 与保存点仍由 plan 绑定。`reduction` 与 `source_policy` 编译进 task，因此改变协议会改变结果签名。旧 manifest 没有这些字段时按历史默认执行。新协议使用新输出目录，已有历史结果保留原位。

| 参数            | 默认与选择                                        | 含义                                                                                                 |
| --------------- | ------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| `reduction`     | 默认 `fixed_cap`；新实验选 `valid_tokens`         | 前者每题除 cap，后者除本题截断后的有效长度；随后按题/组平均                                          |
| `source_policy` | 默认 `training_batches`；独立题目选 `independent` | 前者依据 `batch_index >= stage/16` 判断未来训练 batch；后者只作独立 source，不产生 future-train 结果 |
| `cap`           | 历史默认 1024；新实验 7168                        | 截断响应长度上限，EOS 保留在有效响应中；支持单个或多个 EOS ID                                        |
| `--bank`        | 单 bank 可省略；多 bank 必选                      | 轨迹探针一次只处理一个 bank，不同 bank 使用独立 `--out`                                              |
| `--stages`      | 必须存在于 manifest 的非负保存点                  | 支持 640 及后续阶段，初始模型仍须唯一                                                                |

执行关系为：`batch` 编译与身份核验 → `trajectory` 选择题目/保存点 → `requests` / `response` 调生产 KL/VJP → `pac` 与各 probe 生成小统计和组均值 → report 读取结果。`reductions.py` 只负责归约规则与历史结果兼容；`inputs.py` 统一 EOS、题目身份与 source 暴露协议。新增探针复用这些层，新增模型由配置接入。

新面板把 source32 标为`split=train`，receiver96 标为`split=heldout`；每个 diagnostic group 为 B8。receiver 的前六组固定为 calibration，后六组为 test，记录顺序和题目清单一起冻结。训练侧 source 与 receiver 必须合并核对 prompt 不重复。`updates`与`adam`均可用`--source-indices 0 1 2 3`选择四个 source 组；负索引、越界与重复选择会在模型加载前报错。Adam 路径仍要求与 checkpoint 步数匹配的原生 optimizer state。

归约策略贯通完整轨迹 PAC、live/frozen、分量响应、alignment、小步与 Adam 评分。逐 prefix 的单位置探针继续保持自己的指标定义。原始 hidden/head 缓存不依赖归约方式，可复用；测量结果依赖 task 全部字段，不能跨协议复用。

响应输出记录每题 `token_count` 与 `denominator`。若每题分母为 d，随机符号梯度满足 E||g_sign||² = sum_t||g_t||²/d²；报告的 token 响应能量 U 需先按每题乘 d，再聚合。真正的 token 对角项直接聚合 E||g_sign||²；变长时不能用“平均 U / cap”替代。`aggregation_over_diagonal` 比较轨迹能量与该对角项；`R_agg` 使用 U 作分母，不能独自当作抵消程度。历史固定 H 数据无需长度字段；新有效长度结果缺字段会报错。

保留已有执行优化：生产分块 full-vocab VJP、tied-head 两路导数、bundle 共享 forward、范数 hook、CPU 组均值和分块 FP64 Gram。默认不保存逐题全参数梯度。完整 Gram 的 CPU 内存仍随组数和参数量增长；长响应的新增耗时与峰值显存由 B8 pilot 实测，CPU 回归不能证明 GPU 加速。

首轮实验及其条件补测见[机制探测与干预设计](../../../docs/experiments/teacher-transfer/checkpoint-dynamics/2026-11-13-机制探测与干预设计.md)。自然 checkpoint 窗口位移评分、早期 frame 内/外编辑尚待接入；本节描述的协议改造不代表这些实验已执行。历史 `report-trajectories` / `report-responses` 的主图仍面向 JustRL/Polaris，通用统计函数可供新 pair 使用，新 pair 出图需适配标签与配对身份。

## 批量使用

一个配置描述模型路径、固定 bank 和 checkpoint 序列；新增模型或保存点不需要复制 runner。路径相对于配置文件所在目录解析。

```json
{
  "models": {
    "initial": "/models/student-initial",
    "student320": "/models/student-step320",
    "teacher": "/models/teacher"
  },
  "banks": {
    "initial-bank": {"path": "bank.json", "cap": 1024, "group_size": 8}
  },
  "series": [{
    "id": "teacher-branch", "model": "family", "branch": "teacher-branch",
    "targets": ["teacher"], "banks": ["initial-bank"],
    "checkpoints": [
      {"step": 0, "student": "initial"},
      {"step": 320, "student": "student320"}
    ]
  }]
}
```

先编译并核验全部权重分片、bank 和 token 指纹，再执行显式分配的 GPU lane。以下 `broker.py` 与 `resource_guard.py` 由本机资源管理提供，路径需要显式传入。

```bash
python -m examples.mirroropd.research.studies.teacher_difference batch compile \
  --plan experiment.json --gpus 0 5 6 7 --out runs/study/manifest.json

python -m examples.mirroropd.research.studies.teacher_difference sweep \
  --manifest runs/study/manifest.json --repo "$PWD" --gpus 0 5 6 7 \
  --broker /path/to/broker.py --guard /path/to/resource_guard.py \
  --report --style /path/to/figure-viz/assets/mpl

python -m examples.mirroropd.research.studies.teacher_difference batch status \
  --manifest runs/study/manifest.json

python -m examples.mirroropd.research.studies.teacher_difference report \
  --manifest runs/study/manifest.json --style /path/to/figure-viz/assets/mpl
```

Sweep 对同一 manifest 加锁，每条 lane 顺序处理自己的任务；完成项直接复用，未完成项从已保存的 group 继续。退出后由 broker 验证进程、GPU context 与资源锁释放，再恢复占卡。修改输入后需要重新编译；原输出目录中的不匹配签名会报错，不会覆盖成新条件。`status` 显示已落盘进度，输入是否仍匹配在 worker 执行前再次核验。

`progress.json` 持续记录每项任务的完成状态。`--report` 在结果增加时更新 CSV 与图，并在退出前汇总；绘图日志写入 `report.log`，测量日志按 lane 保存。图按 model、bank、teacher 和 subset 分开，主图包含 loss、P、A、C、Xi、PXi、PXi/L 及辅助的 distinct-record 相干性；CSV 另外保留 batch Gram 统计。测量失败或最终绘图失败时 sweep 返回非零；修复后重跑同一命令即可复用完成项。

无人值守运行也可在配置的 `execution` 中设置 `report: true` 和 `figure_style` 的绝对路径；命令行 `--style` 优先。模型与数据输入编译后固定，绘图配置不改变测量签名。

大模型完整梯度会产生大量临时文件。默认分块 mmap 计算 FP64 Gram；主机 RAM 足够时，可在配置中设置 `"execution": {"resident_gram": true}`，保留 CPU 梯度并避免从网络文件系统重复读取。内存约为同时运行的各 lane 的 `group 数 × 参数数 × 4 bytes` 之和。结果保存完整 Gram 后删除临时梯度。

逐 prefix 探针复用相同加载、缓存和范数实现，并支持任意模型标签、bank、branch 与阶段选择：

```bash
python -m examples.mirroropd.research.studies.teacher_difference prefix \
  --manifest runs/study/manifest.json --model family --branch teacher-branch \
  --bank initial-bank --stages 0 320
```

该命令需在已分配 GPU 的 worker 环境中执行。相同 model/bank 下的各 target 必须共享初始 student。默认每个 split/domain 选 4 条轨迹的首、中、末位置；`--trajectories-per-stratum` 和 `--seed` 控制抽样。

## 实现边界

`runtime.py` 统一模型加载、teacher 缓存和梯度范数 hook；模型加载复用研究后端 `hf_models`，JSON/CSV 原子写入复用研究核心。Teacher 缓存绑定 checkpoint 路径、分片大小和修改时间、完整输入指纹及精度协议；这些元数据用于检测本地替换，不替代内容哈希或不可变模型来源。

`pac.py` 只负责生产 RKL 的梯度与位置级分解，`metrics.py` 负责矩统计，`geometry.py` 负责有限空间诊断。`stages.py` 是唯一批量测量 worker；`batch.py` 展开配置，`sweep.py` 管理资源生命周期，`results.py` 和 `report.py` 读取统一产物。`report_stages.py` 保留 R1 L、MiMo E91 与历史 N3 的专题分析，不参与模型执行。

有限 batch Gram 的谱不是参数 Fisher 或论文的局部 logit-kernel 特征架。有限空间 oracle 使用 receiver 梯度，是描述性上限；同源、SFT 或 RL 标签本身不决定可吸收性。

`PXi=P*Xi` 是固定统计面板的平均梯度平方范数，`PXi_over_L=PXi/loss_mean` 是该目标在欧氏梯度流下的瞬时相对下降率；它们不是测得的 Adam 跨 batch 收益。loss≤0 时后者为 NaN。

## 已有结果的联合分析

以下命令只读取结果并在 manifest 同级 `analysis/` 输出 PNG/PDF、CSV 和输入哈希，无需加载模型或分配 GPU。`--history` 指向包含两类 teacher 风险矩阵 CSV 及 `update-probes.json` 的原 L 分析目录。

```bash
python -m examples.mirroropd.research.studies.teacher_difference.analyze_transfer \
  --manifest runs/teacher-difference-stages-20261109/manifest.json \
  --history ../wandb-logs/MirrorOPD-verl/analysis/twenty-batch-pass-20261027
```

联合图区分原 B16 风险与 B8 梯度子面板；future source 随 checkpoint 移除已见批。Prefix 分层使用配对轨迹 bootstrap，并报告单题压力占比及 delete-one 敏感性。未保存 prefix Gram 时无法恢复分层 C 或跨题共同能量，输出保留该限制，不以逐位置范数推造内积。

## Logit 空间诊断

`signals.py` 维护 native RKL 的 logit 导数、softmax Fisher 衰减和累计 student 概率质量分层；`prefixes.py` 统一固定位置选择与初始请求缓存。PAC 和 prefix runner 保留原有导入兼容与缓存身份。

`logit_geometry.py` 用保留的 forward graph 提供参数 VJP、logit JVP 与 kernel 矩阵向量积，避免显式存储完整词表 Jacobian/kernel。`logit_probe.py` 复用同一模型和请求输入，计算 live/frozen 请求响应、主要 Ritz 谱及跨 prefix 响应。双重反向使用 math SDPA；模型精度继续为 FP32 master/BF16 backbone，因此每项输出同时保存算子不对称度、Ritz 残差和 JVP/VJP 对偶误差。

```bash
python -m examples.mirroropd.research.studies.teacher_difference logit \
  --manifest runs/teacher-difference-stages-20261109/manifest.json \
  --branch justrl --stages 0 80 240 \
  --request-cache runs/teacher-difference-stages-20261109/point-probe/r1-justrl \
  --out runs/teacher-difference-logit-20261109/main \
  --max-sites 8 --iterations 24

python -m examples.mirroropd.research.studies.teacher_difference.report_logit \
  --root runs/teacher-difference-logit-20261109/main
```

测量命令在已分配 GPU 的 worker 中执行。默认从原 prefix 面板中选择 Math 的中间位置，交替抽取 train/heldout，最多 8 个位置；`--positions all` 可纳入已有首、中、末位置；`--positions pressure` 取每题两个 teacher 初始压力最大候选位置的并集，选择结果在所有阶段与 branch 间固定。后者是高压力对照，不代表均匀 token 分布。同一目录内保持抽样、迭代次数与请求身份一致。输出 JSON 保存标量与完整跨位置响应表，PT 保存特征向量和原始请求，报告读取两者并校验身份，生成 CSV、PNG/PDF 与输入哈希。

输出是主要谱段的 Ritz 估计，收敛程度由残差判断；`unresolved_request_fraction` 表示前三维之外的请求能量。交叉响应同时保存幅度、与 receiver 请求的夹角及内积，并用参数梯度 Gram 核对。小面板用于定位机制，完整训练 B16 的实际 Adam 收益仍由原风险与位移探针评价。

### 执行成本与续跑

跨位置阶段按 source/请求字段遍历：只保留一个 source 的全参数 VJP，在同一设备上依次计算 receiver JVP。Source forward 在生成方向后释放，receiver graph 在每对计算后释放。全参数方向不写盘、不常驻 CPU，方向存储从位置数 × 字段数 × 参数数缩减为参数数；代价是重新构建 receiver forward。GPU 实测中，8 位置、24 次迭代的完整条件耗时 88–91 秒，分配显存峰值约 36.3 GiB；包含模型加载以外的谱、四类请求及全部有向位置对。2 位置 pilot 为 6.75 秒、35.9 GiB。

参数内积使用 `metrics.tensor_inner` 在原设备分块转 FP64，只在完整归约后同步一次；正反位置对复用相同 Gram 标量。主 PAC 的磁盘/内存 Gram 与分析端统一使用 `metrics.chunked_gram`，设备与块大小显式传入。Native 请求只计算 RKL 所需概率项，概率分层与 Fisher 统计共用同一组结果；生产训练 loss 仍走原优化算子。

每个 checkpoint 的 `.parts/` 保存带输入身份的任务锁、每位置谱/请求，以及每个 source/字段已完成的 receiver 标量。位置与位置对完成即原子写入；重启复用已完成部分，只重新计算未完成 source 所需的临时 VJP。`logit_site_done`、`logit_source_ready`、`logit_pair_done` 分别记录局部谱、source 方向和传播计算的耗时。聚合 JSON/PT 在全部完成后发布，协议标识为 `logit-kernel-v2`；v1 结果保留独立身份。

中止实验后保持暂停。重新启动任务与追加其他实验均由显式调度决定；占卡恢复由已有 broker/guard 处理。

### 大题目面板

`group_probe` 读取 R1 现有 bank 的全部 160 道 train Math 与 96 道永久 heldout Math，每题由固定种子均匀抽一个 response 位置。`prefixes.select` 的 `domains` 与 `position_mode` 供其他研究复用，默认三位置协议保持兼容。请求缓存保存独立输入身份；大面板与原小面板输出目录分开。

```bash
python -m examples.mirroropd.research.studies.teacher_difference groups \
  --manifest runs/teacher-difference-stages-20261109/manifest.json \
  --branch justrl --stages 0 80 240 \
  --out runs/teacher-difference-logit-20261109/large \
  --spectra-per-split 40 --iterations 24

python -m examples.mirroropd.research.studies.teacher_difference report-groups \
  --root runs/teacher-difference-logit-20261109/large
```

在共同初始化测两个 target，后续参数点测各自 native target。所有题目测局部请求、梯度范数；确定性抽取 train/heldout 各 40 题测主要谱。跨题部分按原始 B8 组构造平均梯度，保存全部组间精确 Gram。`prefix_batch_logits` 使用右侧 padding、attention mask 和逐行最后真实 hidden state；批量 VJP 与逐 prefix 梯度均值通过 CPU 显式对照。Gram 流式保留两个组方向，避免保存全体题目的参数梯度和逐位置二次方数量的 JVP。

`report_groups` 分开输出 prefix A 和 B8 A/C。Bootstrap 使用原 B8 组为单位，train 与 heldout 独立重采样，live/frozen 和各阶段复用相同抽样；区间描述固定 bank 的采样变异。跨组指标使用 basic bootstrap 区间，减轻重采样重复组使均值平方范数上偏的影响；cosine 与 source 自能量归一化使用两端独立 delete-one-group jackknife-t 区间；distinct-B8 相干性另报 delete-one-group jackknife 标准误区间。谱的题目中位数使用 percentile bootstrap。共同初始化只有一次测量，分析时复用给两条 branch。原 L 系列每组 16 updates，报告从原 bank 的 `batch_index` 排除已经训练过的 source 组，另列 future-train→heldout；updates240 剩 5 个 source 组。小面板的按题统计和 24/64 步谱比较由 `analyze_logit` 生成，delete-one 范围只作敏感性诊断。

位置抽样重复使用独立目录与 seed。`--spectra-per-split 0 --iterations 0` 跳过谱，仅测请求和 B8 Gram；新的 seed 必须使用新的请求缓存目录。读完两个目录的 `report-groups` 后，用以下命令比较重复结果，保持相同题目仅计 256 题：

```bash
python -m examples.mirroropd.research.studies.teacher_difference.compare_sampling \
  --roots runs/teacher-difference-logit-20261109/large runs/teacher-difference-logit-20261109/large-replicate \
  --out runs/teacher-difference-logit-20261109/analysis
```

### 完整轨迹 live/frozen 传播

`trajectory_probe` 在原 B8 分组上聚合全部响应位置，生成 live、frozen 两种请求的联合 Gram。每个轨迹/请求只执行一次模型前向与反向；生产 KL/VJP 的分块投影同时提供 loss、P 和请求间内积。Frozen 使用初始 student 的 FP32-master/BF16-autocast 隐状态与 head，不使用 BF16 teacher 副本替代初始 student。

```bash
python -m examples.mirroropd.research.studies.teacher_difference.trajectory_probe \
  --manifest runs/teacher-difference-stages-20261109/manifest.json \
  --branch justrl --stages 0 80 240 320 \
  --out runs/teacher-difference-logit-20261109/trajectory
python -m examples.mirroropd.research.studies.teacher_difference.report_trajectory \
  --root runs/teacher-difference-logit-20261109/trajectory
```

`--max-groups 1` 用于性能与数值试跑，必须使用独立输出目录。`--token-chunk` 控制词表投影的 token 块大小，默认 128。完整面板共 256 道 Math 题、32 个 B8；每条件 64 个 FP32 参数向量约占 455 GB CPU RAM，四任务并行需约 1.9 TB 可用内存。Gram 在 CUDA 上按参数块以 FP64 累积，输出包含跨字段内积，不写完整梯度到共享文件系统。

已完成条件按输入身份跳过；输出目录的 writer lock 禁止两个进程写同一条件。进度文件记录分组耗时、CPU 向量字节数和显存峰值。中断时只保留小型统计与进度，未完成条件会重新计算瞬态向量。Teacher 与初始 student 缓存按模型文件元数据、输入指纹和精度协议寻址。Frozen 标量是初始 KL 的诊断值，其 VJP 是当前 Jacobian 对初始请求的作用，不能解释为当前 KL 的梯度。分析表的 `L` 始终是当前 live KL；frozen 的 `q` 仅是该参照风险下的响应比值。

`alignment_probe` 复用相同 manifest、branch、stages 和 out 参数，单独测逐 token 的方向对照。它缓存当前 student 的隐状态，复用初始 student 与 teacher 缓存，不计算参数梯度；每条记录保存原子检查点。`report_trajectory` 合并相应的 `*.alignment.json`，检查请求压力与主轮一致，再生成逐位置方向图。统一 CLI 的对应命令为 `trajectories`、`alignment`、`report-trajectories`。

### Token 响应、题内聚合与分量传播

`trajectory.py` 共享固定题组、缓存与任务入口，已有完整轨迹和幅度对照也复用该入口。`response.py` 把 native、初始方向平行分量及随机符号请求组成有界 bundle，使用生产 `precompute_kl_rows` 和投影累积实现。一次 backbone forward 保留到该轨迹的全部 VJP 完成；不同 bundle 只重算分块词表投影，不重做 backbone forward。Teacher 额外支持、FP32 master 与 tied head 的双路径梯度保持一致。

```bash
python -m examples.mirroropd.research.studies.teacher_difference responses \
  --manifest runs/teacher-difference-stages-20261109/manifest.json \
  --branch polaris --stages 0 80 240 \
  --out runs/teacher-difference-response-20261109/main \
  --probes 16 --bundle 4

python -m examples.mirroropd.research.studies.teacher_difference report-responses \
  --root runs/teacher-difference-response-20261109/main
```

这些入口使用原 manifest 的 R1 Math 面板。`--max-groups 1` 在独立目录试跑；`--probes` 是每题独立随机符号次数，`--bundle` 控制同时常驻的 hidden/head cotangent 数。所有位置沿用固定 H；随机种子由全局 seed 与 row ID 决定，各 teacher/阶段配对。同一题的多次 probe 不作为额外样本。

每题只保存请求统计、native/parallel 梯度范数、交叉内积和随机符号梯度范数。两个字段的 B8 均值留在 CPU RAM，CUDA 分块计算联合 Gram；不保存 token×参数 Jacobian，也不将随机 probe 参数向量写盘。垂直分量由 native−parallel 的线性关系恢复。分析同时输出 A 的局部/聚合分解、非对角 token 能量、四个分量传播项及逐步增加 probe 次数的收敛表。

结果与部分统计绑定模型、输入、probe 协议和精度身份。完整条件完成后直接复用；中断后的全参数组均值没有持久化，未完成条件需重算。统计区间按原 B8 组抽样，随机 probe 的 Monte Carlo 标准误单独报告。

### 等位移真实更新验证

`update_probe` 对 native、parallel、perpendicular 三个方向施加相同参数位移预算，逐次从不可变 FP32 master 恢复参数，评分仍使用原始 RKL。Source 是预先指定的未见训练组；heldout 前半组用于选择步长，后半组保留为 untouched test。

```bash
python -m examples.mirroropd.research.studies.teacher_difference updates \
  --manifest runs/teacher-difference-stages-20261109/manifest.json \
  --branch polaris --stages 240 \
  --out runs/teacher-difference-response-20261109/updates \
  --source-indices 15 17 19 --displacements 0.003 0.01 0.03

python -m examples.mirroropd.research.studies.teacher_difference report-updates \
  --root runs/teacher-difference-response-20261109/updates \
  --geometry runs/teacher-difference-response-20261109/main
```

每个 trial 完成即保存，续跑跳过已完成评分。分析核对候选方向范数与完整面板 Gram，分别输出等位移比较和 calibration 选择后的 test 结果，保留实际收益与一阶预测的偏差。该探针研究原始分量方向的小步作用，不等同于恢复历史 Adam 状态后继续训练。

### 固定请求的范数对照

`norm_probe` 只计算 frozen 请求的完整梯度范数与随机符号范数，不创建组参数向量或 Gram，适合比较当前 Jacobian 对初始请求的单位压力响应：

```bash
python -m examples.mirroropd.research.studies.teacher_difference norms \
  --manifest runs/teacher-difference-stages-20261109/manifest.json \
  --branch polaris --stages 240 \
  --out runs/teacher-difference-response-20261109/frozen-norms --probes 8
```

逐题统计可续跑。`report-responses` 在结果根目录的同级 `frozen-norms/` 查找身份匹配的结果，补充 `frozen-factors.csv`；面板或 task 不匹配时不混用。若改目录布局，先把范数结果放入对应 study 的同级目录再生成联合分析。

Response 分析另外保存逐题能量份额和 leave-one-question-out 因子，原始全题统计保持不变。正因子和配对正比值用 percentile 题组区间，有符号传播及方向差值用 basic 区间；区间方法记录在产物中。真实更新分析输出相同 source/test 的方向配对差，零更新候选沿用真实 test baseline。

R1-1.5B、H1024 的已测开销：256 题、16 probes、两个字段的完整联合 Gram 每条件约 18–20 分钟，峰值分配显存约 49–50 GB，CPU 瞬态向量约 455 GB；norm-only 的 8 probes 约 6–7 分钟，约 31–33 GB 显存。该数字包含已命中缓存的测量，不是不同硬件或模型的预算保证。详细回执见 [response study](../../../../../runs/teacher-difference-response-20261109/README.md)。

### Teacher × student 交叉与历史 AdamW

交叉配置 `cells.json` 是四个 cell 的列表，每项指定 `student`、`teacher` 标签及 `trajectory`、`updates` 结果路径。同一参数点交换 target；固定同一 teacher 的初始请求再比较两个参数点。分析核对 heldout 题目顺序，输出 live/frozen 的 PAC、future-train→heldout 交叉内积和配对因子比值。

```bash
python -m examples.mirroropd.research.studies.teacher_difference analyze-matrix \
  --cells runs/teacher-difference-crossed-20261109/cells.json \
  --out runs/teacher-difference-crossed-20261109/analysis

python -m examples.mirroropd.research.studies.teacher_difference adam \
  --manifest runs/teacher-difference-crossed-20261109/manifest.json \
  --branch polaris --stages 240 \
  --out runs/teacher-difference-crossed-20261109/adam

python -m examples.mirroropd.research.studies.teacher_difference report-crossed \
  --cells runs/teacher-difference-crossed-20261109/cells.json \
  --adam runs/teacher-difference-crossed-20261109/adam \
  --out runs/teacher-difference-crossed-20261109/update-analysis
```

`adam` 在分配的 GPU 上运行，读取 student 目录的 `optimizer.pt`，核对参数数量、形状和历史 step；恢复全部参数组超参数、动量与二阶矩，clip=1，执行一次 PyTorch AdamW。Source 固定为未来 P/R/T 三个 B8；不是原 B16 训练步的重放。记录实际位移及每个 receiver 的一阶预测，分别评分位移的 0.25/0.5/1 倍；缩放位移不等于改变 Adam 学习率。每个 trial 从相同 FP32 参数恢复，保存后可续跑。Cal/test 继续使用原 48/48 题。

`report-crossed` 合并已有对角与新增非对角结果，输出全部预定干预、calibration 选择后的 test 收益和预测误差。Live/frozen Gram 缺少 parallel 字段时，分量更新仍报告实测收益，对应预测保留 NaN；native 预测照常核对梯度范数。

同一 `analyze-matrix` 命令也接受 `--manifest`，只读汇总任意已完成的 PAC 条件。若找到 model/bank/teacher 一致的唯一初始基线，按相同 heldout 题目计算剩余 KL 比值及配对区间。来源不完整时保留 NaN；不会把不同 teacher 的终点串成连续路径。Qwen 源集合标为固定 train-panel，未从 stage 编号推断历史训练是否见过这些题。
