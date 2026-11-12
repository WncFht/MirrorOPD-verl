# Teacher 变化与 OPD 吸收的谱分析

本包提供只读权重分析、固定 bank 的输出/表示残差 SVD、离线权重重构和原 teacher 的 native RKL 评分。研究配对、论文来源与实验解释见[谱实验设计](../../../docs/experiments/teacher-transfer/absorption/README.md)。上级命令不启动训练或 rollout；`extract`、`head`、`score` 默认使用单张 GPU，其余默认 CPU。

跨模型状态生成、参数干预、全层梯度与资源调度由 [absorption](absorption/README.md) 子包统一提供。

真实窗口更新的空间分解使用`decomposition.right_frame_components(delta, basis)`：`delta`为`W_b-W_a`，`basis`为早期累计更新的右正交基，形状为输入维度×k。算子验证维度、精度与正交性，用两次薄矩阵乘法返回原幅度`in/out`，不构造完整投影矩阵、不自动恢复范数。它与完整权重 base-frame 的`diag/off/null`分解分别使用；全 bank 窗口编辑与评分 runner 尚待接入。科学条件与非目标参数基线见[窗口功能拆分设计](../../../docs/experiments/teacher-transfer/checkpoint-dynamics/2026-11-13-机制探测与干预设计.md#a2-真实窗口更新的早期空间内外功能拆分)。

## 入口与输入

在仓库根目录运行：

```bash
python -m examples.mirroropd.research.studies.spectral --help
python -m examples.mirroropd.research.studies.spectral weights --help
```

HF checkpoint 必须是本地完整 safetensors 导出，支持单文件和索引分片。bank 沿用 `teacher_difference.inputs` 的 JSON/JSONL：每题含非空 `prompt_ids`、`response_ids`、唯一 `row_id` 和显式 `split`。按题划分 calibration/evaluation；历史 heldout 已参与选型时，不能再称未见测试集。下面 `$BASE`、`$TEACHER`、`$OPD`、`$BANK` 由调用者设为已核验的配对和 bank，不自动读取或合并历史 manifest。

## 权重分析入口选择

| 问题                                   | 入口                          | 保存内容                                      |
| -------------------------------------- | ----------------------------- | --------------------------------------------- |
| 多 teacher、全层、训练阶段的完整权重谱 | [scan](scan.md)               | 完整奇异值、δΣ/ρΣ/κ；可追加更新与 missing 谱  |
| base frame 的分量吸收与随机对照        | [weights](weights.md)         | diag/off/null、真实谱随机框架与 Gaussian 标量 |
| 实际导出、补回、删除后输出变化         | 下文 `reconstruct` 与 `score` | 独立模型导出、固定 bank native KL             |

纯谱扫描按同一 base 矩阵归组，在 teacher 与多个 student 阶段间复用一次 base SVD，帧仅驻留当前矩阵。完整诊断保留已有独立三元组接口和历史协议。两个入口都逐矩阵读取 safetensors，默认范围排除 embedding/head/norm，GPU 资源仍由已有 broker/resource guard 管理。

输出随机 SVD 拟合只读取 base/teacher，OPD 数组在评分时读取；校准总能量与捕获能量合并到同一次扫描。当前输出 SVD 使用 CPU 分块，校准 basis 尚未跨 OPD checkpoint 缓存。后续扩展优先复用不可变提取结果，避免为每个新统计量重新前向。

## 本轮统一 checkpoint 几何入口

E94–E111 的新测量使用 [absorption_dynamics/spectral.py](../absorption_dynamics/spectral.py) 组装已完成 pair 的九个 checkpoint 和双 bank 风险，计算由 [trajectory.py](trajectory.py) 完成。历史 `right_dynamics`、`update_dynamics` 与 `scan` 的协议和入口保留；新入口不调用历史汇总中的 step≤320 限制。科学设置只维护在[本轮测量协议](../../../docs/experiments/teacher-transfer/absorption/设置与指标.md#本轮-checkpoint-谱与右-frame-测量协议)。

职责只分两层：清单层负责完成状态、评分／导出验收、矩阵覆盖和显式 teacher 坐标身份；测量层只读一个矩阵的完整轨迹，复用已有主角、ISO 标量、转置 SVD 和 checkpoint 读取函数。一个更新 SVD 服务所有 rank，不为每个指标重复分解。只缓存 checkpoint 索引及当前矩阵的小型 frame；阶段对比较重读 tensor，避免同时驻留九份完整更新。输出压缩 JSON，不持久化 U/V 或权重副本。

```bash
python -m examples.mirroropd.research.studies.absorption_dynamics.spectral prepare \
  --root <formal-training-root> --pairs E94 E95 E96 E97 E98 E99 E100 E101 E102 \
  --teacher-geometry E94 E95 E96 E97 E98 E99 E100 E101 \
  --manifest <new-manifest.json>
python -m examples.mirroropd.research.studies.absorption_dynamics.spectral run \
  --manifest <new-manifest.json> --output <new-output-directory> \
  --devices cpu --workers 2
```

GPU 分析在既有 broker／guard 已分配的四卡内运行，改用 `--devices cuda:0 cuda:1 cuda:2 cuda:3 --workers 1`；编号为 guard 映射后的逻辑设备。每卡一个进程，精确 FP64；命令本身不申请或抢占 GPU。先验收一张卡上的窄矩阵与 CPU 数值一致性，再运行全量。CPU 方阵与矩形矩阵已与历史 frame 指标、直接 SVD 和参数残差恒等式对照；E94 的实际 k_proj 九阶段数据已通过 CPU/GPU 对照。全量 GPU 测量正在执行，吞吐按实际完成矩阵计时，窄矩阵的速度不外推到 MLP。

执行策略由 `--driver gesvd|gesvdj` 显式选择；省略时沿用 PyTorch 默认。solver 与引擎版本保存在 `execution.json` 和新测量记录的 `execution` 中，科学 manifest 保持不变。切换 solver 前先按实际形状核对完整奇异值、谱标量、主角分布和捕获能量，性能验收见[全层谱计算加速](../../../docs/infra/performance/spectral/2026-11-12-全层谱计算加速调研.md)。

调度按设备空闲顺序领取下一个矩阵，每卡最多一个在途任务；CPU 的 `--workers` 控制在途任务数。在输出目录创建 `STOP` 后，活跃矩阵完成并落盘，停止领取新任务；`exit.json` 记录 `stopped=true`、`exit_code=2`，CLI 非零退出。后续新建输出目录并通过 `--resume-from` 续接。这个文件只控制指标测量，不能用于训练队列。rank 投影统一计算最大 rank，再以列能量前缀和生成各 rank 的结果；不同 rank 的交叉 Gram 矩阵也共享一次乘法。

清单和输出目录均须新建。失败保存 `error-*.json` 并让执行命令非零退出，成功项保留；`exit.json` 的 complete 必须等于 total 且 failed=0。续跑使用 `--resume-from <previous-output>`，要求完整 manifest、逐矩阵任务身份、FP64 精度和 checkpoint 覆盖一致；已完成压缩记录复制到新目录，`reused.json` 保留来源，原 attempt 保持原样。多次中断可以传入多个同协议目录，相同任务出现冲突结果时拒绝续跑。持久化结果含输入 tensor 哈希、每阶段完整奇异值、所有阶段对角度、未来增量捕获、teacher 参数几何及计时；图和研究汇总须从这些完成记录派生，不能直接套用历史聚合器。

逐矩阵结果可用 `examples.mirroropd.research.studies.absorption_dynamics.geometry_readouts` 汇总和绘图，参数为 `--source`、`--data`、`--figures`，可选 `--watch`。派生读出保留输入哈希、精确标量、65 点谱能量 CDF、阶段对和增量捕获；只有七个投影齐全的层进入当前曲线，覆盖数量写入 `coverage.json`。跨模型比较须等对应 pair 全部矩阵完成。

## 累计更新的阶段子空间

`python -m examples.mirroropd.research.studies.spectral.update_dynamics --manifest <manifest.json> --output <new-output> --workers 3` 在 CPU FP64 上逐矩阵测量 checkpoint→final 的更新左右子空间，默认入口不使用 GPU。manifest 的 `tasks` 显式指定 `base`、`tensor`、按训练进度排序的 `points`（最后一个为 endpoint）、`ranks` 和每进程 `threads`。不同形状 checkpoint 会拒绝；输出逐矩阵 SHA、完整更新奇异值、能量、左右重叠、随机 k/d 基线及 early 空间捕获率。`output`必须是新目录，失败保存独立 error 记录，已完成项保留。最终查看 `exit.json` 的完成数与失败数，不能仅检查进程退出。

任务可指定 `weight_geometry: true`，追加完整权重的奇异值、相对初始权重的 δΣ／ρΣ／κ，以及左右 top-k 框架重叠。可选的 `teacher_path` 追加同坐标 teacher 更新的幅度、余弦和残差；跨尺寸 teacher 不配置该字段。仍在训练的轨迹只冻结已完成快照，把最后已保存点标为当前参考点，不能当作训练终点。

完成后执行 `python -m examples.mirroropd.research.studies.spectral.summarize_update_dynamics --source <output> --output <report-data>`，验证更新能量、最终自重叠、最优 rank-k 捕获上界和参数残差恒等式，输出压缩逐矩阵数据与模块汇总。模块统计使用矩阵中位数；teacher 整体几何按选定矩阵的能量与内积汇总，不能把层级样本数当作独立模型数。

`collect_update_readouts` 将同 manifest 的历史 fixed-H1024 逐题读出和预先冻结的新训练 metrics 整理到报告附件。入口要求显式传入 `--manifest`、`--history`、`--frozen-metrics`、`--output`；按题、bank 与长度验证历史面板，新的 outer-step 单位保留原义，生成内容不回写运行中的 metrics。

只对有非零中间更新的条件解释 early 动态。请求 rank 超过矩阵短边时显式裁剪；覆盖完整左／右环境空间导致的重叠 1 不作为锁定证据。精确 SVD 采用逐矩阵流式读取，单矩阵计算成本受形状影响，应先用窄矩阵检查吞吐，再扩大模块范围。研究结果见[阶段子空间与能量曲线](../../../docs/experiments/teacher-transfer/absorption/跨模型结果.md)。

### 右子空间的密集阶段角度与增量捕获

`right_dynamics` 对同坐标 checkpoint 执行 CPU FP64 精确 SVD。每项任务显式给定 `condition`、`base`、`tensor`、按进度排序的 `[step,path]` 列表 `points`、`ranks` 和 `threads`。输出所有阶段对的完整主角分布、均值／p90／最大角、重叠率、随机基线与弦距离；另将相邻 checkpoint 增量投影到此前保存的累计更新空间，给出捕获能量及其相对最优 rank-k 捕获的比例。`weight_angles: true` 时增加完整权重相对 initial 的主角。

```bash
python -m examples.mirroropd.research.studies.spectral.right_dynamics \
  --manifest <manifest.json> --output <new-measurement-directory> --workers 6
python -m examples.mirroropd.research.studies.spectral.summarize_right_dynamics \
  --source <complete-measurement-directory> --output <new-report-data-directory> \
  --training <three-source-training-root>
```

两个输出目录均须新建；失败保留独立 `error-*.json`，聚合拒绝未完整完成的结果。逐矩阵数据保留 tensor SHA、形状、阶段和耗时；图层只读报告附件。当前聚合核验参考 step≤320、子空间自重叠、角度范围及捕获不超过最优 rank-k 上界。可选 `--training` 只读取已完成评估事件的风险，不启动训练或前向。单矩阵输入验证包括已知角度、子空间内部换基不变性与真实 checkpoint smoke。

### 首遍风险与多模型几何汇总

`analyze_first_pass`只用 CPU，读取报告附件、历史 clean 索引及其指向的原始逐题 JSON。每个 run 只读取一个来源中的首遍≤320 步；校验 128 道题的身份、group、长度与 bank，拒绝重复题。输出 14 条件 H1024 风险、16 条件端点几何和六条首遍更新空间轨迹。MiMo 风险留空，H128 保留为独立功能协议。

```bash
python -m examples.mirroropd.research.studies.spectral.analyze_first_pass \
  --report examples/mirroropd/docs/experiments/teacher-transfer/absorption \
  --clean examples/mirroropd/docs/experiments/teacher-transfer/figures/fixed-state-clean-20261107 \
  --output runs/spectral-first-pass-new
```

输出必须为新目录，保存逐题压缩数据、来源与输入哈希、按题 bootstrap。报告相邻的`data/first-pass-20261112/plot_first_pass.py`只依赖冻结附件绘图。历史 continuation 分析源码保留，当前主分类和动态图不再使用其晚期读出。

## 各 checkpoint 的初始谱恢复与完整 bank 评分

`spectrum_bank` 读取已完成几何 manifest，绑定九个 checkpoint、原训练的 512 行双 bank RKL 记录与生成轨迹。输入准备核对每个 bank 的内容 SHA256、身份和原始 mask，要求 initial／teacher 各 Math128、Code64、IF64；完整有效响应上限 7168。命令只读现有导出和 bank，不启动训练、生成或 accuracy。

```bash
python -m examples.mirroropd.research.studies.absorption_dynamics.spectrum_bank prepare \
  --geometry-manifest <geometry-plan.json> --output <new-input-directory>
python -m examples.mirroropd.research.studies.absorption_dynamics.spectrum_bank run \
  --plan <new-input-directory>/plan.json --pairs E94 E95 \
  --devices cuda:0 cuda:1 --output <new-score-directory>
```

设备由已有 broker／guard 映射，一卡独立处理一个 pair，pair 内复用教师 hidden 与初始奇异值。全层七类矩阵以精确 FP64 执行 $U_t\Sigma_0V_t^\top$，转回原生 BF16 前向；其他参数保留当前 checkpoint。评分复用主线 `score_distribution_rows` 的稳定全词表 RKL，保留该 pair 的 distribution 定义。每阶段在同一 HF／FA2 执行路径重测未编辑与恢复谱条件；另保存与训练 FSDP 逐题分数的误差，执行差异不归因于换谱。双 bank 三域各一条响应重复无编辑前向，要求最大逐题 RKL 差 ≤1e-8；全层 SVD 重构相对误差要求 ≤1e-10。

可选执行优化由 `--reference-cache-mib <MiB>`、`--token-chunk <tokens>` 和 `--edit-method svd|gram` 控制，默认保持磁盘 hidden、256 token 与 SVD。设备缓存以明确字节上限保存经身份验证的教师 hidden，并跨阶段保留 head；使用 LRU 淘汰，不落盘 logits。`gram` 在较小一侧的 FP64 Gram 矩阵上特征分解，以完整谱比例恢复权重，不做截断；Gram 条件数超过 1e8、非正定或特征残差不合格时回退原 SVD。执行参数绑定 pair 协议，续跑不得改变。切换新后端前，用 `benchmark_spectrum_bank --plan <plan> --pair <pair> --output <new-directory>` 对全部矩阵以及双 bank 三域的长短完整响应验收；基准包含 256/512/1024/2048 token 与缓存开关计时。

`--smoke` 仅测首个 checkpoint、六条完整轨迹，仍编辑全层；`--smoke --probe-only` 只验收重复无编辑前向。正式评分逐行落盘，重新调用 `run_pair` 使用相同 pair 输出目录时校验协议并续接；外层 `run` 输出须新建。输出根或 pair 目录中的 `STOP` 在下一轨迹／矩阵边界停止，保留缓存和完成记录。教师 hidden 缓存仅在 pair 完整完成后删除，不保存全词表 logits 或编辑后模型。

`spectrum_bank_report --source <score-directory> --report <report-directory> --watch` 把完整阶段的逐题结果、位置分段、编辑审计和曲线迁入报告相邻附件，生成绝对 RKL 与成对差值图。正式汇总拒绝 smoke；报告见[恢复初始奇异值后的完整 bank RKL](../../../docs/experiments/teacher-transfer/checkpoint-dynamics/谱恢复与RKL.md)。

## 固定 bank 输出与表示

每个模型单独提取一次，之后不同分析复用缓存：

```bash
python -m examples.mirroropd.research.studies.spectral extract \
  --model "$BASE" --records "$BANK" --precision master \
  --positions 16 --cap 1024 --layers 0 15 \
  --output runs/spectral-analysis/base
python -m examples.mirroropd.research.studies.spectral extract \
  --model "$TEACHER" --records "$BANK" --precision native \
  --positions 16 --cap 1024 --layers 0 15 \
  --output runs/spectral-analysis/teacher
python -m examples.mirroropd.research.studies.spectral extract \
  --model "$OPD" --records "$BANK" --precision master \
  --positions 16 --cap 1024 --layers 0 15 \
  --output runs/spectral-analysis/opd
python -m examples.mirroropd.research.studies.spectral residuals \
  --base runs/spectral-analysis/base --teacher runs/spectral-analysis/teacher \
  --opd runs/spectral-analysis/opd --rank 16 --calibration train --evaluation heldout \
  --output runs/spectral-analysis/output-svd
```

`master` 为 FP32 参数、CUDA backbone BF16 autocast、FP32 head，无梯度与 optimizer；`native` 为 CUDA BF16 参数、FP32 head。CPU 均为 FP32。每题默认最多均匀取 16 个预测位置，包含首尾；预测位置 0 对应 prompt 末 token。该采样用于谱诊断，不能代替 H1024 全位置风险。选中层必须存在；各模型使用相同位置数和 cap。`--layers` 可省略，只保存 post-final-norm 状态；指定层保存 post-block/pre-final-norm 状态。

缓存使用 FP32 `.npy` memmap，完成标记在全部写入后发布；记录 token 映射、题目/位置指纹、模型分片 SHA256 和输出数组 SHA256。已有输出目录拒绝覆盖；失败后保留不完整目录，改用新目录重试。`residuals` 校验完整性与 logprob 内容摘要。Base/OPD 词表和维度须一致；teacher 可有额外原生 head 行，但 token ID 必须覆盖且对齐 student。跨 hidden 尺寸仍可比较输出谱；不同 token 编码不在此适配器的支持范围。

默认 residual 为 base Fisher 白化后的 log ratio。随机 SVD 只拟合 calibration，报告该子空间未解释的 Frobenius 比例；这包含截断误差，不是单独的算法误差上界。`--oversample`、`--iterations` 控制近似，不能用截断结果报告全谱 numerical rank。Heldout 按固定谱 band 和未解析 complement 评分，另有随机同维子空间对照；每题权重相等，吸收系数允许负数及大于 1。`scores.json` 保留逐题结果，供按题 bootstrap；不会将 token 当作独立 seed。

`--metric logit` 使用均匀中心化 logit 差，`--metric probability` 使用原生概率差在 student 支持上的限制，作为敏感性分析。Fisher 与中心化 logit 排除了 teacher 额外支持的常数偏置；另报的 sampled native RKL 保留完整 teacher partition function。这里的 RKL 按采样位置均值，完整 fixed-H 结果由 `score` 给出。

```bash
python -m examples.mirroropd.research.studies.spectral representations \
  --base runs/spectral-analysis/base --teacher runs/spectral-analysis/teacher \
  --opd runs/spectral-analysis/opd --layer final --rank 16 --shared-coordinates \
  --output runs/spectral-analysis/hidden-svd
python -m examples.mirroropd.research.studies.spectral head \
  --base runs/spectral-analysis/base --teacher runs/spectral-analysis/teacher \
  --opd runs/spectral-analysis/opd --base-weights "$BASE" --teacher-weights "$TEACHER" \
  --rank 16 --shared-coordinates --output runs/spectral-analysis/head.json
```

表示与 head 比较要求已核实同源、同 hidden 坐标；`--shared-coordinates` 是调用者声明，不会从等 shape 推断谱系。`representations` 在 calibration 上做精确 CPU SVD；`head` 对 base 中心化 head 做精确 FP64 SVD 并按 split 报告强弱方向能量和 head 恒等式误差，默认 CUDA，可选 CPU。Head 的强弱阈值由 base head 本身确定。该普通 head 谱不是 Fisher 谱。Head 分析拒绝不同词表/hidden shape。

## 离线重构与 native 风险

```bash
python -m examples.mirroropd.research.studies.spectral reconstruct \
  --base "$BASE" --model "$OPD" --mode top --fraction 0.01 --restore-norm \
  --untouched base --output runs/spectral-analysis/top01
python -m examples.mirroropd.research.studies.spectral reconstruct \
  --base "$BASE" --model "$OPD" --anchor "$TEACHER" --mode patch-off --beta 1 \
  --output runs/spectral-analysis/patch-off
python -m examples.mirroropd.research.studies.spectral score \
  --model runs/spectral-analysis/patch-off --teacher "$TEACHER" --records "$BANK" \
  --cache runs/spectral-analysis/native-cache --cap 1024 \
  --output runs/spectral-analysis/patch-off-risk.json
```

`top` 按每矩阵 `ceil(fraction × min(shape))` 保留奇异分量。省略 `--restore-norm` 得未缩放版本；打开后恢复该矩阵原更新范数。`top` 命令默认未选参数回到 base，匹配论文 attention/MLP-only 重构；删除/补回默认保留 target。Python API 的 `untouched` 默认始终是 target，复现论文时应显式指定 base。

`delete-{diag,off,null}` 为 target 减去该更新分量；`patch-{diag,off,null}` 为 target 加上 anchor−target 的对应分量，均用 base frame，支持 `--beta`。`--control gaussian` 将实际干预替换为同形等范数 Gaussian；`--control random_frames` 保留实际干预的奇异值并随机化框架，`--seed` 固定随机源。权重坐标的三方 tensor key/shape 必须匹配；共享绑定的 embedding/head 允许 safetensors 省略 `lm_head.weight` 别名。diag 是一阶改谱分量，off 不等于有限纯旋转。默认 scope 排除 embedding/head/norm；embedding/head 编辑直接拒绝，以保持绑定端点。输出保留架构 config 与 tokenizer，将默认 `torch_dtype` 设为 FP32，避免 `dtype=auto` 重新掩盖微小更新；选中矩阵存 FP32，每 tensor 一个 shard，`spectral-intervention.json` 最后标记完成。输入只读，输出必须是新目录。

`score` 复用已有模型加载、teacher cache 与生产 `full_reverse_kl_result` 的 streaming 后端；每题全部 response 位置的 native KL 求和后除以固定 H。Teacher 额外词表行保留在归一化中。不同 beta 的选择应只看 calibration，再对独立 evaluation 评分；工具不自动搜索 beta 或运行任务生成评测。

## 资源与验证

资源表为完整分析的预算估算；32 题真实模型提取已测 allocator 峰值，数据见[谱实验报告](../../../docs/experiments/teacher-transfer/absorption/README.md)。完整 head SVD 的峰值与吞吐尚未实测。H800 查询显示本机有 8 张约 80 GB 卡，表中的卡数是每个任务的需求，不代表这些卡空闲。

| 工作                                   | GPU 需求                    | 主要资源                                                       |
| -------------------------------------- | --------------------------- | -------------------------------------------------------------- |
| 权重谱、diag/off/null、离线重构        | 最低 0 张；可选 1 张加速    | 逐矩阵精确 SVD；CPU 可先预留 32–64 GB RAM，不包含全模型驻留    |
| 输出 SVD、hidden SVD、投影评分         | 0 张                        | 默认 block32、rank16；建议 16–32 GB RAM，磁盘读取往往是瓶颈    |
| 1.5B/1.7B 固定 bank 提取与 native 评分 | 1 张；24–48 GB 为待验证预算 | FP32 模型权重约 6–7 GB，另加 head、activation、CUDA workspace  |
| 7B teacher 提取                        | 1 张 48–80 GB 较稳妥        | native BF16 权重约 14 GB；若改 master 为约 28 GB，再加前向开销 |
| 精确中心化 head SVD                    | 0 张可运行；建议 1 张 80 GB | 大词表 FP64 SVD 的 U、workspace 和三份 head/hidden；先单独测量 |

首轮建议串行占用 **1 张 H800 80 GB**；两张可并行不同配对，四张或八张只是提高吞吐，无需模型并行。全模型训练、optimizer、full-gradient PAC 不包含在这些预算内。原有 25–69 GB 梯度探针峰值不能套用到这里的 no-grad 分析。

输出磁盘约为 `题数 × 每题采样位置数 × native vocab × 4 bytes × 缓存模型数`。以 256 题、16 位置、152064 词表估算，每模型约 2.49 GB（三模型约 7.47 GB，十进制）；全取 H1024 则约 159 GB/模型。先用 16 位置、32 题和单层 SVD 检查数值/峰值，再扩展完整 bank，避免一开始缓存所有位置。三个 cache 的 logprobs 由 CPU 分块读取，不整体转到 GPU。

CPU 验证入口：

```bash
mkdir -p runs/spectral-implementation-20261109
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest tests/research/test_spectral.py -q \
  --basetemp=runs/spectral-implementation-20261109/pytest
```

测试覆盖正交分解、方阵 null、ISO 缩放、top-rank/范数恢复、随机框架、分片导出与只读输入、输出完整性、SVD 对标、heldout 隔离、额外 teacher 支持、head 恒等式和小模型预测位置。32 题 GPU 提取和单矩阵精确 SVD 已完成；跨模型科学结果与任务读出见谱实验报告。Activation oracle、合成 band teacher 的训练接入、早期子空间约束训练与自动绘图不在此实现中；它们按实验设计中的功能证据条件再接入。

## 全层扩展的成本复算

`costs.py` 只读历史逐矩阵计时，按形状和参考缓存状态匹配成本，并重放整条件／矩阵任务的并行调度。不同工作负载分别报告，缺少同形状数据时保留空值；不会启动 GPU。入口及实测结果见[ISO 计算预算](../../../docs/experiments/teacher-transfer/absorption/附录-定义与复算.md)。
