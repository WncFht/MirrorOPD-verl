# 固定前缀教师迁移实验

该入口执行[跨源边界计划](../../../docs/experiments/teacher-transfer/transfer-mechanism/r1/2026-10-28-跨源蒸馏边界分阶段实验计划.md)的 TB0/TB1/TB2。只使用物理 GPU4–7，共九个分支，每分支 64 次全参数更新；首轮入口不执行续训、fresh 训练或版本扩展；第二轮入口见下文。运行产物位于仓库同级日志目录，由`common.py`明确指定。

<a id="code-path"></a>

## 与生产训练代码的关系

这套入口位于 MirrorOPD-verl 仓库中，但训练循环是为 TB0–TB2 新增的独立研究实现，直接使用 PyTorch、Hugging Face 模型和 FlashAttention2。它没有调用 E20 的生产 trainer/Ray/FSDP2 actor，也没有调用生产 full-vocab backend 或实现 MDPO/BPO 镜像目标。目录归属与运行路径需要区分。

| 部分                                   | 实际来源与调用                                                                                                                                  |
| -------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| 训练与梯度                             | 新增[train.py](train.py)，直接执行 AdamW；手写分块 full-vocab RKL 与 hidden/head VJP                                                            |
| 模型加载、S0 生成、teacher hidden 缓存 | 新增[native.py](native.py)，使用 HF 原生模型；借鉴旧固定前缀研究，未经过生产 rollout manager                                                    |
| 概率距离与 overlap                     | 直接复用已有[research/core/metrics.py](../../core/metrics.py)的`fixed_metrics`                                                                  |
| 数学基线                               | 新增[evaluate.py](evaluate.py)启动 vLLM 服务；复用[evaluate_qwen_moe.py](../qwen_teachers/evaluate_qwen_moe.py)的判分与聚合函数                 |
| verl 复用部分                          | 上述函数调用[数学判分](../../../../../verl/utils/reward_score/one_shot_opd/math.py)与[评估聚合](../../../../../verl/utils/metric/evaluation.py) |
| 运行与分析                             | `assets/prepare/launch/cross/analyze/plot`为本轮新增研究入口，保存在此目录                                                                      |

与 E20 相同的是研究的 full-vocab RKL 目标及部分指标/判分基础；不同的是固定 S0 输入、H1024、有效 batch16、并行和缓存实现。本轮的原生 HF 与分块梯度核验验证了研究实现本身，没有完成研究入口与生产 E20 路径在同一 checkpoint/同一 batch 上的端到端梯度对照。因此，结果用于固定前缀迁移研究，不能自动视为 E20 生产实现的性能或收敛复现。

## 执行顺序

从仓库根目录运行，解释器需具有 PyTorch、Transformers、FlashAttention2、vLLM 和现有数学判分依赖：

```bash
python -m examples.mirroropd.research.studies.boundary.assets
python -m examples.mirroropd.research.studies.boundary.prepare
python -m examples.mirroropd.research.studies.boundary.launch run --phase all
python -m examples.mirroropd.research.studies.boundary.analyze
```

`assets`下载固定 revision 的小文件与缺失权重；本地已有权重仅在其 SHA 与 Hub LFS 元数据一致时通过软链复用。`prepare`冻结输入、检查评测重合与 token-ID 语义。全阶段队列依次生成两套 S0 轨迹、核对 teacher 前向、自教师梯度、缓存 teacher hidden、进行真实单步验收、完成五集 avg@4 基线、九分支训练和 TB2 共同输入评分。

`launch`只停止已确认映射到 GPU4–7 的 gpu occupy。独立恢复守护检查 supervisor 身份，清理本次环境标签所属进程，然后在 GPU4–7 重新启动 gpu occupy。既有其他 GPU 任务不属于其管理范围。阶段失败保留原始日志；修复后可按`--phase`运行未完成阶段。已经有科学训练日志但没有完成标记的分支不会自动重训或追加，需先保存失败 attempt 并明确选择恢复方式。

## 数值与实现

学生使用 FP32 主权重、梯度和 Adam 状态，BF16 backbone autocast、FP32 head 投影；RoPE buffers 保留原生 FP32。全词表 loss 固定除以 16×1024，teacher 多出的 head 行保留在 softmax 分母中。分块 head 反向通过 hidden 叶节点累积 VJP，再对 backbone 反向；CPU 对照保存于运行目录的`numerical/cpu-vjp.json`。

1.5B 学生使用单卡 microbatch1 梯度累计；7B 学生的完整 FP32 训练状态超过单卡容量，使用两卡按层模型并行，仍每更新覆盖全部 16 条轨迹。该实现与计划中的优先数据并行不同，实际 protocol 记录并行方式；不能把硬件差异解释成唯一的学生容量效应。检查点 32/64 保存模型、tokenizer、Adam、RNG 和完整输入身份。

`fixed-000/016/032/048/064.json`是同一 256 题面板上的固定风险。首轮数学 development 面板用于分析，保留题池不会进入梯度；完整 holdout 不在此阶段提前评分。TB2 交叉评分在同一 64 道数学 development 题的两种 S0 轨迹上比较初始与终点参数。

## 结果汇总

TB0–TB2 已完成，结论与图表见[首轮结果](../../../docs/experiments/teacher-transfer/transfer-mechanism/r1/2026-10-28-跨源蒸馏边界分阶段实验计划.md#results)。九分支 576 次更新、18 个检查点、22680 条基线回答和 16 个交叉评分单元均已核对。第二轮使用独立目录和入口，见下文。

`analyze`核对更新计数、训练顺序、固定输入身份、checkpoint 的 Adam step、教师熵不变和完整基线回答槽位，输出原始风险、相对下降、按题 bootstrap 区间和跨来源评分。单训练 seed 的区间不能解释成训练方差。

绘图使用 figure-viz 的 matplotlib 风格；`FIGURE_VIZ_MPL`指向该技能的`assets/mpl`目录：

```bash
python -m examples.mirroropd.research.studies.boundary.plot
```

图输出到对应研究线的`figures/boundary-20261028/`并附审计结果，对应 CSV 在日志目录`analysis/`。执行状态以逐阶段完成标记和恢复后的 gpu occupy 状态为准。

<a id="checkpoint-storage"></a>

## Checkpoint 内容、用途与快速迭代

### 当前保存方式

[checkpoint 函数](train.py)在每个分支 32/64 更新后保存到仓库同级的`wandb-logs/derived/MirrorOPD-verl/teacher-boundary-20261028/checkpoints/<arm>/step032或step064/`。共有 18 个检查点，原始学生没有按 step0 再复制一份。

| 文件                          | 内容与用途                                                                                    |
| ----------------------------- | --------------------------------------------------------------------------------------------- |
| `model-*.safetensors`及 index | FP32 学生权重；重载模型、追加评分或继续训练的参数起点                                         |
| `config.json`、tokenizer 等   | 模型结构和输入协议；支持独立加载                                                              |
| `optimizer.pt`                | AdamW 的一阶矩、二阶矩、step 及 param groups；另含 RNG、分支/版本信息、输入轨迹身份和训练步数 |
| `complete.json`               | 保存完成及 Adam step 计数核对；不是权重或优化器状态本身                                       |

按文件逻辑大小统计，每个 1.5B 检查点约 6.62 GiB 权重＋13.24 GiB optimizer 文件，总计 19.87 GiB；每个 7B 检查点约 28.37＋56.74 GiB，总计 85.12 GiB。18 个合计 618.70 GiB，其中 optimizer 文件约占三分之二。详见[文件大小清单](../../../../../../wandb-logs/MirrorOPD-verl/analysis/teacher-boundary-20261028/analysis/checkpoint-storage.json)。统计不含原始模型、teacher hidden 缓存与日志，也不等同于文件系统压缩后的物理占用。

### Adam 状态的作用

AdamW 更新不仅由当前权重和当前梯度决定，还使用历史梯度的一阶矩`exp_avg`、平方梯度的二阶矩`exp_avg_sq`及用于偏差修正的`step`。只加载权重再建立 fresh Adam，相当于清空这段历史；后续更新方向和步幅通常不同，不能称为原训练轨迹的连续恢复。沿同一状态分叉比较或断点续训时，应保留这些状态；RNG、输入顺序、精度与环境也需对应。

Adam 状态在训练期间始终需要驻留内存或显存。不把它写入 checkpoint，只节省序列化、搬运、写盘和持久存储，不会自动消除优化器在训练中的显存占用。

首轮 `train.py` 没有通用 `--resume`；第二轮 `fast_train.py` 为 J/N/M 提供固定 step064 恢复入口，检查输入身份、原始 Adam 计数和评分一致性。

### 哪些分析需要哪些文件

| 工作                                                                   | 模型权重                                                        | Adam 状态                     |
| ---------------------------------------------------------------------- | --------------------------------------------------------------- | ----------------------------- |
| 重画已记录 RKL/JS/FKL/TV/entropy/overlap 曲线、bootstrap、旧批保持统计 | 不需要，读取逐题指标                                            | 不需要                        |
| 训练时用当前内存中的模型评分                                           | 不需要先写盘                                                    | 不需要保存到磁盘              |
| 训练结束后新增问题、指标或完整任务评测                                 | 需要对应权重，或重新训练                                        | 不需要                        |
| 当前`cross.py`的事后交叉评分                                           | 实际加载 step064 权重；初始点加载原始模型                       | 不加载                        |
| 当前`analyze.py`                                                       | 曲线来自日志；另检查 32/64 的`complete.json`及已生成 cross 结果 | 不加载`optimizer.pt`          |
| 沿原 Adam 历史继续训练或同状态分叉                                     | 需要                                                            | 需要；还要恢复 RNG/输入进度等 |

主分析确实是 RKL、JS 等固定风险，而不是重新加载 checkpoint 计算每一条曲线。32-step 权重和两次 Adam 快照并非这些曲线的必要输入。64-step 权重被本轮独立进程的交叉评分实际使用过；若把该评分放到训练进程释放模型之前完成，它也可以不依赖磁盘 checkpoint。

### 后续建议的轻量策略

| 方案                             | 保存内容                                                                  | 使用场景与代价                                                                    |
| -------------------------------- | ------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| 纯诊断                           | protocol、模型 revision、冻结输入、逐题指标、生成回答及分析源码           | 在内存中完成全部所需评分后不写权重；最快，但以后追加评分或续训需重跑              |
| 只保存最终权重，建议短程筛查默认 | 上述小文件＋终点 FP32 模型和 tokenizer，不保存 Adam 和中间完整 checkpoint | 保留后续任务评测与复查能力；本轮九分支的同等保存量约 103.18 GiB，比原策略少约 83% |
| 完整状态                         | 权重＋Adam＋RNG＋输入进度                                                 | 仅用于明确需要恢复/分叉的代表分支                                                 |

首轮入口保留原来的 32/64 完整保存协议。第二轮 `fast_train.py` 已使用最终权重策略，配套独立评测和分析入口，不新增 Adam 文件；尚未删除旧 checkpoint。纯诊断模式没有实现。

快速迭代还应复用协议和版本完全一致的原始模型基线、S0 银行与 teacher 缓存，避免每轮重复 TB0 的完整生成。保留已有逐题指标，后续仍可重画曲线；若只剩汇总 RKL/JS 而没有模型或完整概率，则不能事后任意增加新的概率指标。

代码中的`gradient_checkpointing_enable`是激活重算，用计算换训练显存；它与磁盘模型 checkpoint 不是同一件事。关闭磁盘保存不会关闭激活重算。最终权重继续保留 FP32，避免为了省空间静默转换 BF16 后改变近目标的读出行为。

## 位移诊断的效率优化

原四个 TB1 进程每步将完整 FP32 参数复制到 CPU 并计算实际位移，诊断耗时超过反向。后续分支保持 Adam 更新不变，改用 GPU 快照；7B 的 embedding/head 大矩阵仍放 CPU 以控制显存，最后再将这些快照分块送 GPU 计算平方差。没有降低评估频率、删除检查点或改变学习率、有效 batch 与 reuse。

同一次 FP32 Adam 更新的两种位移计算已核对，CPU/GPU 结果差为 0，证据在运行目录`numerical/displacement-measurement.json`和`numerical/chunked-displacement.json`。训练日志逐步保存`backward_seconds`、`snapshot_seconds`、`optimizer_seconds`、`displacement_measure_seconds`；新分支还标记`displacement_measurement`。已有进程没有热修改。不同诊断版本的源码快照分别保存，不能把整条 run 的墙钟差直接归为 teacher 效应。

1.5B 诊断从每步约 9–13 秒降到约 0.044 秒；7B 采用 CPU 大矩阵快照加 GPU 分块测量后，测量约 0.87 秒，峰值约 75.22 GiB。完整分阶段中位数见科学报告的效率表；7B 每个检查点约 28.4 GiB 权重和 56.7 GiB Adam 状态，写盘与固定评分仍会产生低 GPU 利用率。

## 后四卡占卡的独立管理

结束时默认占卡状态目录已被另一任务用于前四卡，恢复程序没有停止该任务。后四卡因此使用独立目录；实际状态保存于日志目录`gpu-occupy-restored.json`和`occupy-management.json`。旧 supervisor 的完成后重试循环在后四卡确认恢复后停止，处理记录在`control/recovery-fallback.json`。后续入口已支持独立状态选择，并复用占卡工具的 pidfd 兼容实现，避免当前 Python 缺少 pidfd 封装时失效。

在本机仓库根目录查询本轮后四卡占卡：

```bash
GPU_OCCUPY_STATE_DIR=/mnt/tidalfs-bdsz01/dataset/llm_dataset/zkn_data/fht_data/cluster_tools/gpu_occupy/bsud-quicksilver-hppk-13/uid-0-teacher-boundary-back4 \
  bash ../cluster_tools/gpu_occupy.sh status --json
```

该目录只管理物理 GPU4–7；默认目录对应的前四卡任务保持独立。CPU-only 子进程测试验证了错误启动身份不发信号、正确身份才停止进程，见`numerical/controller-signal-check.json`。

## 第二轮：三组续训、三位新 teacher 与任务回收

[第二轮设置与结果](../../../docs/experiments/teacher-transfer/transfer-mechanism/r1/2026-10-29-同架构教师吸收假说与续训设计.md)固定六个训练分支、384 次新增更新：J/N/M 从各自 step064 连续恢复至 128，DeepScaleR/BroRL/DeepCoder 从原始 R1 student 训练 64 次。Math teacher 只评原生 4K 基线。未安排 Polaris 续训或 MiniMath 旧池分支。

`round2_common.py` 冻结新增 revision、评测身份和预算；`round2_prepare.py` 检查词表、冻结 Math 原生 prompt。`round2_launch.py` 在物理 GPU4–7 上运行依赖队列，优化验收完成后才启动训练。评测覆盖四个新 teacher 和十五个 student 终点，旧的教师及 student-before 基线按同协议复用。

`fast_engine.py` 把当前批的完整 FP32 teacher log-prob 驻留 GPU，在 16 次 reuse 间共享；解析 FP32 head VJP 后，通过原生 backbone 反传，token chunk 不截断词表。`fast_profile.py` 对 S0 和 JustRL step064 的完整参数梯度与旧实现进行比较、计时，并保存验收结果。训练不再复制全参数计算每步位移，只保存梯度范数和计时。`fast_train.py` 仅导出最终 FP32 权重；`round2_analyze.py` 按题配对计算 GapRecovery、置信区间、分母跨零和负教师优势标志。

第二轮来源和输出都独立保留在 `teacher-boundary-20261029`，不会把新默认写回首轮。队列恢复时已完成项跳过，部分完成评测复用已记录回答。存在部分科学训练日志的分支仍拒绝自动重跑；需先保留 attempt，再明确恢复或重跑。最终权重没有新的 Adam 状态，不能直接声称支持 step128 的连续恢复。

```bash
python -m examples.mirroropd.research.studies.boundary.round2_assets
python -m examples.mirroropd.research.studies.boundary.round2_prepare
python -m examples.mirroropd.research.studies.boundary.round2_launch run
python -m examples.mirroropd.research.studies.boundary.round2_analyze
```

队列还要求输出目录的 `optimization-code-ready.json`：这是代码检查完成的条件，出现后才会运行 GPU profile；真正的训练条件是 profile 产出的 `optimization/complete.json`。所有失败都会保留日志，并由独立 guard 恢复后四卡占卡。
