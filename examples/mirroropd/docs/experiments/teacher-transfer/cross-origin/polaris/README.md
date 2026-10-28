# Polaris 跨源蒸馏：E48 Reverse KL / E49 Forward KL

## 背景与 motivation

区分当前批拟合、跨前缀迁移与真实任务收益，并测试KL方向和state覆盖能否解除平台。 综合研究问题见[跨状态迁移与能力获取](../../README.md)。

研究代码与命令已统一迁至 [research 包](../../../../../research/README.md)；公共算法、缓存身份、执行入口与维护约定见该文档。

更新：2026-10-28。正式E48 `grcuotmi` 已完成256个outer、4096次真实更新，本地exit=0、云端finished。末段首次/第16次KL约为0.1875/0.1628。完整日志、RL/SFT迁移文献及Forward/Reverse KL短训练对照见[后期平台与KL方向分析](2026-10-27-E48后期平台与Forward-KL及知识迁移.md)。此前58-outer分析保留其原冻结边界。

研究入口：[跨源蒸馏泛化研究总览](../../transfer-mechanism/2026-10-27-跨源蒸馏泛化研究总览.md)汇总五个已完成系列、47个训练分支与3384次更新。跨批矩阵、来源对照、L 设置及相应图表已迁入[双来源 Rollout 跨批次专题](../../transfer-mechanism/README.md)；二十四幅既有结果图全部保留，四份旧文及其归档、跳转文件仍已删除。

新增：[40批输入、20批单遍训练的完整结果](../../transfer-mechanism/cross-batch/2026-10-28-跨批次迁移与单遍训练结果.md#long-pass)。两teacher各320更新，未回访旧批；含全40批矩阵、实际Adam首步与fresh访问分布分解。

待运行：[双来源四分支重跑协议](../../transfer-mechanism/rollout-sources/2026-10-28-双来源Rollout四分支实验方案.md#protocol)。J/P 各比较 S0 与 teacher 轨迹训练，共1280次计划更新；每批评估40批×两来源，扩大fresh面板并保存全部批次checkpoint。仅文档定稿，等待用户通知有卡后实现和执行。

正式 E49 已完成：[KL 方向与跨源迁移分析](2026-10-28-E49方向对照与跨源迁移.md)同时比较双向 KL、按回答累计的原生 RKL、JS/TV 与任务覆盖。验证截断和答案缺失减少，但新批分布拟合仍处于平台，AIME25 候选覆盖未扩大。

## 运行与研究入口

[OPD 能力筛选假说实验](../../transfer-mechanism/capability-filter/2026-10-28-OPD能力筛选假说的受控实验.md)已完成：18 个训练分支、1984 次更新，以及数学、IF、温度和截断续写评估。RKL 能记住不泛化的随机映射；固定分布拟合与真实任务表现并不等价；被计为“遗忘”的 12 个截断失败中有 10 个续写恢复。报告包含 10 幅图。该实验与下面的双来源 R 重跑协议分开记录。

| 要做的事 | 入口 |
| --- | --- |
| 执行双来源跨批次重跑 | [完整设置、指标、训练与交付协议](../../transfer-mechanism/rollout-sources/2026-10-28-双来源Rollout四分支实验方案.md#execution)，尚未启动 |
| 复查 E49 配置与启动方式 | [Forward KL实现、支持集、指标和命令](../../../../infra/numerics/full-vocab/forward-kl.md) |
| 查找研究数据与复算命令 | [统一数据索引](../../transfer-mechanism/2026-10-27-跨源蒸馏泛化研究总览.md#reproduction)，早期快照另见[本地data目录](../../transfer-mechanism/data/README.md) |
| 理解E48后期平台与KL方向 | [完整日志、文献和FKL短对照](2026-10-27-E48后期平台与Forward-KL及知识迁移.md) |
| 理解state迁移、交集与batch作用 | [统一设置、结果图与研究路线](../../transfer-mechanism/2026-10-27-跨源蒸馏泛化研究总览.md) |
| 检查生产实现和性能证据 | [Forward KL实现验收](../../../../infra/numerics/full-vocab/data/forward-kl-implementation-20261027/README.md) |

E49沿用下述E48的模型、训练与验证预算，目标改为full-vocab Forward KL，默认`precompute_fused`、chunk512、async/prefetch开启，并显式设置`forward_teacher_support=student`。它保留原生Reverse KL及位置指标作为诊断，不建立anchor。正式 256-outer E49 已完成，结果见上方方向对照分析；下述小规模验收仅说明工程可运行。

## E48基线设置

本实验沿用 [E20](../../../reuse-budget/equal-update-budget/README.md) 的训练目标和预算，将教师替换为 `POLARIS-Project/Polaris-7B-Preview`，并开启训练前验证。Student 是 `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`。这是一组跨源师生对，相关论文的配对和实验见[论文笔记](../../../../literature/teacher-transfer/2026-10-26-策略蒸馏泛化的双重性实验与设置.md)；E48 使用 DAPO、full-vocab reverse KL 和 reuse16，不能当作论文 sampled-token 配方的等价复现。

| 参数 | 默认值 |
| --- | --- |
| 方法 | Full-vocab reverse KL；统一接口为 MDPO/full_vocab，κ=0、η=1，无 anchor |
| 数据 | DAPO，全量；shuffle=True；不做 group filtering 或失败组 refill |
| Prompt batch / mini-batch / rollout n | 64 / 64 / 1 |
| Outer / reuse / 计划更新数 | 256 / 16 / 4096；每批全局 mini-batch 一次一轮 |
| LR / optimizer | 1e-6；AdamW，betas=(0.9,0.999)，weight decay=0.01，grad clip=1 |
| Scheduler / warmup | constant / 0 |
| Seed | 数据、actor、rollout 均为 42 |
| Prompt / train response cap | 4096 / 7168 |
| Train decoding | temperature=1，top-p=1，top-k=-1 |
| 验证 | MATH-500 / AMC23 / AIME25，500 / 40 / 30 题；各题 16 条回答 |
| Validation response cap / decoding | 31744；temperature=0.7，top-p=0.9 |
| 验证／保存 | step0 验证；每 32 outer 验证；每 64 outer 保存；保留 4 个 checkpoint |
| 默认共享池 | shared_sync，8 卡；teacher 没有额外独立资源池 |
| Teacher FSDP token budget / dtype | **32768 / BF16**，与 E20 一致 |
| Actor token budget | 32768 |
| Full-KL backend / token chunk / vocab chunk | precompute / 512 / 8192 |
| Hidden IO | async write=True，prefetch=True |
| 学生生成并发／显存比例／TP | 128 / 0.65 / 1 |
| 教师独立评测并发 | 32；不控制训练的 FSDP teacher 评分分批 |

轨迹在每个 outer 开始时由当时的学生生成，随后固定这些前缀做 16 次更新；它不是每次 optimizer update 都重新采样的严格 on-policy 流程。Full-KL 覆盖动作词表，不消除复用前缀的陈旧性。

## 模型与本机路径

Polaris 权重已下载到仓库同级 `model/Polaris-7B-Preview/`，包含四个 safetensors 分片。新增资产目录条目，支持通过[准备脚本](../../../../../assets/prepare_models.py)指定`--models Polaris-7B-Preview`。

本机 student/teacher 均为 Qwen2 架构。Tokenizer 实际`get_vocab()`比较与 chat template 检查确认共用 student token 序列的兼容性；head 行数分别 151936/152064，hidden 宽度分别 1536/3584。Full-KL 接口支持不等 hidden 宽度及 teacher 额外输出行，保留 teacher 原生 softmax 归一化；该师生组合已完成下述四卡短训练验收。

本机已用的训练解释器是 `/root/.local/share/mirroropd/venvs/profile-cu128/bin/python`，数学数据在仓库同级 `MD-OPD/data/`；公共默认 `.venv-native/bin/python` 与 `data/` 是其他部署环境的约定，本机执行需要显式覆盖。

## 启动与独立评测

在仓库根目录先设置本机路径。以下只打印配置，不启动 GPU 任务：

```bash
export PYTHON_BIN=/root/.local/share/mirroropd/venvs/profile-cu128/bin/python
export MODEL_ROOT="$PWD/../model"
export DATA_ROOT="$PWD/../MD-OPD/data"
DRY_RUN=1 bash examples/mirroropd/run_math_opd_ablation_fsdp2.sh E48
```

原始 student/teacher 的独立评测选择器分别是`E48-valid-student`和`E48-valid-teacher`。正式训练前应在同一验证协议下测量教师，不能直接用论文或模型卡分数作为本实验的教师参考。

## 四卡短训练结果

短训练使用 GPU 0–3，完成 3 个 outer step、48 次真实更新。保留 batch64、reuse16、LR1e-6、训练回答上限 7 Ki token、teacher 评分预算 32 Ki token、precompute/512/8192，以及 student 128/0.65。验证在三个数据集各取前两题，每题一个回答、上限 512 token，在训练前和第 3 步后执行；引擎上下文仍保留完整验证所需容量。短训练不用于准确率比较，也不替代完整验证与 checkpoint 验收。

验收检查全部通过：3 步训练均完成，48 次更新连续且实际应用，没有跳过更新；loss 与裁剪前梯度范数有限，teacher 前向成功，训练前后各 6 条验证输出完整且评分失败为零。Loss 范围为 0.186–0.213。每 0.2 秒采样四卡显存，并按 teacher 参数传入与前向的时间区间统计峰值，teacher 阶段共 131 次采样。显存是整卡采样占用，不能当作精确的单模块张量峰值，也可能遗漏瞬时峰值。

| Outer step | 完整步骤耗时（分） | Teacher 前向（秒） | Actor 更新（分） | 累计实际更新 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 2.35 | 5.1 | 1.68 | 16 |
| 2 | 2.14 | 4.7 | 1.61 | 32 |
| 3 | 2.14 | 4.4 | 1.58 | 48 |

| GPU | Teacher 阶段显存峰值（GiB） | 全程显存峰值（GiB） | 总显存（GiB） |
| ---: | ---: | ---: | ---: |
| 0 | 12.7 | 54.1 | 79.6 |
| 1 | 12.4 | 54.1 | 79.6 |
| 2 | 12.4 | 54.1 | 79.6 |
| 3 | 12.6 | 54.1 | 79.6 |

三个训练批的平均回答长度约为 6.2–6.5 千 token，最长回答均达到 7 Ki token 上限。该负载下 teacher 预算无需降低；3 步结果不能证明长程收敛，也不能保证所有后续长度分布下都不会 OOM。

代码采用独立快照。GPU 占用工具完成四卡启动、重复启动保护、心跳与计算进展、停止释放和重新启动验证后，停止占用并启动训练；GPU4–7 不参与。详细结果见短训练报告（本地证据：`outputs/polaris-smoke-20261026-bsud-quicksilver-hppk-13/result.md`）、[逐项验收](../../../../../../../outputs/polaris-smoke-20261026-bsud-quicksilver-hppk-13/result.json)和最终解析配置（本地证据：`outputs/polaris-smoke-20261026-bsud-quicksilver-hppk-13/resolved-config.yaml`）。

上述表格仅覆盖四卡短训练。正式 E48 `grcuotmi` 在8张A100 80GB上正常完成256/256 outer；正式run的指标、边界与诊断见[完整训练与后期平台分析](2026-10-27-E48完整训练与后期平台分析.md)。

## CPU 验证

2026-10-26 已通过 E48 的 Hydra 最终配置解析与配置验证，确认四卡覆盖下仍为 batch64、reuse16、256 outer、LR1e-6、初始验证开启，以及 teacher token budget32768。预设规划测试验证 E48 沿用 E20 预算、teacher 独立评测不继承训练预算；未知编号测试已将原来的 E48 更新为 E49 并通过。模型资产测试 16 项通过，文档链接及 shell 语法检查通过。

实际资产检查确认 Polaris 四个权重分片齐全；师生 `get_vocab()` 的 151665 项 token→ID 映射完全一致，chat template 相同。DAPO 的 17917 行、MATH-500 的 500 行、AMC23 的 40 行、AIME25 的 30 行全部通过数据结构检查。以上均为 CPU 检查，不包含模型前向、梯度更新或显存验收。
