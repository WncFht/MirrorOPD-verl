# 师生验证与等更新预算实验

更新：2026-10-25。本文记录原始 student/teacher 重测入口，以及 E18–E20 的 4096 次更新对照。继续使用已有 DAPO 数据，接受文件内部少量重复；不新增去重、KL/entropy 审计或 trainer 机制。Full-KL 算子和数据文件均未修改。


## 当前运行与结果

日志提交 `ff8b9f0`：E18/E19 均 exit=0，实际成功更新各 4096；E20 云端 failed，本地 partial_snapshot，已完成 outer 189 和 3024 次更新。三个 run 的已记录更新均无 skip。E18/E19 已通过预算验收工具，确认每步更新、全部 outer 汇总，以及 update 512、1024、……、4096 的八个验证节点齐全。关键设置与末次验证见[运行记录](../运行记录.md)，完整 config/history 在日志仓库按 run ID 查询。

| 实验 / run | 计划 outer / update | 已完成 outer / update | 最后验证 outer / update | MATH / AMC / AIME mean（%） |
| --- | --- | --- | --- | --- |
| E18 / 9hx3uegm | 64 / 4096 | 64 / 4096 | 64 / 4096 | 88.1375 / 88.5938 / 35.6250 |
| E19 / d6jhh378 | 128 / 4096 | 128 / 4096 | 128 / 4096 | 88.1875 / 89.2188 / 33.5417 |
| E20 / lwgb5eex | 256 / 4096 | 189 / 3024 | 160 / 2560 | 87.9625 / 89.2188 / 35.6250 |

在完成的 4096-update 两组中，E19 相对 E18 的 MATH/AMC/AIME 差为 +0.0500/+0.6250/−2.0833 个百分点；末点没有逐集合一致的优势。E20 的分数来自 update 2560，不能与前两组的终点当作三臂等更新预算结果。

待办包括补齐 E20 的正式预算、核对原始师生完整 570×16 重测，以及在共同 update 节点比较学习曲线、生成量和验证波动。当前项目与日志未发现这两个新 valid 入口的完整师生参考；旧 One-Shot 参考见[师生分析](../full-kl-comparison/2026-10-24-Full-KL收敛与师生参考评测.md)。入口开发时的测试与排队记录单独保留在[历史验收](../../archive/2026-10-24-师生验证与等预算入口验收.md)。

## 1. 原始 Student / Teacher 验证

新增 [run_math_validation_fsdp2.sh](../../../run_math_validation_fsdp2.sh)，调用既有数学公共脚本和 V1 `_validate()`。`student` 选择 `STUDENT_MODEL`，`teacher` 选择 `TEACHER_MODEL`，然后把所选模型放到 `actor_rollout_ref.model.path` 作为生成模型。默认路径分别为 `MODEL_ROOT/DeepSeek-R1-Distill-Qwen-1.5B` 和 `MODEL_ROOT/JustRL-DeepSeek-1.5B`。

包装脚本设置 `distillation.enabled=False`、`trainer.val_only=True`、`trainer.val_before_train=True`、`trainer.resume_mode=disable`、`trainer.save_freq=-1`；没有蒸馏 teacher worker，也不执行 optimizer 更新。资源变量 `STUDENT_GPUS` 在这里表示被评测模型的共享资源池。沿用现有 trainer 初始化，不新增另一套生成或评分实现。

默认评测预算仍是 MATH-500/AMC23/AIME25 的 500/40/30 题，每题 16 条回答，temperature=0.7、top-p=0.9、回答上限 31744。全量提交、CUDA Graph、TP1、每副本并发 128、显存比例 0.65、调度 token 预算 8192，来自已有公共数学配置，包装脚本不复制这些默认值。

采集现有 mean accuracy、pass@k、majority@k、长度统计、停止原因、截断/答案缺失和完整性指标。逐回答结果沿用 `RUN_DIR/validation/records/0.jsonl`，其他输出及原生环境见[运行与架构](../../infra/2026-10-22-共享八卡运行与架构.md)。生成阶段保持不返回训练专用 log-prob；本轮没有新增 hidden/head 分块或概率重放。

从仓库根目录，在各自已分配 GPU 的作业中运行：

```bash
# CUDA_VISIBLE_DEVICES 使用作业实际获分配的设备；STUDENT_GPUS 与数量一致。
STUDENT_GPUS=8 bash examples/mirroropd/run_math_validation_fsdp2.sh student
STUDENT_GPUS=8 bash examples/mirroropd/run_math_validation_fsdp2.sh teacher

# 检查最终命令或解析配置，不启动 Ray/GPU。
DRY_RUN=1 bash examples/mirroropd/run_math_validation_fsdp2.sh teacher
CONFIG_ONLY=1 bash examples/mirroropd/run_math_validation_fsdp2.sh student
```

本机原有模型、数据、Python 不在默认资产位置时，显式设置：

```bash
export MODEL_ROOT=/mnt/tidalfs-bdsz01/dataset/llm_dataset/zkn_data/fht_data/models
export DATA_ROOT=/mnt/tidalfs-bdsz01/dataset/llm_dataset/zkn_data/fht_data/MD-OPD/data
export PYTHON_BIN=/root/.local/share/mirroropd/venvs/profile-cu128/bin/python
export VERL_USE_UV=0
# 仅在 GPU0–3 空闲、已获得共享使用权时运行本机四卡任务。
export CUDA_VISIBLE_DEVICES=0,1,2,3
export STUDENT_GPUS=4
```

自定义权重可以用 `STUDENT_MODEL` / `TEACHER_MODEL`，或末尾 `actor_rollout_ref.model.path=...` 覆盖。末尾 Hydra 参数仍有最高优先级，包括 val-only 等开关；修改后按实际合成配置判断作业行为。两个模型使用独立实验目录，避免覆盖结果。既有初始化依赖仍会读取训练数据，因此纯 valid 入口也要求正常的数据资产。

## 2. E18–E20：相同更新数，不同复用程度

入口仍为 [run_math_opd_ablation_fsdp2.sh](../../../run_math_opd_ablation_fsdp2.sh)。三组默认如下：

| 预设 | PPO epochs | Outer steps | 更新数 | 消费数据记录数 | TEST_FREQ | SAVE_FREQ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| E18，高复用 | 64 | 64 | 4096 | 4096 | 8 | 16 |
| E19，中复用 | 32 | 128 | 4096 | 8192 | 16 | 32 |
| E20，更多新轨迹 | 16 | 256 | 4096 | 16384 | 32 | 64 |

频率单位为 outer step。三组都在 update 512、1024、……、4096 验证，在 update 1024、2048、3072、4096 保存；保留四份 checkpoint。默认不做训练前验证，原始 student 的独立验证作为共享参考。

共同条件：`TOTAL_EPOCHS=1`，全局 batch=mini-batch=64，每题一条训练回答，seed=42，constant LR=1e-6、零 warmup、与 E17 相同的 AdamW 和梯度裁剪。Full-KL 使用 `precompute`、async write 和 prefetch；不额外加入 sampled-token IS 或 PPO clipping。动态分组筛选和失败补采关闭，采用同步训练。

数据仍是原始 `math/DAPO/train.parquet`，没有派生去重文件。已有 17917 条记录，当前长度配置过滤后数量不变，最多 279 个完整 batch。E20 的 256 个 batch 消费 16384 条，留有 1533 条余量。三组使用同一文件、同一 data seed、`train_max_samples=-1` 和现有独立随机 sampler，较短实验读取较长实验的题目序列前缀；相同题目后续生成的回答不要求一致。

```bash
# 三条命令分别用于独立作业，不自动申请或争用同一组 GPU。
STUDENT_GPUS=8 bash examples/mirroropd/run_math_opd_ablation_fsdp2.sh E18
STUDENT_GPUS=8 bash examples/mirroropd/run_math_opd_ablation_fsdp2.sh E19
STUDENT_GPUS=8 bash examples/mirroropd/run_math_opd_ablation_fsdp2.sh E20

CONFIG_ONLY=1 bash examples/mirroropd/run_math_opd_ablation_fsdp2.sh E20
```

所有预设仍允许环境变量及末尾 Hydra 覆盖，方便小规模 smoke。覆盖 batch、epochs、outer 或数据后，原表的 4096 updates 和单遍容量不再自动成立，应重新计算。没有增加通用 trainer 预算约束，也没有改变数据耗尽后的既有行为；正常预设通过单数据 epoch 和足够的样本容量避免回绕。

## 3. 原数据重复和标注冲突

按完整 prompt 内容精确比较，17917 条记录有 17255 个不同 prompt，多出的重复记录为 662 条，约 3.69%。这是三组共有的数据条件，不宣称每条记录对应唯一问题。

其中 12 组相同题目有不同 ground truth，共 24 条记录。已回查本机上游原始 parquet，题面和标签与处理后的记录一致，冲突并非本地转换引入。逐题核对后的答案如下，ID 省略 `DAPO-`：

| ID 对 | 原标签 | 核对答案 |
| --- | --- | ---: |
| 35 / 179 | 64 / 6 | 64，6 是指数 |
| 282 / 7665 | 15 / 25 | 25 |
| 2040 / 8017 | 99 / 21 | 99 |
| 2129 / 7414 | 30 / 12 | 30 |
| 3470 / 8382 | 36 / 248 | 248 |
| 4850 / 7807 | 239 / 48 | 239 |
| 5785 / 10267 | -1 / 554 | 554 |
| 5885 / 9921 | -1 / 230 | 230 |
| 5915 / 10714 | -1 / 683 | 683 |
| 5937 / 8582 | -1 / 913 | 913 |
| 7694 / 7708 | 108 / 180 | 180 |
| 11659 / 11771 | -2 / -1 | -2 |

保留第一条不能自动保证正确：其中第一条正确 5 组，第二条正确 7 组。这轮不修改标签或去重。纯 Full-KL 的 `use_task_rewards=False`，这些 ground truth 不参与蒸馏梯度，但影响对应训练题的正确率诊断；不能用这项训练集发现推断验证集标签质量。

## 4. 实际更新预算验收

新增 [verify_math_update_budget.py](../../../tools/verify_math_update_budget.py)，只读现有日志导出，不重新实现 .wandb 解码。它检查最终配置、完整逐更新记录、每个 outer 的完成记录，以及验证节点的成功更新坐标。出现跳步、非有限 KL、缺失末步汇总或缺失最终验证时返回非零状态。

```bash
# 日志先通过既有本地导出流程处理；替换 RUN_ID 和新输出目录。
python3 examples/mirroropd/docs/scripts/analyze_run.py export \
  --run wncfht/MirrorOPD-verl/RUN_ID --out outputs/RUN_ID-analysis

"${PYTHON_BIN:-.venv-native/bin/python}" examples/mirroropd/tools/verify_math_update_budget.py \
  --config outputs/RUN_ID-analysis/config.json \
  --history outputs/RUN_ID-analysis/history.jsonl \
  --expected-updates 4096
```

也接受 `resolved_config.yaml` 和未压缩 `history.jsonl`。导出文件的实际扩展名以导出目录为准。只有成功更新数相等，才把三组标记为等更新预算；验收不证明 checkpoint 文件完整、数据唯一或模型效果收敛。效果比较继续同时看 update、生成 token 和实际耗时，相同 update 数不等于相同算力消耗。
