# 长回答 SFT 数据分布对照

本入口构造 RAW-C、GRAPE、RSR、RATIO 四组 teacher，每组两个训练 seed；科学设置与引用见[实验设计](../../../../docs/experiments/teacher-transfer/2026-11-09-SFT数据分布处理与教师构造.md)。复用现有 HF backbone、数值设置、checkpoint 导出和 canonical 数学评分；本包实现共同候选池、分布处理、共同 END 的序列 CE 和分阶段调度。

## 准备与运行

从仓库根目录执行，使用 `envs/mirroropd-fusion` 环境。准备阶段只运行 CPU，要求新目录；训练、生成和评分由独占 GPU 入口执行。

```bash
../envs/mirroropd-fusion/bin/python -m examples.mirroropd.research.studies.fixed_transfer.sft_distribution prepare --root runs/teacher-construction-20261109/sft-distribution-v1 --plan examples/mirroropd/research/studies/fixed_transfer/plans/sft-distribution-long.json
../envs/mirroropd-fusion/bin/python -u -m examples.mirroropd.research.studies.fixed_transfer.sft_distribution wait-run --root runs/teacher-construction-20261109/sft-distribution-v1 --gpus 0,5,6,7
```

`wait-run` 先等待既有长预算评测的完成回执，再等待 GPU0/5/6/7 的原有资源锁和 compute process 全部释放。等待期间仅有 CPU 控制进程；不借用已有队列的卡、不结束其他任务。取得四卡后运行以下顺序：

1. 用真实 1.7B 模型验收 2K/8K/24K 序列的 merged-END 更新、FP32 master/梯度/Adam moments 和峰值显存，测试权重不导出为 teacher。
2. 两条 TP2 专家生成线路运行独立 calibration 512 题，保存正文和两种 END 的 processed log-probabilities；核对 HF 重算、正确率、长度、候选覆盖和截断率。
3. 校准通过后，从训练保留区按固定题目顺序补足 8,192 个有效题，两条线路各补足 4,096 题。每题先 16 条，最多 64 条，至少 8 条正确完整候选；重复回答保留采样次数。无效题在四组中共同排除。
4. 冻结 Base 评分、全局 alpha、候选概率和两个 seed 的四轮抽样表；按公共题目顺序和随机数比较方法。
5. 每卡一组 teacher，两个 seed 依次训练；每次累积 32 题，固定分母 `32*8192`，共 1,024 updates。保存 epoch1/2/4 和共同 response-token 预算首次跨越的 checkpoint；预算读出记录 batch 边界的实际 token 与超出量。

校准验收阈值在计划中显式固定：截断率不超过 5%，至少 50% 题目有 8 条正确完整候选，HF/采样正文 log-probability 差异的最大每 token 绝对值不超过 0.05。后者是输入与数值诊断；RATIO 的专家分母直接使用采样器记录，不用 HF 代理替换。任何验收失败都停止后续阶段并保留日志；调整方案使用新协议。

## 产物与恢复

`protocol.json` 保存实际路径、配置、模型文件清单、依赖版本、源码与题表 hash。精确题面匹配加四词 shingle Jaccard≥0.9 用于 benchmark 排除和近重复分组；这一操作定义不保证识别所有数学改写。模型大权重按固定 revision、大小和 mtime 绑定，小文件及题表按 SHA256 绑定。

`generated/` 保存全部回答和结束原因，`scores/` 保存评分回执，`quality-*.json` 保存候选池验收，`grading-disagreements-*.json` 保存最多 100 条评分分歧供复查。`pool.jsonl`、`weights.jsonl` 和 `selection-<seed>.jsonl` 是训练的冻结输入，修改会阻止训练。`queue.json` 表示等待、运行、需要处理或 teacher 完成，具体运行日志位于 `logs/`。

生成中断按每题的 16 个固定 seed 槽位恢复；已完成的题不再生成。SFT 失败后保持失败轨迹，显式选择带完整状态的 checkpoint：

```bash
../envs/mirroropd-fusion/bin/python -m examples.mirroropd.research.studies.fixed_transfer.sft_distribution train --root runs/teacher-construction-20261109/sft-distribution-v1 --arm RAW-C --seed 20261109 --resume runs/teacher-construction-20261109/sft-distribution-v1/teachers/RAW-C-20261109/step0256
```

该恢复命令须在原资源 guard 保护的单卡环境内执行。恢复加载模型、optimizer、全部 RNG 和抽样位置；checkpoint 之后的旧 metrics 先归档，已有 checkpoint 不覆盖。运行中源码、依赖或来源变化会拒绝继续，需记录迁移或使用新目录。

自动流水线以八个 teacher 完成为终点，**尚不自动运行能力评测、OOD 诊断、长度匹配随机训练对照或 OPD**。后续能力评测复用原生长预算评测协议，并以相同停止集合重测 Base；这些结果与新 teacher 的完整训练回执分开验收。

## 验证

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ../envs/mirroropd-fusion/bin/python -m pytest -q tests/examples/research/test_sft_distribution_on_cpu.py tests/examples/research/test_teacher_construction_llm_on_cpu.py tests/examples/research/test_teacher_construction_reeval_on_cpu.py
```

CPU 测试覆盖 tied/untied embedding 的分块 VJP 与完整 autograd 对照、分布重采样、冻结输入检查、近重复检索、生成分片与恢复、停止事件概率、Adam 续训逐位一致性，以及资源等待和校准失败的阶段阻断。真实长序列 GPU 验收由流水线取得资源后单独记录。
