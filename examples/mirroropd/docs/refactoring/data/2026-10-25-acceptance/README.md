# 架构重构验收数据

实施基点 `dc7037eb0`，开发在隔离worktree完成。这里保存测试摘要、GPU数值结果、真实1.5B短运行的原始指标和解析配置。没有改写已有E系列实验数据，没有上传W&B。

## 证据范围

- `summary.json`：主CPU回归893通过／14跳过，另59项真实TQ replay/reissue通过、9项归一化与传输契约通过，共961个不同CPU用例通过。57项TQ/shared组合回归与主回归存在重叠，不再另行累加。
- `cpu-test-modules.txt`：主回归的69个模块。4个Megatron模块因缺少 `megatron.core` 未收集，名称见summary；不视为通过。
- `baseline-full.json`、`refactored-full.json`：相同测试harness、相同小模型初始化与输入，分别运行基点和重构代码。
- `equivalence.json`：两个rank的所有参数分片与完整optimizer state逐值精确一致。比较使用生成的 `.pt`，这里只保存结果，不把模型状态文件加入Git。
- `fused-qwen3.json`：两卡、Qwen3师生tied head、不同hidden宽度、教师额外词表行；完整checkpoint恢复及下一步更新等价。
- `profile-fused-alias.json`：旧 `fused_candidate` CLI映射到正式fused backend，真实1.5B权重、固定前缀、两次lr=0更新；计时仅作运行记录，不作速度结论。
- `sampled/`、`full/`：四GPU、R1-Distill-1.5B与JustRL-1.5B、两批训练，每批两次更新，三项验证各一题。完整配置与版本见各自文件。

短运行response cap为128；验证分数没有模型效果或failure复现含义。这里验证加载、生成、评分、更新、TQ、日志和验证链路能完成。

## 复现入口

CPU主回归使用 `cpu-test-modules.txt` 中的路径；额外执行：

```bash
CUDA_VISIBLE_DEVICES='' /root/.local/share/mirroropd/venvs/profile-cu128/bin/python -m pytest -q \
  tests/trainer/ppo/v1/test_rejection_denominator_on_cpu.py
```

Replay/reissue测试在独立CPU Ray实例上执行，使用 `address='local'`、4 CPU、0 GPU和128MiB object store；从不连接已有Ray集群。对应模块为 `tests/trainer/ppo/v1/test_replay_buffer_on_cpu.py` 与 `test_reissue_inflight_on_cpu.py`。

两卡检查在空闲GPU上运行：

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 \
  tests/special_distributed/test_full_vocab_fsdp2.py \
  --backend precompute --teacher-extra-rows 4 --checkpoint --state-output /新的状态目录

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 \
  tests/special_distributed/test_full_vocab_fsdp2.py \
  --backend precompute_fused --model-type qwen3 --tied-student --tied-teacher \
  --teacher-extra-rows 4 --teacher-hidden-size 64 --checkpoint

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 \
  tests/special_distributed/test_shared_teacher_fsdp2.py \
  --policy-loss-mode token_is --advantage-source current --ppo-epochs 4 --tie-word-embeddings

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 \
  tests/special_distributed/test_shared_teacher_fsdp2.py \
  --policy-loss-mode token_ppo --advantage-source rollout --ppo-epochs 2
```

本机实际Python／torchrun来自 `/root/.local/share/mirroropd/venvs/profile-cu128/bin/`。基点对照使用独立detached worktree，仅复制同一份增强测试harness，训练实现仍为 `dc7037eb0`。分别保存各rank的参数与optimizer状态，再以 `torch.testing.assert_close(..., rtol=0, atol=0)` 比较。

真实短运行的 `command.sh` 是实际执行命令的副本，只去掉printf产生的最后一个分隔空格。路径对应当时worktree与本机资产；复现时需替换为本机路径并使用新的输出目录。`source_changes.patch` 保留运行当时尚未提交的验证相关差异。

模型和数据未复制或修改；只在输出目录创建三份单题验证parquet。短运行使用独立本地Ray session与file logger，既有GPU任务未被停止。
