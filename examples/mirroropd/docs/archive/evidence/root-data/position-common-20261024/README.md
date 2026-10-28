# 公共位置指标迁移验收

日期：2026-10-24。通用实现位于 `verl/utils/metric/position.py`，Full-KL 的字段名与分桶定义保持一致。

- CPU：54项通过，包括公共位置统计及Full-KL值／梯度检查。
- 两卡FSDP2：`fsdp2-precompute.json`，3次更新；含masked gap、不同response长度、某rank无有效token、teacher额外4行词表。
- 集成测试逐桶核对count：正常时 `[0,2,2,0,0,0,0,2,2,0]`；仅一张卡有效时为其一半。每次全部桶的sum/count均重建全局KL。
- 相对位置模块不依赖teacher/student或input IDs，namespace由caller提供；当前自动接入对象仍为Full-KL。

从源码仓库根目录执行。GPU命令仅在目标设备空闲时运行：

```bash
CUDA_VISIBLE_DEVICES='' /root/.local/share/mirroropd/venvs/profile-cu128/bin/python -m pytest -q \
  tests/utils/test_position_metrics_on_cpu.py \
  tests/trainer/distillation/test_full_vocab_position_on_cpu.py \
  tests/trainer/distillation/test_full_vocab_on_cpu.py

CUDA_VISIBLE_DEVICES=0,1 /root/.local/share/mirroropd/venvs/profile-cu128/bin/torchrun \
  --standalone --nproc-per-node=2 tests/special_distributed/test_full_vocab_fsdp2.py \
  --backend precompute --teacher-extra-rows 4 \
  --output examples/mirroropd/docs/archive/evidence/root-data/position-common-20261024/fsdp2-precompute.json
```
