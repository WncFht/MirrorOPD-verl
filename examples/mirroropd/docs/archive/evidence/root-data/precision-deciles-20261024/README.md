# KL 精度与相对位置十分桶验收

2026-10-24，本机原生 PyTorch 环境。完整语义见[研究记录](../../../../settings/metrics/2026-10-24-评测预算语义与KL精度及位置分桶.md)。这些实验没有训练真实1.5B模型，也不是 E16/E17 末期 checkpoint 的精度消融。

| 文件 | 内容 |
| --- | --- |
| `precision.json` | 合成151936词表、8位置的 FP64/FP32/BF16 对照；相同 BF16 输入上的 head GEMM 检查；浮点相邻数间隔 |
| `fsdp2-precompute.json` | 两卡真实 FSDP2 流程，普通 precompute，teacher额外4行词表；逐桶 sum/count 重建全局 KL |
| `fsdp2-fused-qwen3.json` | 两卡 Qwen3，student/teacher tied head，fused precompute；逐桶汇总与更新检查 |

两个集成实验都覆盖某个 rank 的有效 mask 全零，以及不等长回答、masked gap、microbatch 合并、三次训练更新。位置统计在原loss上附加，未改梯度目标。CPU验收共61项通过。

从源码仓库根目录运行。GPU命令只在选定设备空闲时执行；本次使用空闲0/1卡，未操作4–7卡上的任务。

```bash
CUDA_VISIBLE_DEVICES='' /root/.local/share/mirroropd/venvs/profile-cu128/bin/python -m pytest -q \
  tests/trainer/distillation/test_full_vocab_position_on_cpu.py \
  tests/trainer/distillation/test_full_vocab_on_cpu.py \
  tests/utils/test_evaluation_budget_metrics_on_cpu.py

CUDA_VISIBLE_DEVICES=0 /root/.local/share/mirroropd/venvs/profile-cu128/bin/python \
  examples/mirroropd/research/analysis/legacy/audit_kl_precision.py --device cuda \
  --out examples/mirroropd/docs/archive/evidence/root-data/precision-deciles-20261024/precision.json

CUDA_VISIBLE_DEVICES=0,1 /root/.local/share/mirroropd/venvs/profile-cu128/bin/torchrun \
  --standalone --nproc-per-node=2 tests/special_distributed/test_full_vocab_fsdp2.py \
  --backend precompute --teacher-extra-rows 4 \
  --output examples/mirroropd/docs/archive/evidence/root-data/precision-deciles-20261024/fsdp2-precompute.json

CUDA_VISIBLE_DEVICES=0,1 /root/.local/share/mirroropd/venvs/profile-cu128/bin/torchrun \
  --standalone --nproc-per-node=2 tests/special_distributed/test_full_vocab_fsdp2.py \
  --backend precompute_fused --model-type qwen3 --tied-student --tied-teacher \
  --output examples/mirroropd/docs/archive/evidence/root-data/precision-deciles-20261024/fsdp2-fused-qwen3.json
```

分桶均值是跨卡／microbatch 全局 token 加权均值。只读诊断行不参与反向；CPU测试用独立密集 FP64 公式验证值与 VJP，涵盖教师额外词表支持。图或统计分析不得把合成误差当作真实模型精度误差。
