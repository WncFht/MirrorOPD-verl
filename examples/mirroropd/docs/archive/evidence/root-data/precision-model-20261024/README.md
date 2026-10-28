# KL 精度模型与重构验收数据

2026-10-24。推导、假设与代码走读见[研究记录](../../../../theory/numerical-effects/2026-10-24-KL精度误差模型与诊断接口走读.md)。本目录的合成实验、真实初始checkpoint前向对照、代码集成测试是三个独立证据层次，不能互相替代。

| 文件 | 内容 |
| --- | --- |
| `experiments.json` | 相关量化噪声、BF16输入head误差模型、两种分布上的FP32算术分辨率；包含种子、维度及Monte Carlo标准误 |
| `precision-model.png/pdf` | 理论与合成实测对照；`precision-model.audit.txt`为图表审计 |
| `checkpoint-probe.json` | 本机R1-Distill-1.5B初始权重，两个固定短数学completion的同输入精度对照；保留prompt、response、逐token KL与配置 |
| `fsdp2-precompute.json` | 重构后的两卡precompute测试，教师额外4行词表 |
| `fsdp2-fused-qwen3.json` | 重构后的两卡fused Qwen3测试，师生均tied head |

未读取E16/E17的末期checkpoint，未开展训练精度消融。checkpoint probe保留RoPE buffer的FP32 dtype，只转换计算参数；不能用整个module的BF16转换来冒充FSDP2参数混合精度。FP32参数roundtrip对照检查量化后的参数转回FP32计算是否恢复原logits。

以下命令从源码仓库根目录运行，GPU命令只选择空闲设备。本次使用0/1卡作合成和集成实验，2卡作只读checkpoint probe，没有操作4–7卡上的任务。

```bash
CUDA_VISIBLE_DEVICES=0 /root/.local/share/mirroropd/venvs/profile-cu128/bin/python \
  examples/mirroropd/research/analysis/legacy/audit_kl_precision_model.py --device cuda \
  --out examples/mirroropd/docs/archive/evidence/root-data/precision-model-20261024/experiments.json

FIGURE_VIZ_MPL=/mnt/tidalfs-bdsz01/dataset/llm_dataset/zkn_data/fht_data/dotfiles/.agents/skills/figure-viz/assets/mpl \
  /root/.local/share/mirroropd/venvs/profile-cu128/bin/python \
  examples/mirroropd/research/analysis/legacy/plot_kl_precision_model.py \
  --data examples/mirroropd/docs/archive/evidence/root-data/precision-model-20261024/experiments.json \
  --out examples/mirroropd/docs/archive/evidence/root-data/precision-model-20261024

CUDA_VISIBLE_DEVICES=2 /root/.local/share/mirroropd/venvs/profile-cu128/bin/python \
  examples/mirroropd/research/analysis/legacy/audit_checkpoint_precision.py \
  --model ../models/DeepSeek-R1-Distill-Qwen-1.5B \
  --out examples/mirroropd/docs/archive/evidence/root-data/precision-model-20261024/checkpoint-probe.json

CUDA_VISIBLE_DEVICES='' /root/.local/share/mirroropd/venvs/profile-cu128/bin/python -m pytest -q \
  tests/trainer/distillation/test_full_vocab_position_on_cpu.py \
  tests/trainer/distillation/test_full_vocab_on_cpu.py \
  tests/trainer/distillation/test_full_vocab_adapter_on_cpu.py

CUDA_VISIBLE_DEVICES=0,1 /root/.local/share/mirroropd/venvs/profile-cu128/bin/torchrun \
  --standalone --nproc-per-node=2 tests/special_distributed/test_full_vocab_fsdp2.py \
  --backend precompute --teacher-extra-rows 4 \
  --output examples/mirroropd/docs/archive/evidence/root-data/precision-model-20261024/fsdp2-precompute.json

CUDA_VISIBLE_DEVICES=0,1 /root/.local/share/mirroropd/venvs/profile-cu128/bin/torchrun \
  --standalone --nproc-per-node=2 tests/special_distributed/test_full_vocab_fsdp2.py \
  --backend precompute_fused --model-type qwen3 --tied-student --tied-teacher \
  --output examples/mirroropd/docs/archive/evidence/root-data/precision-model-20261024/fsdp2-fused-qwen3.json
```

CPU回归49项通过后，新增scalar-only后端能力边界测试，所在21项测试全部通过，合计50个不同用例。Liger边界测试使用stub验证结果协议，不声称完成真实Liger GPU验收；普通与fused原生后端完成上述真实FSDP2验收。
