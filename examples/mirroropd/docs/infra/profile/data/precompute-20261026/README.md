# Precompute 四 backend 探索性测量证据

测量日期为 2026-10-26，审阅后的数值与结论维护在[算子性能页](../../operators.md#precompute-20261026)。本目录接收原 `outputs/full-kl-softmax-audit-20261026` 的最终性能证据；文件内容未修改，原位置不再维护另一份最终数据。

| 文件 | 含义 |
| --- | --- |
| [benchmark.json](benchmark.json) | 四 backend 各 9 次未裁剪 wall time、median、allocated peak 和 loss sum |
| [benchmark.log](benchmark.log) | 最终测量输出 |
| [benchmark.py](benchmark.py) | 执行测量的原始脚本快照 |
| [full_vocab.py](full_vocab.py) | 最终测量前保存的生产模块快照，测量后确认内容未变 |

## 输入与执行

H800 单卡，PyTorch 2.11.0+cu128，synthetic BF16 输入，N=4096、V=151936、d=1536。Seed=42，hidden/head 由标准正态样本乘 0.1 构造。Student hidden/head 需要梯度，teacher 冻结；普通 KL，不启用 termination equivalence。Token chunk=512、vocab chunk=8192，TF32 关闭，CUDA 投影输出和梯度累计为 FP32。

每次调用返回 raw loss sum，并对 student hidden/head 求导；不做 token 平均，不执行 optimizer。所有 backend 共享固定输入。分别预热两次，再交替按正序 `[dense,recompute,precompute,precompute_fused]` 与逆序执行 9 轮；每次 wall time 前后 CUDA synchronize。最后为每个 backend 单独清 allocator 缓存、reset peak 并调用一次，记录含常驻输入的进程 allocated 峰值。

源码基点为 `141f26f81a1310dd8e89e67cd721bdf993bd1ca8` 加工作区修改。完整审计保留了[启动时源码差异](../../../../../../../outputs/full-kl-softmax-audit-20261026/source-state.txt)与[数值模块快照](../../../../../../../outputs/full-kl-softmax-audit-20261026/source)。最终计时前 `PrecomputedKL.forward` 增加了 `del output`，释放 endpoint tuple 对临时张量的引用；最终模块以本目录快照为准。首次计时留在[原运行目录](../../../../../../../outputs/full-kl-softmax-audit-20261026/benchmark-first.json)，不合并进最终样本。

## 证据限制

测量运行在物理 GPU 7，已有其他进程驻留；未满足本专题当前 GPU 0–3 共享锁与持续空闲门槛，没有锁频、持续外部 PID 记录或多个独立进程重复。它是探索性样本，不用于正式性能验收或窄置信区间。Raw 样本中的波动完整保留。

四 backend 的 scalar sum 为约 628.837；普通 precompute 与 dense 的绝对差约 6.10e-5，即每 token 约 1.49e-8。测量脚本没有内置计时前独立 VJP 门控，同次审计使用独立测试和数值探针进行检查，仍发现近目标反例；不能用随机输入 loss 接近证明所有 backend 在所有分布上可靠。[审计结果](../../../../../../../outputs/full-kl-softmax-audit-20261026/README.md)独立于性能结论。

不包含 backbone、FSDP 通信、TQ、optimizer 或 rollout。不与历史其他版本同名 backend 混合统计；不把算子速度倍率推广为整个训练的速度倍率。

## 复算与重新测量

以下只用 Python 标准库复算已存数据，不占用 GPU：

```bash
python - <<'PY'
import json
import statistics
from pathlib import Path

path = Path('examples/mirroropd/docs/infra/profile/data/precompute-20261026/benchmark.json')
rows = json.loads(path.read_text())['rows']
for row in rows:
    assert statistics.median(row['wall_ms']) == row['median_ms']
    print(row['backend'], row['median_ms'], min(row['wall_ms']), max(row['wall_ms']), row['allocated_peak_gib'])
PY
```

重新测量前按[测量方法](../../methodology.md#发起新测量)取得可用 GPU，并完成数值门控。原始脚本会写入自身目录，必须先复制到新的运行目录，避免覆盖本目录冻结证据：

```bash
PROFILE_RECHECK_DIR=$(mktemp -d /tmp/precompute-recheck-XXXXXXXX)
cp examples/mirroropd/docs/infra/profile/data/precompute-20261026/benchmark.py "$PROFILE_RECHECK_DIR/benchmark.py"
# 下列 GPU 编号须与已取得的资源相符；本例不自行取得锁或检查空闲。
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \
  /root/.local/share/mirroropd/venvs/profile-cu128/bin/python \
  "$PROFILE_RECHECK_DIR/benchmark.py"
```

脚本导入当前工作区生产实现；它不会自动加载本目录的旧模块快照。因此重新运行得到的是新的测量，应单独记录源码身份和输出目录。
