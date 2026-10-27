# Full-vocab Forward KL实现验收

日期：2026-10-27。生产接口、支持集和启动方法见[Forward KL说明](../../../../full-vocab/forward-kl.md)。本目录按用户要求位于源码docs中，不写入wandb-logs。

## 已完成检查

| 层次 | 结果与输入 |
| --- | --- |
| CPU loss/VJP | 44项通过，覆盖新增Forward与已有Reverse；FP64独立参考、空token、不等head、teacher冻结、全局token归一化 |
| CUDA行运算/VJP | 8项通过；词表31/4097/151936、BF16、近相等及高置信度极小KL、非连续输入 |
| Launcher / 配置 / adapter | E49预算与支持约定验证；既有配置回归中因新增编号产生的两项目录断言已更新并复测 |
| 更新预算检查 | 15项通过，Forward必须同时有FKL目标和原生RKL诊断；E49进入Polaris实验分类 |
| 四卡FSDP2 | `fsdp4.json`，Forward、融合backend、teacher额外4行、hidden宽度32/48，包含一个rank全空mask |
| 两卡tied Qwen3 / resume | `fsdp2-tied-resume.json`，teacher额外4行、hidden宽度32/64；模型/Adam/scheduler回读及随后更新逐项一致 |
| 真实冻结logits | `real-logits.json`，student/Polaris生成的数学、代码、IF前缀共384位置；最大FKL绝对误差5.60e-7、logit梯度误差2.35e-7 |
| 配置对照 | `comparison-config.json`；同版本、同fused backend下，E48/E49仅目标与支持约定不同，模型/训练/验证预算一致 |
| 真实模型链路 | `smoke/`，详见下文 |

`fsdp4.json`的执行命令明确为`--direction forward`；该文件生成时输出尚未加direction字段。后续分布式结果直接包含该字段。

## 真实模型链路

物理GPU0、3，DS-distill-1.5B student、Polaris-7B teacher，真实rollout→teacher hidden→FSDP2更新→权重发布→短验证。保留batch64、reuse16和teacher token budget32768，只执行一个outer。训练回答cap128；验证在MATH/AMC/AIME各取2题、每题1答、cap128。`foreach=false`避免共享显存下AdamW的临时批量buffer，未改变LR、betas、weight decay或KL目标。

- 16次实际更新，0次跳过；loss与grad全部有限。
- 首次/第16次观测的FKL为0.13283890/0.11889962；原生RKL为0.20523178/0.18958441。
- FKL与RKL位置sum/count重建全局均值的最大误差分别为6.22e-9与1.02e-8。
- 6条验证记录全部完成评分，`failed_count=0`；短cap下的任务分数不用于效果比较。
- 已记录actor allocated峰值15.62GiB；该值不是整卡总占用，也不是正式长窗峰值。

`smoke/result.json`保留验收统计；`history.jsonl`、`resolved_config.yaml`、`command.sh`、`train.log`是本次实际运行证据。此前在共享GPU上有多次显存不足的尝试，不计作成功；没有终止其他任务。用户重新分配 GPU0–3 后完成的默认优化器、两个 outer 与末轮评分验收见[四卡记录](../forward-kl-four-gpu-20261027/README.md)。

## Head阶段性能

`head-benchmark.json`采用实际head尺寸：student151936×1536、teacher152064×3584，512 token、BF16输入。预热后交错运行7轮。

| 方法 | 中位时间 | min–max | 峰值allocated |
| --- | ---: | ---: | ---: |
| Forward Torch，chunk512 | 69.70ms | 47.98–108.09ms | 6.56GiB |
| Forward fused，chunk512 | 11.46ms | 6.78–16.65ms | 3.52GiB |
| Forward fused，chunk128 | 26.63ms | 11.89–26.91ms | 2.80GiB |
| Reverse fused，chunk512 | 16.97ms | 7.33–17.01ms | 3.37GiB |

所有Forward行均计算相同目标和诊断。Reverse保留原生teacher支持，其loss值不应与Forward直接比较。该基准包含head投影、行运算、dH/dW和backward应用，不包括backbone、rollout、IO、通信；共享GPU存在时间波动，因此不作端到端速度承诺。

## 复算

从源码根目录执行：

```bash
FKL_PY=/root/.local/share/mirroropd/venvs/profile-cu128/bin/python
FKL_DATA=examples/mirroropd/docs/experiments/polaris-transfer/data/forward-kl-implementation-20261027

CUDA_VISIBLE_DEVICES='' "$FKL_PY" -m pytest -q \
  tests/trainer/distillation/test_forward_vocab_on_cpu.py \
  tests/trainer/distillation/test_full_vocab_on_cpu.py \
  tests/examples/test_mirroropd_update_budget_on_cpu.py
CUDA_VISIBLE_DEVICES=0 "$FKL_PY" -m pytest -q tests/trainer/distillation/test_forward_vocab_on_gpu.py

CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 \
  /root/.local/share/mirroropd/venvs/profile-cu128/bin/torchrun --standalone --nproc-per-node=4 \
  tests/special_distributed/test_full_vocab_fsdp2.py --direction forward --backend precompute_fused \
  --teacher-extra-rows 4 --teacher-hidden-size 48 --output "$FKL_DATA/fsdp4-new.json"

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. "$FKL_PY" \
  examples/mirroropd/docs/infra/profile/scripts/benchmark_forward_kl.py \
  --output "$FKL_DATA/head-benchmark-new.json"
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. "$FKL_PY" \
  examples/mirroropd/docs/infra/profile/scripts/audit_forward_logits.py \
  --cache outputs/polaris-transfer-cache/state-transfer-20261027 --out "$FKL_DATA/real-logits-new.json"
```

真实模型smoke入口为`examples/mirroropd/smoke/smoke_full_forward_kl.sh`。它使用分配到的GPU，默认小batch、两步、console+file日志；模型和数据路径按本机环境设置。上面保存的`smoke/command.sh`记录了实际验收的全部覆盖参数。正式E49预算由主launcher提供，不能把smoke的短窗口结果冒充完整训练。
