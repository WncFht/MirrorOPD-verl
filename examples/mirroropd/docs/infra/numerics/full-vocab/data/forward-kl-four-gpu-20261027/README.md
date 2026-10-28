# Forward KL 四卡验收

日期：2026-10-27。代码版本为 `a6c1ee729`，使用物理 GPU 0–3。CUDA 数值检查、四卡 FSDP2 恢复检查，以及真实 DS-distill-1.5B／Polaris-7B 的训练与评测链路均通过。生产配置和支持集定义见 [Forward KL](../../forward-kl.md)。

## 结果

| 检查 | 结果 | 证据 |
| --- | --- | --- |
| CUDA 数值与梯度 | 8 项通过；覆盖大词表、BF16、不等输出行数、极小 KL 和非连续输入 | [测试日志](cuda-tests.log) |
| 四卡 FSDP2 | Qwen3、tied student embedding、teacher 额外 4 行、hidden 32/64；包含单 rank 空 mask | [结果](fsdp4-tied-resume.json)、[日志](fsdp4-tied-resume.log) |
| Checkpoint 恢复 | 模型、Adam、scheduler 回读一致，恢复后的下一次更新与连续训练一致 | 同上，三个 roundtrip 字段均为 true |
| 真实模型更新 | 2 outer × reuse16，共 32 次更新，0 次跳过；loss 与梯度均有限 | [汇总](result.json)、[逐步指标](smoke/history.jsonl) |
| 真实模型评测 | 更新 32 次后执行 MATH500、AMC23、AIME25 各 2 题，共 6 条记录；评分失败和非有限分数均为 0 | [验证输出](smoke/validation/2.jsonl) |
| 指标归约 | FKL／原生 RKL 的位置桶重建误差分别不超过 1.03e-8／1.64e-8 | [复算脚本](summarize.py) |

真实模型保持 batch=mini-batch=64、reuse16、LR=1e-6、seed42、teacher token budget32768、fused chunk512、async write/prefetch。优化器使用默认 AdamW 配置，`override_optimizer_config=null`，没有关闭 foreach。冻结输入是 64 个数学 prompt，两轮遍历同一批 prompt，每轮由当时的学生重新生成回答；训练和验证的回答上限均为 512，验证每题采样一次。输入文件保存在 [inputs](inputs)，完整配置和实际命令见 [resolved_config.yaml](smoke/resolved_config.yaml) 与 [command.sh](smoke/command.sh)。

| Outer | FKL 首／末观测 | 原生 RKL 首／末观测 | JS 首／末观测 | TV 首／末观测 |
| --- | --- | --- | --- | --- |
| 1 | 0.17593 → 0.16773 | 0.26196 → 0.25352 | 0.03894 → 0.03765 | 0.12860 → 0.12605 |
| 2 | 0.17007 → 0.14500 | 0.24630 → 0.21907 | 0.03765 → 0.03349 | 0.12690 → 0.11847 |

每轮首末观测之间跨越 15 次更新，末次观测不是第 16 次更新后的独立重评。两轮的生成轨迹不同，表中批次起点不能作为固定 state 上的泛化测量。

梯度裁剪前范数为 0.43531–0.78405，低于阈值 1。Teacher 原生分布落在 student 输出行之外的质量为 7.80e-10–8.30e-10。Teacher 在每轮固定轨迹上的 entropy 不随 inner update 改变，student entropy、JS、TV 与两个 KL 均保留在逐步指标中。

Actor 记录的 allocated 峰值约 9.95 GiB。第二次运行期间存在同卡其他进程，[gpu.csv](gpu.csv) 的 52.6–56.6 GiB 整卡峰值包含这些进程，不能解释为 Forward KL 的独占显存。两轮 actor update 阶段分别为 26.50／22.34 秒；这些共享设备上的时间只作运行记录，不用于性能对照。

验收覆盖真实 rollout、teacher hidden、全参数更新、权重发布、下一轮 rollout 和末轮评分。它采用短回答窗口；正式 E49 的训练 cap7168、验证 31744×16 及 256 outer 尚未在这组验收中运行，6 题的分数不用于判断任务准确率。

## 复算与重跑

在源码根目录，只复算现有证据，不使用 GPU：

```bash
FKL_DATA=examples/mirroropd/docs/infra/numerics/full-vocab/data/forward-kl-four-gpu-20261027
/root/.local/share/mirroropd/venvs/profile-cu128/bin/python "$FKL_DATA/summarize.py" \
  --run "$FKL_DATA/smoke" --gpu-csv "$FKL_DATA/gpu.csv" --output /tmp/fkl4-result.json
```

四卡分布式恢复检查：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 PYTHONPATH=. \
  /root/.local/share/mirroropd/venvs/profile-cu128/bin/torchrun --standalone --nproc-per-node=4 \
  tests/special_distributed/test_full_vocab_fsdp2.py --direction forward --backend precompute_fused \
  --model-type qwen3 --tied-student --teacher-extra-rows 4 --teacher-hidden-size 64 --checkpoint \
  --output /tmp/fkl4-resume.json
```

[run-smoke.sh](run-smoke.sh) 固定使用前四卡，默认读取已冻结的 inputs，并显式设置 `trainer.test_freq=2`。运行前确保这四张卡已分配给该任务；使用新名称，避免覆盖结果：

```bash
EXPERIMENT_NAME=full-fkl-four-gpu-check-new \
  bash examples/mirroropd/docs/infra/numerics/full-vocab/data/forward-kl-four-gpu-20261027/run-smoke.sh
```

新运行输出写入 `outputs/$EXPERIMENT_NAME`。可通过 `PYTHON_BIN`、`MODEL_ROOT` 和 `DATA_ROOT` 覆盖资产位置。Ray 临时目录采用短路径，避免 Unix socket 文件名超过 107 字节。

## 辅助记录

[training-only](training-only) 保存此前关闭评测的独立两轮运行：32 次更新全部执行，第二轮 FKL 为 0.16886→0.14409，原生 RKL 为 0.25338→0.22410。该运行没有验证记录，不计为评测验收。其后一次启动因 Ray socket 路径过长，在模型加载前退出，日志保存在 [attempts](attempts/ray-socket-path.log)。上文结果来自修正临时目录后成功完成的运行。
