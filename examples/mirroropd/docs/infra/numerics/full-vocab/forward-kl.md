# Full-vocab Forward KL

生产训练入口为 `distillation.distillation_loss.loss_mode=full_forward_kl`，数学配方为 `E49`。它沿用 E48 的 DS-distill-1.5B student、Polaris-7B teacher、batch64、reuse16、256 outer、LR1e-6、训练回答 7168 和验证 31744×16，只改变目标及其明确的词表支持约定。E49 默认启用融合行运算；对照 E48 时建议两组使用同一个代码版本和 backend。

## 目标和输出支持

令 student 输出行集合为 S，teacher 原生分布为 q。Forward KL 优化：

$$
L_F=\frac1N\sum_{t\in\mathrm{valid}}
\sum_{v\in S}q_S(v\mid s_t)\left[\log q_S(v\mid s_t)-\log p_\theta(v\mid s_t)\right].
$$

N 是全局有效 response token 数。Teacher 冻结，每次 inner update 使用当前 student 分布；梯度是 $p-q_S$，通过 backbone 和 lm_head 更新所有 student 参数。

`full_vocab.forward_teacher_support`有两个值：

| 值            | 语义                                                                    |
| ------------- | ----------------------------------------------------------------------- |
| `equal`，默认 | 要求 teacher 和 student head 行数一致，使用原生 teacher 分布            |
| `student`     | 明确使用 $q_S=q(\cdot\mid S)$，将 teacher 在 student 输出行上重新归一化 |

E49 使用`student`：Polaris head 有 152064 行，student 有 151936 行，共享 token 映射已由已有模型接入检查。若 teacher 在额外行有正概率，而 student 在这些行的概率为 0，原生 FKL 为无穷，不能把直接裁掉这些项称为原生 FKL。当前实现要求 teacher 覆盖全部 student 输出行，并记录`distill/teacher_outside_student_mass`以监控裁掉的概率质量。

同时记录保留 teacher 原生归一化的 Reverse KL：

$$
L_R^{\mathrm{native}}=\mathrm{KL}(p\Vert q_S)-\log q(S).
$$

这使`distill/full_kl`仍能与 E48 的原字段比较。FKL 和 RKL 的数值大小不能直接作为谁更优的依据；应在同一评估前缀上分别看变化，再看任务准确率、覆盖和完成率。

## 后端和内存

实现位于[full_forward_kl_result](../../../../../../verl/trainer/distillation/forward_vocab.py)、[融合行运算](../../../../../../verl/trainer/distillation/forward_vocab_triton.py)。现有的 shared teacher、hidden 写入、异步 prefetch、FSDP2 head hooks、DP token 归一化和 checkpoint 通路继续复用。

| Backend            | 行为                                                                 |
| ------------------ | -------------------------------------------------------------------- |
| `dense`            | 可微参考实现，用于小输入与正确性检查                                 |
| `recompute`        | 按 token 块计算，backward 时重新计算块的 VJP                         |
| `precompute`       | Torch 稳定行运算；forward 预计算 dH/dW，backward 仅施加上游缩放      |
| `precompute_fused` | CUDA FP32 logits 使用 Triton 融合行运算，其他设备/dtype 使用参考路径 |

Forward 暂不接入`cached/streaming/liger`；不支持终止 token 类别合并。配置会明确拒绝这些组合。已有 Reverse KL 和 MDPO/BPO 的行为保持原有语义。

`token_chunk_size`控制投影与行运算的峰值。上述 Forward 后端在块内覆盖完整 student 词表，不再按`vocab_chunk_size`切词表；该字段保留在公共配置中。E49 默认为 512。Precompute 保存 dH/dW，不保留跨全部 response 的 logits 计算图，也不向 CPU 传输完整词表概率。

融合实现共享 student/teacher 投影，复用稳定的相对 logit 和 cumulant 归约，同时产生 FKL、RKL、entropy、JS、TV。Teacher 额外行单独做小投影，避免为原生 RKL 诊断重复整个 teacher 投影。梯度使用稳定的`expm1`处理接近相等的概率，并重建 pivot 梯度，保留高置信度位置的小非零梯度。

## 指标

全部标量使用真实有效 token 的 sum/count 归约。两套位置桶使用相同 mask 和相对位置定义，保留 mask 空洞。

| W&B / file 字段                                            | 含义                                            |
| ---------------------------------------------------------- | ----------------------------------------------- |
| `optim/actor/loss`、`distill/full_forward_kl`              | 实际优化的 FKL                                  |
| `distill/full_kl`                                          | 对原生完整 teacher 的 Reverse KL，沿用 E48 口径 |
| `distill/full_forward_kl_position/p00_10/...` 至 `p90_100` | FKL 位置 sum/count/mean                         |
| `distill/full_kl_position/...`                             | 同一批位置上的原生 RKL                          |
| `distill/teacher_outside_student_mass`                     | 原生 teacher 在 student 输出行之外的概率质量    |
| `distill/student_entropy`                                  | Student entropy                                 |
| `distill/teacher_entropy`                                  | 目标 $q_S$ 的 entropy                           |
| `distill/js`、`distill/tv`                                 | p 和 $q_S$ 的 JS、TV                            |

它们复用已有投影，不新增 teacher backbone 前向；诊断量不贡献梯度。Step 轴、实际更新计数、nonfinite/skip 等沿用 policy-v3。

## 运行 E49

在源码仓库根目录，开发机使用已有 `.venv-native`、同级`model/`与`data/`时：

```bash
# 查看最终命令，不启动训练。
DRY_RUN=1 bash examples/mirroropd/run_math_opd_ablation_fsdp2.sh E49

# 与历史E48相同的8卡、batch和更新预算。
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 STUDENT_GPUS=8 \
  bash examples/mirroropd/run_math_opd_ablation_fsdp2.sh E49
```

本机四卡运行需要明确解释器和数据目录：

```bash
export PYTHON_BIN=/root/.local/share/mirroropd/venvs/profile-cu128/bin/python
export MODEL_ROOT="$PWD/../model"
export DATA_ROOT="$PWD/../MD-OPD/data"
export VERL_USE_UV=0
CUDA_VISIBLE_DEVICES=0,1,2,3 STUDENT_GPUS=4 \
  bash examples/mirroropd/run_math_opd_ablation_fsdp2.sh E49
```

同版本公平对照可分别运行以下两条；不要在同一组 GPU 上同时启动：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3 STUDENT_GPUS=4
FULL_KL_BACKEND=precompute_fused FULL_KL_TOKEN_CHUNK=512 \
  bash examples/mirroropd/run_math_opd_ablation_fsdp2.sh E48
FULL_KL_BACKEND=precompute_fused FULL_KL_TOKEN_CHUNK=512 \
  bash examples/mirroropd/run_math_opd_ablation_fsdp2.sh E49
```

E48 历史 run 使用更早的代码与`precompute`。和历史曲线比较有参考价值，但不能把新旧代码、backend、卡数等差异全归因于 KL 方向。`E48 --method full_forward_kl`也可用，但不自动选择支持条件化；不等 head 需显式设置`FULL_KL_FORWARD_TEACHER_SUPPORT=student`。直接使用 E49 最省步骤。

E48 原路径不额外计算 FKL、JS 和 TV。跨目标比较这些量时，应在同一固定 state 面板上给两组 checkpoint 评分；不能把 E49 新增的训练诊断当作 E48 已有历史字段。两组都可直接比较原生 RKL、更新预算和完整任务评测。

本机使用当前工作区无需传输。开发机运行须先提交要传输的源码，再按[离线源码同步](../../../../../../meituan/README.md)执行`make archive-pack TARGET=dev`；接收端使用`make archive-unpack ARCHIVE=/上传目录/代码包.bundle`。未提交内容不会进入 bundle。大型模型与 tensor 缓存位于 outputs，不作为源码包的一部分。

## 验证和成本

数值检查覆盖 FP64 独立参考、teacher 不参与梯度、不等 head/hidden、空 token、高置信度近零损失；CUDA 检查覆盖大词表、非整块长度和 BF16 参数 VJP。四卡 FSDP2 覆盖不均匀 mask；两卡 tied-Qwen3 检查验证模型、Adam、scheduler 的 checkpoint 回读及后续更新逐项一致。

真实 DS-distill-1.5B/Polaris-7B 链路已在 GPU0、3 完成一次 batch64、reuse16 更新及 6 条短验证：16 次更新全部执行，FKL 为 0.13284→0.11890，原生 RKL 为 0.20523→0.18958，loss/grad 有限，验证评分失败数为 0。该验收将 response cap 设为 128、validation n 设为 1，并关闭 AdamW foreach 临时批量运算以适应共享显存；正式 E49 默认不改该优化器选项。它不是 256-outer 或完整任务准确率验收。

随后在物理 GPU0–3 完成了[四卡默认优化器验收](data/forward-kl-four-gpu-20261027/README.md)：batch64、reuse16、两个 outer、回答 cap512，未覆盖 AdamW 选项。32 次更新全部执行，末轮 6 条验证记录全部完成评分；第二轮 FKL 为 0.17007→0.14500，原生 RKL 为 0.24630→0.21907。两套位置指标重建误差均小于 1.7e-8。四卡 tied-Qwen3 的模型、Adam、scheduler 恢复及下一次更新一致性也通过。该验收补齐了四卡真实链路，正式长窗预算仍需单独运行。

同一 H800 共享 GPU 上，512 token、student head151936×1536、teacher head152064×3584、BF16 输入的 head 阶段测量为：

| Forward 后端               | 中位时间 | 峰值 allocated 显存 |
| -------------------------- | -------: | ------------------: |
| Torch precompute，chunk512 | 69.70 ms |            6.56 GiB |
| Fused precompute，chunk512 | 11.46 ms |            3.52 GiB |
| Fused precompute，chunk128 | 26.63 ms |            2.80 GiB |

方法交错测量 7 次，已排除首次编译。此数值包含 head 投影、行诊断、dH/dW 和 backward 应用，不包含 backbone、rollout、IO 与通信；共享 GPU 的时间有波动，不能称为端到端 6 倍提速。原始样本和分布式结果见[实现数据](data/forward-kl-implementation-20261027/README.md)。
