# Full-vocab KL 数据流与算子

完整词表反向 KL 在 student 生成的固定前缀上，对全部输出动作求和。它与 sampled 目标使用同一批前缀，但不依赖采样到的单个动作估计条件 KL。理论推导见[固定前缀目标](../theory/2026-10-22-固定前缀下的OPD目标与梯度.md)。

## 接口与运行约束

[full_reverse_kl_result](../../../../verl/trainer/distillation/full_vocab.py) 接收 student hidden/head 和冻结 teacher hidden/head，返回 [FullKLResult](../../../../verl/trainer/distillation/full_vocab.py)：可反传的 `loss_sum` 以及可选、detached 的 `token_kl`。标量兼容入口 `full_reverse_kl` 只返回前者。Kernel 不负责分布式通信、训练归一化或日志命名。

若有效预测行数为 $n$，输入形状为 `hs[n, d_s]`、`ws[V_s, d_s]`、`ht[n, d_t]`、`wt[V_t, d_t]`。师生 hidden 宽度可以不同；输出 token ID 必须对齐，且教师支持需覆盖学生每一行输出。

当前生产路径要求 shared-sync、FSDP2、token-mean、temperature 1，无额外任务奖励或 sampled PG 目标。参考 [DistillationLossConfig](../../../../verl/workers/config/distillation.py) 和 [PPOTrainer._setup](../../../../verl/trainer/ppo/v1/trainer_base.py)。

## 从教师 hidden 到 student head

1. [FSDPEngineWithLMHead._full_vocab_output](../../../../verl/workers/engine/fsdp/transformer_impl.py) 用公共 `prediction_positions` 选出有效预测 state。Teacher 将选中 hidden detach 并搬到 CPU，按样本发布 `teacher_hidden/teacher_positions`。
2. [HiddenBatchIO](../../../../verl/trainer/distillation/hidden_io.py) 按 `sample_keys/sample_partition` 读取数据。Actor 校验每条样本保存的位置与当前 `loss_mask` 一致。
3. Student 每个更新重新执行 backbone，选中相同位置。模型通过 [install_full_vocab_adapter](../../../../verl/models/transformers/full_vocab.py) 接入回调；调用包装后的 head 保留 FSDP2 的 all-gather/backward hooks。
4. Kernel 返回 token sum；[distillation_loss](../../../../verl/trainer/distillation/losses.py) 用 LossContext 归一化，detached 行值另送[位置统计](metrics.md#相对位置统计)。

[HiddenBatchIO.publish](../../../../verl/trainer/distillation/hidden_io.py) 的异步写只重叠 CPU→TQ 与下一批 teacher 计算：提交下一笔前等待上一笔，正常退出 flush 最后一笔。`__iter__` 的预取只把下一批 TQ 数据读到 CPU。两者均不表示专用 CUDA stream 的异步 D2H/H2D，也不让师生 backbone 同时执行。

[FrozenTeacherHead](../../../../verl/trainer/distillation/hidden.py) 从 dense Qwen2/Qwen3 safetensors 读取输出权重，tied head 使用 embedding 权重；CPU 副本与按需 GPU 缓存均独立于 student module。每个 actor rank 持有完整冻结 head，`ActorRolloutRefWorker.update_actor` 在 `finally` 中释放 GPU 引用。它不进入 student optimizer、checkpoint 或 rollout 权重同步。

## 梯度与精度

令 $p_\theta$ 为当前 student 分布，$T$ 为冻结教师分布，$z_v$ 为 student logit。每个前缀的目标和 logit 梯度为：

$$
K=\sum_v p_\theta(v)\left[\log p_\theta(v)-\log T(v)\right],
\qquad
\frac{\partial K}{\partial z_v}=p_\theta(v)\left[\log\frac{p_\theta(v)}{T(v)}-K\right].
$$

[PrecomputedKL](../../../../verl/trainer/distillation/full_vocab.py) 在 forward 中按 token 块计算行 KL 和 logit VJP，累积 student hidden/head 的梯度并缓存。Backward 施加上游缩放并传回 student backbone；教师无梯度，`token_kl` 标为不可微。当前自定义 autograd 支持一阶梯度。

CUDA BF16 hidden/head 投影使用 FP32 输出，log-softmax 和 KL 在 FP32 中计算。梯度矩阵乘法的输入及缓存仍有 BF16 转换，不能把 FP32 loss 等同于整条计算链的 FP32 精度。精度模型和小 KL 探针见[精度研究](../research/2026-10-24-KL精度误差模型与诊断接口走读.md)。

## Backend 与输出支持

| Backend | 计算方式 |
| --- | --- |
| `dense` | 完整投影和 autograd，适合小规模参考 |
| `recompute` / `streaming` | 保存行统计，backward 重算；streaming 还按 vocab 块遍历 |
| `cached` | 缓存投影结果供 backward 使用 |
| `precompute` / `precompute_fused` | 共享 PrecomputedKL，分别使用普通或融合行归约 |
| `liger` | 外部融合实现，当前不返回逐 token 诊断行 |

`precompute_fused` 在 CUDA BF16 条件下选择 Triton 行实现，其余条件使用普通行实现。Precompute 主路径按 token 分块，`vocab_chunk_size` 不等于主投影会切词表；该参数在其他后端及额外教师支持修正中有作用。

当 $V_t>V_s$，学生在额外教师输出行的概率为零。[teacher_support_correction](../../../../verl/trainer/distillation/full_vocab.py) 补入 `log(Z_teacher / Z_teacher_on_student_support)`，同时修正标量与行 KL，保持教师完整归一化；该修正与 student 参数无关。$V_t<V_s$ 被拒绝。

通用 [run_full_reverse_kl.sh](../../run_full_reverse_kl.sh) 的未覆盖默认 backend 是 `recompute`，IO 开关默认关闭。特定实验预设可选择 `precompute` 和异步 IO，不能把某次验收的配置当作所有入口的默认值。当前设置见[训练配方](../训练配方.md)。

## 验证入口

[test_full_vocab_on_cpu.py](../../../../tests/trainer/distillation/test_full_vocab_on_cpu.py) 覆盖目标和 VJP；[test_full_vocab_adapter_on_cpu.py](../../../../tests/trainer/distillation/test_full_vocab_adapter_on_cpu.py) 覆盖 adapter；[test_full_vocab_position_on_cpu.py](../../../../tests/trainer/distillation/test_full_vocab_position_on_cpu.py) 覆盖诊断接入；[test_full_vocab_fsdp2.py](../../../../tests/special_distributed/test_full_vocab_fsdp2.py) 检查分片、head 更新、额外支持、空 rank 和 checkpoint。性能测量仍从 [profile](../infra/profile/README.md) 进入。
