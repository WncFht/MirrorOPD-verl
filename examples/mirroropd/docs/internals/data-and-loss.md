# 样本身份、张量布局与归一化

本页定义从 TQ 到目标函数的数据约定。Teacher 的执行顺序见[教师评分](teacher.md)，指标的归约方式见[指标系统](metrics.md)。

## 样本身份

[transferqueue_utils.py](../../../../verl/utils/transferqueue_utils.py) 在 `retain_sample_identity=True` 时保留 `sample_keys/sample_partition`。如果 dispatch 把 key 转成全局索引，bridge 会先查回原始 key。身份与样本一起经过 shuffle 和 micro-batch 切分；读取 teacher hidden 时不能假设师生拥有相同的切分边界。

[HiddenBatchIO](../../../../verl/trainer/distillation/hidden_io.py) 优先读取通用字段，只在自身入口兼容 `full_vocab_keys/full_vocab_partition`。新生产者使用通用字段。身份负责定位样本，`teacher_positions` 则负责验证该样本内部的预测位置，二者不能互相替代。

## 两种布局与预测坐标

| 数据 | 布局 | 转换入口 |
| --- | --- | --- |
| Engine 产生的 log-prob、teacher 预测概率 | 完整输入序列的预测位置 | [no_padding_2_padding](../../../../verl/workers/utils/padding.py) |
| behavior、anchor、response mask | 已对齐 response target | [as_padded_response](../../../../verl/workers/utils/padding.py) |
| Full-KL hidden | 选中的预测 state 行 | [prediction_positions](../../../../verl/workers/utils/padding.py) |
| 位置诊断坐标 | 原始 response 内的 target 下标 | [response_target_coordinates](../../../../verl/workers/utils/padding.py) |

令一条输入长度为 $N$，mask 长度为 $M$，其中有效 mask 下标为 $j$。预测 state 下标为：

$$
i=j+N-M-1.
$$

再以 response 长度 $L$ 换算 target 下标：

$$
r=i+1-(N-L).
$$

例如 prompt 长度 3、response 长度 3，mask 为 `[1,0,1]`，预测 state 下标是 `[2,4]`，response target 坐标是 `[0,2]`。中间被 mask 的位置保留空洞。已经 response-aligned 的 behavior 或 anchor 只能补 padding，不能再次按完整序列切片或 shift。

`prediction_positions` 的定义位于公共 padding 模块；`trainer/distillation/hidden.py` 只保留历史导入兼容。

## 三种 mask 与分母

| 字段 | 负责的语义 |
| --- | --- |
| `loss_mask` | 原始训练资格及全局 token 归一化总体；Full-KL 也据此选择预测行 |
| `response_mask` | 当前目标实际应用的位置；拒绝采样可以减少它 |
| `metrics_response_mask` | 筛选前的诊断总体 |

[FSDPEngine.forward_backward_batch](../../../../verl/workers/engine/fsdp/transformer_impl.py) 在切 micro-batch 前，对 `loss_mask` 求和并跨 DP 归约，写入 `batch_num_tokens` 和 `dp_size`。训练时原始全局有效 token 为零会报错；某个 rank 局部为空可以参与计算。

[LossContext.from_batch](../../../../verl/workers/utils/loss_context.py) 读取调用级分母和缩放信息。它不会根据筛选后的 mask 重建分母。`aggregate` 委托已有聚合函数；Full-KL 的 token sum 通过 `normalize_token_sum` 归一化。若 DP 大小为 $D$，原始全局 token 数为 $N$，当前 rank 的 loss sum 为 $S_r$，局部返回：

$$
\ell_r=\frac{D}{N}S_r.
$$

DP 梯度平均后得到全局 token-mean 的梯度。显式 sequence 分母同样保留配置的全局 batch 数，包括零贡献序列。

拒绝采样后被拒绝 token 是原总体中的零贡献。例如原来 6 个 token，保留 3 个，保留位置的单位 loss 得到均值 $3/6$，而不是 $3/3$。全部被拒绝但原总体非空时可以产生零 loss；这本身不等于跳过 optimizer，动量或 weight decay 仍可能改变参数。

## 配置兼容边界

[LossContext.bind_legacy_config](../../../../verl/workers/utils/loss_context.py) 为旧 policy loss 构造浅复制配置，只替换调用级 `global_batch_info`，原配置不随 batch 修改。新增目标直接接收上下文即可，不应将运行时分母塞进共享静态配置。

## 验证入口

[test_loss_context_on_cpu.py](../../../../tests/trainer/distillation/test_loss_context_on_cpu.py) 验证配置隔离与分母；[test_rejection_denominator_on_cpu.py](../../../../tests/trainer/ppo/v1/test_rejection_denominator_on_cpu.py) 使用真实 TQ/Trainer 发布路径及两 rank 归约验证拒绝采样总体。Full-KL 的不等长输入、空 rank、mask 空洞和位置计数由[四卡集成测试](../../../../tests/special_distributed/test_full_vocab_fsdp2.py)覆盖。
