# Full-vocab 验证

本页维护 full-vocab 数据通路、模型接入和数值正确性的证据入口。性能结果由[性能专题](../../performance/full-vocab/README.md)维护，实际算法与运行支持以[代码走读](../../architecture/full-vocab.md)和[mirror 支持范围](../mirror-objectives/reference.md#settings)为准。

## 验证矩阵

| 层次          | 核对内容                                                                        | 测试入口                                                                                                                                                                                                    |
| ------------- | ------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Endpoint 数值 | Dense/FP64 对照、hidden/head VJP、额外 teacher 行、空局部 mask、非单位 upstream | [test_full_vocab_on_cpu.py](../../../../../../tests/trainer/distillation/test_full_vocab_on_cpu.py)                                                                                                         |
| Mirror 数值   | Target、eta、MDPO/BPO 值与 VJP、q/T 无梯度、诊断开关                            | [test_mirror_on_cpu.py](../../../../../../tests/trainer/distillation/test_mirror_on_cpu.py)、[test_mirror_on_gpu.py](../../../../../../tests/trainer/distillation/test_mirror_on_gpu.py)                    |
| 近目标精度    | 大行常数、小残差、实际混合精度链路                                              | [数值审计](../optimization/2026-10-26-数值链路审计.md)                                                                                                                                                      |
| 模型          | Qwen adapter、tied head、普通 forward 回退、teacher 重建                        | [test_full_vocab_adapter_on_cpu.py](../../../../../../tests/trainer/distillation/test_full_vocab_adapter_on_cpu.py)、[verify_teacher_hidden.py](../../../../benchmarks/full_vocab/verify_teacher_hidden.py) |
| 数据与 IO     | Key 重排、预测位置、mask 空洞、同步/异步读写与异常传播                          | [test_hidden_io_on_cpu.py](../../../../../../tests/trainer/distillation/test_hidden_io_on_cpu.py)                                                                                                           |
| 指标          | 行值复用、位置 sum/count、全局归约                                              | [test_full_vocab_position_on_cpu.py](../../../../../../tests/trainer/distillation/test_full_vocab_position_on_cpu.py)                                                                                       |
| 分布式        | FSDP head/backbone 更新、全局 token 分母、空 rank、checkpoint 下一步一致性      | [test_full_vocab_fsdp2.py](../../../../../../tests/special_distributed/test_full_vocab_fsdp2.py)、[test_mirror_fsdp2.py](../../../../../../tests/special_distributed/test_mirror_fsdp2.py)                  |

CPU 的 `precompute_fused` 请求现在应明确拒绝；参考数值检查使用 precompute/dense，真实 CUDA BF16 kernel 单独检查；小模型分布式恢复不替代 V1/Ray 任意故障点恢复；短训练通过不证明长期质量或目标单调下降。

## 已有证据

| 测量批次              | 覆盖内容                                                                              | 原始记录                                                                                                                                                                                          |
| --------------------- | ------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2026-10-22 原型       | 24 组 FP64 值/VJP 对照，最大误差约 5.44e-15；self-teacher、中心化身份；GPU 原型精度   | [CPU](data/probe-20261022/cpu.json)、[H800](data/probe-20261022/h800.json)                                                                                                                        |
| 2026-10-22 接入       | 多 backend 两卡、cached 四卡、不等长/空 rank、teacher hidden 重建、IO 与 checkpoint   | [验证汇总](data/implementation-20261022/verification.json)                                                                                                                                        |
| 2026-10-22 真实 smoke | 四卡 H800、1.5B 师生、precompute、batch8、response cap128，两次参数更新；未做正式验证 | [指标](data/implementation-20261022/smoke4.json)、[配置](data/implementation-20261022/smoke4-resolved_config.yaml)、[命令](data/implementation-20261022/smoke4-command.sh)                        |
| 2026-10-23 融合入口   | CUDA 值/VJP、四卡更新与 checkpoint、真实 Ray/TQ/vLLM 两步非零 LR 训练                 | [FSDP2](../../performance/full-vocab/data/final/production-validation/fsdp4-checkpoint/result.json)、[真实入口](../../performance/full-vocab/data/final/production-validation-retry/summary.json) |
| 2026-10-24 位置统计   | 公共分桶接口、mask 空洞、不等长/空 rank 与 teacher 额外行                             | [位置验收](../../../archive/evidence/root-data/position-common-20261024/README.md)                                                                                                                |
| 2026-10-24 模型支持   | Qwen3 tied head、输出支持、tokenizer/EOS 与保存协议                                   | [对齐诊断](../../model-compatibility/probes/2026-10-24-OPD实现验收与对齐诊断.md)                                                                                                                  |
| 2026-10-25 集成       | 四组分布式测试、response cap512 的真实 sampled/full 短训练                            | [四卡验收](../../../archive/engineering/2026-10-25-主分支四卡验收.md)                                                                                                                             |
| 2026-10-26 Mirror     | 近目标数值修正、CPU/CUDA 检查、MDPO checkpoint 与 eta=1 endpoint                      | [审计与范围](../optimization/2026-10-26-数值链路审计.md)、[实现验证](../mirror-objectives/reference.md#validation)                                                                                |

每行只描述对应输入与代码的完成事实，不把不同批次的测试项数相加作为当前覆盖率。需要验证当前代码时，使用上方矩阵选择与改动相关的检查，并在新的结果中记录版本与条件。

## 聚合与未来信用验收（2026-11-04）

本次在本地 H800 与 `mirroropd-fusion` 环境中执行。[结构化验收记录](../../performance/full-vocab/data/aggregate-future-20261104/validation.json)保留命令、版本、源码 SHA256、运行日志目录、失败和各 FSDP2 结果；[性能结果](../../performance/full-vocab/operators.md#聚合与未来信用复测2026-11-04)单独解释计时。没有将以下层次的数量相加作为独立覆盖率：扩展 trajectory suite 包含首轮 suite 的部分用例。

| 检查                              | 结果与实际范围                                                                                                                                                                                                               |
| --------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 首轮 CUDA distillation            | 259 passed；现有 GPU 数值测试，包括 EOS、近目标、FP16、forward/common support 与 MDPO/BPO                                                                                                                                    |
| 扩展 trajectory 与 stable logprob | 560 passed；其中 trajectory 540 组覆盖两种 precompute backend、FP32/BF16/FP16、六种 future、五种 aggregate，以及 teacher-only、weighted teacher 和 EOS；与独立 FP64 概率参考比较值/VJP，teacher/anchor 无梯度                |
| Rollout CUDA                      | 17 passed；dense/MoE CUDA graph、Flash KV、compact、grouped MM、attention 不变量与 optimizer 卸载缓存                                                                                                                        |
| 指标设备回传                      | 17 passed；16 项 CPU 与 1 项真实 CUDA 混合设备回归                                                                                                                                                                           |
| 入口校验                          | 74 passed；含新增十组 estimator/aggregate 启动前合同                                                                                                                                                                         |
| 双卡 FSDP2 主矩阵                 | 22 组中 21 组通过；两种 precompute backend × 六种 future，轮换五种 aggregate 与 Qwen2/Qwen3/Llama；另覆盖 dense/recompute/cached/streaming/Liger、forward teacher 额外行、置换 common support、BPO tied FP16 与 sampled MDPO |
| MDPO 拆分复核                     | 三组通过：BF16/FP32 混合精度 checkpoint 与重算；纯 FP32 checkpoint 加 backend 对比；普通 BF16 backend 对比                                                                                                                   |
| 四卡 FSDP2                        | precompute_fused、future mean、seq-mean-token-sum-norm 通过三次更新、空局部 rank、参数与 optimizer/scheduler 恢复及恢复后下一步等价                                                                                          |
| 算子性能门控                      | 512-token 七 backend 对 dense；4096-token 六分块 backend 一致性；72 组 aggregate/future/遍历控制对 dense，详见算子性能页                                                                                                     |

FSDP2 使用本地随机小模型，真实包装 head、teacher capture、CPU offload、collective 和 optimizer。它不启动 Ray/vLLM，也不替代完整科学评估或长期训练质量验证。五种 aggregate 的最终全局缩放在 FSDP2 测试中另与 dense 结果比较；trajectory 的五种行权重测试本身不代表五种全局分母都不同。

两组 weighted-MDPO future 补充检查也通过：η=0.3、κ=0.5、EOS `[1,4]`，分别为 fused/BF16/discounted_sequence/seq-mean-token-mean/checkpoint，与 precompute/FP16/mean/seq-mean-token-sum-norm/tied head。二者均经过真实 `update_actor`、三次 reuse、下一 outer 的 anchor 刷新和空 rank；checkpoint 组还检查恢复后更新与 optimizer state 等价。首次运行因新增测试入口未把 aggregate 传给 engine 而缺少 sequence 分母；`8837db9c` 修正测试接线后，两组重跑通过，生产设置未改。

另完成[真实模型短训练](../../performance/full-vocab/data/aggregate-future-20261104/smoke-native-summary.json)：Qwen3-1.7B-Base 学习 Qwen3-8B，两卡 H800、batch4、两轮各一次非零学习率更新，precompute_fused/chunk512、seq-mean-token-mean、future mean、EOS 合并 `[151643,151645]`。使用 Ray/TQ/vLLM 与默认 evaluator，outer 0/1/2 各验证 4 条 MATH-500 记录，两种 JSONL 产物均生成，主记录的题目／sample 身份唯一、status 全为 ok，进程返回 0。训练与验证 response cap 都是 32，属于通路验收，不解释任务质量。此运行没有覆盖 public suite 或 source/history 的 GPU 事件生命周期，也没有提供完整 actor 的成对性能结论。

首轮真实 FSDP2 在指标回传发现 CPU/CUDA state 混拼，已由 `4f68f9fe` 按源设备打包修复；相同配置重跑及混合设备回归通过。真实训练入口还发现 `validate_sampled_token_distillation` 对 full-vocab 沿用 sampled 的 token-mean 限制，`2584da10` 将限制收窄到 sampled，保留其原行为。

### Public suite 与 source/history 的 GPU 生命周期

两卡 H800、Qwen3-1.7B-Base ← Qwen3-8B 的两条独立新运行均完成 batch4、reuse2、2 outer，共 4 次非零学习率更新。训练使用 precompute_fused/chunk512、seq-mean-token-mean、future mean、EOS `[151643,151645]`；读出仍按 native、未合并 RKL 口径。评分 payload 为 16 MiB、host budget 为 4 GiB、score chunk128，启用两 rank 状态 hash 审计。训练与生成 response cap 均为 32。

| 路径                                                        | outer 0/1/2 的有效 RKL 行 | 有效评分 token | optimizer steps | 结果                                                 |
| ----------------------------------------------------------- | ------------------------- | -------------- | --------------- | ---------------------------------------------------- |
| Public suite，source2/history4；Math2、Code1、Science1、IF1 | 12/21/21                  | 384/626/626    | 0/2/4           | 三事件及根 manifest complete；四域 accuracy 产物可读 |
| 独立 SourceHistoryReadout，source2/history4                 | 2/6/6                     | 64/192/192     | 0/2/4           | 三事件及根 manifest complete；正常继续第二轮训练     |

[离线复核摘要](../../performance/full-vocab/data/aggregate-future-20261104/evaluation-gpu-audit.json)逐事件验证 `(bank,id)` 唯一覆盖、轨迹身份、原始 response mask token 数，以及两 rank 的 `unchanged/state_digest`；从 Parquet 独立复算 source 题均、history 批均和 public 分域／域均指标，误差阈值为 $10^{-12}$。当前与初始 student 的 accuracy 样本身份唯一且与生成覆盖一致，指标由评分表复算；32-token 回答只验证执行与产物，不能解释科学质量。两条运行均未启用最终 HF export，不作为导出／重载复测。

[SourceLifecycleProbe](../../../../benchmarks/evaluation/lifecycle_probe.py)在实际训练边界核对同步前 pending、重复 validate 不改写事件、同步后驻留版本正确并发布完成。在 outer1 的本地 `event.finish` 写入前注入一次发布错误，错误上抛、pending 保留，恢复后显式重试发布且不重新评分；outer2 正常完成。这覆盖本地发布失败，不模拟未完成远端 GPU RPC。原始 lifecycle JSONL、冻结源码、配置、退出状态与复核命令见[测量身份](../../performance/full-vocab/data/aggregate-future-20261104/evaluation-gpu-manifest.json)。

GPU 路径暴露的两处缺陷分别由 `0138972f` 与 `8e173e90` 修复：生成协议预检改由各 readout owner 声明；source 初始 RKL bank 显式跳过任务奖励，空生成结果不再被空 keys 清理错误覆盖。默认验证继续使用原 reward 协议。包含新增回归的评估／rollout CPU 选择为 98 passed；两个失败运行保留在测量身份中，没有覆盖原证据。

### 混合精度 MDPO 参数差异

**保留的未通过项：**MDPO＋EOS＋BF16/FP32 混合精度，三个 Adam 更新后的 fused/precompute 严格参数对比失败，attention key bias 的最大绝对差约 $1.63\times10^{-5}$，原门槛为 atol=$10^{-6}$、rtol=$10^{-5}$。没有修改该门槛。拆分复核说明混合精度恢复和 FP32/BF16 backend 对比各自通过；追加[逐参数梯度诊断](../../performance/full-vocab/data/aggregate-future-20261104/mdpo-diagnostic.json)复现原差异，出问题的 key bias 在前两步 optimizer 前梯度完全一致，第三步梯度最大绝对差仅约 $3.73\times10^{-9}$。这支持微小数值差经 Adam 放大的解释，但没有证明所有差异只来自此处；逐 rank 的参数、梯度和 optimizer 原始快照保留在运行目录，后续仍需分离 attention/backbone 与 head 的误差。原失败与后续通过均保留，不宣称全部 GPU 验收完成。

2026-11-04 的追加诊断先执行同 backend 的 A/A checkpoint 对照，最大参数差为 0，恢复后的参数与 optimizer/scheduler 检查通过。随后两 rank 保存三个主路径更新和三个恢复后参考更新，包含裁剪前／后梯度、参数、Adam 状态及每块 head 输入／输出。两路径的 head 输入逐元素相同；head 的最大 $dZ$ 差为 $2.27\times10^{-9}$，相对相同输入的 CPU FP64 投影和行公式，fused／precompute 的最大绝对误差分别为 $2.23\times10^{-9}$／$1.74\times10^{-9}$。该参考同时包含投影与行公式，不单独把差值归为 Triton kernel 误差。

前两步的全部参数与 optimizer 前梯度完全一致，第三步更新前参数也完全一致。第三步 key bias 的裁剪前／后梯度差均为 $3.73\times10^{-9}$，因此裁剪没有造成该差异。独立 FP64 AdamW 按已保存的前态、moment 和实际梯度复算

$$
\theta_t=(1-\alpha\,w_d)\theta_{t-1}-\alpha\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon},
$$

其中学习率 $\alpha=10^{-3}$、weight decay $w_d=0.01$、$\epsilon=10^{-8}$，$\hat m_t$／$\hat v_t$ 为 bias-corrected moments。复算预测该 bias 的两路径参数差为 $1.63338947\times10^{-5}$，实测为 $1.63338846\times10^{-5}$；该 bias 单路径 FP64 复算与 FP32 实际更新的最大残差为 $1.41\times10^{-10}$。head 的微小差异经低精度主干反传后进入小 moment 的 Adam 除法，足以解释这次参数差；不需要假设 checkpoint 漏恢复或 optimizer 换算法。这里的学习率比正式实验 $10^{-6}$ 大 1000 倍，不能据此外推正式训练的漂移幅度或质量。

[A/A 结果](../../performance/full-vocab/data/aggregate-future-20261104/mdpo-aa-fused.json)、[逐参数复算](../../performance/full-vocab/data/aggregate-future-20261104/mdpo-adam-replay.json)及[命令／哈希](../../performance/full-vocab/data/aggregate-future-20261104/mdpo-adam-replay-manifest.json)保存证据。[FSDP2 测试](../../../../../../tests/special_distributed/test_mirror_fsdp2.py)的 `--diagnostic-dir` 在断言前落盘；[复算入口](../../../../benchmarks/full_vocab/analyze_optimizer_drift.py)读取这组两 rank、三 update 的 weighted-MDPO fixture。原严格参数门槛仍失败且保持原值；该诊断没有修改生产 kernel、optimizer 或精度，也不宣称多步低精度轨迹逐位等价。

## 运行检查

CPU 数值和 IO 检查可从仓库根目录执行：

```bash
python -m pytest -q \
  tests/trainer/distillation/test_full_vocab_on_cpu.py \
  tests/trainer/distillation/test_mirror_on_cpu.py \
  tests/trainer/distillation/test_hidden_io_on_cpu.py
```

GPU/分布式测试需要已分配的设备与匹配依赖。使用现有原生环境，按具体测试的参数指定 world size、backend 与输出目录；短 smoke 可使用 [smoke_full_reverse_kl.sh](../../../../smoke/smoke_full_reverse_kl.sh)，其短长度、验证关闭和 attention override 不能作为正式性能设置。

Frozen teacher 的 SDPA 单模型重建只验证 logits/位置关系，不替代多轮 IF 模板、监督 mask 和完整任务验证。性能比较前的数值门控见[测量方法](../../performance/full-vocab/methodology.md#正确性门控)。

## EOS 融合路径的回归

[停止类映射与独立参考](../../../../../../tests/trainer/distillation/test_termination_action_map_on_cpu.py)核对合并概率、logsumexp 链式梯度和极小修正保留；[CUDA EOS 融合](../../../../../../tests/trainer/distillation/test_termination_fused_on_gpu.py)使用独立 80 位概率空间 oracle，覆盖公共偏移、小 η、κ=0/0.5/1、FP16/BF16/FP32 与真正的 fused 分派。近零误差同时约束相对量和绝对量；新增 EOS 行梯度采用 rtol=2e-4、atol=1e-22，标量采用 rtol=2e-4、atol=1e-24，不放宽既有非 EOS 数值测试。

[生产词表](../../../../../../tests/trainer/distillation/test_mdpo_termination_on_gpu.py)、[teacher-only EOS](../../../../../../tests/trainer/distillation/test_termination_kl_on_gpu.py)和[FP16 AMP](../../../../../../tests/trainer/distillation/test_full_vocab_fp16_on_gpu.py)同时覆盖 precompute 与 precompute_fused。[原生两卡验收](../../../../../../tests/special_distributed/test_mirror_fsdp2.py)可加 `--termination-ids 1 4 --compare-backend precompute`，检验相同初态下的实际参数／Adam 更新；`--sequence-length 129 --token-budget 129` 强制多 micro-batch 累积，并覆盖 tied head、空局部 mask、冻结 anchor、梯度检查点和恢复后的下一步。η=1 使用 `--compare-backend precompute` 检查原 Full-KL 入口。

实际数值、完整 actor 性能与运行边界由对应测量报告记录；通过这些测试不等于长期任务质量得到提升。
