# Full-vocab 验证

本页维护 full-vocab 数据通路、模型接入和数值正确性的证据入口。性能结果由[性能专题](../../performance/full-vocab/README.md)维护，实际算法与运行支持以[代码走读](../../architecture/full-vocab.md)和[mirror 支持范围](../mirror-objectives/实现方案.md#settings)为准。

## 验证矩阵

| 层次 | 核对内容 | 测试入口 |
| --- | --- | --- |
| Endpoint 数值 | Dense/FP64 对照、hidden/head VJP、额外 teacher 行、空局部 mask、非单位 upstream | [test_full_vocab_on_cpu.py](../../../../../../tests/trainer/distillation/test_full_vocab_on_cpu.py) |
| Mirror 数值 | Target、eta、MDPO/BPO 值与 VJP、q/T 无梯度、诊断开关 | [test_mirror_on_cpu.py](../../../../../../tests/trainer/distillation/test_mirror_on_cpu.py)、[test_mirror_on_gpu.py](../../../../../../tests/trainer/distillation/test_mirror_on_gpu.py) |
| 近目标精度 | 大行常数、小残差、实际混合精度链路 | [数值审计](../../../experiments/teacher-transfer/optimizer-precision/mechanism/2026-10-26-MDPO内层KL反弹与数值链路审计.md) |
| 模型 | Qwen adapter、tied head、普通 forward 回退、teacher 重建 | [test_full_vocab_adapter_on_cpu.py](../../../../../../tests/trainer/distillation/test_full_vocab_adapter_on_cpu.py)、[verify_teacher_hidden.py](../../../../benchmarks/full_vocab/verify_teacher_hidden.py) |
| 数据与 IO | Key 重排、预测位置、mask 空洞、同步/异步读写与异常传播 | [test_hidden_io_on_cpu.py](../../../../../../tests/trainer/distillation/test_hidden_io_on_cpu.py) |
| 指标 | 行值复用、位置 sum/count、全局归约 | [test_full_vocab_position_on_cpu.py](../../../../../../tests/trainer/distillation/test_full_vocab_position_on_cpu.py) |
| 分布式 | FSDP head/backbone 更新、全局 token 分母、空 rank、checkpoint 下一步一致性 | [test_full_vocab_fsdp2.py](../../../../../../tests/special_distributed/test_full_vocab_fsdp2.py)、[test_mirror_fsdp2.py](../../../../../../tests/special_distributed/test_mirror_fsdp2.py) |

CPU 的 `precompute_fused` 回退测试不替代真实 CUDA BF16 kernel；小模型分布式恢复不替代 V1/Ray 任意故障点恢复；短训练通过不证明长期质量或目标单调下降。

## 已有证据

| 测量批次 | 覆盖内容 | 原始记录 |
| --- | --- | --- |
| 2026-10-22 原型 | 24 组 FP64 值/VJP 对照，最大误差约5.44e-15；self-teacher、中心化身份；GPU 原型精度 | [CPU](data/probe-20261022/cpu.json)、[H800](data/probe-20261022/h800.json) |
| 2026-10-22 接入 | 多 backend 两卡、cached 四卡、不等长/空 rank、teacher hidden 重建、IO 与 checkpoint | [验证汇总](data/implementation-20261022/verification.json) |
| 2026-10-22 真实 smoke | 四卡 H800、1.5B 师生、precompute、batch8、response cap128，两次参数更新；未做正式验证 | [指标](data/implementation-20261022/smoke4.json)、[配置](data/implementation-20261022/smoke4-resolved_config.yaml)、[命令](data/implementation-20261022/smoke4-command.sh) |
| 2026-10-23 融合入口 | CUDA 值/VJP、四卡更新与 checkpoint、真实 Ray/TQ/vLLM 两步非零 LR 训练 | [FSDP2](../../performance/full-vocab/data/final/production-validation/fsdp4-checkpoint/result.json)、[真实入口](../../performance/full-vocab/data/final/production-validation-retry/summary.json) |
| 2026-10-24 位置统计 | 公共分桶接口、mask 空洞、不等长/空 rank 与 teacher 额外行 | [位置验收](../../../archive/evidence/root-data/position-common-20261024/README.md) |
| 2026-10-24 模型支持 | Qwen3 tied head、输出支持、tokenizer/EOS 与保存协议 | [对齐诊断](../../model-compatibility/probes/2026-10-24-OPD实现验收与对齐诊断.md) |
| 2026-10-25 集成 | 四组分布式测试、response cap512 的真实 sampled/full 短训练 | [四卡验收](../../../archive/engineering/2026-10-25-主分支四卡验收.md) |
| 2026-10-26 Mirror | 近目标数值修正、CPU/CUDA 检查、MDPO checkpoint 与 eta=1 endpoint | [审计与范围](../../../experiments/teacher-transfer/optimizer-precision/mechanism/2026-10-26-MDPO内层KL反弹与数值链路审计.md)、[实现验证](../mirror-objectives/实现方案.md#validation) |

每行只描述对应输入与代码的完成事实，不把不同批次的测试项数相加作为当前覆盖率。需要验证当前代码时，使用上方矩阵选择与改动相关的检查，并在新的结果中记录版本与条件。

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
