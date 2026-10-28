# Full-vocab 性能

本专题维护 OPD Full-KL、full MDPO、full BPO 的计算、显存和训练成本证据。性能结论按实际测量配置解释；运行默认值由[训练配方](../../../settings/训练设置.md)维护，数学与实现由[代码走读](../../architecture/代码走读.md)维护。证据核对日期：2026-10-26。

## 阅读入口

| 需要回答的问题 | 正文 |
| --- | --- |
| 测量范围、计时、显存和可比性怎样定义；如何复算 | [测量方法](methodology.md) |
| head-loss 的 backend、chunk、融合与容量代价 | [算子性能](operators.md) |
| teacher/actor、完整 outer、reuse、IO 与 full/sample 的实际成本 | [训练性能](training.md) |

三页分别维护方法、算子测量和训练测量。原始 JSON、配置、图与 trace 由各项结果直接链接；新增实验进入对应主题，不再为排队、首次检查、复测和完成分别新增正文。

## 可用于决策的结论

| 判断 | 适用范围与依据 |
| --- | --- |
| Precompute 的主要收益同时包含容量与避免重算 | [2026-10-26 四 backend 复测](operators.md#precompute-20261026)：N=4096 时与 dense 耗时接近，allocated 峰值减少约 76%；相对 recompute 耗时减少约 37%。共享 GPU 探索性样本，尚非正式性能或近目标数值验收。 |
| 融合的 head 收益需要在 actor 层复核 | [OPD endpoint 固定输入对照](operators.md#opd-endpoint-融合)：单卡 head-loss 耗时减少 43.5%；[四卡管线](training.md#固定输入管线)的 actor 耗时减少 6.4%。二者属于不同层级。 |
| 多次 reuse 配方应优先测 actor | [E44–E46 观察](training.md#多次复用的完整训练)：四卡 H800、reuse=16 时 actor 占 outer 的 74%–80%；下一项对照是 chunk、micro-batch 与梯度同步。 |
| 一次更新的配方仍需关注 rollout | [单次更新的完整训练](training.md#单次更新的完整训练)：四卡 graph 基线中 rollout 占 69.1%。不能把这个排序套到 reuse=16。 |
| Full-KL 整轮成本不能由词表大小推断 | [Full-KL 与 sampled 对照](training.md#full-kl-与-sampled)：已有八卡观察中 full actor 略慢、outer 较短、进程 allocated 较低；当前 fused 与显式 sampled token-IS 的固定输入对照仍缺失。 |
| 完整 target 缓存需要单独容量设计 | [容量预算](operators.md#容量预算)：64×7168 个有效位置、V=151936 时，一份 FP32 target 约 259.66 GiB；现有方案只保留 frozen hidden/head 并分块重建。 |

这些结论按各项已保存测量的源码和输入解释。2026-10-26 的 precompute 复测保存了当时工作区的模块快照，其他读数没有因此变成新版本结果。MDPO/BPO 的数值层 A/B 测量早于近目标梯度修正；版本边界见[算子测量](operators.md#mdpo-bpo-target-与诊断融合)及[数值审计](../../../experiments/teacher-transfer/optimizer-precision/mechanism/2026-10-26-MDPO内层KL反弹与数值链路审计.md)。

## 下一组有判别力的测量

以相同真实前缀、冻结监督、student/optimizer 初态为输入，在同一设备上比较参数变动；通过 loss/VJP 检查后再回到自然生成流程。

| 优先级 | 变量 | 希望分辨的代价 |
| --- | --- | --- |
| 1 | Head token chunk：128、256、512 | dW 累积次数、GEMM 形状与临时显存 |
| 2 | Actor token budget；梯度累积同步开关 | backbone 分批次数、通信与未分片梯度驻留 |
| 3 | Activation checkpointing | 重算开销与激活峰值；保持 token budget 不变 |
| 4 | q/T 投影、归一化与有界复用 | 冻结计算占比及缓存的内存/传输代价 |

这些是待测变量，不是已验证的最优配置。无配对数据时，不给出预计加速百分比。Full/sample 对照应明确 sampled 是 vanilla、token-IS 还是其他目标，并分别报告 actor 与完整 outer。

## 维护约定

每项测量在所属正文中记录输入、源码身份、硬件、采样协议、统计量和原始证据。结果更新时保留仍有解释价值的配置差异，删除已被覆盖的等待状态与临时建议。测量条件或代码变化后，旧数据仍按原条件解释，不改写成新版本结果。

原始数据、生成图和复算脚本保持一份；自动运行摘要写入对应 evidence 目录，人工维护页面只引用审阅后的结果。没有新增模型 benchmark 时，不刷新性能数字或宣称新版本加速。

GPU busy、MFU 和训练日志字段分别见[利用率口径](../2026-10-22-MFU与利用率口径.md)与[指标系统](../../architecture/metrics.md)。Rollout/验证参数选择见[推理性能](../2026-10-27-学生模型推理性能与参数选择.md)，数值正确性见[Full-vocab 验证](../../numerics/full-vocab/validation.md)。
