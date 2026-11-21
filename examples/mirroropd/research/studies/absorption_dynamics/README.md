# 吸收实验的 SGD 校准协议

科学选择规则由 `sgd_protocol.py` 维护：256 道独立校准训练题、32 道开发题、两类固定轨迹 bank 的 RKL 相对改善和有界 LR 边缘扩展。正式 held-out 面板不参与选择，预算与实际来源见[优化器配置及校准运行](../../../docs/infra/training/optimizers.md)。

执行、资源管理、配置生成和完整训练队列位于 `examples/mirroropd/tools/absorption_queue.py` 与 `sgd_campaign.py`，核心优化器位于 `verl/utils/fsdp_optimizers.py`。
