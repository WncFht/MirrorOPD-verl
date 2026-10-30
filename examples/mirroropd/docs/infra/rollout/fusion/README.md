# Rollout 来源与双融合实现

当前主线是原生 vLLM 双模型融合，使用已分配的 GPU 6、7，先完成单卡验收，再比较双卡拓扑；HF 保留作参考和回退。当前状态及通过范围以[实现与验收记录](2026-10-28-Rollout双融合实现与验收记录.md)为准。设计、实施和验证不是同一状态，不把早期短测作为最终数值路径或正式性能的验收。

- [实施入口](2026-10-28-Rollout双融合实施交接说明.md)。
- [配置与生命周期设计](2026-10-28-可组合Rollout策略架构与实现方案.md)。
- [实现细节与验收计划](2026-10-28-Rollout双融合实现细节与验收计划.md)。
- [vLLM 后端改造方案](2026-10-28-vLLM双模型融合重构与优化方案.md)。

开发实现已汇入 `feat/vllm-probability-fusion` 统一开发分支，main 的最新已提交研究工作也已整合；尚未交付回 main。科学实验继续在独立冻结 worktree 运行。当前 GPU 6、7 优先运行[十二组正式前缀实验](../../../experiments/rollout-policy/2026-10-30-E20设置下的六种前缀对照.md)；其余扩展验收后补，旧 HF 全套性能矩阵不再作为迁移前置条件。

## 阅读与更新顺序

查当前可用路径、限制和测量结果，先读实现与验收记录；继续实施时读交接说明。架构方案负责策略语义与生命周期，验收计划负责判定条件，vLLM 方案负责后端改造设计，实际完成范围以验收记录为准。方案中的源码基点描述设计当时的输入，不代表当前工作区。

新增测量更新验收记录及其证据入口；配置或生命周期改变时同步架构说明。已完成的要求不再作为新计划重复提出，未完成的 gate 保留。来源/混合策略的最终 acc 比较归 [rollout-policy](../../../experiments/rollout-policy/README.md)，此处只维护工程实现与验收。

## 本机开发入口

持久环境通过 `examples/mirroropd/benchmarks/fusion/run_fusion_setup_fsdp2.sh` 重建。单卡开发命令通过 `run_fusion_local_fsdp2.sh` 包装，该入口默认选择 GPU 7，也接受 `FUSION_GPUS=6` 或 `FUSION_GPUS=6,7`，核对所选 GPU 的 UUID、拒绝已有计算任务并使用独立 Ray 实例。其他入口须显式设置 `CUDA_VISIBLE_DEVICES`，不再默认选择 GPU 0–3。

真实训练、验证、保存恢复入口为 `run_fusion_native_lifecycle_fsdp2.sh dense current <唯一运行名>`，其中 `dense` 可替换为 `moe`，`current` 可替换为 `initial`。先用同目录 `prepare_vllm_workload.py` 固定开发数据；完整命令与结果以[验收记录](2026-10-28-Rollout双融合实现与验收记录.md)及原始运行配置为准。当前入口属于开发验收，不代表全部组合已经通过。
