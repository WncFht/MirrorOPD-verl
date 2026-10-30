# Rollout 来源与双融合实现

当前主线是原生 vLLM 双模型融合，本机已授权使用物理 GPU 0–7。GPU 0、1、2、3、6、7 分配给科学实验，GPU 4、5 分配给工程验收；HF 保留作参考和回退。当前状态及通过范围以[实现与验收记录](2026-10-28-Rollout双融合实现与验收记录.md)为准。设计、实施和验证不是同一状态，不把早期短测作为最终数值路径或正式性能的验收。

- [实施入口](2026-10-28-Rollout双融合实施交接说明.md)。
- [配置与生命周期设计](2026-10-28-可组合Rollout策略架构与实现方案.md)。
- [实现细节与验收计划](2026-10-28-Rollout双融合实现细节与验收计划.md)。
- [vLLM 后端改造方案](2026-10-28-vLLM双模型融合重构与优化方案.md)。

开发实现已汇入 `feat/vllm-probability-fusion` 统一开发分支，并整合 main 的已提交研究工作；尚未交付回 main。工程资源入口固定在 `a34724a3`，科学执行代码独立冻结为 `c52e3eb6`，避免工程修改改变运行中的研究条件。工程范围仍包括 MoE 完整长度 refit/训练容量复测、TP2 组件数值与长历史，以及 TP2/双 TP1 的性能比较；资源扩展不代表这些检查已经通过。GPU4 已启动 MoE current、GPU5 已启动 MoE initial 的 batch64/cap7168 单卡生命周期复测，均使用 graph、活跃请求 4、KV10 GiB；各执行两轮更新、五来源短验证和保存，再独立恢复执行第三轮。实际退出与验收结果见[工程执行清单](../../../../../../../runs/vllm-fusion-expanded-20261030/acceptance-01/manifest.json)和同目录事件回执，启动或模型加载不计为通过。

[十二组前缀实验](../../../experiments/rollout-policy/README.md)复用已核对的三套完整任务基线，按六个独立 TP1 通道执行。此前两卡 v3 S0 的 304 次更新因模型路径迁移失败而单列保留；新尝试使用研究目录自有模型绑定，配置、模型内容与运行身份分别核对。当前分配和实际状态见[科学执行清单](../../../../../../../runs/prefix-fusion-scientific-v2/expansion-20261030/manifest.json)与[实时状态](../../../../../../../runs/prefix-fusion-scientific-v2/live-status.json)。旧 HF 全套性能矩阵不再作为 vLLM 迁移前置条件。

## 阅读与更新顺序

查当前可用路径、限制和测量结果，先读实现与验收记录；继续实施时读交接说明。架构方案负责策略语义与生命周期，验收计划负责判定条件，vLLM 方案负责后端改造设计，实际完成范围以验收记录为准。方案中的源码基点描述设计当时的输入，不代表当前工作区。

新增测量更新验收记录及其证据入口；配置或生命周期改变时同步架构说明。已完成的要求不再作为新计划重复提出，未完成的 gate 保留。来源/混合策略的最终 acc 比较归 [rollout-policy](../../../experiments/rollout-policy/README.md)，此处只维护工程实现与验收。

## 本机开发入口

持久环境通过 `examples/mirroropd/benchmarks/fusion/run_fusion_setup_fsdp2.sh` 重建。运行命令通过 `run_fusion_local_fsdp2.sh` 包装，由 `resource_guard.py` 持有物理 GPU 的锁、核对 UUID，并管理独立的 CUDA 映射、IPC 命名空间和 Ray 临时目录。入口支持授权范围内的 GPU 集合，默认值仍为 7；当前工程任务须显式选择 `FUSION_GPUS=4`、`5` 或 `4,5`，与六条科学通道分开。已有计算进程、UUID 不匹配或无法确认清理完成时，入口拒绝下一阶段。

模型路径使用可核对内容身份的固定绑定。科学入口通过 `PREFIX_MODEL_ROOT` 选择 `runs/prefix-fusion-scientific-v2/model-bindings/`，保持原模型 basename；原资产回执中的历史来源路径不再作为加载入口。路径迁移不改变模型文件、路由 seed 或训练目标，相关证明保留在研究运行目录。工程模型同样使用工程运行目录下独立的 `model-bindings`，具体路径由工程执行清单固定。

真实训练、验证、保存恢复入口为 `run_fusion_native_lifecycle_fsdp2.sh dense current <唯一运行名>`，其中 `dense` 可替换为 `moe`，`current` 可替换为 `initial`。先用同目录 `prepare_vllm_workload.py` 固定开发数据；完整命令与结果以[验收记录](2026-10-28-Rollout双融合实现与验收记录.md)及原始运行配置为准。当前入口属于开发验收，不代表全部组合已经通过。
