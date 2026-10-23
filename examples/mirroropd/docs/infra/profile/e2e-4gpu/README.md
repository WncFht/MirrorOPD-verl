# 四卡单轮数学 reasoning 训练现状

本专题先测量当前完整训练过程，交付现状报告和离线交互 HTML，再决定是否优化。单 teacher、FSDP2 shared-sync、SP=1、原 `precompute`、async write/prefetch 开启，使用正式 graph rollout；不改变目标、全局 batch、训练长度或学习率。GPU 仅限本机 0–3。

## 测量设计

- **baseline**：14 个真实训练步；保留完整 token 预算，步骤 4–13 提供 10 个稳态候选样本；预热是否充分由实际曲线判断。最后一步额外保存一次同步 checkpoint，并做 8 题 × 2 次、最大 1024 response token 的验证窗口，单独记账，不作全量准确率或验证耗时结论。训练的模型上下文配置保持原预算。
- **torch-profile**：同一配方运行 5 步，第 4 步采 actor/teacher 四 rank trace；vLLM 单独开启 discrete capture，截取前 128 个 engine iteration。该窗口是引擎迭代，不等同于每条请求的前 128 token；不拿详细采集步当吞吐基准。
- **nsys-profile**：短程同配方的训练 worker 系统时间线，补充跨线程 NVTX/通信/传输；vLLM 不会因全局工具为 nsys 自动被其现有 adapter 采集。此边界在报告中明确。

所有运行记录阶段 host span、真实 request 起止与 token 数、后端可用时间字段、每 2 秒一次 NVML/主机观测与 HTTP 指标。观测通过 `VERL_E2E_TIMELINE_DIR` 启用，默认关闭，不插入 CUDA 同步、不记录 prompt 文本或权重。四卡数据在同一主机时间命名空间中对齐；Torch trace 的时间基准另行检查，未确认的 trace 不强行合并。

运行目录计划为 `outputs/reasoning-e2e-profile-20261023`，冻结源码和实际参数由各任务 `launch.json`/`resolved_config.yaml` 确认。此前 Full-KL 算子实验位于上级目录，作为历史材料引用，不混入本次基线。

## 交付

最终报告解释配置、模型/进程关系、完整生命周期、阶段工作量、耗时/显存/等待、采集覆盖与未知项。交互 HTML 提供 step 选择、阶段和四卡时间线、rollout 并发/完成分布、显存曲线和配置/原始数据入口；静态图提供 PNG/PDF/SVG。性能优化候选与现状事实分开，第一阶段不自动修改性能配置。
