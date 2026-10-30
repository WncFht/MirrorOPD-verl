# Rollout 组合启动说明

日常训练使用 `examples/mirroropd/run_math_opd_ablation_fsdp2.sh`。模型引用与组合含义见[统一语义](2026-10-31-Rollout统一模型来源与组合语义.md)，分阶段验证见[实施计划](2026-10-31-Rollout统一接口实施计划.md)。开发期间在 `MirrorOPD-verl-rollout` 执行；main 是否完成发布以[工作区说明](workspaces.md)为准。

## 配置入口

下面的命令通过 `CONFIG_ONLY=1` 解析最终配置，不分配 GPU。真实运行时去掉该变量，并设置分配给作业的 `CUDA_VISIBLE_DEVICES`、`STUDENT_GPUS` 和持久环境的 `PYTHON_BIN`。共享机器使用已有资源保护入口管理 GPU 锁；训练入口本身不选择或抢占 GPU。

默认行为是 current student 单模型生成，fusion 配置不会改变验证引擎：

```bash
CONFIG_ONLY=1 bash examples/mirroropd/run_math_opd_ablation_fsdp2.sh E20 --method full_kl
```

固定 initial 生成：

```bash
CONFIG_ONLY=1 bash examples/mirroropd/run_math_opd_ablation_fsdp2.sh E20 --method full_kl \
  'rollout_policy.model={kind:initial}'
```

Current 与监督 teacher 的概率融合：

```bash
CONFIG_ONLY=1 bash examples/mirroropd/run_math_opd_ablation_fsdp2.sh E20 --method full_kl \
  'rollout_policy.mode=token_mixture' \
  'rollout_policy.left={kind:current}' \
  'rollout_policy.right={kind:teacher}' \
  'rollout_policy.right_weight=0.5' \
  'rollout_policy.fusion_space=probability'
```

几何融合将 `fusion_space` 设为 `logit`。Batch fusion 将 mode 设为 `trajectory_mixture` 并移除 `fusion_space=probability`。两种 mixture 的 left/right 均可指定 `{kind:checkpoint,ref:模型资产ID或完整目录}`；固定 A/B 不需要改变 learner 的初始化模型。Current 可以放在任意一侧。

Single 使用 `model`，mixture 使用 `left/right`，不能混用。未指定引用时，single 默认为 current，mixture 默认为 current＋teacher。Alpha 始终表示右侧权重；精确的 0/1 会自动使用对应单模型。批内来源、权重版本和系数保持不变。

## 后端与资源

`rollout_policy.execution.backend` 默认为 `vllm_fusion`，`hf_fusion` 只能显式选择，失败时不会自动回退。HF 保留数值参考能力。当前原生融合支持 Qwen2 dense 配对、Qwen3 dense 第一组件与 dense/MoE 第二组件；current 映射到第一组件。模型引用独立不代表任意结构、词表或模型族都已支持。

原生 fusion 的 `execution.max_num_seqs` 是活跃请求数，`execution.kv_cache_memory_bytes` 是组合引擎的 KV 预算。HF 的 `micro_batch_size` 是静态微批；两者不共用配置。普通 single 与验证使用 `actor_rollout_ref.rollout` 配置。此轮保持既有执行参数，不进行性能调参。

## 手动保存与恢复

保存仍使用 `trainer.save_freq`，恢复使用 `trainer.resume_mode=resume_path` 和 `trainer.resume_from_path`。非默认 rollout checkpoint 增加 `rollout_state.json`，核对策略、原始 initial 和固定模型的内容身份。恢复配置继续指向原始初始化模型；不要把 actor 恢复检查点当作 initial。

旧运行缺少新身份文件时，用其冻结代码继续。新运行遇到旧字段会直接报错；按[配置迁移表](2026-10-31-Rollout统一接口实施计划.md#配置迁移)修改后重启。模型不受支持、无法确认请求排空、权重接收失败仍会终止运行；此轮不加入自动重试。
