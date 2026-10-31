# Rollout 组合启动说明

日常训练使用 `examples/mirroropd/run_math_opd_ablation_fsdp2.sh`。模型引用与组合含义见[统一语义](2026-10-31-Rollout统一模型来源与组合语义.md)，分阶段验证见[实施计划](2026-10-31-Rollout统一接口实施计划.md)。日常从主仓库启动，开发使用仓库内 `.worktrees/rollout-maintenance`；版本和发布回执见[工作区说明](workspaces.md)。

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

非默认 rollout 当前用于单节点、FSDP2、全参数 dense learner、shared_sync full-vocabulary 蒸馏和单轮文本。两侧生成模型与 learner 使用相同 token-ID 映射，不隐式重分词。默认 single(current) 保留原训练路径及其能力范围。

`rollout_policy.execution.backend` 默认为 `vllm_fusion`，`hf_fusion` 只能显式选择，失败时不会自动回退。HF 保留数值参考能力。当前原生融合支持 Qwen2 dense 配对、Qwen3 dense 第一组件与 dense/MoE 第二组件；current 映射到第一组件。模型引用独立不代表任意结构、词表或模型族都已支持。

原生 fusion 的 `execution.max_num_seqs` 是活跃请求数，`execution.kv_cache_memory_bytes` 是组合引擎的 KV 预算。HF 的 `micro_batch_size` 是静态微批；两者不共用配置。普通 single 与验证使用 `actor_rollout_ref.rollout` 配置。固定 single 和 native fusion 的私有引擎窗口限制在训练 prompt cap＋response cap 内，保留更小的显式窗口；current/validation 引擎仍保留验证所需窗口。默认 1 GiB KV 预算用于保守的小规模配置，不承诺覆盖任意配对的最长上下文；较长上下文应显式设置足够的 KV 预算。P/Q dense 配对的独立生成候选和实测成本见[完整自然生成测量](2026-10-28-vLLM双模型融合重构与优化方案.md#85-完整自然生成的并发与拓扑选择)。这些候选仍需对应的训练驻留验证，不自动改写全局默认值。

## 手动保存与恢复

保存仍使用 `trainer.save_freq`，恢复使用 `trainer.resume_mode=resume_path` 和 `trainer.resume_from_path`。策略路径和采用公共采样合同的默认 single checkpoint 增加 `rollout_state.json`，核对策略、原始 initial 和固定模型的内容身份。恢复配置继续指向原始初始化模型；不要把 actor 恢复检查点当作 initial。

`rollout_state.json` 现在使用 schema 2：模型身份由加载所需的权重、配置和 tokenizer 文件决定；普通说明文件和优化器文件不参与身份。恢复比较模型绑定、系数时钟与采样参数，超时和执行预算单独记录并接受当前能力检查。旧 schema 1 或缺少身份文件的运行，用其冻结代码继续。新运行遇到旧字段会直接报错；按[配置迁移表](2026-10-31-Rollout统一接口实施计划.md#配置迁移)修改后重启。模型不受支持、无法确认请求排空、权重接收失败仍会终止运行；此轮不加入自动重试。

资产别名在公共启动脚本解析成固定路径；直接调用通用 trainer 时，`model/left/right.ref` 应提供本地目录。

## 采样合同迁移

单节点、FSDP2、colocated vLLM V1、单轮文本、本地原生模型配置的默认 single，现在与策略 single、mixture 端点和策略验证共用 tokenizer 支持、top-k/top-p 边界与 processed behavior log-prob。默认传输、训练更新和验证参数仍由既有入口执行。准确的适用条件见 [uses_common_single_sampling](../../../../../../verl/experimental/rollout_policy/validation.py)；多轮、自定义服务、LoRA、分离配置或其他后端保留原有执行合同，不据此扩大融合支持范围。

这是一项行为变更：在两个相同 logits、top-p=0.5 的边界例子中，公共过滤规则保留两个 token。默认 single 旧运行的自然轨迹可能改变；旧 checkpoint 应用原冻结代码恢复。新运行保存 schema 2 合同，缺少该合同的历史 checkpoint 不会静默切换采样规则。数学合同一致不承诺不同后端、执行形状或请求调度产生逐 token 相同的随机轨迹。

恢复可以调整 deadline、drain、微批和已支持的执行资源设置，但仍需通过后端能力检查；模型身份、采样参数及实际融合后端变化会拒绝恢复。调资源不保证数值逐位一致。新结构的 CPU/RPC 检查与真实 GPU 训练验收分开记录，旧 GPU 验收不能自动覆盖这项采样迁移。
