# DeepSeekMath 同构 SFT 与 SFT+RL 教师对照

状态：方案已撤销，E69／E70 已从活动实验目录移除，未启动 bootstrap 或训练。以下保留原 4K 提案的条件和限制；当前执行方案见[MiMo 三类教师对照](../../../experiments/teacher-transfer/post-training-stages.md#mimo)。文中的 E69／E70 命令只描述撤销前的提案，当前入口将明确拒绝这两个编号。

E69、E70 固定 DeepSeekMath-7B-Base student，分别学习 Instruct（SFT）和 RL（SFT+RL）teacher，比较相同初始化、问题覆盖和学生更新预算下的教师吸收。配置采用 main 的原生 V1／FSDP2、shared-sync 和统一评估，参考 [JustRL 数学长程实验](../../../experiments/teacher-transfer/cross-domain-transfer.md)。两组均关闭 source heldout 与 training history RKL。当前状态为预设与 CPU 配置检查；尚未启动 bootstrap 或训练。

## 研究问题与模型身份

DeepSeekMath 的公开阶段链为 Base → SFT/Instruct → GRPO/RL。[^deepseekmath] 这里比较从同一个 Base 学习两个完整教师；Instruct student 学习 RL teacher 的增量实验留待后续，不计入 E69、E70。

| 编号 | Student                             | Teacher                                 | 教师阶段 |
| ---- | ----------------------------------- | --------------------------------------- | -------- |
| E69  | `deepseek-ai/deepseek-math-7b-base` | `deepseek-ai/deepseek-math-7b-instruct` | SFT      |
| E70  | `deepseek-ai/deepseek-math-7b-base` | `deepseek-ai/deepseek-math-7b-rl`       | SFT+RL   |

| 模型     | 固定 revision                              |
| -------- | ------------------------------------------ |
| Base     | `036a8c6189aac6e2fc4e07b46e1e57c6b647bca5` |
| Instruct | `0a5828f800a36df0fd7f0ed581b983246c0677ff` |
| RL       | `f3cd419a172ff5a100e9a563ebfe7900d78e9740` |

版本及规范路径由[模型注册表](../../../../assets/model_registry.json)维护。三份公开配置均为 Llama 架构、30 层、hidden4096、intermediate11008、32 个 attention/KV heads、102400 维输出头、不绑定 embedding/head、RoPE theta10000、上下文 4096；tokenizer 的 vocab 与 added-token ID 映射一致。[^base][^instruct][^rl]

两组统一使用固定 Instruct revision 的 tokenizer 和原生聊天模板，即 BOS 后的 `User: …\n\nAssistant:`；student 参数仍从 Base 初始化。Continuous-token family 显式设为 `default`，采用整个回答判分，不注入 R1/Qwen3 thinking 标记。训练、bootstrap 与 current 评估均停止于 EOS100001；RKL 使用完整原生输出头，不合并终止动作。

## 训练设置

| 项目           | E69、E70 的共同默认值                                                                       |
| -------------- | ------------------------------------------------------------------------------------------- |
| 数据           | 复用 JustRL 新协议的冻结 `questions-v2`，6400 道去重、去公共题家族污染的 DAPO 数学题        |
| 题目顺序       | 相同 prepared 清单与顺序，`shuffle=False`，跨 outer 不重复                                  |
| Rollout        | 每 outer 用 current student 生成，每题 1 答；T1、top-p1、top-k−1、seed42                    |
| 学生预算       | batch=mini-batch128、reuse32、50 outer，共 1600 次 optimizer 更新                           |
| 目标           | Full-vocab RKL，无任务奖励、无 anchor；`precompute_fused`、token chunk256、异步写与预取     |
| 优化器         | 全参数 AdamW，LR1e-6、constant、无 warmup、β=(0.9,0.999)、wd0.01、clip1                     |
| 精度           | 学生 FP32 主权重、BF16 混合计算；teacher BF16；rollout BF16                                 |
| 训练长度       | prompt2048／response2048，共 4096；超长 prompt 报错                                         |
| 资源           | 默认单节点 4 卡 shared-sync，teacher 无独立池；参数／optimizer offload，激活重算            |
| 推理与计算预算 | rollout TP1、每副本并发 16、prefill token budget16384；actor／teacher token budget32768     |
| 保存           | `save_freq=-1`；统一评估器在终点导出一个 HF 模型并做重载校验，无 optimizer／中途 checkpoint |

`DEEPSEEK_MATH_QUESTIONS` 默认指向仓库 `runs/unified-evaluation-20261031/questions-v2`，可改为等价的完整 prepared artifact。入口读取真实 panel IDs、采样次数和训练清单，并记录 manifest SHA256；覆盖不足以支撑预算或任一公共领域缺少 128 道 panel 题时直接失败。旧 source parquet 可留在 prepared artifact 中，两个新实验不会生成 source bank 或评分该文件。

已用固定 Instruct tokenizer 对现有题池做 CPU 长度检查：6400 道训练题最长 1436 tokens；公共 Math 最长 870、Code1818、Science2747、IF363。以下长度容纳全部公共题目，保留原生 4K 上下文，不静默裁题。

## 公共 RKL 与任务评估

| 领域    | Prompt cap | Response cap | Accuracy 覆盖                         |
| ------- | ---------: | -----------: | ------------------------------------- |
| Math    |       1024 |         3072 | MATH-500×1；AMC23、AIME24/25/26 各×16 |
| Code    |       2048 |         2048 | LCBv5×10                              |
| Science |       3072 |         1024 | GPQA-Diamond×4                        |
| IF      |       1024 |         3072 | IFEval×1                              |

公共生成均为 T1、top-p0.95、top-k−1、seed42、repetition penalty1。每模型／节点共 1648 题、6703 答；固定 RKL panel 为四域各 128 题，共 512 题，取 sample0。训练 loss 按 token 平均，公共 RKL 先在题内按有效 response token 平均，再按题和领域汇总。

| 读出                                   | 调度                                                        |
| -------------------------------------- | ----------------------------------------------------------- |
| Source heldout RKL                     | `source_enabled=False`；不生成、不评分                      |
| Training history RKL                   | `history_enabled=False`；不抽样、不评分；原始训练轨迹仍保留 |
| 固定 initial-student／teacher bank RKL | outer0、1、2 及每 5 outer，含终点 50                        |
| Current-student RKL 与全量 accuracy    | outer0、25、50；current sample0 复用同轮 accuracy 轨迹      |
| 初始 student 与 teacher accuracy       | 各自 bootstrap 阶段完成                                     |

两组共用相同公共题目、panel IDs、生成协议和初始 Base。每个 teacher 的 bootstrap 单独绑定模型身份；比较初始状态上的拟合时，应复用同一份 initial 生成产物，经各组 finalize 验证。不同 teacher 银行上的风险分别解释，不能把不同状态集合上的绝对 KL 当成相同测量。最终结果同时报告绝对残差、相对降幅与任务表现，不预设“完全吸收”的数值门槛。

## 入口与运行前提

原 DeepSeekMath 协议组合源码已随预设撤销；[公共 shell 入口](../../../../run_math_opd_ablation_fsdp2.sh)与[实验目录](../../../../launchers/math_experiment.py)现在维护 MiMo 等活动实验。

```bash
DRY_RUN=1 bash examples/mirroropd/run_math_opd_ablation_fsdp2.sh E69
DRY_RUN=1 bash examples/mirroropd/run_math_opd_ablation_fsdp2.sh E70
```

准备本地模型和公共银行后，在明确分配的四卡上运行对应编号：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash examples/mirroropd/run_math_opd_ablation_fsdp2.sh E69
CUDA_VISIBLE_DEVICES=0,1,2,3 bash examples/mirroropd/run_math_opd_ablation_fsdp2.sh E70
```

上面两个命令分别启动实验，不应在同一组 GPU 上同时运行。默认 bootstrap 路径分别为 `runs/deepseek-math-native/E69/bootstrap`、`runs/deepseek-math-native/E70/bootstrap`；可通过 `DEEPSEEK_MATH_BOOTSTRAP`显式指定。正式入口要求 bootstrap 的完整回执，`DRY_RUN=1`／`CONFIG_ONLY=1`仅规划或解析配置。原始模型评估使用同一 YAML 的 bootstrap，不开放这两组的 `-valid-student/-valid-teacher`旧验证入口。

首次运行先指定固定 `RUN_DIR`与实际`PYTHON_BIN`，以`CONFIG_ONLY=1`输出 resolved YAML，再按[统一评估 bootstrap](../../../../../../verl/trainer/evaluation/README.md#公共-suite-与-bootstrap)依次执行`initial`、`teacher`、`finalize`。两个 producer 会使用 GPU；本配置修改没有执行这些步骤。

模型权重尚待本地准备。固定 Base／Instruct revision 发布的是 PyTorch `.bin` shards，当前 full-vocab 接口要求 safetensors，因此须在模型准备阶段保真转换并保留来源及转换记录；RL revision 已有 safetensors。[^base][^instruct][^rl] 模型文件、兼容性预检、完整 bootstrap 和训练资源均通过后，才能报告 GPU 验收或正式结果。

### 参考文献

[^deepseekmath]: Shao et al. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. arXiv 2024. [论文 §3–4](https://arxiv.org/html/2402.03300)。

[^base]: DeepSeek-AI. DeepSeekMath-7B-Base. [固定版本](https://huggingface.co/deepseek-ai/deepseek-math-7b-base/tree/036a8c6189aac6e2fc4e07b46e1e57c6b647bca5)。

[^instruct]: DeepSeek-AI. DeepSeekMath-7B-Instruct. [固定版本](https://huggingface.co/deepseek-ai/deepseek-math-7b-instruct/tree/0a5828f800a36df0fd7f0ed581b983246c0677ff)。

[^rl]: DeepSeek-AI. DeepSeekMath-7B-RL. [固定版本](https://huggingface.co/deepseek-ai/deepseek-math-7b-rl/tree/f3cd419a172ff5a100e9a563ebfe7900d78e9740)。
