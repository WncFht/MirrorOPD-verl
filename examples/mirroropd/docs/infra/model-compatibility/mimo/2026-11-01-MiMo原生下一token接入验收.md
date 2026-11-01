# MiMo 原生下一 token 接入验收

MiMo-7B 的下一 token 主干已在本页记录的验收源码版本中接入 FSDP2 Full-vocab RKL 与原生统一评估。适配保留公开 MiMo 模型类型、词表和主干，不另建 Qwen 权重副本。本文记录执行合同及工程证据；三个教师的科学比较、预算和采样见[MiMo 实验设置](../../../experiments/teacher-transfer/post-training-stages.md#mimo)。

## 1. 执行合同

四份发布配置均包含自定义 `MiMoConfig` 和 `MiMoForCausalLM`。配置和模型源码随固定 revision 下载，checkpoint identity 同时绑定根目录的 Python 源码。师生配置、tokenizer、bootstrap、Ray worker 和导出重载均显式传递 `trust_remote_code=True`。Ray 的运行环境还须包含 HF 动态模块缓存目录，避免在反序列化配置时无法导入 `transformers_modules`。

MiMo 发布的模型类继承 Qwen2；普通下一 token forward 不调用 MTP 层。原生 `HFModelConfig` 在关闭 MTP 时将层数归零，训练不载入 MTP 参数；直接 HF 模型保留 MTP 模块时由 Full-vocab 适配器冻结。推测解码关闭，生成与评分均使用主干和无偏置线性 head。论文中的 MTP 加速成绩不作为本协议的吞吐参照。

词表支持为原生 151680 维。Base／RL-Zero 与 SFT／RL 的默认 EOS 不同，公共停止集合显式固定为 `[151643,151645]`，评分保留两个独立动作。共同 tokenizer 模板和 `qwen25` continuous-token builder 用于训练与公共生成。最大总上下文为 32768，训练长度与各域公共长度分别在科学设置中固定。

## 2. 验收证据与边界

实际环境为 H800、PyTorch 2.11.0+cu129、Transformers 5.12.1、vLLM 0.25.1+cu129、Ray 2.59.0、FlashAttention 2.8.3。环境版本保存在[原生验收包记录](../../../../../../runs/mimo-stage-transfer-20261101/native-a03/packages.json)。代码来自 main `6b736cbe` 加工作区兼容补丁，实际执行源码以[冻结哈希](../../../../../../runs/mimo-stage-transfer-20261101/native-a03/source-hashes.json)为准。

| 验收项                               | 状态与证据                                                                                                                                                                                                                                                |
| ------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| HF 下一 token 与 Full-vocab callback | 正常 logits 与 callback logits 逐元素相等，合成目标 RKL 反传有限，保留的 MTP 无梯度；[结果](../../../../../../runs/mimo-stage-transfer-20261101/hf-adapter-a02/complete.json)                                                                             |
| HF 总长 32768                        | 完整主干前向有限，检查位置 0、1023、7167、24575、32767；峰值 allocated 31.75 GiB，仍属于上述 HF 检查                                                                                                                                                      |
| 原生师生 bootstrap                   | Base initial、RL teacher、accuracy 与 finalize 均完成；[短程完成记录](../../../../../../runs/mimo-stage-transfer-20261101/native-a03/complete.json)                                                                                                       |
| 原生更新与读出                       | batch128、reuse32、outer1，共 32 次 optimizer 更新；公共固定银行、current RKL 和 accuracy 执行成功；[评估 manifest](../../../../../../runs/mimo-stage-transfer-20261101/native-a03/run/evaluation/manifest.json)                                          |
| 原生模型导出与重载                   | 两次短程验收的固定轨迹 RKL 最大误差均为 0；[修复后导出审计](../../../../../../runs/mimo-stage-transfer-20261101/native-a04/run/evaluation/final-model/export-audit/manifest.json)                                                                         |
| 独立 HF 可移植性                     | 修复后 HF 目录独立加载成功，435 个张量均以 FP32 存储，BF16 前向有限，普通生成输出 `42`；[结果](../../../../../../runs/mimo-stage-transfer-20261101/hf-export-load-complete.json)                                                                          |
| vLLM 编译路径与 32K 边界             | `enforce_eager=False`、TRITON_ATTN；32767 个输入 token 加 1 个生成 token 成功；[结果](../../../../../../runs/mimo-stage-transfer-20261101/compiled-vllm-complete.json)                                                                                    |
| CPU 合同                             | 配对、预设、禁用读出、MTP 梯度及 identity 共 80 项通过；FSDP2 与 PEFT 源码导出 2 项通过                                                                                                                                                                   |
| 模板长度                             | 6400 道训练题最长 1388 token；1648 道公共题最长 2802 token；[训练题](../../../../../../runs/mimo-stage-transfer-20261101/train-prompt-lengths.json)与[公共题](../../../../../../runs/mimo-stage-transfer-20261101/public-prompt-lengths.json)审计均无超长 |

短程验收只选 8 道真实评估题、每题 1 答，公共回答 cap256；训练只取 128 道题、回答 cap128。这个独立工程预算验证调用链和优化，不提供三个教师的正式吸收结论，也不计入 E71–E73 的训练预算。

源码打包修复另用同一个不可变 bootstrap、batch128／reuse1／outer1 执行真实保存与原生重载，源码见[该轮冻结哈希](../../../../../../runs/mimo-stage-transfer-20261101/native-a04/source-hashes.json)。随后从新 HF 目录独立加载，未从原始模型目录补代码。独立 HF 检查验证可移植加载与有限前向，未将其标成独立 HF 与 FSDP 的逐题 RKL 数值相等检查；数值相等证据来自表中的原生重载。

## 3. 已修复的兼容问题

普通单模型采样的预检原来拒绝所有自定义模型代码，导致统一评估在分配 worker 前退出。现在只在模型配置声明 `model_type=mimo`、`architectures=[MiMoForCausalLM]` 且显式信任代码时允许该采样路径，继续拒绝未验收的其他自定义架构和配置覆盖。MiMo 组合模型采样不在此验收范围。

自定义配置最初只在 driver 能导入，Ray worker 反序列化时报 `No module named transformers_modules`。MiMo launcher 现在将项目与 HF 动态模块缓存加入 driver 和 Ray 的 `PYTHONPATH`，bootstrap 同时转发配置中的 Ray 运行环境。Tokenizer 加载也显式信任该固定模型代码，避免交互询问和回退。

FSDP2 动态改变模型的 Python 类，原有 `custom_object_save` 把包装层的 PyTorch 源码写入 HF 导出，漏掉 `modeling_mimo.py`。源码打包现在从模型继承链取得原始 HF 类，创建只供源码检查的对象，不修改运行中模型的类型；PEFT 先解包到 Base。回归检查验证原始源码被复制、HF 自动类能加载、运行对象类型不变，并保留原有 PEFT 用例。旧导出保持冻结，其审计中的 `hf_reload_forward=not_tested` 不改写为通过。

正式运行使用[修复后的冻结源码哈希](../../../../../../runs/mimo-stage-transfer-20261101/formal-source-hashes.json)，沿用原生终点 HF 导出与固定轨迹重载校验。[三组完整配置及队列计划](../../../../../../runs/mimo-stage-transfer-20261101/formal-plan.json)已提交，指标同时保留文件记录与离线 W&B。正式科学结果产生前，只报告已完成的工程检查及实际运行状态。
