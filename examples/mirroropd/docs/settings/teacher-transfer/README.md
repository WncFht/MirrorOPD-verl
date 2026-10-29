# 教师来源差异与 OPD 可学习性：模型与实验设置

更新时间：2026-10-28。本目录按模型谱系和可控因素组织候选，不以权重是否已经下载为筛选条件。先复查已有 docs 的研究，再扩展到作者论文、模型卡、实际仓库文件、配置与 tokenizer。

[跨源蒸馏边界实验](../../experiments/teacher-transfer/transfer-mechanism/2026-10-28-跨源蒸馏边界分阶段实验计划.md#results)的 TB0–TB2 已完成：固定 Nemotron v1，九个配对各 64 更新，后续阶段未启动。本目录的候选范围不等于默认运行清单。论文结果与预算比较见[配套调研](../../literature/teacher-transfer/2026-10-28-蒸馏边界实验的论文对照与预算依据.md)。

| 文档／数据                                                                                                                                          | 内容                                                                                               |
| --------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| [模型谱系图与论文配对扩展](2026-10-28-模型谱系图与论文配对扩展.md)                                                                                  | 11 张 Mermaid 图；两篇 OPD 论文的模型配对；Nemotron 版本、Qwen3 分支、DeepCoder 等扩展及文末参考池 |
| [详细调研与推荐 setting](2026-10-28-异源教师分级对照与Student选择.md)                                                                               | 用户提出的六类对照；R1、Qwen2.5、Qwen3、Llama/Marin、Olmo/Hybrid 的具体候选与限制                  |
| [候选 setting CSV](settings.csv)                                                                                                                    | 58 个候选配对；固定 student、teacher、阶段、下载位置、优先级和特定 teacher revision                |
| [模型登记 CSV](../../literature/teacher-transfer/sources/model-registry.csv)／[JSON](../../literature/teacher-transfer/sources/model-registry.json) | 92 个公开 HF 仓库的元数据核对；包含原始结构配置、revision、可见权重文件状态，非 92 个推荐实验      |
| [接口与架构核对](../../literature/teacher-transfer/sources/tokenizer-and-architecture-audit.json)                                                   | 规范化 BPE merges 后的 tokenizer 比较、共享 ID 冲突、控制 token 增补与结构差异                     |
| [Semi-OPD 全部 17 组配对](../../literature/teacher-transfer/sources/semi-opd-17-pairs.csv)                                                          | 精确模型链接、T/NT 模式、共同训练 cap、初始 top-64 overlap、末步/AUC 方法差值                      |
| [Nemotron 固定版本](../../literature/teacher-transfer/sources/nemotron-revisions.json)                                                              | v1、v2、brorl 的完整 SHA 与配置；不可只写 repo 名                                                  |
| [新增 token 映射核对](../../literature/teacher-transfer/sources/additional-tokenizer-audit.json)                                                    | 三个 Nemotron 版本、Polaris-4B、GRPO-4B、R1-0528-Qwen3-8B                                          |
| [图谱扩展检索记录](../../literature/teacher-transfer/sources/2026-10-28-图谱扩展完整检索记录.md)                                                    | 本轮 21 条原始论文检索结果；模型卡另由官方 Hub 核对                                                |
| [图谱扩展来源快照](../../literature/teacher-transfer/sources/graph-expansion-sources)                                                               | 28 个仓库/版本的官方模型卡、配置、元数据快照；无权重文件                                           |
| [Mermaid 解析验收](../../literature/teacher-transfer/sources/mermaid-validation.json)                                                               | 两份文档中的图语法检查结果                                                                         |
| [第一轮 tokenizer 核对](../../literature/teacher-transfer/sources/tokenizer-comparisons.json)                                                       | 原始 JSON 比较；其中 merges 的字符串／数组序列化差别已在上项区分                                   |
| [完整论文检索记录](../../literature/teacher-transfer/sources/2026-10-28-完整论文检索记录.md)                                                        | 25 条原样检索记录，不作为人工推荐排序                                                              |

调研只读取公开论文和小型元数据，未下载模型权重、未运行 GPU 实验、未修改数学实验启动预设。用户会另行通知 GPU 可用时间；大型设置先讨论，再决定接入 `run_math_opd_ablation_fsdp2.sh`。

返回[研究目录](../../literature/README.md)；现象与机制证据见[谱滤波诊断](../../experiments/teacher-transfer/transfer-mechanism/spectral/README.md)。
