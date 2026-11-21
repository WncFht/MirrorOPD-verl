# 研究证据

报告、正式图和冻结绘图数据见各专题。本目录保存复算和后续测量所需的原始记录。运行新分析时请另指定输出位置。

| 证据目录                                      | 内容                                       |
| --------------------------------------------- | ------------------------------------------ |
| [absorption-geometry](absorption-geometry/)   | 吸收实验的逐矩阵 SVD、frame、谱和范数      |
| [absorption-models](absorption-models/)       | 最终模型及保留的补测权重                   |
| [optimizer-geometry](optimizer-geometry/)     | E124–E129 九时刻测量                       |
| [teacher-difference](teacher-difference/)     | 教师差异的绘图输入和逐条件测量             |
| [teacher-spectrum](teacher-spectrum/)         | 正式谱扫描、配对及配置                     |
| [teacher-construction](teacher-construction/) | 教师构造语料、权重及评测                   |
| [mimo-stage-transfer](mimo-stage-transfer/)   | MiMo、E91/E93 的逐题与逐回答评测           |
| [olmo-stage-transfer](olmo-stage-transfer/)   | OLMo 阶段对照及协议                        |
| [post-training-stages](post-training-stages/) | 九组后训练端点的复算数据                   |
| [latent-state](latent-state/)                 | E100/E102/E109 状态分析数据                |
| [prefix-s0](prefix-s0/)                       | J/P 的 1,280 条更新、配置与完成审计        |
| [prefix-expanded](prefix-expanded/)           | J/P 十二组正式结果与共同基线               |
| [qwen-prefix-six](qwen-prefix-six/)           | Qwen 六来源报告的训练指标和逐回答记录      |
| [qwen-four-gpu](qwen-four-gpu/)               | Qwen 三来源四卡对照                        |
| [reuse-and-optimizers](reuse-and-optimizers/) | Reuse、Inner Bank、Prefix IS 和 SGD 的测量 |
| [evaluation-inputs](evaluation-inputs/)       | 公共题集及跨域对照数据                     |
| [engineering-evidence](engineering-evidence/) | 正式验收和少量独有源码补丁                 |
| [tracking-logs](tracking-logs/)               | W&B 和 Trackio 数据                        |

Absorption 图从[专题 readout 和绘图表](../docs/experiments/teacher-transfer/checkpoint-dynamics/README.md#数据与复算)重画。E94–E104 缺失的旧逐矩阵数据、E81/E82 缺失的训练日志，以及本次删除的恢复 checkpoint 无法用于原位重算或续跑。
