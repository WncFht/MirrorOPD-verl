# OPD checkpoint 的谱与 frame 动力学

本报告独立维护 E94–E104 的 checkpoint 分析，结果、曲线、图与复算输入均保存在本目录。研究重点是：输出吸收较好与较差的 pair，在更新谱形状、更新子空间和参数残差上有哪些差异。后续 pair 使用相同协议接入；已完成的测量按实际覆盖更新。

**十一组全层结果显示：较好 pair 更早形成接近自身终点的更新方向，但不需要恢复 teacher 参数或完整谱；后期是否继续旋转、更新是否更低秩，都不能统一区分成败。** E94/E99/E100 全域匹配最好，E95/E97/E98 主要留下跨域缺口，E102/E104 在 Math 域内也有明显残差。完整证据见[指标综合分析](#已测结果的综合分析)，五张综合图的测量定义见[综合图索引](#g-全指标综合图5-类)。

恢复初始奇异值后的功能检验见[完整 bank RKL 换谱测量](谱恢复与RKL.md)，覆盖各 checkpoint 的未编辑／恢复谱对照。

后续实验的假设、对照和分阶段设置见[机制探测与干预设计](2026-11-13-机制探测与干预设计.md)，当前为讨论稿，尚未启动。

## 交互式 HTML 对比

在当前目录打开本地生成页面 `explorer.html`。页面内嵌 E94–E104 的矩阵级标量轨迹和绘图库，可下载后用现代 Chrome、Edge 或 Firefox 直接打开；无需服务器、网络或 GPU。它是已有测量的离线快照，新 checkpoint 需要重新构建才会出现。原始图模式读取相邻 `figures` 目录，分享这个模式时一并保留图片。

| 视图                  | 操作与用途                                                                                            |
| --------------------- | ----------------------------------------------------------------------------------------------------- |
| 多 pair 轨迹叠加      | 多选 pair，选择 RKL、frame、更新维度、参数残差或谱指标；按 rank、模块及单层筛选，显示矩阵中位数与 IQR |
| 层 × 模块热图         | 选定 checkpoint，逐 pair 展示层和模块；所选 pair 使用共同色标                                         |
| 几何指标 × RKL 散点   | 横轴为所选矩阵的几何指标中位数，纵轴为同 checkpoint 的整模型 held-out RKL；每点一个 pair              |
| Checkpoint 两两 frame | 选择 rank、层和模块，逐 pair 查看所有 checkpoint 的调整后 overlap；共同色标                           |
| 原始 28 类静态图对照  | 按图集原顺序选图，同屏展示多个 pair；这些图片的 rank 和层范围保持原测量定义                           |

建议先点 `E99 / E100 / E101`，比较同 student 的不同 teacher。RKL 可切换 Math/Code/IF/三域均值、initial/teacher bank，以及绝对值或终点轨迹相对各自 step0 的剩余比例。层和模块筛选只影响几何汇总，不改变整模型 RKL。跨尺寸 E102 的 teacher 参数指标缺失时显示“不适用”，不会补零。不同 pair 的同名矩阵可能形状不同；相同 k、原始 rank 和 SD 距离的跨尺寸比较须保留维度差异。

轨迹横轴保留实际更新步数，两两热图按保存时刻等距排列。阴影是所选矩阵间四分位区间，不是 seed 置信区间；角度 P90 先在每个矩阵的主夹角分布内计算，再跨矩阵汇总。对数轴隐藏零值和负值，CSV 保留原值。工具栏可导出 SVG，页面按钮可导出当前数值 CSV，并把筛选条件写入地址栏片段以便保存或分享。

交互轨迹目前覆盖 RKL 与 27 类几何标量，其中 frame 指标支持已有五档 rank。完整逐 rank 谱曲线、角度 ECDF、增量空间内外能量等保留在原始图模式，尚未全部改成交互叠加；其精确测量仍在原始 readout 中。浏览器数据见 [explorer-data](data/explorer-data.json.gz)，构建记录见 [explorer-provenance](data/explorer-provenance.json)。

CPU 构建入口为 `absorption_dynamics/build_explorer.py`，页面模板与交互逻辑分别在同目录的 `checkpoint_explorer.html`、`checkpoint_explorer.js`。构建环境需要 `ijson`、`plotly` 和本仓库依赖；不需要修改训练环境。生成 HTML 后，浏览器端没有第三方网络依赖。在仓库根目录运行：

```bash
python -m examples.mirroropd.research.studies.absorption_dynamics.build_explorer \
  --readout examples/mirroropd/docs/experiments/teacher-transfer/checkpoint-dynamics/data/readout.json.gz \
  --synthesis examples/mirroropd/docs/experiments/teacher-transfer/checkpoint-dynamics/data/metric-synthesis-20261113.json \
  --output examples/mirroropd/docs/experiments/teacher-transfer/checkpoint-dynamics/explorer.html
```

## 实验设置

| 项目          | 实际设置                                                                                                                    |
| ------------- | --------------------------------------------------------------------------------------------------------------------------- |
| 训练          | current student rollout；DAPO 每 outer 64 prompts，reuse 16；40 outer、640 次更新；学习率 $10^{-6}$；seed 42；不测 accuracy |
| 长度          | prompt 上限 4096；response 上限 7168；固定 bank 使用完整有效响应                                                            |
| Held-out bank | initial 与 teacher 各生成一套固定 bank；每套 Math 128、Code 64、IF 64；每 outer 评分共 512 行                               |
| RKL 聚合      | 每题有效 token 平均，再域内平均；all 为 Math、Code、IF 三域等权平均                                                         |
| Checkpoint    | 更新步 16、32、48、64、128、192、320、480、640；FP32 导出；原训练快照经过 native reload RKL 检查                            |
| 几何覆盖      | 每模型 28 层，每层 q/k/v/o 与 gate/up/down 七类矩阵，共 196；十一组共 2156 个矩阵任务                                       |
| 分解          | 读取 FP32 权重，FP64 精确 SVD；完整奇异值；frame 使用右子空间，k=16/32/64/128/256                                           |
| Teacher 比较  | E94–E101、E103–E104 使用对应坐标的同尺寸矩阵；E102 为跨尺寸，只比较 student 自身训练动态                                    |
| 统计单位      | 每个矩阵先计算指标，再汇总；阴影为矩阵间四分位区间，不是多 seed 置信区间                                                    |

初始 student 记作 $W_0$，teacher 记作 $W_T$，checkpoint 记作 $W_t$。累计更新为 $\Delta W_t=W_t-W_0$。它的奇异值 $\sigma(\Delta W_t)$ 与完整权重奇异值之差 $\sigma(W_t)-\sigma(W_0)$ 是两个不同对象。

## 扩展至 30 组的配对面板

保留原 E94–E111 的 18 组，新增 E112–E123，总计 **30 组：25 组同尺寸、5 组跨尺寸**。E94–E104 的 11 组训练已完成；其余 E105–E123 共 19 组在 2026-11-13 完成训练设置审计并恢复执行，每组四卡、两组并行，首批为 E105/E106。两组先生成各自的双固定 bank，再进入训练；已完成的 11 组不重跑。**19 组均通过 CPU 配对、完整题单和展开配置验收；GPU 更新与导出验收按各组实际完成状态记录。** 审计设置、模型协议和题单 SHA256 见[19 组正式训练协议](data/training-protocol-19-20261113.json)。

推荐重点是补齐容量、初始化、teacher 训练来源和目标域的对照。原 18 组中，R1/Qwen 多个 student 已经经过后训练，不能将其追加 SFT/RL 的差别直接推广到真正的 Base；跨尺寸当前仅 E102 完成，也不足以解释容量效应。新增模型不预设“好／坏”标签，按完成训练后的绝对 RKL、剩余比例与两套 bank 一起分组。

### 新增的 12 组

| 实验 ID | Initial student      | Teacher                       | Teacher 与 student 的关系                | 核心对照                                                                     |
| ------- | -------------------- | ----------------------------- | ---------------------------------------- | ---------------------------------------------------------------------------- |
| E112    | Qwen3-0.6B           | Qwen3-4B-Instruct-2507        | 跨尺寸；官方独立后训练终点               | 与 E102、E113 固定同一个 teacher，比较 0.6B/1.7B/4B student                  |
| E113    | Qwen3-4B             | Qwen3-4B-Instruct-2507        | 同尺寸；不是已证明的单一后训练延续       | 检验 E102 的高残差在同尺寸 student 下是否仍然存在                            |
| E114    | R1-Distill-Qwen-1.5B | Light-R1-7B-DS                | 跨尺寸；teacher 为 R1-7B 追加长 CoT SFT  | 与 E108/E109 补齐大 teacher 的 R1／RL／SFT 终点，与 E111 固定同一 teacher    |
| E115    | MiMo-7B-Base         | MiMo-7B-RL                    | Base → SFT → RL                          | 与 E107 固定同 teacher，改变 student 是否经过 SFT                            |
| E116    | MiMo-7B-SFT          | MiMo-7B-RL-Zero               | SFT student 学习另一条 Base 直接 RL 分支 | 与 E106 固定同 teacher；和 E115 补齐 MiMo 2×2 对照                           |
| E117    | Qwen3-1.7B-Base      | Thinking-Space/Qwen3-1.7B-SFT | 真正 Base → 全参数数学 SFT               | 补充 Base 起点；SFT 数据为 4B non-thinking 回答，不称为与 GT 同数据          |
| E118    | Qwen3-1.7B-Base      | GT-Qwen3-1.7B-Base-MATH       | 真正 Base → ground-truth reward GRPO     | 与 E117 共 Base，比较已公开 SFT 与直接 RL 路径                               |
| E119    | R1-Distill-Qwen-1.5B | DeepCoder-1.5B-Preview        | 同尺寸，代码 GRPO+ RL                    | 与 E94 数学 RL、E97/E98 多域 RL 保持同 student；观察 Code 与 Math 的吸收差别 |
| E120    | R1-Distill-Qwen-1.5B | UltraData-IF-1.5B             | 同尺寸，指令跟随 GRPO RL                 | 补充 IF teacher，检验“RL 易吸收”是否依赖目标域与 Math 训练域接近             |
| E121    | Tulu-3-8B-SFT        | Tulu-3-8B-DPO                 | SFT → DPO 偏好优化                       | 单独观察偏好阶段形成的 teacher 更新，不把 DPO 归成 RLVR                      |
| E122    | Tulu-3-8B-DPO        | Tulu-3-8B                     | DPO → RLVR；PPO、可验证奖励              | 观察直接后续 RL 增量在独立 8B 系列上是否容易吸收                             |
| E123    | Tulu-3-8B-SFT        | Tulu-3-8B                     | SFT → DPO → RLVR                         | 与 E122 固定同 teacher，检验缺少 DPO 初始化时的吸收与动态                    |

Qwen 官方资料支持各尺寸与发布身份，但未给出 April 4B 到 July 4B 的逐阶段父 checkpoint；这里把 E113 定义为同尺寸终点对照，不画直接继承箭头。[^panel-qwen-small][^panel-qwen-four][^panel-qwen-target] Light 的 7B 版明确为追加 SFT，不按同名 14B 的阶段链标为 RL。[^panel-light] Qwen Base 的两个分支分别来自公开 SFT 与 GT-GRPO 模型卡，数据、回答来源和预算并不相同。[^panel-sft][^panel-gt] DeepCoder 与 UltraData 分别是代码 RL 和 IF RL；UltraData 的数据集名字含 SFT，不改变该权重使用 GRPO 的训练来源。[^panel-code][^panel-if]

Tulu 三阶段的公开父型号为 SFT→DPO→RLVR，模型卡与固定配置支持这条链。[^panel-tulu-sft][^panel-tulu-dpo][^panel-tulu-final] 这里使用 **Tulu-3**，不混入 Tulu-3.1 的新发布版本。DPO 和 RLVR 结果按各自类别报告。

### 扩展后形成的关键对照

| 问题                                                     | Pair 组合                                    | 实际比较                                                                                   |
| -------------------------------------------------------- | -------------------------------------------- | ------------------------------------------------------------------------------------------ |
| E102 的困难来自小 student，还是 teacher 本身很远         | E112 / E102 / E113                           | 同一 4B teacher、0.6B/1.7B/4B student；容量变化也伴随初始化变化，不能称为纯参数量因果实验  |
| R1 大 teacher 中，蒸馏底座差异与追加 SFT/RL 如何关联吸收 | E108 / E109 / E114；E109 / E110；E114 / E111 | 前三组固定 1.5B student；后两对分别固定 Polaris／Light teacher，比较 1.5B 与 7B student    |
| SFT 初始化是否帮助吸收任意 RL teacher                    | E106 / E116 与 E115 / E107                   | MiMo 完整 2×2：两种初始化 × 两种 RL 路径                                                   |
| 真正 Base 上是否也出现 SFT teacher 更难吸收              | E117 / E118；另与 E99–E101 并列              | Base→SFT／直接 RL；官方后训练 student 的追加 SFT/RL 单列，不合并训练来源                   |
| 现有 RL 结论是否主要来自数学任务                         | E94 / E119 / E120；E97/E98 补充              | 同 R1 student、Math／Code／IF／多域 RL teacher，统一 Math-only OPD；报告域内匹配与跨域迁移 |
| DPO 与 RLVR 增量是否具有不同的 frame 动态                | E121 / E122 / E123                           | 连续阶段与相同最终 teacher 的不同 student 初始化                                           |

MiMo 的完整设计为：

| Initial student | Teacher：MiMo-RL-Zero，Base 直接 RL | Teacher：MiMo-RL，SFT 后 RL |
| --------------- | ----------------------------------- | --------------------------- |
| MiMo-Base       | E106，原计划                        | E115，已登记                |
| MiMo-SFT        | E116，已登记                        | E107，原计划                |

E105 的 Base→SFT 保留为 SFT 目标参考。四份 MiMo 模型的公开阶段关系与权重身份均已核实。[^panel-mimo] 这组对照尤其重要：历史 MiMo 中 SFT 初始化没有普遍改善 direct-RL teacher 的吸收；统一 640 更新、长响应和九快照后，可以检查这种差异伴随怎样的 frame 演化，而不是预设 SFT 初始化一定有益。

### 模型与输入协议核查

已核查 26 个候选及相关模型的固定版本公开模型卡、配置和本地资产。新增 12 组涉及的权重都已在共享模型目录中，未下载新权重。原始模型卡文本、revision、哈希、配置、候选理由和 CPU token 检查保存在[候选来源与核查数据](sources/pair-expansion-30-20261112.json)，不依赖运行目录。

| 对照组                     | 已核实的条件                                                                 | 当前输入协议与后续验收                                                                                         |
| -------------------------- | ---------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| Qwen 尺寸组 E112/E113      | head151936；0.6B 为 28 层、4B 为 36 层                                       | 与 E102 使用 `enable_thinking=false`；E99–E101 的 thinking=true 单列。跨尺寸只分析 student 自身参数动态        |
| R1→Light，E114             | 共享 ID 无冲突；student head151936，teacher head152064                       | 保留两侧原生 head；full-vocab RKL 不裁切为相同宽度，teacher 的额外概率质量保留在原生归一化中                   |
| MiMo，E105–E107、E115/E116 | 四模型 token-ID 一致、head151680、36 层                                      | 后续新启动统一双停止符 151643/151645，原生 EOS 元数据保持各自身份；GPU 短程仍须验收 remote-code/Ray 加载与导出 |
| Qwen Base，E117/E118       | head151936；teacher 多定义 4 个 token；新协议资产通过完整 tokenizer 定义检查 | Base 原生 EOS151643、pad151643、模板保留；定义额外 token，不改权重或 head；两组共用协议资产与双停止符          |
| DeepCoder / UltraData      | 与 R1 student 已定义 ID 一致、head151936、28 层                              | DeepCoder 固定公开 revision103033da，与既有 local-content 版本区分；统一 Math-only OPD，单列 Code/IF 域残差    |
| Tulu，E121–E123            | head128264、32 层、相同 ID；上下文 131072；EOS128001                         | 使用 `default` 文本协议与 student 模板；不套 Llama 专用连续 token 拼接；8B 四卡执行与导出待短程验收            |

正式 CPU 验收逐 pair 检查完整 tokenizer 分段／归一化／解码及 added-token 定义、共享 ID、safetensors 输出头、上下文、teacher 生成支持能否回送 student，并遍历 **2560 条训练题及 Math128／Code64／IF64 题单**。12 组均通过；逐组 head、EOS、题长最大值和题单 SHA256 见[接入验收数据](data/pair-expansion-acceptance-final-20261112.json)。早期七段文本／两个 prefix 的抽样结果仍保留在来源数据中，完整验收补足了其覆盖。

#### Base EOS 与新增 token 的处理

Qwen Base 原生 EOS 是 151643，SFT teacher 为 151645；MiMo Base／RL-Zero 为 151643，SFT／RL 为 151645。**停止生成与合并 KL 概率是两件事。** 后续 MiMo 五组及 Qwen Base 两组在 training rollout 和双 bank 都显式接受 `[151643,151645]`，而 `FULL_KL_TERMINATION_TOKEN_IDS=[]`，仍分别比较两个 token 的概率。已冻结 attempt 的配置和旧 bank 不改；停止协议变化后必须按新的生成身份重建 bank。

Qwen Base 缺少的定义是 `<tool_response>`、`</tool_response>`、`<think>`、`</think>`，对应 151665–151668。新资产 `Qwen3-Base-AbsorptionProtocol` 为这些已有 head 行添加与 teacher 一致的 token 定义，**不 resize、不重排或初始化参数**；权重链接到固定 revision 的原始 Base 文件，保留 Base 原生 EOS、pad 和 chat template。其含义是“Base 权重＋显式扩展的输入／合法生成协议”，结果不标为完全未改协议的原生 Base。两组共用这个资产，因此 SFT／GT 的比较不掺入不同适配。

三份 Tulu 均为 EOS128001、head128264，可直接构成阶段对照；原生 Llama-3.1 Base 的 head128256 不接入这条链。所有新增组都使用 student 渲染的共同 prefix 给 teacher 评分；teacher 的原生模板不替换训练 prefix。

### 统一训练、bank 与几何读出

新增组延续本报告 40 outer／640 更新、DAPO batch64/reuse16、current rollout、seed42、response cap7168、不测 acc 与九个 FP32 保存点。保持每 outer 双 bank 的 512 行读出，不增加 seed 或 outer 内快照。Code/IF teacher 仍用 Math 训练，因此其非 Math 残差首先解释为这套训练分布下的迁移边界。

训练数据的角色必须分开：**30 组 OPD 全部使用同一 DAPO Math 题单；teacher 原始训练数据则各不相同；held-out 三域只用于固定 bank RKL。** E119 的 Code teacher、E120 的 IF teacher 不会暗中改成 Code/IF OPD。若其 Code/IF 残差较高，先解释为 Math-only 训练下未充分迁移；这一面板不能证明它们在对应领域无法蒸馏。

| Teacher 分支         | 公开 teacher 训练来源                                     | 本轮 OPD 输入 | 解释时保留的差异                                                  |
| -------------------- | --------------------------------------------------------- | ------------- | ----------------------------------------------------------------- |
| E117 Qwen Base→SFT   | OpenThought3-Qwen3-4B 的 non-thinking 回答，完整 SFT 两轮 | DAPO Math     | 与 GT 分支的数据、回答来源、训练预算不同，不能归因为只改变 loss   |
| E118 Qwen Base→GT    | MATH train、ground-truth reward、GRPO                     | DAPO Math     | 属于直接 RL 分支；不能假定与 SFT 使用同一题集                     |
| E119 DeepCoder       | Code RL、GRPO+                                            | DAPO Math     | 同时报告 Math 与 Code，区分域内匹配和跨域迁移                     |
| E120 UltraData-IF    | IF 子集、GRPO；数据集名含 SFT                             | DAPO Math     | 权重按 RL 分类；IF bank 不是 IF 训练数据                          |
| E121 Tulu-DPO        | 公开偏好混合数据、DPO                                     | DAPO Math     | DPO 独立分类，不与可验证奖励 RL 混称                              |
| E122/E123 Tulu final | SFT→DPO→RLVR；GSM/MATH/IF 可验证奖励、PPO                 | DAPO Math     | 两组固定 teacher，改变 student 起点；teacher 最终阶段具有多域覆盖 |

固定同 teacher 的容量／初始化组优先使用相同题目、相同模板与相同 teacher-bank 轨迹；只有生成身份完全匹配才能复用。Initial bank 随 student 改变，两者分别报告。原生输出支持、条件输出支持和不同 thinking 模式具有不同测量身份，不把数值直接拼成同协议下降曲线。

几何仍分析完整层、七类矩阵、全部保存点、k16/32/64/128/256。Qwen0.6B/R1-1.5B/Base1.7B 每组 196 矩阵；Qwen4B/MiMo 每组 252；Tulu 每组 224。新增 12 组共 **2604 个矩阵轨迹任务**，每个包含九个 checkpoint。跨尺寸组比较 student 自身动态；逐参数 teacher 差分和同坐标 frame 只在有效同坐标条件下计算。跨宽度比较使用随机基线校正 overlap，并同时报告 k/q、能量覆盖和模块分布。

按当前本地权重张量形状、每组九份稠密 FP32 快照推算，新增 12 组约需 **1.75 TiB** 裸快照预算，尚不含 optimizer、bank、缓存和文件系统开销；实际 tied-weight 别名的保存方式可能改变大小。MiMo 每组约 263 GiB，Tulu 每组约 269 GiB，是主要存储项。不要把九份完整快照和额外重复的终点导出同时长期保留。这里没有新阶段实测耗时，不给未经测量的整批完成时间。

### 接入与开训前检查

公共入口仍是 `run_math_opd_ablation_fsdp2.sh E112` 至 `E123`，需要已准备的题单和完成的对应双 bank；登记编号不会自动提交训练。配对表、长度、预算与停止协议统一由 [absorption launcher](../../../../launchers/absorption.py) 提供；[队列](../../../../research/studies/absorption_dynamics/queue.py) 按实际 prepared manifest 分组，兼容原 18 组、全 30 组与子集，不重排历史组的分配。`queue prepare --pairs E105 ... E123` 只准备显式选中的未完成组；未指定 `--pairs` 时保留完整面板入口。

[CPU 接入工具](../../../../research/studies/absorption_dynamics/acceptance.py) 接受 `--questions`、`--experiments` 与新的 `--output`，可用于后续增补；只读本地模型和题单，不分配 GPU，不接受覆盖已有验收记录。Base 协议资产由 `python -m examples.mirroropd.assets.stage_protocols --models Qwen3-Base-AbsorptionProtocol` 构建，原始资产不被修改。

正式执行还需逐 student 验收生成／评分／更新／导出：确认 native/remote-code 加载、有效响应 token、EOS 与截断比例、finite loss/grad、16更新/outer、双bank512行和FP32导出重载。Base 出现空响应、很早 EOS 或高截断率时先查模板和停止身份，不能临时改变停止符后继续拼接同一条曲线。未实测的 student 并发 32；已实测的 R1-1.5B 用 128，既有 Qwen3-1.7B 用 64；Base1.7B 是不同生成协议，暂用 32。

几何读出还需遵守两点：k 不超过该矩阵可用秩，各模型同时报告 k/维度与随机基线；跨尺寸不计算无对应坐标的 teacher 参数残差。E113 的两个官方 4B 终点没有公开逐阶段继承证明，其差分应称为终点参数位移，不称为已知单次后训练更新。

### 备选及取舍

| 备选 pair                              | 价值                                                       | 本轮取舍                                                                                          |
| -------------------------------------- | ---------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| Qwen3-8B → 同一 4B-Instruct-2507       | 为固定 teacher 增加大于 teacher 的 student，补反向尺寸对照 | 下一轮补充容量曲线；本轮保留 Tulu 同 teacher 初始化对照，不替换 E123                              |
| Qwen3-1.7B-Base → e3-1.7B              | 与 E100 固定 teacher，检验官方后训练初始化的作用           | 下一轮补充 Qwen 初始化；本轮保留 E120 的 IF 域对照                                                |
| R1-7B → DeepScaleR-1.5B                | 大 student 学小 teacher，补反向尺寸                        | student head 大于 teacher，需显式共同条件支持；不与原生 head RKL 混排                             |
| Qwen Base → Co-rewarding-I／II         | 与 GT-MATH 比较自监督奖励来源，已有公开权重                | 待直接 RL 基线 E118 完成后扩展；不能假定不同公开分支的数据子集和预算相同                          |
| Olmo3 Base → IVON-SFT；IVON-SFT → C3PO | 新架构、贝叶斯 SFT 与后续 RL 链                            | 模型卡与资产已核查；使用已有协议资产仍需单独验收，C3PO 不标为 vanilla GRPO                        |
| R1-7B → Skywork-OR1-Math-7B            | 同一 7B student 的另一数学 RL teacher                      | 已有 Polaris 与 Light，新增容量／目标域对照的信息优先级更高；其直接父模型是 R1-7B，不是 Math Base |

备选模型的固定卡片和配置也保存在同一核查数据中。Gemma/Phi 等不同 tokenizer 家族、30B MoE 和 70B 暂不加入：前者需要另立跨词表蒸馏目标，后者显著增加四卡 teacher 执行成本，均偏离这轮统一 full-vocab RKL 动力学比较。这些备选保留为下一轮扩展清单，本轮不新增编号或启动任务。

下一轮扩展时优先考虑以下独立轴，不覆盖已登记的 Math-only 条件：

| 后续问题                       | 建议新增条件                                                            | 需要保留的协议信息                                                                                |
| ------------------------------ | ----------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| Code/IF 残差来自领域覆盖不足吗 | E119/E120 的 teacher 与 student 不变，另建 Code-only／IF-only／混合 OPD | 使用新实验 ID，匹配更新与 token 预算，重做 train/held-out 去重；分别报告原 Math-only 与对应域条件 |
| Base 的聊天包装是否限制了蒸馏  | 固定 Base 权重，另做原生 completion 与 chat prefix 对照                 | 模板与合法 token 支持是实验条件；两套 bank 单独生成，不能直接复用                                 |
| 容量效应在反方向也成立吗       | Qwen8B→固定 4B teacher；R1-7B→DeepScaleR1.5B                            | 后者需要显式共同条件输出支持，独立报告其条件 RKL                                                  |
| teacher 的来源是否重要         | Qwen Base 的 Co-rewarding I/II；R1-7B 的 Skywork OR1                    | 核对直接父模型、数据与预算；不凭模型名字归为 SFT 或 RL                                            |
| 新架构的阶段变化是否一致       | Olmo3 Base→IVON-SFT→C3PO                                                | 单独标明贝叶斯训练与协议适配，先验收，再决定是否与当前阶段组汇总                                  |

## Sigma variation 的定义

采用 SoSD 论文[^sosd]的归一化方式：对降序奇异值向量除以其元素和。

$$
p(W)=\frac{\sigma(W)}{\sum_i\sigma_i(W)},\qquad
D_\Sigma(a,b)=\Vert p(W_a)-p(W_b)\Vert_2.
$$

整体缩放矩阵不改变 $D_\Sigma$，因此同时保留核范数与 Frobenius 范数。完整权重分析包括初始 step 0；累计更新从 step 16 开始，因为零更新没有归一化谱。

| 测量                          | 解释                                                                             |
| ----------------------------- | -------------------------------------------------------------------------------- |
| $D_\Sigma(0,T)$               | 初始 student 与 teacher 的谱形状差距                                             |
| $D_\Sigma(t,T)$               | 训练过程中与 teacher 的谱距离；不等于能力差距                                    |
| $D_\Sigma(t_a,t_b)$           | 相邻或任意两个 checkpoint 的净谱变化                                             |
| $D_\Sigma(t_a,t_b)/(t_b-t_a)$ | 区间平均净变化速度；区间内部往返运动无法恢复                                     |
| $D_\Sigma(t,640)$             | 谱形状何时接近本次训练终点                                                       |
| $D_\Sigma^\Delta(t_a,t_b)$    | 累计更新自身的谱形状变化                                                         |
| Hellinger 距离                | $\Vert\sqrt{p_a}-\sqrt{p_b}\Vert_2/\sqrt{2}$；用于矩阵奇异值个数不同时的补充汇总 |

初始 teacher 距离大于 $10^{-12}$ 时保留剩余比例 $D_\Sigma(t,T)/D_\Sigma(0,T)$；基线更小时存为 null，并展示绝对距离。原始 SD 距离随谱维度及分布形状改变，Attention 与 MLP 的中位数曲线是描述统计；模块间比较同时查看 Hellinger、逐层结果与矩阵幅度。

完整权重 ISO 指标继续保留：$\delta_\Sigma=\Vert\sigma(W_t)-\sigma(W_0)\Vert_2/\Vert W_0\Vert_F$，$\rho_\Sigma$ 将分母改为 $\Vert W_t-W_0\Vert_F$；$\kappa_{\rm spec}$ 度量更新在初始奇异向量配对方向上的集中度。它们与归一化的 SD variation 分别回答幅度、更新构成和谱形状问题。

## 结果与图

<!-- results:start -->

数据更新于 2026-11-13 00:02:49（北京时间）。矩阵完成后包含全部九个 checkpoint；完整层要求七类投影齐全。

| Pair | Student → teacher                                                 | 已测矩阵 | 完整层 | 全模型几何覆盖 |
| ---- | ----------------------------------------------------------------- | -------: | -----: | -------------- |
| E94  | DeepSeek-R1-Distill-Qwen-1.5B → deepscaler                        |  196/196 |     28 | 完成           |
| E95  | DeepSeek-R1-Distill-Qwen-1.5B → JustRL-DeepSeek-1.5B              |  196/196 |     28 | 完成           |
| E96  | DeepSeek-R1-Distill-Qwen-1.5B → minimath                          |  196/196 |     28 | 完成           |
| E97  | DeepSeek-R1-Distill-Qwen-1.5B → nemotron                          |  196/196 |     28 | 完成           |
| E98  | DeepSeek-R1-Distill-Qwen-1.5B → brorl                             |  196/196 |     28 | 完成           |
| E99  | Qwen3-1.7B → Qwen3-1.7B-GRPO-KL-math-reasoning                    |  196/196 |     28 | 完成           |
| E100 | Qwen3-1.7B → e3-1.7B                                              |  196/196 |     28 | 完成           |
| E101 | Qwen3-1.7B → Qwen3-1.7B-Math                                      |  196/196 |     28 | 完成           |
| E102 | Qwen3-1.7B → Qwen3-4B-Instruct-2507                               |  196/196 |     28 | 完成           |
| E103 | Llama-3.2-3B-Instruct → Co-rewarding-I-Llama-3.2-3B-Instruct-MATH |  196/196 |     28 | 完成           |
| E104 | Llama-3.2-3B-Instruct → Llama-3.2-3B-Instruct-arithmetic-full     |  196/196 |     28 | 完成           |

所有 pair 的 held-out RKL 均来自完成的 640 更新训练。initial 和 teacher 两种 bank 分开报告；括号为终点相对初始的剩余百分比。

| Pair | initial bank：平均 RKL | teacher bank：平均 RKL | initial bank：Math RKL | teacher bank：Math RKL |
| ---- | ---------------------: | ---------------------: | ---------------------: | ---------------------: |
| E94  |   0.000484536（1.54%） |    0.00046542（1.59%） |   0.000257067（0.60%） |   0.000221113（0.57%） |
| E95  |    0.0361471（25.73%） |    0.0833731（44.30%） |    0.00211075（1.92%） |     0.0018512（2.01%） |
| E96  |    0.0269825（25.22%） |    0.0302335（21.05%） |    0.00637784（9.66%） |    0.00820156（6.25%） |
| E97  |    0.0622798（22.07%） |    0.0523476（23.78%） |    0.00356046（1.45%） |    0.00178394（0.97%） |
| E98  |    0.0910948（26.57%） |     0.121539（38.05%） |    0.00478825（1.85%） |    0.00468845（2.02%） |
| E99  |   0.000629341（2.62%） |   0.000674672（2.37%） |   0.000331525（1.07%） |   0.000342272（0.81%） |
| E100 |   0.000958098（2.30%） |   0.000863318（2.37%） |   0.000337136（0.72%） |   0.000343142（0.83%） |
| E101 |     0.0179311（8.58%） |     0.0421213（7.31%） |     0.0170987（9.97%） |    0.00910418（1.39%） |
| E102 |     0.846753（90.71%） |    0.990329（104.56%） |     0.252773（60.15%） |     0.274474（65.28%） |
| E103 |   0.00766375（12.95%） |    0.00423648（9.07%） |    0.00972591（8.62%） |   0.000378029（0.52%） |
| E104 |     0.245294（45.46%） |     0.250099（19.63%） |     0.135539（30.11%） |    0.0877396（12.50%） |

以下几何统计使用每个 pair 当前完整层的相同矩阵集合，均为逐矩阵中位数。不同 pair 尚未测齐时不据此排序。

| Pair | 纳入矩阵 | 完整权重：0→640 SD 距离 | 累计更新：16→640 SD 距离 | k64:16→640 调整后 overlap | OPD/teacher 更新幅度 | 参数残差能量比 |
| ---- | -------: | ----------------------: | -----------------------: | ------------------------: | -------------------: | -------------: |
| E94  |      196 |               7.521e-07 |                 0.005734 |                    0.5680 |               0.9374 |          1.167 |
| E95  |      196 |               1.429e-06 |                 0.007744 |                    0.4552 |               0.4576 |         0.9741 |
| E96  |      196 |               1.754e-06 |                 0.008998 |                    0.3800 |               0.0996 |         0.9963 |
| E97  |      196 |               1.462e-06 |                  0.00897 |                    0.4224 |               0.4794 |          1.001 |
| E98  |      196 |               1.801e-06 |                 0.009847 |                    0.3951 |               0.4983 |         0.9421 |
| E99  |      196 |               5.137e-07 |                  0.00936 |                    0.5034 |                1.465 |          2.561 |
| E100 |      196 |               5.593e-07 |                   0.0066 |                    0.5288 |                1.573 |          2.882 |
| E101 |      196 |               1.622e-06 |                 0.009431 |                    0.3646 |               0.3307 |         0.9739 |
| E102 |      196 |               1.624e-06 |                  0.01302 |                    0.3318 |       跨尺寸，不比较 | 跨尺寸，不比较 |
| E103 |      196 |                4.47e-07 |                  0.01551 |                    0.4649 |               0.9699 |           1.48 |
| E104 |      196 |               1.536e-06 |                   0.0135 |                    0.2728 |              0.08121 |         0.9929 |

### 跨 pair 总览

![overview-geometry-rkl](figures/overview-geometry-rkl.png)

![overview-rkl-endpoints](figures/overview-rkl-endpoints.png)

![overview-rkl-trajectories](figures/overview-rkl-trajectories.png)

![overview-shared-student-E103](figures/overview-shared-student-E103.png)

![overview-shared-student-E94](figures/overview-shared-student-E94.png)

![overview-shared-student-E99](figures/overview-shared-student-E99.png)

完整分类图集见[图集](图集.md)。下列四组图优先展示更新维度、方向和后续学习。

### E94: DeepSeek-R1-Distill-Qwen-1.5B → deepscaler

196/196 个纳入矩阵，28 个完整层；全模型覆盖。

![E94：update-dimension](figures/E94-update-dimension.png)

![E94：frame-rank-time](figures/E94-frame-rank-time.png)

![E94：frame-increments](figures/E94-frame-increments.png)

![E94：singular-energy-rank](figures/E94-singular-energy-rank.png)

### E95: DeepSeek-R1-Distill-Qwen-1.5B → JustRL-DeepSeek-1.5B

196/196 个纳入矩阵，28 个完整层；全模型覆盖。

![E95：update-dimension](figures/E95-update-dimension.png)

![E95：frame-rank-time](figures/E95-frame-rank-time.png)

![E95：frame-increments](figures/E95-frame-increments.png)

![E95：singular-energy-rank](figures/E95-singular-energy-rank.png)

### E96: DeepSeek-R1-Distill-Qwen-1.5B → minimath

196/196 个纳入矩阵，28 个完整层；全模型覆盖。

![E96：update-dimension](figures/E96-update-dimension.png)

![E96：frame-rank-time](figures/E96-frame-rank-time.png)

![E96：frame-increments](figures/E96-frame-increments.png)

![E96：singular-energy-rank](figures/E96-singular-energy-rank.png)

### E97: DeepSeek-R1-Distill-Qwen-1.5B → nemotron

196/196 个纳入矩阵，28 个完整层；全模型覆盖。

![E97：update-dimension](figures/E97-update-dimension.png)

![E97：frame-rank-time](figures/E97-frame-rank-time.png)

![E97：frame-increments](figures/E97-frame-increments.png)

![E97：singular-energy-rank](figures/E97-singular-energy-rank.png)

### E98: DeepSeek-R1-Distill-Qwen-1.5B → brorl

196/196 个纳入矩阵，28 个完整层；全模型覆盖。

![E98：update-dimension](figures/E98-update-dimension.png)

![E98：frame-rank-time](figures/E98-frame-rank-time.png)

![E98：frame-increments](figures/E98-frame-increments.png)

![E98：singular-energy-rank](figures/E98-singular-energy-rank.png)

### E99: Qwen3-1.7B → Qwen3-1.7B-GRPO-KL-math-reasoning

196/196 个纳入矩阵，28 个完整层；全模型覆盖。

![E99：update-dimension](figures/E99-update-dimension.png)

![E99：frame-rank-time](figures/E99-frame-rank-time.png)

![E99：frame-increments](figures/E99-frame-increments.png)

![E99：singular-energy-rank](figures/E99-singular-energy-rank.png)

### E100: Qwen3-1.7B → e3-1.7B

196/196 个纳入矩阵，28 个完整层；全模型覆盖。

![E100：update-dimension](figures/E100-update-dimension.png)

![E100：frame-rank-time](figures/E100-frame-rank-time.png)

![E100：frame-increments](figures/E100-frame-increments.png)

![E100：singular-energy-rank](figures/E100-singular-energy-rank.png)

### E101: Qwen3-1.7B → Qwen3-1.7B-Math

196/196 个纳入矩阵，28 个完整层；全模型覆盖。

![E101：update-dimension](figures/E101-update-dimension.png)

![E101：frame-rank-time](figures/E101-frame-rank-time.png)

![E101：frame-increments](figures/E101-frame-increments.png)

![E101：singular-energy-rank](figures/E101-singular-energy-rank.png)

### E102: Qwen3-1.7B → Qwen3-4B-Instruct-2507

196/196 个纳入矩阵，28 个完整层；全模型覆盖。

![E102：update-dimension](figures/E102-update-dimension.png)

![E102：frame-rank-time](figures/E102-frame-rank-time.png)

![E102：frame-increments](figures/E102-frame-increments.png)

![E102：singular-energy-rank](figures/E102-singular-energy-rank.png)

### E103: Llama-3.2-3B-Instruct → Co-rewarding-I-Llama-3.2-3B-Instruct-MATH

196/196 个纳入矩阵，28 个完整层；全模型覆盖。

![E103：update-dimension](figures/E103-update-dimension.png)

![E103：frame-rank-time](figures/E103-frame-rank-time.png)

![E103：frame-increments](figures/E103-frame-increments.png)

![E103：singular-energy-rank](figures/E103-singular-energy-rank.png)

### E104: Llama-3.2-3B-Instruct → Llama-3.2-3B-Instruct-arithmetic-full

196/196 个纳入矩阵，28 个完整层；全模型覆盖。

![E104：update-dimension](figures/E104-update-dimension.png)

![E104：frame-rank-time](figures/E104-frame-rank-time.png)

![E104：frame-increments](figures/E104-frame-increments.png)

![E104：singular-energy-rank](figures/E104-singular-energy-rank.png)

<!-- results:end -->

## 已测结果的综合分析

**当前最明确的结论是：输出吸收并不要求恢复 teacher 参数或 teacher 的完整谱；较好 pair 的早期累计更新方向更接近自身终点，但“后期不旋转”“更低秩”“更接近 teacher 更新”都不是统一的成功标准。** 完整权重谱在好坏组中都很稳定，更新谱与 frame 则持续演化。困难 pair 至少要区分训练域外缺口、teacher 目标距离大，以及同预算下的跨尺寸匹配困难。

以下统一使用 **E94–E104 十一组、每组全部 28 层×7 类矩阵、九个 checkpoint**，共 2156 个矩阵；Sigma0 干预覆盖 99 个完整阶段。全层重分析替代此前 E104 的 12 层预览，旧预览保留在其历史数据文件中。标量汇总含不同 rank、锚点展开后的 216 个字段，RKL 另保留两套 bank、三域、逐 outer 和相对位置统计。独立输入与计算结果见[综合分析数据](data/metric-synthesis-20261113.json)，全层基础汇总见[cohort 数据](data/cohort-comparison-20261113.json)。

### 输出结果：好坏至少分成五种情况

训练统一为 Math-only current-rollout OPD、640 更新。下表全部是终点的绝对 question-mean RKL，三域均值等权；不是正确率。完整初始值、相对剩余比例和逐 outer 曲线保存在综合分析数据中。

| Pair | Math initial | Math teacher | Code teacher | IF teacher  | 三域 teacher | 表现                    |
| ---- | ------------ | ------------ | ------------ | ----------- | ------------ | ----------------------- |
| E94  | 0.000257067  | 0.000221113  | 0.000291325  | 0.000883822 | 0.00046542   | 全域好                  |
| E95  | 0.00211075   | 0.0018512    | 0.00456613   | 0.243702    | 0.0833731    | Math 好，IF 缺口大      |
| E96  | 0.00637784   | 0.00820156   | 0.0113329    | 0.0711661   | 0.0302335    | 部分吸收                |
| E97  | 0.00356046   | 0.00178394   | 0.056659     | 0.0985998   | 0.0523476    | Math 好，Code/IF 有缺口 |
| E98  | 0.00478825   | 0.00468845   | 0.0675552    | 0.292374    | 0.121539     | Math 好，Code/IF 有缺口 |
| E99  | 0.000331525  | 0.000342272  | 0.00061074   | 0.001071    | 0.000674672  | 全域好                  |
| E100 | 0.000337136  | 0.000343142  | 0.000698185  | 0.00154863  | 0.000863318  | 全域好                  |
| E101 | 0.0170987    | 0.00910418   | 0.0318708    | 0.0853888   | 0.0421213    | 部分吸收；绝对残差仍在  |
| E102 | 0.252773     | 0.274474     | 1.35649      | 1.34002     | 0.990329     | 明显差                  |
| E103 | 0.00972591   | 0.000378029  | 0.0104639    | 0.0018675   | 0.00423648   | 对 bank 来源敏感        |
| E104 | 0.135539     | 0.0877396    | 0.346137     | 0.31642     | 0.250099     | 明显差                  |

- **全域匹配最好的核心组是 E94/E99/E100。** 两套 bank 的 Math 均在约 $2\times10^{-4}$–$4\times10^{-4}$，Code/IF 也保持低绝对残差。
- **E95/E97/E98 的主要问题是跨域残差。** E95 的 Code 已到约 0.0046，三域差距主要被 IF 的 0.2437 拉高；E97/E98 的 Code 和 IF 都有缺口。Math 终点剩余仅约 1%–2%，不能据此称为全域吸收。
- **E96/E101 是部分吸收。** E101 的 teacher-bank Math 剩余只有 1.39%，但绝对 RKL 为 0.00910，仍是 E99 的约 27 倍。相对下降大反映初始差距大，不代表终点已接近。
- **E103 对轨迹来源敏感。** teacher-bank Math 为 0.000378，initial-bank 为 0.00973，相差约 26 倍；它在 teacher 访问的前缀上匹配很好，在初始 student 轨迹上还有明显缺口。两套 bank 都需要展示。
- **E102/E104 在 Math 域内也明显差。** E102 的 teacher-bank 三域 RKL 比初始还高约 4.56%，其中 Code 剩余 121.58%、IF 剩余 102.66%；E104 有较大改善，但终点绝对残差仍高。

![两套 bank、三域绝对 RKL](figures/metric-synthesis/domain-rkl.png)

### 训练动态：E102 更像高残差变缓，E104 仍在明显改善

终点差并不表示过程相同。E102 的 teacher-bank Math 在 step128/320/640 分别为 0.3061/0.2857/0.2745，后半程只再下降 3.9%；initial bank 同期只下降 3.5%。**在当前训练分布和预算下，E102 已表现出明显的高残差变缓。** 这不等于已经证明容量是唯一原因。

E104 的 teacher-bank Math 在 step128/320/640 为 0.1478/0.1157/0.08774；后半程仍下降 24.1%，initial bank 下降 31.8%。**E104 的“难吸收”更准确地说是同预算下慢且不完整，尚不能写成延长训练也无效。**

E101 的两套 bank 则呈现不同时间尺度：step320→640，initial-bank Math 仅从 0.01820 降至 0.01710，而 teacher-bank 从 0.04937 降至 0.00910，下降 81.6%。一个终点比例掩盖了这种轨迹依赖。E99/E100 在 step320 后已到低残差，后半程 Math 只再下降约 5%–10%；相同“下降变慢”对应的绝对水平完全不同。

逐 outer 曲线见[全部训练轨迹](figures/overview-rkl-trajectories.png)。这些数据支持区分低误差收敛、高误差变缓和仍在改善三类过程。

### Frame：早期方向质量是最稳定的关联

这里的 frame 属于累计更新 $\Delta_t=W_t-W_0$，比较 $\Delta_t$ 的 top-k 右奇异子空间与自身 $\Delta_{640}$；step16 是第一个 outer 完成后的快照，不是第一次 optimizer update。

| Pair | k64 overlap：16→640 | 主角中位数 | 主角 p90 | 相对最优能量捕获 | k64 overlap：480→640 |
| ---- | ------------------- | ---------- | -------- | ---------------- | -------------------- |
| E94  | 0.568               | 35.350     | 76.389   | 0.870            | 0.969                |
| E95  | 0.455               | 45.005     | 80.094   | 0.763            | 0.948                |
| E96  | 0.380               | 52.977     | 82.812   | 0.675            | 0.927                |
| E97  | 0.422               | 47.894     | 82.519   | 0.719            | 0.958                |
| E98  | 0.395               | 50.494     | 83.207   | 0.671            | 0.952                |
| E99  | 0.503               | 40.534     | 80.158   | 0.821            | 0.908                |
| E100 | 0.529               | 37.788     | 79.246   | 0.847            | 0.908                |
| E101 | 0.365               | 54.224     | 84.471   | 0.535            | 0.940                |
| E102 | 0.332               | 57.055     | 84.615   | 0.543            | 0.861                |
| E103 | 0.465               | 45.589     | 84.235   | 0.804            | 0.931                |
| E104 | 0.273               | 64.109     | 86.405   | 0.483            | 0.878                |

同 student 的对照方向一致：R1 的 E94 优于 E96；Qwen 的 E99/E100 优于 E101，跨尺寸 E102 更低；Llama 的 E103 优于 E104。E95/E97/E98 位于 R1 两端之间，与其 Math 好、跨域仍有缺口的表现相符。E103 的 teacher-bank 成绩很好，但 early overlap 低于全域好组；两套 bank 的差异也提醒我们不要把一个几何量当作所有输出的标签。

Overlap 衡量平均 $\cos^2\theta_i$；我们减去了同维随机子空间的期望 $k/n$ 后再归一化。主角来自 $V_a^\top V_b$ 的奇异值，能量捕获则给方向按最终更新能量加权。这三者互补，不能当作三个独立发现。较好组的主角中位数更小，早期空间相对同 rank 最优的最终能量捕获也更高：E94 为 87.0%，E99/E100 为 82.1%/84.7%，E101/E104 为 53.5%/48.3%。

**“部分主方向较早形成”成立，“整个 top64 空间早早完全固定”不成立。** 即使 E94 的主角中位数只有 35.35°，p90 仍为 76.39°；各组最大主角的逐矩阵中位数都在约 89°。最大角度对好坏几乎没有区分度，尾部弱方向仍可大幅变化。

还要区分方向形成与累计幅度进度。按每个矩阵的 $\Vert\Delta_{16}\Vert_F/\Vert\Delta_{640}\Vert_F$ 再取中位数，E99/E100 已达到 49.9%/49.4%，E96 仅 14.5%，因此较好组的 early overlap 高也部分伴随学习进度更快。但 E95/E104 的幅度进度都约 18.5%，overlap 仍分别为 0.455/0.273；E103 的幅度进度最高、约 54.1%，overlap 却低于 E99/E100。幅度不是全部解释，也不能将方向关联直接写成独立因果机制。该比例可由综合数据中每个矩阵的`spectral_trajectory`更新能量直接复算。

奇异值边界间隔也已核对：$g_k=(\sigma_k-\sigma_{k+1})/\sigma_k$。early top64 的 gap 中位数仅约 0.0051–0.0086，终点约 0.0062–0.0075；它与 Math 终点 RKL 的相关很弱，early 在两套 bank 上分别为 +0.191/−0.100。各组 top-k 边界都不尖锐，邻近方向互换会影响硬截断 frame，但 gap 本身解释不了好坏排序。这也是同时查看能量加权捕获、多个 k 及主角分布的理由；k 已达到完整奇异维度时 gap 没有定义，汇总保留有效计数而不填零。

![不同 rank 的早期、后期 overlap 和能量覆盖](figures/metric-synthesis/frame-rank-comparison.png)

k16/32/64/128/256 的主要同 student 对照都保留早期差异。例如 E94 的 k16/k64/k256 overlap 为 0.661/0.568/0.432，E96 为 0.465/0.380/0.268；E99 为 0.602/0.503/0.418，E101 为 0.385/0.365/0.288。更大的 k 纳入较弱、后续变化更多的方向，因此 overlap 下降本身不代表测量失败。

### 后期旋转与区间增量：不能据此统一区分成败

**“学得差的后期 frame 一定转得更多”有明确反例。** E101 的 480→640 overlap 为 0.940，高于 E99/E100 的约 0.908；但其绝对 RKL 更差。E102/E104 后期 overlap 较低，分别 0.861/0.878，说明这两组后期仍有较多方向调整；这个现象不能推广成所有失败 pair 的判据。

累计更新稳定也不等于后续梯度都在同一个空间里。step16 top64 对最后 480→640 增量的实际能量捕获，E94 为 22.6%、E99 为 20.2%、E101 为 16.4%、E104 为 19.7%；对应约 77%–84% 的新增能量仍在早期空间之外。E94/E95 在第 16→32 区间的空间外能量分别仅 29.8%/36.2%，到最后区间则都约 78%。**早期空间覆盖较多最终累计能量，与晚期小增量继续探索新方向，可以同时发生。**

换成 step320 的 top64 空间，最后增量相对同 rank 最优的捕获，E94 为 54.3%，E102/E104 反而为 66.9%/62.6%。因此增量“更容易被旧空间覆盖”也不是成功条件。区间长短不同，比较方向捕获时同时参考增量范数和每更新净幅度；这些快照不能恢复区间内部的往返运动。

### 层和模块：差异跨七类矩阵存在，深层常更晚定向

早期 k64 overlap 的模块中位数如下，选取同 student 的代表性对照；全部模块统计保存在 cohort 数据中。

| 模块 |   E94 |   E96 |   E99 |  E100 |  E101 |  E103 |  E104 |
| ---- | ----: | ----: | ----: | ----: | ----: | ----: | ----: |
| q    | 0.488 | 0.336 | 0.461 | 0.463 | 0.320 | 0.415 | 0.265 |
| k    | 0.386 | 0.229 | 0.450 | 0.461 | 0.334 | 0.439 | 0.240 |
| v    | 0.449 | 0.279 | 0.503 | 0.522 | 0.345 | 0.456 | 0.268 |
| o    | 0.613 | 0.504 | 0.519 | 0.538 | 0.389 | 0.444 | 0.306 |
| gate | 0.715 | 0.544 | 0.645 | 0.672 | 0.437 | 0.540 | 0.379 |
| up   | 0.686 | 0.498 | 0.605 | 0.633 | 0.408 | 0.507 | 0.341 |
| down | 0.581 | 0.379 | 0.490 | 0.534 | 0.324 | 0.441 | 0.235 |

Gate/up通常较早稳定，q/k/v相对更晚；但down并不总比attention更稳定。**不能把结果概括成“MLP 稳定、attention 不稳定”，也不是只在 k_proj 上才能看到好坏差异。** R1、Qwen、Llama 所列好坏对照在七类模块的中位数上全部方向一致。

将 28 层分为 0–8、9–18、19–27 三段，early overlap 为：

| Pair | 浅层 0–8 | 中层 9–18 | 深层 19–27 |
| ---- | -------- | --------- | ---------- |
| E94  | 0.569    | 0.581     | 0.550      |
| E95  | 0.506    | 0.455     | 0.431      |
| E96  | 0.392    | 0.370     | 0.367      |
| E97  | 0.473    | 0.444     | 0.391      |
| E98  | 0.409    | 0.407     | 0.388      |
| E99  | 0.529    | 0.499     | 0.478      |
| E100 | 0.509    | 0.538     | 0.529      |
| E101 | 0.408    | 0.355     | 0.315      |
| E102 | 0.341    | 0.324     | 0.311      |
| E103 | 0.532    | 0.441     | 0.430      |
| E104 | 0.315    | 0.273     | 0.257      |

E95/E97/E99/E101/E103/E104在深层较低，E101从0.408降至0.315，E104从0.315降至0.257；但E100深层并不低于浅层，E94也只有轻微差异。较难pair的深层定向更慢是常见现象，不是普遍的单调深度规律。这里比较的是参数更新结构，未测激活或反传误差信号，不能直接把原因归于某层的特定语义功能。

### 参数幅度、方向和残差：功能吸收走到了另一套权重

记 $\Delta_T=W_T-W_0$，$a=\Vert\Delta_t\Vert_F/\Vert\Delta_T\Vert_F$，$c=\cos(\Delta_t,\Delta_T)$。每个矩阵严格满足：

$$
R=\frac{\Vert W_t-W_T\Vert_F^2}{\Vert W_0-W_T\Vert_F^2}=1+a^2-2ac.
$$

2156 个矩阵中的可比项、全部九个 checkpoint 已核对该恒等式，最大数值偏差为 $2.7\times10^{-15}$。幅度前两列按所有被测矩阵的能量求和；a、c、R 则分别为逐矩阵中位数，不能将三列中位数代入公式求等号。

| Pair | Teacher 更新/W0 | OPD 更新/W0 | OPD/teacher 范数比 a | 更新余弦 c | 参数残差能量 R |
| ---- | --------------- | ----------- | -------------------- | ---------- | -------------- |
| E94  | 0.1236%         | 0.1194%     | 0.937                | 0.373      | 1.167          |
| E95  | 0.4572%         | 0.2076%     | 0.458                | 0.256      | 0.974          |
| E96  | 2.7358%         | 0.2649%     | 0.100                | 0.070      | 0.996          |
| E97  | 0.4531%         | 0.2171%     | 0.479                | 0.246      | 1.001          |
| E98  | 0.5457%         | 0.2710%     | 0.498                | 0.317      | 0.942          |
| E99  | 0.0537%         | 0.0824%     | 1.465                | 0.209      | 2.561          |
| E100 | 0.0508%         | 0.0834%     | 1.573                | 0.185      | 2.882          |
| E101 | 0.7400%         | 0.2439%     | 0.331                | 0.206      | 0.974          |
| E102 | 跨尺寸          | 0.2722%     | —                    | —          | —              |
| E103 | 0.1127%         | 0.1099%     | 0.970                | 0.238      | 1.480          |
| E104 | 4.8468%         | 0.3898%     | 0.081                | 0.085      | 0.993          |

**E99/E100 是“输出吸收好，但参数没有恢复 teacher”的最清楚证据。** OPD 更新范数已超过 teacher 更新，a 为 1.465/1.573，而更新方向余弦只有 0.209/0.185，最终参数残差能量反而扩大到 2.56/2.88。它们不是因为 OPD 更新太小才 R≈1，而是找到了不同于 teacher 增量的功能匹配解。

E96/E104则确实有“相对于teacher，OPD走得很短”的成分：a仅0.100/0.081，且余弦0.070/0.085，因此R接近1。E101的a约0.331，余弦约0.206，$a^2$与$2ac$量级接近，也会使R靠近1。**R≈1 可能来自不同的幅度和方向组合，本身不是失败原因。**

与 teacher 更新 frame 更接近也没有统一排序：终点 k64 overlap，E99/E100/E101 分别 0.317/0.236/0.283；E100 匹配更好，方向集合却不比 E101 更接近 teacher。整体更新余弦可用于残差恒等式的方向项，不能仅因高维接近零就丢弃；但当前数据已表明它不适合作为独立的好坏分类器。

![相同全层覆盖下的几何与吸收特征](figures/cohort-comparison-full/absorption-features.png)

### 完整权重谱：保谱成立，恢复 teacher 谱不是必要条件

| Pair | 完整权重 SD：0→640 | 更新 SD：16→640 | OPD ρΣ  | OPD κspec | 终点/初始 teacher 谱距离 |
| ---- | ------------------ | --------------- | ------- | --------- | ------------------------ |
| E94  | 7.521e-07          | 0.005734        | 0.02608 | 1.531     | 1.022                    |
| E95  | 1.429e-06          | 0.007744        | 0.02674 | 1.717     | 0.9725                   |
| E96  | 1.754e-06          | 0.008998        | 0.02672 | 1.77      | 0.9943                   |
| E97  | 1.462e-06          | 0.00897         | 0.02629 | 1.658     | 0.9905                   |
| E98  | 1.801e-06          | 0.009847        | 0.0263  | 1.778     | 0.9364                   |
| E99  | 5.137e-07          | 0.00936         | 0.02202 | 1.084     | 1.636                    |
| E100 | 5.593e-07          | 0.0066          | 0.02221 | 1.124     | 1.783                    |
| E101 | 1.622e-06          | 0.009431        | 0.02192 | 1.144     | 0.9792                   |
| E102 | 1.624e-06          | 0.01302         | 0.02187 | 1.069     | 跨尺寸                   |
| E103 | 4.47e-07           | 0.01551         | 0.01809 | 1.145     | 1.218                    |
| E104 | 1.536e-06          | 0.0135          | 0.01761 | 1.142     | 0.9946                   |

完整权重的 0→640 SD 仅约 $4.5\times10^{-7}$–$1.8\times10^{-6}$，各组最优统一缩放系数都非常接近 1；去除缩放后的奇异值形状残差约 $1.6\times10^{-5}$–$5.9\times10^{-5}$。ISO 的 $\rho_\Sigma$ 约 0.018–0.027，说明完整奇异值净变化的范数远小于参数更新范数；$\kappa_{spec}$ 约 1.07–1.78，未出现好坏组的清晰分界。

**谱漂移小不等于完全没有统计关联。** 完整权重 SD 与 teacher-bank Math RKL 的描述性相关为 +0.755，难 pair 往往也走得更多。但它同时与 OPD 幅度有关，而且好坏组都处于极小漂移范围。结合换谱干预，现有证据不支持把“奇异值没学到 teacher”当作残差的主要原因。

更直接的反例是：E99/E100 完整权重到 teacher 的归一化谱距离，终点反而是初始的 1.64/1.78 倍，E94 为 1.02 倍，输出依然匹配很好。E96/E101/E104 的剩余谱距离约 0.994/0.979/0.995，参数谱并没有明显靠近 teacher，输出改善程度却相差很大。**归一化谱的接近程度，不是这里输出匹配的可靠替代指标。** E102 跨尺寸不计算这些对应参数距离。

完整权重 top64 frame 相对初始的主角中位数仅约 0.032°–0.167°；同时累计更新自身的主角却可达几十度。两者不矛盾：大幅度预训练权重的主空间几乎不动，其上很小的更新仍可以改变自己的方向结构。后文所有 early-to-final frame 结论都指更新，不能换成完整权重 frame。

### 更新谱和有效秩：更集中不等于“只需 16 维”

| Pair | Teacher top16 能量 | OPD top16 能量 | Teacher rank90/q | OPD rank90/q | OPD rank95/q |
| ---- | ------------------ | -------------- | ---------------- | ------------ | ------------ |
| E94  | 0.335              | 0.567          | 0.495            | 0.364        | 0.570        |
| E95  | 0.269              | 0.521          | 0.633            | 0.398        | 0.596        |
| E96  | 0.070              | 0.470          | 0.758            | 0.422        | 0.609        |
| E97  | 0.175              | 0.559          | 0.699            | 0.334        | 0.537        |
| E98  | 0.235              | 0.566          | 0.602            | 0.298        | 0.502        |
| E99  | 0.151              | 0.380          | 0.603            | 0.370        | 0.534        |
| E100 | 0.190              | 0.430          | 0.595            | 0.353        | 0.518        |
| E101 | 0.272              | 0.454          | 0.390            | 0.252        | 0.412        |
| E102 | —                  | 0.271          | —                | 0.343        | 0.493        |
| E103 | 0.175              | 0.427          | 0.626            | 0.311        | 0.479        |
| E104 | 0.098              | 0.322          | 0.623            | 0.259        | 0.411        |

OPD 更新的 top16 能量比例高于对应 teacher 更新，但这是更新总能量内部的比例；范数、能量和能量占比要分开。E96 的 OPD/teacher 范数比约 0.10，意味着该矩阵的总更新能量量级仅约 teacher 的 1%；即使 top16 占比从 teacher 的 7.0% 升至 OPD 的 47.0%，前 16 方向的绝对能量仍可小很多。上述平方关系逐矩阵成立，表中分别取中位数不强行作精确换算。

“更低秩吸收更好”被 E101 直接否定：其 rank90/q 约 0.252，低于 E99/E100 的 0.370/0.353，绝对 RKL 却更差。E104 的 rank95/q 仅 0.411，也低于 E94 的 0.570，却并未匹配好。各组 90% 能量仍需完整奇异维度的约 25%–42%，不能把 top16 能量较高写成 rank16 已足够表达所有学习。

累计更新的归一化谱从 step16 到 640 的 SD 约 0.006–0.016，稳定秩、rank90/95 和熵有效秩也随训练改变。例如 E99/E100 的 top16 占比从 step16 的 55.5%/57.0% 降至终点 38.0%/43.0%，E94 则大致保持 56.8%→56.7%。**成功 pair 也可以在训练中扩展更新维度；不需要保持固定的谱形状。**

### Teacher 来源：幅度差异清楚，κ和秩不能统一标记 SFT/RL

新增阶段按相对于指定 initial student 的来源标注。R1 已是蒸馏 SFT 模型，Qwen3-1.7B 也不是原始 Base，因此下面不是“同一预训练 Base 只改训练目标”的因果对照。来源依据见[师生配对说明](../../../settings/模型与师生配对.md)。

| Pair | 新增阶段 | Teacher 更新/W0 | Teacher top16 能量 | Teacher rank90/q | Teacher ρΣ | Teacher κspec |
| ---- | -------- | --------------- | ------------------ | ---------------- | ---------- | ------------- |
| E94  | RL       | 0.1236%         | 0.3352             | 0.4954           | 0.02565    | 1.356         |
| E95  | RL       | 0.4572%         | 0.2694             | 0.6328           | 0.4517     | 386.1         |
| E96  | SFT      | 2.7358%         | 0.07015            | 0.7578           | 0.02219    | 1.437         |
| E97  | RL       | 0.4531%         | 0.1746             | 0.6992           | 0.02493    | 1.159         |
| E98  | RL       | 0.5457%         | 0.2354             | 0.6016           | 0.0253     | 1.223         |
| E99  | RL       | 0.0537%         | 0.1514             | 0.6025           | 0.02164    | 1.007         |
| E100 | RL       | 0.0508%         | 0.1897             | 0.5947           | 0.02149    | 1.008         |
| E101 | SFT      | 0.7400%         | 0.2717             | 0.3896           | 0.02136    | 1.06          |
| E103 | RL       | 0.1127%         | 0.1747             | 0.626            | 0.01778    | 1.035         |
| E104 | SFT      | 4.8468%         | 0.0979             | 0.6226           | 0.02104    | 2.004         |

同家族 SFT teacher 相对 student 的更新幅度更大：R1 的 E96 为 2.736%，RL 为约 0.124%–0.546%；Qwen 的 E101 为 0.740%，RL 约 0.051%–0.054%；Llama 的 E104 为 4.847%，E103 仅 0.113%，约 43 倍。这是 teacher 端最一致的区别，也与同预算 OPD 覆盖 teacher 增量的比例有关。

秩的方向并不统一：R1 的 SFT teacher 能量更分散，Qwen 的 SFT teacher 却更集中。JustRL 是 RL，但 teacher 的 $\rho_\Sigma=0.452$、$\kappa_{spec}\approx386$，是明显谱变化异常值；它的 Math 仍能吸收至 0.00185，而 IF 残差很高。E104 的κ仅约 2 却更难匹配。因此“大κ就是难吸收”“RL 的κ一定小”均不成立。

![全层 teacher 幅度、能量集中度及 ISO 指标](figures/cohort-comparison-full/teacher-origins.png)

### 响应位置：部分缺口在后段，另一些从开头就存在

位置指标将每条保存响应按自身长度分为十段，再在段内按 token 加权；它与正文的每题平均 RKL 权重不同。这里的“最后 10%”不是固定第 6451–7168 个 token，也不能单靠这个图证明绝对上下文长度外推失败。

E95 的 initial-bank IF，从前 10% 的 0.0178 升至最后 10% 的 0.646；teacher bank 从 0.195 升至 0.844。长段落、约束维持或终止分布都可能贡献这部分缺口，现有指标尚未区分原因。E98 的 teacher-bank Math 也从 0.00413 升至 0.01385，但 E102 的 Math 在开头就已有约 0.329，后段仍约 0.263。**困难并不统一来自响应后半段：有的是后段放大，有的是整段分布都不匹配。**

![Math 与 IF 的相对位置 RKL](figures/metric-synthesis/position-rkl.png)

### Sigma0 干预：不是残差的主要来源，但也不是严格零作用

将各 checkpoint 全部被测 Transformer 矩阵改成 $U_t\Sigma_0V_t^\top$，保持当前 frames 及 embedding/head/norm；11 组×9 阶段均完成 512 条成对评分和无编辑 gate。配对评分使用同一 HF BF16 路径；与训练 FSDP 执行的差值另列，不能拿跨引擎数值充当干预效应。

终点 teacher-bank Math 的代表结果为：E94 的 0.000221113→0.000221298，E99 的 0.000342272→0.000349330，E100 的 0.000343142→0.000348521，E102 的 0.274415→0.274833，E104 的 0.0877205→0.0877268。好的仍好，差的也没有因为恢复初始谱而明显改善。

**当前大部分 RKL 改善，在重置完整权重奇异值后仍能保留。** 但效应不严格为零：594 个域×bank×阶段的对照中，最大相对变化为 E99/step320/teacher-bank Code 的 11.59%，对应绝对 RKL 只从 0.000631844 升至 0.000705064，变化 $7.32\times10^{-5}$。绝对与相对效应需并列，不能把好组低基线上的百分比变化夸大为整体能力丢失，也不能称所有谱变化都无关紧要。

![全部 99 阶段的最大换谱影响](figures/metric-synthesis/sigma0-effects.png)

图中每格取同阶段两套 bank×三域的最大绝对／相对变化，原始正负号及逐域数值见综合数据。完整成对曲线及逐题结果由[谱恢复与 RKL](谱恢复与RKL.md)维护。该干预只重置指定矩阵的奇异值，没有单独证明某一 frame 方向的因果作用，也未编辑 embedding/head/norm。

### 指标综合判断：保留关联和反例，不把相关当机制

| 指标                          | Math initial | Math teacher | 三域 initial | 三域 teacher |
| ----------------------------- | ------------ | ------------ | ------------ | ------------ |
| 早期 overlap                  | -0.891       | -0.973       | -0.836       | -0.836       |
| 早期能量捕获/同 rank 最优     | -0.864       | -0.945       | -0.809       | -0.827       |
| 后期 overlap                  | -0.491       | -0.436       | -0.273       | -0.273       |
| Teacher 相对更新幅度          | +0.709       | +0.903       | +0.721       | +0.721       |
| OPD 相对更新幅度              | +0.755       | +0.891       | +0.864       | +0.845       |
| 与 teacher 更新余弦           | -0.503       | -0.479       | -0.127       | -0.079       |
| 与 teacher 更新 frame overlap | -0.321       | -0.552       | -0.406       | -0.333       |
| Teacher κspec                 | +0.236       | +0.503       | +0.564       | +0.576       |
| OPD rank90/q                  | -0.564       | -0.400       | -0.355       | -0.400       |
| OPD top16 比例                | -0.555       | -0.400       | -0.245       | -0.264       |
| 完整权重 SD 漂移              | +0.509       | +0.755       | +0.691       | +0.664       |
| 参数残差能量                  | -0.515       | -0.758       | -0.697       | -0.770       |

相关的统计单位是 pair：student 自身特征 n=11，对应 teacher 参数特征 n=10，跨尺寸 E102 不参与参数相减。相同 student 的多个 teacher 并非完全独立，多个指标又是同一量的 rank 或时间展开；表中是描述性 Spearman 筛查，不做显著性或多重检验后的预测声明。最终 frame 是事后参照，early-to-final overlap 也不是训练前可用的预测器。

![几何指标与终点 RKL 的描述性关联](figures/metric-synthesis/metric-screen.png)

**现有数据最支持的解释是：较容易匹配的 teacher，OPD 在很小的权重变化内较早形成持续有用的更新方向，随后在近似保谱的邻域内完善输出分布；困难 pair 的早期方向与本次最终解更不一致，但困难来源并不唯一。**

具体来说：E95/E97/E98 主要暴露 Math 训练的跨域覆盖问题；E96/E104 的 teacher 增量远大于同预算 OPD 增量，且 E104 仍在持续改善；E102 既有跨尺寸/初始化差异，又表现为高残差变缓，当前不能只归因容量；E101 和 E103 揭示了前缀来源的重要性。功能匹配与参数恢复明显分离，完整谱、单一秩、κ、更新余弦和后期旋转都不足以单独解释好坏。

尚缺的机制证据是：激活／误差信号协方差、逐层功能干预和不改变其它设置的训练域或预算对照。当前没有这些测量，也没有 accuracy 或投影后继续训练结果，不能把 RKL 残差直接换成专项能力损失。

### 数据与图的复算

分析与绘图入口为[metric_synthesis.py](../../../../research/studies/absorption_dynamics/metric_synthesis.py)。下面命令只读取文档附近的自包含数据，不分配 GPU，不改变训练；输出五张综合图，全部同时保存 PNG/PDF 及机器审计。

```bash
python -m examples.mirroropd.research.studies.absorption_dynamics.metric_synthesis \
  --data examples/mirroropd/docs/experiments/teacher-transfer/checkpoint-dynamics/data/metric-synthesis-20261113.json \
  --figures examples/mirroropd/docs/experiments/teacher-transfer/checkpoint-dynamics/figures/metric-synthesis
```

重画前按仓库绘图约定设置 `FIGURE_VIZ_MPL`。teacher 来源、Math-vs-all 与全层散点三图由现有[cohort_comparison.py](../../../../research/studies/absorption_dynamics/cohort_comparison.py)读取`data/cohort-comparison-20261113.json`复算。原有逐 pair 全部指标图仍见[图集](图集.md)。

## 分析方法与判断

优先按两套固定 bank 的绝对 RKL 和剩余比例评价吸收，Math 与三域平均分别展示。当前实验测量输出分布匹配，不将其直接写成任务 accuracy 或能力完全恢复。

完整权重谱稳定与累计更新谱稳定分开判断。前者受预训练权重占主导的影响；后者直接描述 OPD 更新的组织方式。重点检查 RKL 快速下降与变缓的阶段，谱变化速度、frame 与终点的相似度以及后续更新的能量捕获是否同步改变。

谱形状相似不要求 frame 相同，frame 相似也不要求参数幅度相同。若完整权重谱几乎不变但 RKL 明显下降，意味着这段输出适配不需要大幅重塑完整谱；是否由子空间变化解释，要结合 frame 和能量捕获，不能由小谱距离单独推出。

参数残差按同坐标矩阵计算。令 $a=\Vert\Delta W_t\Vert_F/\Vert W_T-W_0\Vert_F$，$c$ 为两种更新的余弦，则

$$
\frac{\Vert W_T-W_t\Vert_F^2}{\Vert W_T-W_0\Vert_F^2}=1+a^2-2ac.
$$

因此更新幅度较大时，残差仍可接近或大于 1；幅度、方向和输出 RKL 必须并排看。

SoSD 论文的主体实验是预训练，其理论针对带假设的简化 Transformer。这里检验 OPD 的谱与输出动态关联；观察到谱稳定不能直接认定是吸收失败的原因。Checkpoint 间隔不等，曲线的每步速度是区间净变化速度，不是逐更新测得的波动。

## 图表索引与指标定义

[完整图集](图集.md)展示已经生成的图片。本节列出 **28 类单 pair 图和 4 类跨 pair 总览**，逐项说明实际计算的指标。单 pair 文件名为 `E编号-下表中的后缀.png`，另有同名 PDF。每类图通常包含多个面板；图的类别数不等于图片文件数。各 pair 按测量覆盖发布，不能把目录中的图类清单当成全部模型已经测齐。

### 比较对象与统计单位

| 记号               | 实际对象                                                           |
| ------------------ | ------------------------------------------------------------------ |
| $W_0,W_t,W_T$      | 初始 student、更新步 t 的 OPD checkpoint、teacher 的同坐标权重矩阵 |
| $\Delta_t=W_t-W_0$ | 从初始 student 出发的累计更新；参照时刻从 step16 开始              |
| $I_{a,b}=W_b-W_a$  | 相邻保存时刻之间的净增量，等于 $\Delta_b-\Delta_a$                 |
| $V_{t,k}$          | $\Delta_t$ 的前 k 个右奇异向量构成的正交基，位于矩阵输入空间       |
| $V^W_{t,k}$        | 完整权重 $W_t$ 的前 k 个右奇异向量，与更新 frame 分开              |
| $n,q$              | 矩阵输入维度 n；完整奇异值个数 $q=\min(m,n)$；实际 k 按 q 截断     |

除特别标明外，几何折线先对每个矩阵计算指标，再画矩阵间中位数和四分位区间。Attention 包含 q/k/v/o，MLP 包含 gate/up/down。只有七类矩阵都已测齐的层进入该 pair 的图。层 × 模块热图每格对应一个矩阵；主角分布图先计算单个矩阵内部的角度分布，再跨矩阵汇总。阴影体现不同矩阵的差异，不是多 seed 置信区间。

### A. 输出与 teacher 对齐：5 类

| 文件后缀                 | 实际测量指标                                                                                                                 | 横轴、分组与读法                                                                                |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| `sigma-teacher`          | 上排为 initial/teacher 两套固定 bank 的绝对 RKL：all、Math、Code、IF；下排为 $H(W_t,W_T)$ 与 $H(\Delta_t,\Delta_T)$          | 横轴为 optimizer 更新步；谱距离按 Attention/MLP 汇总。跨尺寸 pair 的 teacher 谱面板标为不可比较 |
| `sigma-teacher-distance` | 完整权重 $D_\Sigma(W_t,W_T)$、相对初始剩余距离 $D_\Sigma(W_t,W_T)/D_\Sigma(W_0,W_T)$、累计更新 $D_\Sigma(\Delta_t,\Delta_T)$ | 横轴为更新步；相对距离跳过近零分母。需要 teacher 与 student 同坐标                              |
| `frame-teacher-final`    | $V_{t,k}$ 与 teacher 更新 frame 的调整后 overlap，以及与自身终点 $V_{640,k}$ 的调整后 overlap                                | 横轴为更新步；k=16/64/256 三面板。区分“趋近自己的终点”和“趋近教师方向”                          |
| `parameter-residual`     | OPD/teacher 更新范数比 a、更新余弦 c、参数残差能量比 R                                                                       | 横轴为更新步；三面板分别展示幅度、方向和剩余参数差距                                            |
| `residual-layer-modules` | 终点的 a、c、R                                                                                                               | 横轴为七类模块，纵轴为层；定位参数差异集中在哪些矩阵，各面板独立色标                            |

RKL 先对每题有效响应 token 求平均，再在域内对题目平均；all 为三个域等权平均。它测量固定 bank 上的输出分布匹配。上表中的 H 是 Hellinger 距离，定义见 E 类；a、c、R 的公式见本节末尾。

### B. Frame 的方向与时间结构：8 类

| 文件后缀               | 实际测量指标                                                                                                                | 横轴、分组与读法                                                                                |
| ---------------------- | --------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| `early-final`          | 调整后 overlap $\widetilde O(V_{t,k},V_{640,k})$；早期 frame 捕获最终累计更新能量相对于最优 rank-k 的比例 $C^*_{t\to640,k}$ | 横轴为 anchor 更新步；曲线为 k=16/64/256                                                        |
| `frame-angles`         | 各 anchor 到终点的完整主角分布：33 点分位曲线，以及每 5 度精确统计的 ECDF                                                   | 上排横轴为分位点、纵轴为角度；下排横轴为角度、纵轴为累计比例。anchor=16/64/192/480，k=16/64/256 |
| `frame-angle-time`     | 到终点主角的 median、P90、最大值；各值先在单个矩阵内部计算                                                                  | 横轴为更新步，纵轴为角度（度）；k=16/64/256。区分典型方向和转动最大的方向                       |
| `frame-pairwise`       | 任意两个保存 checkpoint 累计更新 frame 的调整后 overlap                                                                     | 横纵轴均为更新步，颜色为矩阵间中位数；k=16/64/256，三面板共同色标                               |
| `frame-rank-time`      | 到终点的调整后 overlap、原始能量捕获 C、相对最优捕获 $C^*$                                                                  | 横轴为 anchor 更新步，纵轴为五档 k=16/32/64/128/256；三种量分别成图                             |
| `frame-modules`        | 七类模块各自到终点的调整后 overlap                                                                                          | 横轴为更新步，曲线为 k=16/64/256；每个模块单独一个面板，跨完整层汇总                            |
| `frame-layer-modules`  | anchor=16/64/192、k64 到终点的调整后 overlap                                                                                | 横轴为模块、纵轴为层；三个 anchor 共用色标，每格一个矩阵                                        |
| `weight-update-frames` | 完整权重 $V^W_{0,k}\leftrightarrow V^W_{t,k}$ 的主角中位数，以及累计更新 $V_{16,k}\leftrightarrow V_{t,k}$ 的主角中位数     | 横轴为更新步，k=16/64/256。两条线参照对象不同，完整权重参照 W0，更新参照 step16                 |

设 A、B 为同一输入空间内实际 k 维正交基，$c_i=\sigma_i(A^\top B)$，则主角和 overlap 定义为

$$
\theta_i=\arccos(c_i),\qquad
O(A,B)=\frac1k\sum_i c_i^2,\qquad
\widetilde O(A,B)=\frac{O(A,B)-k/n}{1-k/n},\qquad
d_{\mathrm{chord}}(A,B)=\sqrt{1-O(A,B)}.
$$

角度以度展示。调整后 overlap 扣除了同维随机子空间的期望 $k/n$；0 表示达到该基线，1 表示空间重合，可以略小于 0。当 k=n 时校正分母为零，调整值不定义。这里的 chordal distance 除以了 $\sqrt{k}$，取值在 0–1；与使用未归一化距离的论文对照时需换算。

捕获指标定义为

$$
C_{a\to b,k}=\frac{\Vert\Delta_bV_{a,k}\Vert_F^2}{\Vert\Delta_b\Vert_F^2},\qquad
F_{b,k}=\frac{\sum_{i=1}^{k}\sigma_i(\Delta_b)^2}{\Vert\Delta_b\Vert_F^2},\qquad
C^*_{a\to b,k}=\frac{C_{a\to b,k}}{F_{b,k}}.
$$

C 表示最终总能量的多少被捕获；$C^*$ 表示达到同 rank 最优捕获量的多少。Overlap 不按目标能量加权，因此与两种捕获量回答不同问题。

### C. 早期空间对后续增量的覆盖：2 类

| 文件后缀               | 实际测量指标                                                                                                                                                                 | 横轴、分组与读法                                                                                |
| ---------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| `frame-increments`     | 将目标 $\Delta_b$ 换成区间增量 $I_{a,b}$ 后的相对最优捕获：$\Vert I_{a,b}V_{h,k}\Vert_F^2/\sum_{i\le k}\sigma_i(I_{a,b})^2$                                                  | 横轴为相邻 checkpoint 区间，纵轴为历史 anchor h；k=16/64/256。只计算 $h\le a$，未来 anchor 留白 |
| `frame-outside-energy` | step16 frame 内绝对能量 $E_{\rm in}=\Vert I_{a,b}V_{16,k}\Vert_F^2$、外部能量 $E_{\rm out}=\Vert I_{a,b}\Vert_F^2-E_{\rm in}$、外部比例 $E_{\rm out}/\Vert I_{a,b}\Vert_F^2$ | 横轴为区间终点；上排为对数绝对能量，下排为外部比例，k=16/64/256                                 |

增量是区间净更新，区间内的往返变化未保存。区间长度不等，绝对能量曲线没有按更新步数归一化；因此不能把这些区间当作等预算训练。

### D. Singular spectrum 与更新维度：7 类

| 文件后缀                   | 实际测量指标                                                                                               | 横轴、分组与读法                                                                                                            |
| -------------------------- | ---------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| `update-dimension`         | $\Vert\Delta_t\Vert_F$、$\Vert\Delta_t\Vert_F/\Vert W_0\Vert_F$、stable rank、rank90、rank95、能量熵有效秩 | 横轴为更新步，六面板；Attention/MLP 分组                                                                                    |
| `singular-energy-rank`     | 完整权重、累计更新、区间增量各自的前 r 个奇异值平方累计比例 $\sum_{i\le r}s_i^2/\sum_i s_i^2$              | 横轴为保留 rank/完整奇异维度，纵轴为能量比例；完整权重和累计更新选 step16/64/192/640，增量选 16→32、48→64、128→192、480→640 |
| `singular-profiles`        | 完整权重的实际奇异值 $\sigma_i(W_t)$，以及同尺寸 teacher 的完整谱                                          | 横轴为奇异值序号，纵轴为奇异值，双对数；step0/16/64/640；七类模块分别展示                                                   |
| `singular-update-profiles` | 累计更新的实际奇异值 $\sigma_i(\Delta_t)$，以及同尺寸 teacher 更新谱                                       | 双对数；step16/64/192/640；零奇异值从对数坐标中略去                                                                         |
| `singular-rank-time`       | 完整权重逐 rank 的有符号变化 $[\sigma_i(W_t)-\sigma_i(W_0)]/\sigma_1(W_0)$                                 | 横轴为保存时刻，纵轴为线性抽取的 80 个精确 rank；七类模块分别画热图，各面板独立对称色标                                     |
| `singular-scale-shape`     | 最佳整体谱缩放 $\alpha$、去除该缩放后的形状残差、最大逐 rank 扰动 $\max_i\lvert s_i-r_i\rvert/r_1$         | 横轴为更新步；完整权重参照 step0，累计更新参照 step16；Attention/MLP 分组                                                   |
| `singular-layer-modules`   | step16 与 step640 的累计更新 stable rank、rank95、相对 W0 的更新范数                                       | 横轴为模块，纵轴为层；每格一个矩阵，各面板独立色标                                                                          |

原始奇异值曲线和 rank–时间热图确定性地选择当前完整层集合中的中间层，七类模块全部展示，没有按效果挑层。能量–rank 汇总将每个矩阵的 r/q 插值到公共网格，增量使用已导出的 65 点能量 CDF。

对奇异值向量 s，三种维度统计分别为

$$
\mathrm{stable\ rank}=\frac{\sum_i s_i^2}{s_1^2},\qquad
r_p=\min\left\{r:\frac{\sum_{i=1}^{r}s_i^2}{\sum_i s_i^2}\ge p\right\},\qquad
r_{\mathrm{entropy}}=\exp\left(-\sum_i q_i\log q_i\right),\quad q_i=\frac{s_i^2}{\sum_j s_j^2}.
$$

rank90/95 分别取 p=0.90/0.95。Stable rank 受最大奇异值影响，rank90/95 描述累计能量覆盖，熵有效秩描述全谱能量分散程度。它们均测更新维度，不直接衡量输出能力。

谱缩放和形状分解令 s 为当前、r 为参照奇异值向量：

$$
\alpha=\frac{r^\top s}{r^\top r},\qquad
e_{\mathrm{shape}}=\frac{\Vert s-\alpha r\Vert_2}{\Vert s\Vert_2}.
$$

### E. 谱形状变化及其与 frame 的关系：6 类

| 文件后缀               | 实际测量指标                                                                                       | 横轴、分组与读法                                                                                |
| ---------------------- | -------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| `weight-spectrum`      | 完整权重相对 W0 的 ISO 三指标：$\delta_\Sigma$、$\rho_\Sigma$、$\kappa_{\rm spec}$                 | 横轴为更新步，三面板；矩阵间中位数和四分位区间                                                  |
| `sigma-dynamics`       | 相邻 checkpoint 的 SD 距离、SD 距离/区间更新步数、到最终 checkpoint 的 SD 距离                     | 横轴为区间终点或 checkpoint 更新步；完整权重、累计更新两排，Attention/MLP 分组                  |
| `sigma-pairwise`       | 完整权重和累计更新各自的任意 checkpoint 两两 SD 距离                                               | 横纵轴均为保存时刻，颜色为逐矩阵距离的中位数；两面板独立色标                                    |
| `sigma-layers`         | 每层相邻区间 Hellinger 距离/区间更新步数                                                           | 横轴为区间，纵轴为层；先计算每个矩阵的距离，再对该层七类矩阵取中位数；完整权重和累计更新分开    |
| `singular-gap-angles`  | 起点累计更新的边界相对谱隙 $g_k=(s_k-s_{k+1})/s_k$，与相邻累计更新 frame 的归一化 chordal distance | 横轴为 g（对数），纵轴为转角距离；每点一个矩阵，k=16/64/256；选 16→32、64→128、192→320、480→640 |
| `singular-frame-phase` | 相邻区间的完整权重/累计更新 SD 距离，与该区间累计更新 frame 的 k64 chordal distance                | 横轴为谱距离（对数），纵轴为 frame 距离；每点一个矩阵；选 16→32、48→64、128→192、480→640        |

SD 使用奇异值和归一化 $p_i=s_i/\sum_j s_j$，Hellinger 使用同一 p：

$$
D_\Sigma(A,B)=\Vert p(A)-p(B)\Vert_2,\qquad
H(A,B)=\frac{\Vert\sqrt{p(A)}-\sqrt{p(B)}\Vert_2}{\sqrt2}.
$$

这与 D 类能量熵采用的 $q_i\propto s_i^2$ 不同。谱速度都是区间净距离除以更新步数，不是实际路径长度。

ISO 三指标在本实现中为

$$
\delta_\Sigma=\frac{\Vert\sigma(W_t)-\sigma(W_0)\Vert_2}{\Vert W_0\Vert_F},\qquad
\rho_\Sigma=\frac{\Vert\sigma(W_t)-\sigma(W_0)\Vert_2}{\Vert\Delta_t\Vert_F},\qquad
\kappa_{\rm spec}=\frac{mn}{q}\frac{\sum_{i=1}^{q}(u_{0,i}^{\top}\Delta_t v_{0,i})^2}{\Vert\Delta_t\Vert_F^2}.
$$

$u_{0,i},v_{0,i}$ 来自 W0 的配对奇异向量，q 为完整奇异维度。$\kappa_{\rm spec}$ 是相对随机方向基线缩放后的配对方向能量集中度，可以大于 1；它不是奇异值最大值/最小值的条件数，也不是 0–1 的能量比例。

### F. 跨 pair 总览：4 类

| 文件名                          | 实际测量指标                                                                                                                                                          | 纳入条件与读法                                                                                          |
| ------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| `overview-rkl-endpoints`        | step640 的绝对 RKL（颜色为 log10）和终点/初始 RKL 比例；all、Math、Code、IF，initial/teacher 两 bank 分开                                                             | 所有已纳入报告的完整训练 pair，不要求几何测齐；横轴为域，纵轴为 pair；不同面板色标可能不同              |
| `overview-rkl-trajectories`     | 两套 bank 的 all 与 Math 绝对 RKL 随更新步变化                                                                                                                        | 每个完整训练 pair 一个面板，纵轴为对数；各面板独立纵轴，比较绝对值需读刻度                              |
| `overview-geometry-rkl`         | 横轴六指标：k64 step16→640 调整后 overlap、相对最优捕获、终点 stable rank、相对更新范数、完整权重 0→640 SD 距离、终点参数残差比；纵轴统一为 teacher-bank 终点平均 RKL | 只纳入几何全层测齐的 pair；每点代表一个 pair 的逐矩阵中位数，不拟合因果关系；跨尺寸 pair 略去参数残差点 |
| `overview-shared-student-E编号` | 同一初始 student、不同 teacher 的 k64 到终点 overlap、累计更新 stable rank、相对更新范数                                                                              | 只纳入全层测齐的 pair，并匹配相同 tensor 坐标；横轴为更新步，曲线为不同 pair                            |

### G. 全指标综合图：5 类

| 文件名                                   | 指标与统计单位                                                                         | 用途                                                 |
| ---------------------------------------- | -------------------------------------------------------------------------------------- | ---------------------------------------------------- |
| `metric-synthesis/domain-rkl`            | 十一组×三域×两套 bank 的终点绝对 question-mean RKL；共同 log10 色标                    | 区分域内匹配、跨域缺口和 bank 来源差异               |
| `metric-synthesis/frame-rank-comparison` | k16/32/64/128/256的16→640、480→640调整后overlap及早期能量捕获/同rank最优；逐矩阵中位数 | 比较早期定向、后期稳定和 rank 依赖                   |
| `metric-synthesis/metric-screen`         | pair 级几何特征与两套 bank 终点 Math/三域 RKL 的描述性 Spearman 相关                   | 汇总关联、反例和相互依赖的指标；不是因果或显著性检验 |
| `metric-synthesis/position-rkl`          | Math/IF、两套bank的相对响应十等分位置token-mean RKL；共同 log10 色标                   | 区分整段不匹配与后段残差放大                         |
| `metric-synthesis/sigma0-effects`        | 每 pair 每阶段两套 bank×三域中最大的配对绝对/相对换谱效应                              | 同时检查小绝对效应和低基线上的百分比变化             |

### 参数残差三指标

A 类图中的 teacher 更新为 $\Delta_T=W_T-W_0$，三个指标为

$$
a=\frac{\Vert\Delta_t\Vert_F}{\Vert\Delta_T\Vert_F},\qquad
c=\frac{\langle\Delta_t,\Delta_T\rangle_F}{\Vert\Delta_t\Vert_F\Vert\Delta_T\Vert_F},\qquad
R=\frac{\Vert W_T-W_t\Vert_F^2}{\Vert W_T-W_0\Vert_F^2}=1+a^2-2ac.
$$

a 比较更新范数，R 比较残差能量，两者不是同一个量。R=1 表示参数残差能量等于初始值，R>1 表示更大；它衡量同坐标参数距离。三项都需要同尺寸、同坐标的 teacher，跨尺寸 E102 仅展示 student 自身动态和输出 RKL。

## 尚未绘制的内容

| 内容                                                       | 当前缺口与后续处理                                                                       |
| ---------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| 后续 pair 的全模型对照                                     | E94–E104 已全层完成；E105–E123 按训练和测量实际完成后接入，不将未完成组纳入终点比较      |
| Teacher 与 OPD 更新相互捕获的能量                          | 已有更新 frame 主角，但缺少双向、能量加权的投影量；需要新增投影测量，不能由 overlap 代算 |
| 按相同能量选择不同 rank 的 frame 比较                      | 已有固定五档 rank；可计算 rank90/95，但对应可变 rank 的主角需要补测                      |
| 每个区间增量的完整奇异值曲线                               | 导出数据保留 65 点能量 CDF；可画能量–rank，完整增量谱需从原始测量迁入                    |
| PNT 的坐标 mask 与 GRRR 的 diagonal/off-diagonal/null 通道 | 当前任务未测这些分解；需新测量和明确定义，不能从整体谱图推导                             |
| 谱尾 Hill 指数及尾部拟合稳定性                             | 尚未实现；需要先定义拟合区间并做区间敏感性检查，避免单条拟合线制造结论                   |
| 早期子空间投影后继续训练                                   | 属于新干预实验；按当前约定暂不进行                                                       |

现有 checkpoint 只能恢复九个保存时刻及相邻区间的净变化。逐更新瞬时角速度、区间内往返轨迹以及多 seed 误差条均不在这些数据中。

## 数据与复算

源码提交保留[结果摘要](data/result-summary.json)、正式图表、分析与页面生成代码。[本地数据清单](data/artifact-inventory.json)记录大型明细、原始 readout、下载论文源码与生成 HTML 的路径、字节数和 SHA256。这些文件按仓库源文件体积规则留在本地，未删除或迁移；交互页面和完整复画依赖其中的对应数据。结果摘要保留十一组的矩阵指标汇总、完整层覆盖及双 bank RKL 轨迹，便于仅凭源码提交复核正文总表。

[完整 readout](data/readout.json.gz) 保存每个矩阵的归一化谱向量、核范数、Frobenius 范数、SD/Hellinger 距离、所有 checkpoint 对、ISO 指标、角度摘要、能量捕获、参数残差、输入 tensor 哈希及全部 RKL 事件。[覆盖表](data/coverage.json) 标明哪些 pair、哪些层已齐。图以 PNG 与 PDF 同时输出，并附审计结果。

代码将测量、数据后处理、绘图和报告更新分开：GPU 测量保留现有 `spectral/trajectory.py`；CPU 后处理入口为 `absorption_dynamics/geometry_readouts.py`；纯指标在 `sigma_variation.py`，绘图按 `frame_figures.py`、`singular_figures.py`、`overview_figures.py` 及原有 `geometry_figures.py`、`sigma_figures.py` 分工；公共布局在 `geometry_plotting.py`，图目录在 `figure_catalog.py`，表格更新在 `geometry_report.py`。

在仓库根目录运行，`FIGURE_VIZ_MPL` 指向 figure-viz 的 `assets/mpl`。以下命令只读取本目录数据即可复画全部图并更新结果表：

```bash
python -m examples.mirroropd.research.studies.absorption_dynamics.geometry_readouts \
  --readout examples/mirroropd/docs/experiments/teacher-transfer/checkpoint-dynamics/data/readout.json.gz \
  --figures examples/mirroropd/docs/experiments/teacher-transfer/checkpoint-dynamics/figures \
  --report examples/mirroropd/docs/experiments/teacher-transfer/checkpoint-dynamics
```

测量尚未结束时使用 `--source` 指向任务输出目录、`--data` 指向本报告的数据目录，并加 `--watch`。多组并行任务用 `--extra-source` 追加目录；各目录须包含互不重叠的 pair，重复 pair 或目录直接报错，防止混入重复测量。默认每分钟检查新矩阵，可用 `--interval` 调整刷新间隔。它在 CPU 上计算指标、刷新图表；已发布且内容未变的 pair 跳过重绘；测量结束时再读取 exit 并完成最后一次发布。这个进程不创建巡检 hook，也不申请 GPU。测量失败会保留错误数量与 exit 信息，已有成功矩阵仍可复算。

### 参考文献

[^sosd]: The Stability of Singular Distribution: A Spectral Perspective on the Two-Phase Dynamics of Language Model Pre-training. arXiv, 2026. [2605.26489](https://arxiv.org/abs/2605.26489).

[^panel-qwen-small]: Qwen/Qwen3-0.6B. 固定版本 c1899de289a04d12100db370d81485cdf75e47ca 的公开模型卡与配置；2026-11-12 核查。[Hugging Face](https://huggingface.co/Qwen/Qwen3-0.6B/tree/c1899de289a04d12100db370d81485cdf75e47ca).

[^panel-qwen-four]: Qwen/Qwen3-4B. 固定版本 1cfa9a7208912126459214e8b04321603b3df60c 的公开模型卡与配置；2026-11-12 核查。[Hugging Face](https://huggingface.co/Qwen/Qwen3-4B/tree/1cfa9a7208912126459214e8b04321603b3df60c).

[^panel-qwen-target]: Qwen/Qwen3-4B-Instruct-2507. 固定版本 cdbee75f17c01a7cc42f958dc650907174af0554 的公开模型卡与配置；2026-11-12 核查。[Hugging Face](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/tree/cdbee75f17c01a7cc42f958dc650907174af0554).

[^panel-light]: qihoo360/Light-R1-7B-DS. 固定版本 06c9338cdbec28416e15f28a325b1439729ec237 的公开模型卡与配置；2026-11-12 核查。[Hugging Face](https://huggingface.co/qihoo360/Light-R1-7B-DS/tree/06c9338cdbec28416e15f28a325b1439729ec237).

[^panel-sft]: Thinking-Space/Qwen3-1.7B-SFT. 固定版本 f0065babe7440257d7db331a5aa442ef7c3cf266 的公开模型卡与配置；2026-11-12 核查。[Hugging Face](https://huggingface.co/Thinking-Space/Qwen3-1.7B-SFT/tree/f0065babe7440257d7db331a5aa442ef7c3cf266).

[^panel-gt]: TMLR-Group-HF/GT-Qwen3-1.7B-Base-MATH. 固定版本 679b8b3a531200f6748eb782c265c83a58b1417d 的公开模型卡与配置；2026-11-12 核查。[Hugging Face](https://huggingface.co/TMLR-Group-HF/GT-Qwen3-1.7B-Base-MATH/tree/679b8b3a531200f6748eb782c265c83a58b1417d).

[^panel-code]: agentica-org/DeepCoder-1.5B-Preview. 固定版本 103033da54f104b0d0a3570547fb3721fba99962 的公开模型卡与配置；2026-11-12 核查。[Hugging Face](https://huggingface.co/agentica-org/DeepCoder-1.5B-Preview/tree/103033da54f104b0d0a3570547fb3721fba99962).

[^panel-if]: Thinking-Space/UltraData-IF-1.5B. 固定版本 be67d8ba9be0b9f405da4481f7dab1045a0256e1 的公开模型卡与配置；2026-11-12 核查。[Hugging Face](https://huggingface.co/Thinking-Space/UltraData-IF-1.5B/tree/be67d8ba9be0b9f405da4481f7dab1045a0256e1).

[^panel-tulu-sft]: allenai/Llama-3.1-Tulu-3-8B-SFT. 固定版本 f2a0b46b0cfda21003c6141b1ff837b7e165524d 的公开模型卡与配置；2026-11-12 核查。[Hugging Face](https://huggingface.co/allenai/Llama-3.1-Tulu-3-8B-SFT/tree/f2a0b46b0cfda21003c6141b1ff837b7e165524d).

[^panel-tulu-dpo]: allenai/Llama-3.1-Tulu-3-8B-DPO. 固定版本 a7beb67e33ffd01cc87ac3b46cadc1000985b8db 的公开模型卡与配置；2026-11-12 核查。[Hugging Face](https://huggingface.co/allenai/Llama-3.1-Tulu-3-8B-DPO/tree/a7beb67e33ffd01cc87ac3b46cadc1000985b8db).

[^panel-tulu-final]: allenai/Llama-3.1-Tulu-3-8B. 固定版本 666943798adbde0b1aff34626007e26986a3c107 的公开模型卡与配置；2026-11-12 核查。[Hugging Face](https://huggingface.co/allenai/Llama-3.1-Tulu-3-8B/tree/666943798adbde0b1aff34626007e26986a3c107).

[^panel-mimo]: XiaomiMiMo/MiMo-7B-RL. 固定版本 7c7707596b0ef1032ef82970d7a4a484ca457c4d 的公开模型卡与配置；2026-11-12 核查。[Hugging Face](https://huggingface.co/XiaomiMiMo/MiMo-7B-RL/tree/7c7707596b0ef1032ef82970d7a4a484ca457c4d).
