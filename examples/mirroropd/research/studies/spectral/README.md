# Teacher 变化与 OPD 吸收的谱分析

本包提供只读权重分析、固定 bank 的输出/表示残差 SVD、离线权重重构和原 teacher 的 native RKL 评分。研究配对、论文来源与实验解释见[谱实验设计](../../../docs/experiments/teacher-transfer/absorption/README.md)。上级命令不启动训练或 rollout；`extract`、`head`、`score` 默认使用单张 GPU，其余默认 CPU。

跨模型状态生成、参数干预、全层梯度与资源调度由 [absorption](absorption/README.md) 子包统一提供。

## 入口与输入

在仓库根目录运行：

```bash
python -m examples.mirroropd.research.studies.spectral --help
python -m examples.mirroropd.research.studies.spectral weights --help
```

HF checkpoint 必须是本地完整 safetensors 导出，支持单文件和索引分片。bank 沿用 `teacher_difference.inputs` 的 JSON/JSONL：每题含非空 `prompt_ids`、`response_ids`、唯一 `row_id` 和显式 `split`。按题划分 calibration/evaluation；历史 heldout 已参与选型时，不能再称未见测试集。下面 `$BASE`、`$TEACHER`、`$OPD`、`$BANK` 由调用者设为已核验的配对和 bank，不自动读取或合并历史 manifest。

## 权重分析入口选择

| 问题                                   | 入口                          | 保存内容                                      |
| -------------------------------------- | ----------------------------- | --------------------------------------------- |
| 多 teacher、全层、训练阶段的完整权重谱 | [scan](scan.md)               | 完整奇异值、δΣ/ρΣ/κ；可追加更新与 missing 谱  |
| base frame 的分量吸收与随机对照        | [weights](weights.md)         | diag/off/null、真实谱随机框架与 Gaussian 标量 |
| 实际导出、补回、删除后输出变化         | 下文 `reconstruct` 与 `score` | 独立模型导出、固定 bank native KL             |

纯谱扫描按同一 base 矩阵归组，在 teacher 与多个 student 阶段间复用一次 base SVD，帧仅驻留当前矩阵。完整诊断保留已有独立三元组接口和历史协议。两个入口都逐矩阵读取 safetensors，默认范围排除 embedding/head/norm，GPU 资源仍由已有 broker/resource guard 管理。

输出随机 SVD 拟合只读取 base/teacher，OPD 数组在评分时读取；校准总能量与捕获能量合并到同一次扫描。当前输出 SVD 使用 CPU 分块，校准 basis 尚未跨 OPD checkpoint 缓存。后续扩展优先复用不可变提取结果，避免为每个新统计量重新前向。

## 固定 bank 输出与表示

每个模型单独提取一次，之后不同分析复用缓存：

```bash
python -m examples.mirroropd.research.studies.spectral extract \
  --model "$BASE" --records "$BANK" --precision master \
  --positions 16 --cap 1024 --layers 0 15 \
  --output runs/spectral-analysis/base
python -m examples.mirroropd.research.studies.spectral extract \
  --model "$TEACHER" --records "$BANK" --precision native \
  --positions 16 --cap 1024 --layers 0 15 \
  --output runs/spectral-analysis/teacher
python -m examples.mirroropd.research.studies.spectral extract \
  --model "$OPD" --records "$BANK" --precision master \
  --positions 16 --cap 1024 --layers 0 15 \
  --output runs/spectral-analysis/opd
python -m examples.mirroropd.research.studies.spectral residuals \
  --base runs/spectral-analysis/base --teacher runs/spectral-analysis/teacher \
  --opd runs/spectral-analysis/opd --rank 16 --calibration train --evaluation heldout \
  --output runs/spectral-analysis/output-svd
```

`master` 为 FP32 参数、CUDA backbone BF16 autocast、FP32 head，无梯度与 optimizer；`native` 为 CUDA BF16 参数、FP32 head。CPU 均为 FP32。每题默认最多均匀取 16 个预测位置，包含首尾；预测位置 0 对应 prompt 末 token。该采样用于谱诊断，不能代替 H1024 全位置风险。选中层必须存在；各模型使用相同位置数和 cap。`--layers` 可省略，只保存 post-final-norm 状态；指定层保存 post-block/pre-final-norm 状态。

缓存使用 FP32 `.npy` memmap，完成标记在全部写入后发布；记录 token 映射、题目/位置指纹、模型分片 SHA256 和输出数组 SHA256。已有输出目录拒绝覆盖；失败后保留不完整目录，改用新目录重试。`residuals` 校验完整性与 logprob 内容摘要。Base/OPD 词表和维度须一致；teacher 可有额外原生 head 行，但 token ID 必须覆盖且对齐 student。跨 hidden 尺寸仍可比较输出谱；不同 token 编码不在此适配器的支持范围。

默认 residual 为 base Fisher 白化后的 log ratio。随机 SVD 只拟合 calibration，报告该子空间未解释的 Frobenius 比例；这包含截断误差，不是单独的算法误差上界。`--oversample`、`--iterations` 控制近似，不能用截断结果报告全谱 numerical rank。Heldout 按固定谱 band 和未解析 complement 评分，另有随机同维子空间对照；每题权重相等，吸收系数允许负数及大于 1。`scores.json` 保留逐题结果，供按题 bootstrap；不会将 token 当作独立 seed。

`--metric logit` 使用均匀中心化 logit 差，`--metric probability` 使用原生概率差在 student 支持上的限制，作为敏感性分析。Fisher 与中心化 logit 排除了 teacher 额外支持的常数偏置；另报的 sampled native RKL 保留完整 teacher partition function。这里的 RKL 按采样位置均值，完整 fixed-H 结果由 `score` 给出。

```bash
python -m examples.mirroropd.research.studies.spectral representations \
  --base runs/spectral-analysis/base --teacher runs/spectral-analysis/teacher \
  --opd runs/spectral-analysis/opd --layer final --rank 16 --shared-coordinates \
  --output runs/spectral-analysis/hidden-svd
python -m examples.mirroropd.research.studies.spectral head \
  --base runs/spectral-analysis/base --teacher runs/spectral-analysis/teacher \
  --opd runs/spectral-analysis/opd --base-weights "$BASE" --teacher-weights "$TEACHER" \
  --rank 16 --shared-coordinates --output runs/spectral-analysis/head.json
```

表示与 head 比较要求已核实同源、同 hidden 坐标；`--shared-coordinates` 是调用者声明，不会从等 shape 推断谱系。`representations` 在 calibration 上做精确 CPU SVD；`head` 对 base 中心化 head 做精确 FP64 SVD 并按 split 报告强弱方向能量和 head 恒等式误差，默认 CUDA，可选 CPU。Head 的强弱阈值由 base head 本身确定。该普通 head 谱不是 Fisher 谱。Head 分析拒绝不同词表/hidden shape。

## 离线重构与 native 风险

```bash
python -m examples.mirroropd.research.studies.spectral reconstruct \
  --base "$BASE" --model "$OPD" --mode top --fraction 0.01 --restore-norm \
  --untouched base --output runs/spectral-analysis/top01
python -m examples.mirroropd.research.studies.spectral reconstruct \
  --base "$BASE" --model "$OPD" --anchor "$TEACHER" --mode patch-off --beta 1 \
  --output runs/spectral-analysis/patch-off
python -m examples.mirroropd.research.studies.spectral score \
  --model runs/spectral-analysis/patch-off --teacher "$TEACHER" --records "$BANK" \
  --cache runs/spectral-analysis/native-cache --cap 1024 \
  --output runs/spectral-analysis/patch-off-risk.json
```

`top` 按每矩阵 `ceil(fraction × min(shape))` 保留奇异分量。省略 `--restore-norm` 得未缩放版本；打开后恢复该矩阵原更新范数。`top` 命令默认未选参数回到 base，匹配论文 attention/MLP-only 重构；删除/补回默认保留 target。Python API 的 `untouched` 默认始终是 target，复现论文时应显式指定 base。

`delete-{diag,off,null}` 为 target 减去该更新分量；`patch-{diag,off,null}` 为 target 加上 anchor−target 的对应分量，均用 base frame，支持 `--beta`。`--control gaussian` 将实际干预替换为同形等范数 Gaussian；`--control random_frames` 保留实际干预的奇异值并随机化框架，`--seed` 固定随机源。权重坐标的三方 tensor key/shape 必须匹配；共享绑定的 embedding/head 允许 safetensors 省略 `lm_head.weight` 别名。diag 是一阶改谱分量，off 不等于有限纯旋转。默认 scope 排除 embedding/head/norm；embedding/head 编辑直接拒绝，以保持绑定端点。输出保留架构 config 与 tokenizer，将默认 `torch_dtype` 设为 FP32，避免 `dtype=auto` 重新掩盖微小更新；选中矩阵存 FP32，每 tensor 一个 shard，`spectral-intervention.json` 最后标记完成。输入只读，输出必须是新目录。

`score` 复用已有模型加载、teacher cache 与生产 `full_reverse_kl_result` 的 streaming 后端；每题全部 response 位置的 native KL 求和后除以固定 H。Teacher 额外词表行保留在归一化中。不同 beta 的选择应只看 calibration，再对独立 evaluation 评分；工具不自动搜索 beta 或运行任务生成评测。

## 资源与验证

资源表为完整分析的预算估算；32 题真实模型提取已测 allocator 峰值，数据见[谱实验报告](../../../docs/experiments/teacher-transfer/absorption/README.md)。完整 head SVD 的峰值与吞吐尚未实测。H800 查询显示本机有 8 张约 80 GB 卡，表中的卡数是每个任务的需求，不代表这些卡空闲。

| 工作                                   | GPU 需求                    | 主要资源                                                       |
| -------------------------------------- | --------------------------- | -------------------------------------------------------------- |
| 权重谱、diag/off/null、离线重构        | 最低 0 张；可选 1 张加速    | 逐矩阵精确 SVD；CPU 可先预留 32–64 GB RAM，不包含全模型驻留    |
| 输出 SVD、hidden SVD、投影评分         | 0 张                        | 默认 block32、rank16；建议 16–32 GB RAM，磁盘读取往往是瓶颈    |
| 1.5B/1.7B 固定 bank 提取与 native 评分 | 1 张；24–48 GB 为待验证预算 | FP32 模型权重约 6–7 GB，另加 head、activation、CUDA workspace  |
| 7B teacher 提取                        | 1 张 48–80 GB 较稳妥        | native BF16 权重约 14 GB；若改 master 为约 28 GB，再加前向开销 |
| 精确中心化 head SVD                    | 0 张可运行；建议 1 张 80 GB | 大词表 FP64 SVD 的 U、workspace 和三份 head/hidden；先单独测量 |

首轮建议串行占用 **1 张 H800 80 GB**；两张可并行不同配对，四张或八张只是提高吞吐，无需模型并行。全模型训练、optimizer、full-gradient PAC 不包含在这些预算内。原有 25–69 GB 梯度探针峰值不能套用到这里的 no-grad 分析。

输出磁盘约为 `题数 × 每题采样位置数 × native vocab × 4 bytes × 缓存模型数`。以 256 题、16 位置、152064 词表估算，每模型约 2.49 GB（三模型约 7.47 GB，十进制）；全取 H1024 则约 159 GB/模型。先用 16 位置、32 题和单层 SVD 检查数值/峰值，再扩展完整 bank，避免一开始缓存所有位置。三个 cache 的 logprobs 由 CPU 分块读取，不整体转到 GPU。

CPU 验证入口：

```bash
mkdir -p runs/spectral-implementation-20261109
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest tests/research/test_spectral.py -q \
  --basetemp=runs/spectral-implementation-20261109/pytest
```

测试覆盖正交分解、方阵 null、ISO 缩放、top-rank/范数恢复、随机框架、分片导出与只读输入、输出完整性、SVD 对标、heldout 隔离、额外 teacher 支持、head 恒等式和小模型预测位置。32 题 GPU 提取和单矩阵精确 SVD 已完成；跨模型科学结果与任务读出见谱实验报告。Activation oracle、合成 band teacher 的训练接入、早期子空间约束训练与自动绘图不在此实现中；它们按实验设计中的功能证据条件再接入。

## 全层扩展的成本复算

`costs.py` 只读历史逐矩阵计时，按形状和参考缓存状态匹配成本，并重放整条件／矩阵任务的并行调度。不同工作负载分别报告，缺少同形状数据时保留空值；不会启动 GPU。入口及实测结果见[ISO 计算预算](../../../docs/experiments/teacher-transfer/absorption/附录-定义与复算.md)。
