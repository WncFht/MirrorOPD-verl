# 吸收实验的优化器分支

`absorption_queue.py` 准备和运行原生 Full-RKL 吸收实验。`sgd_campaign.py` 为 E94–E114 的 21 个师生对先选择 SGD 学习率，再启动两条四卡完整训练队列。完整预算为 batch64、40 outer、每批 reuse16、回答上限 7168；固定初始学生／教师 bank 沿用各 pair 的模板协议，最终面板不参与学习率选择。

本地曲线可通过[自动刷新监控页](../../../tools/metrics_dashboard/README.md)查看，服务读取 JSONL，无需 W&B 联网。

## 编号化 SGD 平台对照

`run_math_opd_ablation_fsdp2.sh E124` 和 `E125` 使用 E102 的 Qwen3-1.7B ← Qwen3-4B-Instruct-2507 配对，分别默认 SGD LR=0.01 和 0.03，零动量、零 weight decay；完整预算为 40 outer × reuse16。编号独立于历史 AdamW E102。环境变量仍可覆盖 LR，实际运行以 resolved config 为准。

在线记录显式设置 `WANDB_MODE=online`，使用机器已有 netrc；连通性失败不能自动当作 offline 成功。开发集与正式 math 面板同时评分时，读取 `eval/rkl/public_initial/dataset/development_math/question_mean` 与 `dataset/math/question_mean`，教师 bank 同理。原有 `math` domain 聚合包含该域全部 dataset，不能当作独立的正式面板。

## SGD 校准与完整训练

在仓库根目录、已有原生训练依赖和本地模型的环境中运行。`--root` 必须是不存在的新目录；所有配置、冻结源码、题目来源、日志和状态写入该目录。下面两条命令分别准备配置和运行持久化监督进程，运行不会自动转到后台。

```bash
python -m examples.mirroropd.tools.sgd_campaign prepare \
  --repo . --root ./rebuild/sgd-study
python -m examples.mirroropd.tools.sgd_campaign run \
  --root ./rebuild/sgd-study --broker /path/to/idle-occupancy/broker.py
```

题目来源为已有 `examples/mirroropd/research-evidence/qwen-prefix-six/prepared-03`。正式训练使用前 2560 行。校准从后续题目中按规范化文本去重，剔除正式训练前缀和永久 held-out 面板，取 256 题更新、再取 32 题开发评分。每档 LR 从原始学生重新开始，运行 4 outer × reuse16；开发集只使用同一 pair 首次生成的固定初始学生／教师轨迹，后续候选校验 bank 内容身份后复用。

初始候选为 0.003、0.01、0.03。选择指标是两个开发 bank 在 outer3、outer4 的相对初始 RKL 降幅均值。若最佳候选在边界，向该方向乘或除 3，最多增加两档；仍在边界会保留 `edge_winner`，不宣称找到了全局最优 LR。评分要求 32 道开发题在每个事件、每个 bank 都完整有效，且初始 RKL 为正有限数。无早期改善但评分有效的 pair 仍按最佳实测值进入完整训练，早期排序不等于长期上限。

所有 21 对校准结束后，`selected-lrs.json` 保存实测值与候选证据，`full/partition.json` 保存完整训练的两条 lane。`templates/prepared.json` 的 `unselected_template` 只用于配置生成，不能当作已校准结果。SGD 使用无动量、零 weight decay、恒定 LR 和无 warmup；历史 AdamW 使用 weight decay=0.01，因此这组历史对比包含衰减差异。模型精度、rollout 和完整训练预算沿用原协议。

`status.json` 记录总状态，`front4-status.json`、`back4-status.json` 记录当前 pair／LR。每个 native job 由已有 broker 和 resource guard 管理资源，等待已有占用退出，不释放其他 owner。创建根目录的 `STOP` 文件会停止活动 job；任一非零退出也会触发停止并保留原始失败。控制进程退出后，可在确认旧控制进程已退出的前提下加 `--resume` 接管已有成功或仍运行的校准 job；接管检查配置 hash、PID／启动时刻与退出回执，不重新执行已有更新。失败 job 不自动重试，完整训练已准备时也拒绝从校准入口重启；需检查原始日志与 native checkpoint 后单独处理。

评测 coordinator 与 CPU preflight 共用 checkpoint 配置检查：周期保存要求 save/load 均包含 model、optimizer、extra；评测导出另要求 model+hf_model 与 FP32 master weights。运行 checkpoint 与模型读出快照可以并存。

完整训练每 5 outer 保存一次含模型、optimizer、scheduler／RNG 与 dataloader 的运行 checkpoint，最多保留两个 actor checkpoint。评测的 FP32 模型快照仍按原 9 个时刻导出，只保存／回载模型，保留 optimizer、RNG 与运行 checkpoint 的留存记录。校准关闭模型导出和运行 checkpoint；运行 checkpoint 和评测快照的用途不同。

## 显式优化器配置

[公共 launcher](../../../launchers/absorption.py) 默认仍为 AdamW。直接使用 `run_math_opd_ablation_fsdp2.sh` 时，设置 `ABSORPTION_OPTIMIZER=sgd` 或 `muon`，并显式给出 `ABSORPTION_OPTIMIZER_LR`。Muon 还需要独立的 `ABSORPTION_AUX_LR`。分支默认写入 `training-sgd/EXX` 或 `training-muon/EXX`，拒绝覆盖对应的历史 `training/EXX`。完整训练的双 bank 必须先通过 bootstrap 生成和 finalize。

手动准备已校准的分支时，`queue.py prepare --optimizer sgd --calibration selected-lrs.json --pairs E94 ...` 接受按实验 ID 索引的 JSON，每项至少含正有限 `lr`；Muon 每项另含 `aux_lr`。`--pairs` 省略时仍沿用队列原有全部 pair 范围，SGD campaign 明确限制为 E94–E114。

## FSDP2 Muon

[MuonAdamW](../../../../../verl/utils/fsdp_optimizers.py) 通过 [optimizer factory](../../../../../verl/workers/config/optimizer.py) 和 FSDP2 engine 接入，要求安装提供原生 `torch.optim.Muon` 的 PyTorch。实现依赖原生 Muon 的私有 Newton–Schulz 函数；升级 PyTorch 后应重新运行原生数值对照。

通过模块结构和 Hugging Face 的输入／输出 embedding accessor 排除 embedding、输出头及其 tied weights，其余可训练二维矩阵使用 Muon，bias 和 norm 使用辅助 AdamW。自定义模型可直接传入显式 `use_muon` 参数组；多个组、各组独立 LR、scheduler、closure 和 `add_param_group` 使用标准 optimizer 接口。分片不单独正交化。Muon 使用原生 Nesterov、五步 Newton–Schulz 和 `match_rms_adamw` 的形状缩放，动量状态保持分片。参数组 LR、辅助 AdamW 状态和 Muon 动量都进入同一个 optimizer state dict，恢复时重新连接辅助组和状态。

| `ABSORPTION_MUON_DISTRIBUTED_MODE` | 完整矩阵计算与适用范围                                                                                              |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| `owner`，launcher 默认             | 一维 FSDP2 mesh、按行分片；轮流选择 owner 收集梯度方向，执行一次 NS，scatter BF16 更新分片；支持不均匀和空行分片    |
| `replicated`                       | DTensor 收集完整矩阵，各 rank 执行 NS，再恢复原 placement；用于参考与其他 placement，重复计算和完整矩阵显存成本较高 |

`owner` 在 gather 前转为 BF16，沿用原生 NS 的首步转换；连续 gather buffer 避免拼接，整除分片直接 scatter 正交化结果。所有 rank 必须保持相同参数组顺序和梯度存在性。`owner` 减少重复正交化和非 owner 的完整矩阵工作区，但逐矩阵 gather／scatter 仍为同步通信，不能据此保证端到端更快。当前没有 bucket、通信计算重叠或张量并行的 NS。辅助 AdamW 与 Muon 的 LR 应分别校准；SGD campaign 不启动 Muon 的 21 组完整训练。

CPU 测试为 `tests/workers/test_fsdp_muon_on_cpu.py` 和 `tests/examples/test_sgd_campaign_on_cpu.py`。四卡原生矩阵对照为 `tests/special_distributed/test_mirror_muon_reference.py --device cuda --distributed-mode owner --out ...`；需使用 torchrun。真实 engine 的 offload／checkpoint 路径由 `test_mirror_fsdp2.py --optimizer SGD` 或 `MuonAdamW --precision bf16_fp32 --checkpoint --output ...` 验证。矩阵对照通过只证明更新与状态一致，不代替完整训练或吞吐验收。

## 实现依据与性能验收

参数组沿用 Muon 上游的 `use_muon` 接口[^upstream]，辅助参数交给原生 AdamW；Newton–Schulz 和形状缩放直接复用 PyTorch Muon[^torch]。Moonlight 提供了大模型混合优化器和 BF16 通信的参考[^moonlight]；其分布式参数布局与 FSDP2 不同，不能直接套用通信实现。原生 DTensor Muon 的探针没有复现完整矩阵数值基准，因此保留显式完整矩阵路径。

四张 H800、PyTorch 2.11.0+cu128 下，对 owner 通信修改前后做 5 次预热、每轮 20 步、5 轮计时，以下是每轮最慢 rank 耗时的中位数。owner 临时显存为 warmup 后峰值新增分配。该测量不包含 backward、梯度归约或 offload。

| 矩阵      | 修改前 ms | 修改后 ms | owner 临时显存 MiB，前→后 |
| --------- | --------: | --------: | ------------------------: |
| 1536×1536 |     2.771 |     1.340 |               40.50→26.75 |
| 8960×1536 |     1.710 |     1.497 |             192.75→107.44 |
| 4096×4096 |     3.853 |     3.642 |             288.00→184.00 |

原始数据与脚本在仓库 `examples/mirroropd/research-evidence/reuse-and-optimizers/optimizer-infra-20261114/`。CPU／CUDA 四 rank 对照覆盖不均匀和空分片，更新及恢复逐元素匹配原生完整矩阵 Muon＋AdamW。真实 FSDP2 engine 另验 BF16 计算、FP32 主参数、tied embedding、offload 与模型／optimizer checkpoint。微基准没有给出完整训练的吞吐结论。

调度实现位于 `examples/mirroropd/tools/absorption_queue.py` 和 `sgd_campaign.py`，科学选择规则位于 `research/studies/absorption_dynamics/sgd_protocol.py`。新准备的执行快照同时冻结核心代码、launchers、tools 和 shell 入口；完整训练沿用校准时的完整源码快照。冻结源码 hash 不匹配时停止。部分 reserve 失败时，只收回本队列尚未 attach 的 reservation，已 attach 的资源由 broker 在进程退出后释放。

### 参考文献

[^upstream]: Keller Jordan 等。Muon implementation. GitHub. [Muon](https://github.com/KellerJordan/Muon).

[^torch]: PyTorch. Muon optimizer. [torch.optim.Muon](https://docs.pytorch.org/docs/stable/generated/torch.optim.Muon.html).

[^moonlight]: Liu 等。Muon is Scalable for LLM Training. 2025. [arXiv:2502.16982](https://arxiv.org/abs/2502.16982).

## E102 配对的 SGD 结果与 Muon LR 扫描

2026-11-15，E124（SGD 0.01）和 E125（SGD 0.03）均完成 40 outer、640 次更新，无跳过。原始日志位于 `examples/mirroropd/research-evidence/reuse-and-optimizers/sgd-e102-platform-20261114-retry1/`；历史 AdamW 为 dashboard catalog 登记的 E102。以下为 outer31–40 均值，训练 loss 读取每批第 1／16 次更新前，固定 bank 为 Math held-out 按题 RKL。

| 运行            | 初始学生 bank | 教师 bank | 首次更新前 loss | 第 16 次更新前 loss | 批内相邻 loss 上升比例 |
| --------------- | ------------: | --------: | --------------: | ------------------: | ---------------------: |
| E102 AdamW 1e-6 |        0.2546 |    0.2761 |          0.2425 |              0.1939 |                     0% |
| E124 SGD 0.01   |        0.2788 |    0.3100 |          0.2720 |              0.2435 |                   0.7% |
| E125 SGD 0.03   |        0.2923 |    0.3189 |          0.2718 |              0.2267 |                  21.3% |

AdamW 同时改善批内拟合和下一批起点，不能把差距完全解释为 SGD 泛化较差。SGD 0.03 相比 0.01 有更深的批内拟合，却没有改善下一批起点，并出现更多批内振荡。训练首次更新前 loss 来自各模型自己的 rollout；不同运行的轨迹和回答长度不同，不能把它们当作同一个固定测试集。训练 loss 按 token 聚合，也不能直接与按题 held-out RKL 相减。

SGD 全程触发 norm=1 梯度裁剪的比例为 99.7%／100%，AdamW 为 25.5%。SGD 裁剪保留原始梯度方向；AdamW 同时使用动量与逐坐标二阶矩缩放。当前比较没有分离动量、预条件、LR、独立 bank 采样和 weight decay。历史 AdamW 的每步衰减系数为 `1 - 1e-6 × 0.01`，640 步累计直接缩放约 0.9999936，因此直接衰减幅度很小，但仍不是严格的 optimizer-only 对照。原始指标聚合另存于 `examples/mirroropd/research-evidence/reuse-and-optimizers/optimizer-analysis-20261115/batch-summary.json`。

用户授权的 Muon 扫描采用相同 Qwen3-1.7B ← Qwen3-4B-Instruct-2507 配对，E126–E129 的矩阵 LR 依次为 3e-7、1e-6、3e-6、1e-5；辅助 AdamW 固定 1e-6，WD=0，momentum=0.95、Nesterov、五步 NS、`match_rms_adamw`、owner 分布式路径。每档从原始学生重新开始，保持 batch64、reuse16、40outer。两条队列分别执行 E126→E128（GPU0–3）、E127→E129（GPU4–7），W&B offline。

此 LR 使用原生形状缩放 `0.2 × sqrt(max(rows, cols))`；矩阵更新 RMS 量级与 AdamW LR 对齐，不照搬 SGD 或原始 Muon 缩放约定的 LR。辅助 LR 本轮固定，扫描结果只代表该条件下的矩阵 LR。选择依据是独立开发集双 bank 的后段表现；held-out 用于报告。四档共享 E124 原有 finalized bootstrap，并校验模型、数据和协议身份；与历史 AdamW 的独立 bank 差异仍保留。

执行入口为 `examples/mirroropd/tools/optimizer_sweep.py`，运行目录为 `examples/mirroropd/research-evidence/muon-e102-platform-20261115/`。`prepare` 固定源码、配置和银行身份，`run` 使用现有 native queue 与 broker 顺序占用两组四卡。每小时及结束时写入观察记录；任一失败设置 STOP 并停止另一条队列，不自动重试。根目录 STOP 也会传递至活动任务。源码、配置和日志保留，禁止复用已有 launch 记录静默重跑。前端目录已登记四档，出现指标日志后可以勾选加载。

## 多配对平台比较与定时巡检

`optimizer_campaign.py` 准备 E130–E151，共 22 个新配置，运行目录 `examples/mirroropd/research-evidence/reuse-and-optimizers/optimizer-platform-20261115/`。执行依赖 E126–E129 扫描成功结束，之后使用两个四卡 lane。当前训练、冻结来源与旧指标保持原位。

| 配对                                        | Muon 矩阵 LR                        | AdamW LR         | 新编号    |
| ------------------------------------------- | ----------------------------------- | ---------------- | --------- |
| E102：Qwen3-1.7B ← Qwen3-4B-Instruct-2507   | 3e-5；合并已有 E126–E129 的结果分析 | 3e-7、1e-6、3e-6 | E130–E133 |
| E95：R1-Distill-1.5B ← JustRL-DeepSeek-1.5B | 1e-6、3e-6、1e-5                    | 3e-7、1e-6、3e-6 | E134–E139 |
| E109：R1-Distill-1.5B ← Polaris-7B          | 1e-6、3e-6、1e-5                    | 3e-7、1e-6、3e-6 | E140–E145 |
| E117：Qwen3-1.7B-Base ← Qwen3-1.7B-SFT      | 1e-6、3e-6、1e-5                    | 3e-7、1e-6、3e-6 | E146–E151 |

各配置从原始学生开始训练 40outer、batch64、reuse16，统一 WD=0、恒定 LR；Muon 辅助 AdamW LR 固定 1e-6。各 pair 复用同一个 finalized bootstrap，新增 pair 首次生成包括独立 development_math 在内的完整 bank，后续配置校验身份后复用。E102 复用 E124 原 bank。AdamW 不以历史 WD=0.01 配置替代新对照。公共训练 checkpoint 每 5outer 保存，保留最近两个；模型读出和回载验收保留 outer20／40，减少其他快照对长队列的存储压力。

`optimizer_campaign report --root <root>` 输出 `analysis.json`：记录每个 bank 最后 10 个 outer 的均值、线性斜率和标准差。LR 选择只使用 development_math 两个 bank 相对初始 RKL 的后段均值；要求完整 32 题、无无效行、outer31–40 均存在。held-out 只报告，不参与选择。若后段拟合直线在 10outer 的降幅超过后段均值 1%，标记仍在下降；这是执行规则，不是统计平台检验，也不等于最终上限。

每 2 小时巡检由 `training_review.py` 调用本机 Codex CLI 执行，任务目录 `examples/mirroropd/research-evidence/reuse-and-optimizers/optimizer-platform-reviews-20261115/`，连通性已验证。巡检保存分析文本、CLI 日志与退出回执，单次超时 20 分钟，失败保留错误并在下一周期重试；后台训练控制器独立运行。巡检授权范围包括读取指标、核实进程、修复明确 infra 错误及有界追加实验，不向外发送消息。运行依赖本机进程和 Codex 服务可用；机器重启后需显式恢复，不能承诺服务中断期间仍有检查。

失败修复保留原始证据和冻结源码，新建 attempt。已有更新的任务仅从完整且经验证的 model／optimizer／extra 及 dataloader checkpoint 恢复，恢复语义不清时停止该分支并记录。健康任务不中断。后续按开发集结果，每 pair 每 optimizer 最多补两次边缘 LR 档；Muon 主 LR 定位后再比较辅助 LR 3e-7／3e-6。若最佳配置尚未平台，比较双方采用相同延长预算，先到 80outer，必要时最多 120outer；这些扩展需要巡检按实际结果准备与验收，尚非已启动配置。

## LR 矩阵入口与几何读出

运行 `bash examples/mirroropd/run_math_opd_ablation_fsdp2.sh --optimizer-matrix`，只显示配对×优化器×LR 矩阵及完整编号展开，不启动训练。矩阵、`--overview`和 shell 顶部注释均由公共 launcher catalog 生成。单元格中的 `E158(E136)` 表示新 attempt 编号和旧协议编号，不表示两个运行同时有效；实时状态仍以 dashboard 和各 study 为准。辅助 AdamW LR、WD 与历史默认值分别标注，环境变量覆盖后的实际值以 resolved config 为准。

已完成的优化器配置可复用 checkpoint-dynamics 的精确 FP64 SVD。`optimizer_geometry.py prepare --catalog <catalog.json> --output <manifest.json>` 审核完整训练凭据、41 个 RKL 事件、所选时刻的导出验收与权重文件，清点真正保留的 actor 分片，生成供现有 `spectral.trajectory` 使用的全层任务。原始 global_step 目录可能只剩 data／trainer 状态；只有 actor 模型及 optimizer 分片仍在的目录才可作为恢复候选。存在分片不等于已经验证连续恢复。

`optimizer_geometry.py pilot --manifest <manifest.json> --output <pilot.json>` 在 CPU 对 Qwen 跨尺寸 pair 的首／中／末三层 k_proj 执行精确 SVD，分析最终累计更新 `W40-W0`；输出完整奇异值、stable rank、rank90，以及跨配置左右子空间主角、随机基线校正 overlap 和更新矩阵余弦。输入 tensor 保留 hash。小范围试测只用于检验测量链路，不推断全部层结论；完整任务同时读取 outer20／40，能测后半程净增量，不能恢复区间内逐更新往返。

正式比较应区分累计更新谱、区间增量谱和权重本身的谱；同时报告更新范数与归一化能量谱，避免把 LR 放大误读成形状变化。奇异值接近时单个奇异向量不可稳定识别，按多个 rank 比较子空间并检查边界谱隙。Muon 正交化的是每步动量矩阵，不保证 40outer 累计参数更新具有平坦谱；累计更新还受不同方向的旋转与相消影响。跨尺寸 teacher 不与 student 做参数 frame 对齐；同坐标 teacher 比较须单独核验来源。所有几何指标应与开发集／held-out 后段结果关联，不能由谱更平坦直接推出泛化更好。

E124–E129 的[九时刻权重动力学](../../experiments/teacher-transfer/checkpoint-dynamics/optimizer-comparison/README.md)复用原 checkpoint-dynamics 的全层测量和分类图，E123 标记缺失。通过 `--experiments E124 E125 E126 E127 E128 E129 --outers 1 2 3 4 8 12 20 30 40` 显式选择快照；每组 196 矩阵，每矩阵覆盖九个时刻。报告按共同完整层比较 SGD／Muon 与 LR，同时保留两套 bank、后段趋势和覆盖状态。CPU 发布入口为 `python -m examples.mirroropd.tools.optimizer_geometry_report --source <measurements> --report <report-directory> --watch --interval 600`；绘图使用独立 plotting 环境和 `FIGURE_VIZ_MPL`，不修改运行中的训练环境。
