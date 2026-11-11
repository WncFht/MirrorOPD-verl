# ISO 谱恢复、几何、受控训练与合并

本包实现 ISO 论文的完整权重谱恢复、四类 rank-r 重构、固定谱参数化和专家合并，供已有 OPD 模型作方法复核与扩展。论文范围、实验配对和结果见 [ISO 专题](../../../../docs/experiments/teacher-transfer/absorption/附录-定义与复算.md)。原 `spectral weights`、`scan` 和冻结运行保持各自协议。

## 执行层次

| 模块                       | 责任                                                            | 持久化内容                                          |
| -------------------------- | --------------------------------------------------------------- | --------------------------------------------------- |
| `math.py`                  | polar、Stiefel 切空间、四类重构、主角度和 retention-aware merge | 纯函数，无文件                                      |
| `geometry.py`              | 全层、阶段过渡、rank 敏感性                                     | 逐矩阵误差、边界谱隙、子空间余弦                    |
| `functional.py`            | 完整权重恢复、反向谱交换、α 插值及配对风险与生成                | 输入哈希、无编辑 gate、逐题/逐 token 风险、完整生成 |
| `training.py`              | online OPD 的权重 AdamW、固定谱 frame AdamW、仅谱 AdamW         | 每步采样/损失/耗时/显存、固定阶段读出、最终更新     |
| `merger.py`                | ISO、uniform、keep-all、task arithmetic、weight average         | 每矩阵系数/Gram、谱 gate、逐题风险与生成            |
| `analyze.py`、`monitor.py` | 完成项聚合与本地持续巡检                                        | 覆盖、失败、来源、配对统计及文档附件                |

任务调度复用 `absorption.queue`。扩展 task 显式给 `module`、`arguments` 和 `complete_marker`；`{attempt}` 替换为独立新 attempt 路径，模块本身不申请卡。无 `module` 的旧 task 行为不变。依赖没有完成时保留等待；失败 attempt 不覆盖。每个 GPU worker 始终由已有 broker/guard 保护，STOP 使用新运行自身路径。

## 离线干预与几何

记 $W_0=U_0\Sigma_0V_0^\top$，$W_1=U_1\Sigma_1V_1^\top$。谱恢复沿 $U_1[(1-\alpha)\Sigma_0+\alpha\Sigma_1]V_1^\top$，固定 α 为 0、0.25、0.5、0.75、1；α=1 是无编辑基线。反向控制为 $U_0\Sigma_1V_0^\top$。默认只改全层七类 attention/MLP 投影，embedding/head/norm 保留目标模型值。所有编辑由不可变原始 tensor 重设，不在前一次干预结果上叠加。

四类几何使用完整权重 top-r 截断，$r=\lfloor fq\rfloor$，$q=\min(m,n)$，$f\in\{0.5,0.75,0.9,0.95,0.99\}$。分别测 incoming 双侧 span 内 remix、固定左 span、固定右 span、固定 incoming 谱但两侧可动；分母是两端 rank-r 权重差的 Frobenius 范数。记录谱隙和 admissibility，退化谱边界不进入唯一子空间解释。额外保存 top16/top64 左右主角余弦。

功能评分复用冻结 initial-student/teacher 两个 H128 bank，保持原始 teacher 身份和全词表归一化。α=1 endpoint 与旧 matched endpoint 评分逐题核对，最大误差要求 ≤10⁻⁶；teacher-self floor 同阈值。生成使用同 16 题（8 MATH-500、8 AMC23）、模型自身模板、greedy cap4096，记录截断和原始回答。该面板为诊断读出，原始 canonical grader 保留，不以最低 heldout KL 选择 α。

`functional --scale-controls-only` 把整体尺度与归一化谱形状分开。令 $c=\|W_1\|_F/\|W_0\|_F$，尺度恢复为 $W_1/c$，保留目标模型归一化谱；形状恢复为 $cU_1\Sigma_0V_1^\top$，保留目标模型 Frobenius 范数。两者均保留目标奇异框架，逐矩阵重设。该入口在相同 bank 上评分 teacher/student 和无编辑基线，不追加生成。

离线分析由已保存的 FP64 范数复算径向投影 $\langle W_0,\Delta W\rangle=(\|W_1\|_F^2-\|W_0\|_F^2-\|\Delta W\|_F^2)/2$，报告径向能量占比。另拟合 $c_*=\langle\sigma_0,\sigma_1\rangle/\|\sigma_0\|^2$，报告 $\|\sigma_1-c_*\sigma_0\|/\|\sigma_1-\sigma_0\|$。投影和最佳谱缩放使用不同的空间与分母，分别保存；不据此反推训练中采用了 weight decay。

## Online OPD 训练入口与历史 pilot

64 步终点、seed42/43、学习率 10⁻⁵ 和零 weight decay 预先固定。历史 pilot 误从三域 12 个 calibration prompt 抽题，由当前 student 以 temperature1、cap128 重新生成，再计算该轨迹上 full-vocabulary reverse KL。记录在线 response token，便于分析训练状态覆盖；固定 heldout 只用于读出。该 pilot 不能解释原 DAPO 训练的平台；六个未完成任务已取消，完整结果仅保留为诊断。当前入口要求 `--train-prompts` 显式给出 DAPO JSON，含 source_family=dapo、source、rows；每行含 row_id、domain=math、prompt_ids、eos_token_id，并拒绝与 heldout token prompt 重叠。输入文件内容哈希写入 protocol。此处移植 ISO 的参数化，目标是 OPD，未复现论文 RLVR 的奖励、GRPO 与长训练配方。

- 权重 AdamW：直接更新七类投影。
- 固定谱：分别对 U/V 建立 AdamW 状态，使用链式梯度 $G_U=G_WV\Sigma_0$、$G_V=G_W^\top U\Sigma_0$，更新后分别 polar 回缩，再重构权重。
- 仅谱：U/V 固定，仅对非负奇异值做 AdamW。

模型、因子和 optimizer 状态为 FP32，backbone BF16 autocast，polar 回缩使用 FP64。超过 22 亿个选中参数时，因子和 Adam 状态驻留 CPU，每个矩阵依次传到 GPU 计算梯度/回缩，控制 Llama 的显存峰值；性能数据分别报告，不与全 GPU 状态混算。初始化和预定义读出点核验谱和正交误差，阈值 10⁻⁵。少量更新参数的 smoke 先验收反向和两步更新，完整模型仍需检查自身峰值。

## 共享 base 合并

在同一个 base 下用 teacher 专家作输入，符号对齐后投到共同 Stiefel 切空间，按 keep0.9 保留位移列；Gram 系统 ridge10⁻¹²，系数裁剪至 [0,1.5]，汇总后再投影与 polar 回缩。Uniform 和 keep-all 是原方法消融；task arithmetic 与 weight average 为权重空间对照。

二维参数包含 embedding/head，绑定参数按模型唯一叶参数处理，一维参数用专家均值。每矩阵核验 base 谱保持误差 ≤10⁻⁸。风险同时对每个输入专家评分；小任务面板检验是否保留互补行为，不能仅凭拟合任一专家宣称成功合并。

## 存储与运行

Frames、干预方向和合并权重仅驻留内存，不导出每个条件的完整模型。保留谱值、哈希、误差、逐题评分和生成。训练只保留最终选中参数更新，不保存每步全模型或完整梯度；旧归档张量不会自动恢复。

```bash
python -m examples.mirroropd.research.studies.spectral.iso.geometry --help
python -m examples.mirroropd.research.studies.spectral.iso.functional --help
python -m examples.mirroropd.research.studies.spectral.iso.training --help
python -m examples.mirroropd.research.studies.spectral.iso.merger --help
```

本地 monitor 每两分钟核对任务、子进程身份、心跳和 broker，完成项才进入结果统计；结束核验所有卡自动恢复占用。失败保留原 attempt，由新版本或新 attempt 重试；依赖 smoke 未通过时完整任务不放行。停止本轮写运行根目录 `STOP`，仅停止巡检写 `MONITOR_STOP`。

有依据的后续实验另建冻结队列，通过 `continue_queue` 等待指定前驱全部完成（仅接受逐任务、逐 attempt 匹配审计回执的设置取消例外）、exit 已写且八卡回到 occupying 后才申请卡。`queue-chain.json` 显式列出队列；分析与 monitor 同时覆盖全部队列，保留每个任务的所属 ledger，不覆盖前驱身份或重复启动前驱 supervisor。

训练 loss 复用 `verl.trainer.distillation.reverse_vocab.full_reverse_kl_result`，默认 precompute、token chunk128，保留固定 H128 归约；可显式选择其他已支持 backend。新 rollout 的 teacher hidden 在设备上重算，避免原 CPU 往返。冻结运行不受源码更新影响。该入口是独立参数化诊断，尚未接入生产 trainer 的 DAPO batch/reuse/rollout 配置；正式训练按报告中的设置审计先验收。

## 可选回缩加速

`training --retraction newton` 使用经过邻域与完整正交检查的 FP64 Newton–Schulz 极分解；不收敛时回退精确 SVD，默认仍为 `svd`。每步记录调用、成功与回退计数。仅谱梯度使用对角收缩，避免构造完整 UᵀGV。真实形状、四步优化器与 DAPO 两步读出结果见[性能验收](../../../../docs/infra/performance/spectral/固定谱优化器加速.md)。
