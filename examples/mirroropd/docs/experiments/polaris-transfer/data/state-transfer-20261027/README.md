# State 分布与跨源迁移实验，2026-10-27

正文：源码仓库 `examples/mirroropd/docs/experiments/polaris-transfer/2026-10-27-跨源蒸馏泛化研究总览.md`。这是 E48 问题的独立局部机制实验，不是 E48 checkpoint 回放，也不是正式 benchmark 准确率评测。

## 数据和完成范围

- 24个固定prompt：16道数学、4道代码、4道IF；三种来源各生成一次，共72条轨迹，3376个固定评分state。
- 初始student与两个teacher的完整词表比较，共6752条state-target记录。
- 八组主训练：两种teacher × student/对应teacher两种state来源 × 两个互不重叠训练题fold。每组16次全参数更新。
- 两组Polaris温度1.5补充实验，各16次更新；所有迁移评价仍使用原始温度1的teacher。
- 一次LR=0对照；3376个state的前后KL最大变化为0。总计161次完整optimizer调用。
- 八条student轨迹、两个teacher、每条四个位置块的全参数梯度几何；各块保持完全相同的248-token response输入，只改变评分位置。梯度完成后转存CPU，不改变其数值。

模型目录未被写入。训练状态在进程退出后释放，不保存生产checkpoint。所有更新从初始student和fresh Adam重新开始。每组四道训练题、128个稀疏位置，与E48长回答batch64负载不同；两个训练fold不能当作两个独立完整训练seed。

## 产物

| 文件 | 含义 |
| --- | --- |
| `prompts.json`、`protocol.json` | 数据选择和生成协议 |
| `rollouts-*.json` | 完整prompt/token IDs和生成轨迹；结束原因与实际长度 |
| `initial-state-metrics.json` | 每state的KL、entropy、top-k交集与概率质量、TV、信号符号、logit梯度等 |
| `initial-summary.json`、`phase-summary.json` | 按teacher/source/domain/位置或推理阶段归约 |
| `fp32-master-baseline-*.json` | 与更新后评估完全同口径的初始FP32-master基线 |
| `*-split*-t*-lr*/` | 训练参数、逐更新loss/grad、step0/1/8/16固定state评估 |
| `transfer-summary.json` | prompt配对变化；bootstrap只表示固定模型条件下的prompt不确定性 |
| `within-trace-gradient-geometry.json` | 同一输入上四个位置块的全参数梯度范数、C、范数加权pairwise cosine |
| `verification.json` | 独立数值、输入、分割与更新边界核验 |
| `papers/` | 已核对的2610.11291v1、2604.13016v2、2608.16647v3源文件 |
| `attempts/` | 中断或被更严格输入控制替代的尝试；不计入最终结果 |

完整FP32 logits和逐token log-prob/entropy放在忽略的 `outputs/polaris-transfer-cache/state-transfer-20261027/`，可由模型和冻结轨迹重算。它们不是模型checkpoint，不需要随报告传输。正文PNG/PDF与图形审计位于源码仓库报告旁。

`attempts/` 包含一个因GPU0新增共享任务而主动迁移的训练尝试、变长前向输入的初版梯度检查，以及共享GPU余量变化导致OOM的梯度检查。最终梯度检查用固定前向输入及CPU梯度存储重新执行。它们不是任务loss发散或模型训练失败的证据。

## 复算

从源码仓库根目录执行。已有冻结输入可以直接做统计；生成/训练会使用GPU和模型资产。新配置必须选新OUT，不能将旧rollout cache当作新配置的输出。

```bash
PROBE_PY=/root/.local/share/mirroropd/venvs/profile-cu128/bin/python
PROBE_SCRIPTS=examples/mirroropd/docs/experiments/polaris-transfer/scripts
PROBE_OUT=examples/mirroropd/docs/experiments/polaris-transfer/data/state-transfer-20261027
PROBE_CACHE=outputs/polaris-transfer-cache/state-transfer-20261027

python3 "$PROBE_SCRIPTS/summarize_state_transfer.py" --data "$PROBE_OUT"
python3 "$PROBE_SCRIPTS/audit_state_transfer.py" --data "$PROBE_OUT"

# 以下以一张被分配的GPU为例；代码里的cuda:0指CUDA_VISIBLE_DEVICES映射后的设备。
export CUDA_VISIBLE_DEVICES=1
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false

# 新目录首次准备和生成：先三种来源全部生成，再评分。
"$PROBE_PY" "$PROBE_SCRIPTS/state_transfer_probe.py" prepare \
  --models ../model --data ../MD-OPD/data --out "$PROBE_OUT" --cache "$PROBE_CACHE"
for PROBE_SOURCE in student justrl polaris
do
  "$PROBE_PY" "$PROBE_SCRIPTS/state_transfer_probe.py" generate \
    --models ../model --out "$PROBE_OUT" --cache "$PROBE_CACHE" --source "$PROBE_SOURCE"
done
for PROBE_SOURCE in student justrl polaris
do
  "$PROBE_PY" "$PROBE_SCRIPTS/state_transfer_probe.py" score \
    --models ../model --out "$PROBE_OUT" --cache "$PROBE_CACHE" --source "$PROBE_SOURCE"
done
"$PROBE_PY" "$PROBE_SCRIPTS/state_transfer_probe.py" diagnose \
  --models ../model --out "$PROBE_OUT" --cache "$PROBE_CACHE"

# 示例主训练臂。teacher取justrl/polaris；source取student或对应teacher；split取0/1。
"$PROBE_PY" "$PROBE_SCRIPTS/state_transfer_probe.py" train \
  --models ../model --out "$PROBE_OUT" --cache "$PROBE_CACHE" \
  --teacher polaris --source student --split 0 --steps 16
# 温度消融追加 --temperature 1.5；无更新对照使用 --lr 0 --steps 1。
"$PROBE_PY" "$PROBE_SCRIPTS/state_transfer_probe.py" geometry \
  --models ../model --out "$PROBE_OUT" --cache "$PROBE_CACHE"

FIGURE_VIZ_MPL=../dotfiles/.agents/skills/figure-viz/assets/mpl \
  "$PROBE_PY" "$PROBE_SCRIPTS/plot_state_transfer.py" \
  --data "$PROBE_OUT" --out examples/mirroropd/docs/experiments/polaris-transfer/figures/state-transfer-20261027
```

图形中误差范围表示两个训练fold的范围。审计仅对held-out图豁免`palette-overflow`：Matplotlib将4组errorbar的帽线/范围线计成16个artist，实际只有2个teacher色相与2种来源线型；其他规则仍检查。

正式长训练的六组对照及判断标准见正文第9节。尚未实现或运行生产trainer的固定cache切换，不能把上述短probe命令当成完整Semi-OPD训练入口。
