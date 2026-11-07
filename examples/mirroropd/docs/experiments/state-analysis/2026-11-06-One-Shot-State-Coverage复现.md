# One-Shot OPD State Coverage 复现设置

本文记录 arXiv:2609.04172 的 rollout state coverage 复现方案和工具接口。当前交付包含提取、PCA、K-means 与 coverage 代码，以及 CPU 单元测试；尚未对真实 teacher 执行 hidden-state 提取，也没有论文结果复现数值。论文方法的解读见[One-Shot OPD State 分析](../../literature/opd/2026-11-06-One-Shot-OPD的State分析与相关技术.md)。[^oneshot]

## 复现目标与论文设置

论文把 state 定义为 $s_i=(x,y_{<i})$，并以 teacher 在该前缀最后 token 处的最终层 hidden vector 作为表示。它从 full-data OPD rollout 中每条轨迹均匀抽 8 个相对位置；每个训练 step 的 5 条 rollout 中，3 条构成参考集，2 条作为 held-out full-data。合计参考状态 12,000 个、held-out 状态 8,000 个。参考表示经 PCA 降至 50 维，报告解释方差为 71.9%；K-means 使用 $K=200$。待测 state 按欧氏距离归入最近质心，coverage 是被访问质心数量除以 $K$。论文的训练曲线累计到 step 300，每步抽 2/5 rollout，重复 30 次并报告标准差。[^oneshot]

| 环节               | 论文设置                         | 实现设置                                                       |
| ------------------ | -------------------------------- | -------------------------------------------------------------- |
| state 采样         | 每条 rollout 均匀抽 8 个相对位置 | `extract --positions 8`，在有效位置间作端点包含的等间隔取样    |
| state 表示         | teacher 最后一层 hidden，1536 维 | `output_hidden_states[-1]`，取预测目标 token 前的 prefix state |
| reference/held-out | 每 step 5 条中按 3/2 切分        | `fit --split-field <训练步字段>`，按 rollout 整组哈希切分      |
| 降维               | PCA 50 维，71.9% 方差            | centered full-SVD PCA，默认 `--pca-dim 50`                     |
| 聚类               | K-means，$K=200$                 | NumPy Lloyd K-means++，默认 `--clusters 200`                   |
| coverage           | 唯一命中 cluster 数除以 $K$      | 共享 reference codebook，默认每步抽 2 条、重复 30 次           |

若每个训练步恰有 5 条 reference-source rollout，0.6 切分会得到 3 条参考 rollout 和 2 条 held-out rollout。实现按完整 rollout 切分，并在 state 级别继承其归属，避免相邻前缀跨入两侧。输入没有训练步字段时，代码要求调用者先用可审计的元数据补齐，不能把 `outer` 自动解释成优化 step。

## 代码与产物

实现位于 [`research/state_analysis`](../../../research/state_analysis/)。`extract` 生成 `vectors.npy`、逐 state 的 `states.jsonl` 和 `metadata.json`；`fit` 生成 `pca.npz`、`centroids.npy`、`rollout_split.json` 和 `fit.json`；`score` 生成每次重采样的 `coverage-repeats.csv`、汇总表 `coverage-summary.csv` 和 `coverage.json`。

`states.jsonl` 中每行与 `vectors.npy` 同索引，包含来源、轨迹 ID、位置、有效 response 长度，以及输入中可用的标量元数据。提取对 `prompt_ids + response_ids[:i]` 做一次因果前向，并读取位置 `prompt_len + i - 1` 的最后层向量；因此 state 表示不包含当前待预测 token。批处理使用左 padding 和 attention/position mask，前缀位置的因果表示与逐前缀单独前向一致。论文指定均匀间隔的位置，但未在正文中规定端点约定；实现将有效 target 的首尾都纳入间距计算，并以 `linspace` 确定位置，后续若取得作者代码或更细节定义，应对齐该规则。

没有 GPU 时可以只运行 CPU 测试，不会加载 teacher：

```bash
python -m pytest -q examples/mirroropd/research/state_analysis/test_oneshot_state_coverage.py
```

PCA 与质心只在指定 reference-source 的 60% rollout 上拟合一次；全部待测 source 共享相同变换和质心。`fit` 与 `score` 会核对 extraction metadata 中的 teacher 标识和模型配置摘要，阻止同维度但不同架构的表征混用。评分时 reference-source 只使用其 held-out rollout，其他来源使用各自完整 state 集。K-means 输入是参考 state 的等权样本，每条 rollout 固定抽取相同数量的位置；coverage 对每个被选 rollout 的 state 等权计数。若任务长度不同，八个分位位置代表每条轨迹的整体进度，不代表每个 token 的训练频率。

## 输入与执行命令

每个来源单独运行一次 `extract`，`--source` 是稳定的来源标签。输入支持仓库 trajectory Parquet 和 JSONL，至少需要 `id`、`prompt_ids`、`response_ids`，以及可选的等长 `response_mask`。提取将 mask 为真的 response 位置视作可评分的预测目标；mask 为假位置仍可留在更后续状态的上下文中。EOS 是否纳入由原始 response mask 决定，padding 不作为有效 target。

```bash
python -m examples.mirroropd.research.state_analysis.oneshot_state_coverage extract \
  --input /path/to/full-data-trajectories \
  --output runs/state-coverage/full-data \
  --model /path/to/frozen-teacher \
  --source full-data \
  --positions 8 \
  --batch-size 1 \
  --device cuda
```

所有来源都提取完成后，在 full-data rollout 上拟合共享空间。`step` 必须是每条 rollout 所属的真实优化步，且每组有 5 条 full-data rollout，才能对应论文的 3/2 切分。

```bash
python -m examples.mirroropd.research.state_analysis.oneshot_state_coverage fit \
  --extractions runs/state-coverage/full-data \
  --output runs/state-coverage/codebook \
  --reference-source full-data \
  --split-field step \
  --pca-dim 50 \
  --clusters 200 \
  --seed 17
```

给定多个已提取来源，按 step 做累计 coverage。默认要求每来源每步至少有两条 rollout；`--allow-fewer` 仅用于不平衡的小规模 smoke，不能用于论文预算的对照结论。

```bash
python -m examples.mirroropd.research.state_analysis.oneshot_state_coverage score \
  --extractions runs/state-coverage/full-data runs/state-coverage/one-shot \
    runs/state-coverage/four-shot runs/state-coverage/sixteen-shot \
  --fit runs/state-coverage/codebook \
  --output runs/state-coverage/results \
  --step-field step \
  --rollouts-per-step 2 \
  --repeats 30 \
  --max-step 300 \
  --seed 29
```

## 数据映射与证据边界

仓库 trajectory Parquet 的字段包括 `outer`、`prompt_ids`、`response_ids` 和 `response_mask`；该 schema 不声明 `step`。`outer` 表示采集/评估外循环编号，只有经具体 run 元数据确认它与训练 step 一一对应时，才可将它传给 `--split-field` 或 `--step-field`。训练更新号也不能由 `min_global_steps` / `max_global_steps` 区间任意取值代替。应在提取源数据前建立 `trajectory_id -> optimizer_step` 对照，并保留来源 manifest、teacher checkpoint/revision、prompt 集、解码参数与 rollout seed。

现存 `runs/evaluation-parquet-implementation/roundtrip-d670f78b/trajectories/` 和 Qwen source/initial/teacher banks 可用于验证 Parquet 读取与样本字段，但它们没有论文所需的 500 step full-data OPD rollout 及已核实的 optimizer-step 映射。它们不能直接复现 71.5% 或 98.9%。现有数据读取 smoke 和 NumPy CPU 测试只验证接口、位置采样、PCA、K-means、切分和 coverage 算法；teacher 前向、真实维度 1536、71.9% 方差以及论文 coverage 曲线仍待模型与完整数据可用后验证。

复现差异至少应检查 teacher checkpoint/tokenizer、prompt 组成、temperature/top-p、response 截断、有效 token mask、step 到 rollout 的映射、相对位置采样、PCA 方差占比、K-means 初始化和 coverage 重采样预算。论文对 $K=50,100,200,500$ 的排序稳定性以及 10-NN 距离诊断可作为下一阶段敏感性分析；当前最小管线暂不实现近邻距离或语义分层占用，以优先复现核心 coverage 指标。

### 参考文献

[^oneshot]: Fu, Zixuan, et al. Rethinking On-Policy Distillation of Large Language Models II: One Training Example. arXiv:2609.04172, 2026. §4.1、§4.2 与 Appendix B.1. [arxiv.org](https://arxiv.org/abs/2609.04172)
