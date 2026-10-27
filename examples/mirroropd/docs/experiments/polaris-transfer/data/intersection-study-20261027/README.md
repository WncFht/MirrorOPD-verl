# 交集采样与扩大held-out实验，2026-10-27

正文位于源码仓库 `examples/mirroropd/docs/experiments/polaris-transfer/2026-10-27-交集采样与扩大前缀评估.md`。

## 完成范围

- 128道数学训练题；独立held-out为数学64题、代码16题、IF16题。标准化空白后的prompt也无训练/held-out交集。
- 每个held-out题都有初始student、JustRL、Polaris参考，各1024 cap，共288条冻结轨迹。
- 五个主臂各4个outer、batch32、reuse16，共64次更新；每个outer重新生成，随后固定该批前缀更新16次。
- 三个high/low/random state选择对照，在同一32题student初始轨迹上各做16次更新、每次8192个位置。
- 总计368次正式实验更新；工程smoke使用的模型被丢弃，不计入预算。
- Fresh评估：固定32道held-out数学题×两个seed，每次64条新的unrestricted student轨迹，H=1024。
- 独立任务检查：32道MATH-500题，cap4096，初始模型与五个endpoint各一答，共192条输出；不是标准全量benchmark。
- 对初始/最终参数、初始/最终fresh state做完整2×2交叉评分，分别测固定state参数收益和访问分布项。

所有GPU工作在物理GPU0–3上执行。模型原始资产没有被改写，GPU4–7没有参与本实验。五个endpoint保存在日志仓库的 `outputs/polaris-transfer-cache/intersection-study-20261027/checkpoints/`，不随报告证据复制到源码仓库。

Endpoint 为 HF 模型权重、配置和 tokenizer，不含 Adam 状态；可用于继续评估，但不能据此无缝恢复相同优化器历史。

## 指标与数据归属

`protocol.json`固定实验协议。Student参数与LM head为FP32，backbone使用BF16 autocast；teacher为BF16权重、FP32 head投影。固定参考由BF16 HF生成器预先产生，不随训练方法变化。Fresh生成和在线累计KL使用同一套实际采样概率。

固定风险按领域分别计算，题目与三个参考来源等权，采用固定H=1024分母。EOS后的固定参考位置贡献零，不能删除该来源或该prompt再平均。Fresh序列KL是实际student轨迹上的完整条件reverse KL之和；所有fresh和任务评估均取消intersection/top-k限制。JS/TV只作为条件分布指标，不冒称序列JS/TV。

类别编码：0=两边都不在，1=仅teacher top-k，2=仅student top-k，3=都在。`token_categories`保留k=8/16/64；`stats`保留计数、概率质量、集合overlap、Jaccard和距离的逐位置区间和。按token数算的采样比例与按prompt/source等权的固定风险是不同归约。

| 产物 | 内容 |
| --- | --- |
| `prompts.json`、`reference-*.json` | 训练/held-out题、完整冻结token IDs和来源 |
| `task-prompts.json` | 独立32题MATH-500诊断集 |
| `baseline-justrl/`、`baseline-polaris/` | 相同初始student、目标teacher独立的固定与fresh基线 |
| `*-k16/` | 主训练配置、64次更新、四批真实rollout、每16步固定评估、32/64步fresh评估、完成标记 |
| `selection-high/low/random/` | 同一完整输入上的三种loss位置mask、16次更新、固定与fresh结果 |
| `task-*.json` | 原始生成、自然stop、逐token概率诊断和完整grader返回 |
| `decomposition/` | 参数×state来源四格交叉评分，评分实现统一 |
| `summary-*.json` | 固定风险、fresh风险、实际token类别、训练成本、任务与交叉分解统计 |
| `paired-method-contrasts.json` | intersection/student/top-k及state选择的配对差异 |
| `entropy-by-source.json` | 固定来源上的entropy、RKL、JS、TV关系 |
| `verification.json` | 数值与预算独立核验；非文件校验或摘要清单 |
| `environment.json` | 解释器和依赖版本；完成时仓库HEAD不代表工作区无其他修改 |

配对区间按prompt bootstrap，同一题的两个fresh seed先平均；它不包含训练seed不确定性，也没有做多重检验校正。任务`pred`为空不必然等于评分失败，grader可能经fallback判为正确。

Teacher固定前缀hidden和endpoint权重属于可再生的大缓存，位于 `derived/`。原始teacher额外输出行保留在评分归一化中；teacher-source采样限制到student可输入的输出行。Intersection使用student概率在交集上条件化，空交集回退student top-k并记录；此次空交集次数为0。

## 复算

从源码仓库根目录执行。新实验选择新的OUT/CACHE；已有目录中的统计可直接重算，不需要GPU。下列生成和训练命令会使用模型资产和GPU，不是仅导出报告。

```bash
STUDY_PY=/root/.local/share/mirroropd/venvs/profile-cu128/bin/python
STUDY_SCRIPTS=examples/mirroropd/docs/experiments/polaris-transfer/scripts
STUDY_OUT=examples/mirroropd/docs/experiments/polaris-transfer/data/intersection-study-20261027
STUDY_CACHE=outputs/polaris-transfer-cache/intersection-study-20261027

python3 "$STUDY_SCRIPTS/summarize_intersection.py" --data "$STUDY_OUT"
python3 "$STUDY_SCRIPTS/analyze_intersection_controls.py" --data "$STUDY_OUT"
python3 "$STUDY_SCRIPTS/audit_intersection_study.py" --data "$STUDY_OUT"

# 以下为从新目录重新执行的设置；CUDA_VISIBLE_DEVICES按实际分配填写。
export CUDA_VISIBLE_DEVICES=0
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
"$STUDY_PY" "$STUDY_SCRIPTS/intersection_study.py" prepare \
  --models ../model --data ../MD-OPD/data --out "$STUDY_OUT" --cache "$STUDY_CACHE"
for STUDY_SOURCE in student justrl polaris
do
  "$STUDY_PY" "$STUDY_SCRIPTS/intersection_study.py" reference \
    --models ../model --out "$STUDY_OUT" --cache "$STUDY_CACHE" --source "$STUDY_SOURCE"
done

# 所有参考生成完成后再训练。这里给出单卡顺序复算写法。
for STUDY_PAIR in justrl:student polaris:student polaris:teacher polaris:intersection polaris:topk
do
  "$STUDY_PY" "$STUDY_SCRIPTS/intersection_study.py" run \
    --models ../model --out "$STUDY_OUT" --cache "$STUDY_CACHE" \
    --teacher "${STUDY_PAIR%%:*}" --mode "${STUDY_PAIR#*:}" --k 16 --save
done
# 多卡并行时每个进程独立设置可见GPU；相同teacher的初始baseline有独立writer锁。

for STUDY_SELECTION in high random low
do
  "$STUDY_PY" "$STUDY_SCRIPTS/intersection_study.py" selection \
    --models ../model --out "$STUDY_OUT" --cache "$STUDY_CACHE" --selection "$STUDY_SELECTION"
done

# 所有endpoint保存以后执行交叉分解。
"$STUDY_PY" "$STUDY_SCRIPTS/intersection_study.py" decompose \
  --models ../model --out "$STUDY_OUT" --cache "$STUDY_CACHE"

# 初始任务评测不传checkpoint。
"$STUDY_PY" "$STUDY_SCRIPTS/intersection_study.py" task_eval \
  --models ../model --out "$STUDY_OUT" --cache "$STUDY_CACHE" --teacher polaris \
  --tag initial
for STUDY_PAIR in justrl:student polaris:student polaris:teacher polaris:intersection polaris:topk
do
  STUDY_ARM="${STUDY_PAIR%%:*}-${STUDY_PAIR#*:}-k16"
  "$STUDY_PY" "$STUDY_SCRIPTS/intersection_study.py" task_eval \
    --models ../model --out "$STUDY_OUT" --cache "$STUDY_CACHE" --teacher "${STUDY_PAIR%%:*}" \
    --tag "$STUDY_ARM" --checkpoint "$STUDY_CACHE/checkpoints/$STUDY_ARM"
done

FIGURE_VIZ_MPL=../dotfiles/.agents/skills/figure-viz/assets/mpl \
  "$STUDY_PY" "$STUDY_SCRIPTS/plot_intersection_study.py" \
  --data "$STUDY_OUT" --out examples/mirroropd/docs/experiments/polaris-transfer/figures/intersection-study-20261027
```

这份脚本与统计实现固定服务于本报告的H=1024、k=16主实验；任务检查另为4096。改变长度、参考来源、预算或采样规则时，应同步定义新的评估协议与统计分母，不直接混入本快照。
