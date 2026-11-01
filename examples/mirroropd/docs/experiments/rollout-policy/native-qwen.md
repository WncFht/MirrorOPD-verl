# Qwen 六来源：原生执行与评估协议

本页维护 E63–E68 的实际预设与原生银行准备条件。它与[旧六组结果](qwen-sources.md)具有独立的执行和评估身份；结果产生后在本页更新，不将旧 HF 读出替换成新原生结果。

<a id="native-e63e68qwen3-八卡前缀对照与原生评估"></a>

## E63–E68：Qwen3 八卡前缀对照与原生评估

这组六个新编号使用 Qwen3-1.7B non-thinking learner 和 Qwen3-4B-Instruct-2507 teacher，每个实验默认占用 8 张卡，分别启动。公共入口仍为 `run_math_opd_ablation_fsdp2.sh`，配置由[编号规划器](../../../launchers/math_experiment.py)及[Qwen 资源与评估配方](../../../launchers/qwen_prefix.py)生成，直接进入共享 trainer，不再经过旧六组 benchmark 封装。

| 编号 | 生成来源                               | 训练 rollout 拓扑                |
| ---- | -------------------------------------- | -------------------------------- |
| E63  | current student                        | 8 个 TP1 副本                    |
| E64  | 固定 initial student                   | 8 个 TP1 副本                    |
| E65  | 固定 teacher                           | 8 个 TP1 副本                    |
| E66  | 0.5 current＋0.5 teacher，logit 融合   | 4 个 TP2 副本                    |
| E67  | 0.5 current＋0.5 teacher，概率融合     | 4 个 TP2 副本                    |
| E68  | batch fusion，current/teacher 各 32 条 | 两个单模型阶段，各 8 个 TP1 副本 |

<a id="native-资源默认值"></a>

### 资源默认值

| 参数                              | 八卡默认值                | 解释                                                                               |
| --------------------------------- | ------------------------- | ---------------------------------------------------------------------------------- |
| `STUDENT_GPUS`                    | 8                         | FSDP2 共享池总卡数；不是同时启动 8 个实验                                          |
| `TRAIN_BATCH_SIZE` / `PPO_EPOCHS` | 64 / 16                   | 每 outer 生成 64 条回答，在同批上做 16 次更新；40 outer 共 640 次更新              |
| `FUSION_TP`                       | 2                         | LM/PM两模型共用TP2，每两卡一个组合副本                                             |
| `FUSION_MAX_NUM_SEQS`             | 16                        | 每个组合副本的活跃请求上限；4 个副本合计 64                                        |
| `FUSION_KV_BYTES`                 | 24 GiB/卡                 | 组合引擎 KV 预算；不是整个训练进程的显存上限                                       |
| `ROLLOUT_MAX_NUM_SEQS`            | 16                        | 普通 current/评估引擎的每副本上限；训练 batch64 分到 8 卡后实际每卡只有 8 条       |
| `PREFIX_SINGLE_KV_BYTES`          | 64 GiB/卡                 | current 引擎同时服务长上下文原生 accuracy；固定训练来源另按短窗口和实际路由计算 KV |
| 原生评估全局并发                  | 128                       | 8 个 TP1 副本×16，按请求完成补入，不等待每 64 条全部结束才补下一批                 |
| `PPO_MAX_TOKEN_LEN_PER_GPU`       | 32768                     | learner 前反向的动态计算微批 token 预算，不是生成并发                              |
| `TEACHER_FSDP_TOKEN_BUDGET`       | 32768                     | 共享 teacher 评分的计算微批预算                                                    |
| Full-vocab backend / chunk        | `precompute_fused` / 1024 | 使用 main 的 fused 实现；不沿用较慢参考 backend                                    |
| 参数/优化器卸载                   | 开启                      | 当前 `shared_sync` 的已支持生命周期要求；关闭卸载尚未作为支持能力交付              |

学习目标保持原生完整词表 reverse KL、`termination_token_ids=[]`，训练 temperature1、top-p1、top-k-1，prompt cap4096、response cap7168，FP32 master＋BF16 backbone，LR1e-6、seed42。没有为了八卡改大全局 batch 或增加学习更新数。参数优先级仍为预设＜环境变量＜方法选项＜末尾 Hydra 覆盖，实际配置必须以 `CONFIG_ONLY=1` 的输出核对。

8 个 TP1 训练副本各只有 8 条请求，单纯把每卡并发写成 128 不会增加这一批的有效并发。LM/PM 选择 4 个 TP2 副本，让每个副本处理 16 条；依据是此前 Q 配对的双卡生成比较，尚不能据此宣称八卡训练达到某个加速倍数。较小的每卡 fusion KV 来自 TP 分片及请求分配，不意味着总资源比单卡少。

<a id="native-新评估协议及版本边界"></a>

### 新评估协议及版本边界

新编号默认使用 main 的 `EvaluationCoordinator`：复用 trainer 的 actor/teacher 进行 RKL 读出，复用普通 current rollout 引擎进行多卡连续生成，并保留逐请求身份和 Parquet 产物。训练来源为 teacher 或 fusion 时，accuracy 仍评估纯 current student，不使用混合模型。原生服务会在评估引擎与训练策略引擎之间明确释放、同步和切换。

训练题表复用旧六组的去重 DAPO 文件，630 道数学 accuracy 题也复用相同文本。新协议重新生成原生银行：Math64 source held-out、五个数学数据集各 8 题的 initial/teacher 固定 panel、64 条按训练 batch 均衡抽样的 history；每 5 轮评分，outer20/40 执行 630 题 avg@4 及 current-state 评分，outer0 使用新银行基线。最终保存 FP32 模型并做原生加载前后 RKL 校验，不再每 20 轮保存 Adam 恢复状态。

这与[旧六组报告](qwen-sources.md)的 HF 固定状态读出有明确区别：原生随机流、银行轨迹、RKL 按题/有效 token 平均，以及 source/history/public 分组替代旧 H1024 固定分母；旧 Code/IF 面板不作为这组新数学配置的评估项。因此旧银行与旧 accuracy 结果不能冒充新 bootstrap，不把两种 RKL 曲线直接拼接。旧六组继续使用冻结 `462e3f3a` 跑完；E63–E68 是新的执行和评估版本。

main 在 `402f6353` 相对冻结版新增了 EOS 补偿融合、原生统一评估和最终 model-only 导出等实现。当前原生完整词表 loss 没有启用 EOS 合并分支；若另行设 `FULL_KL_TERMINATION_TOKEN_IDS=[151643,151645]`，那是目标定义的改变，需要独立标记。多卡归约、TP、动态微批与采样调度仍可能改变浮点结果和自然轨迹，不保证与旧单卡逐 token 或逐位一致。

<a id="native-准备与启动"></a>

### 准备与启动

下面先生成 CPU 题表，再解析新配置并生成共用银行。`PYTHON_BIN` 指向已安装项目依赖的 Python 环境，`MODEL_ROOT`、`DATA_ROOT` 按本机资产位置配置；`QWEN_EVAL_ROOT` 放在仓库内。GPU 命令仅在八张卡已分配给该作业时执行，不中断正在运行的旧队列。

```bash
export QWEN_EVAL_ROOT="$PWD/runs/qwen-prefix-native"

"$PYTHON_BIN" -m examples.mirroropd.research.studies.prefix_fusion.prepare_native questions \
  --source "$PWD/runs/qwen-prefix-six-20261031/prepared-03" \
  --out "$QWEN_EVAL_ROOT/questions"

CONFIG_ONLY=1 bash examples/mirroropd/run_math_opd_ablation_fsdp2.sh E66 \
  > "$QWEN_EVAL_ROOT/config.yaml"

"$PYTHON_BIN" -m examples.mirroropd.research.studies.prefix_fusion.prepare_native configs \
  --source "$QWEN_EVAL_ROOT/config.yaml" --out "$QWEN_EVAL_ROOT/configs"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "$PYTHON_BIN" -m verl.trainer.evaluation.bootstrap \
  --config "$QWEN_EVAL_ROOT/configs/initial.yaml" --producer initial
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "$PYTHON_BIN" -m verl.trainer.evaluation.bootstrap \
  --config "$QWEN_EVAL_ROOT/configs/teacher.yaml" --producer teacher
"$PYTHON_BIN" -m verl.trainer.evaluation.bootstrap \
  --config "$QWEN_EVAL_ROOT/config.yaml" --producer finalize

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 STUDENT_GPUS=8 \
  bash examples/mirroropd/run_math_opd_ablation_fsdp2.sh E66
```

准备目录必须是新目录；已完成的银行直接复用，不覆盖。Teacher bootstrap 使用每副本 8 并发/KV40 GiB，initial 使用 16 并发/KV64 GiB；这些资源差异不会改变声明的生成协议。后续运行把 E66 替换为 E63–E68，并保持同一模型、输入与评估协议。缺少完整 native bootstrap 时，公共入口会在启动 Ray 之前报错。

六个编号的八卡配置已完成[解析检查](../../../../../runs/qwen-ablation-20261101/configs/acceptance.json)。两卡短检查完成 2 轮、32 次更新、TP2 融合、原生评估和最终 FP32 导出；原生加载前后 RKL 误差为 0，见[验收回执](../../../../../runs/qwen-ablation-20261101/gpu-acceptance.json)。初次因旧验证题表检查而提前退出的记录保留，修复后复测通过。这些结果不替代八卡完整负载容量和性能验收；版本比较范围见[提交差异清单](../../../../../runs/qwen-ablation-20261101/version-comparison.json)。
