#!/usr/bin/env bash
# BEGIN GENERATED EXPERIMENT OVERVIEW
# 用法与配置优先级
#   bash examples/mirroropd/run_math_opd_ablation_fsdp2.sh --help
#   bash examples/mirroropd/run_math_opd_ablation_fsdp2.sh --overview
#   bash examples/mirroropd/run_math_opd_ablation_fsdp2.sh --list-presets
#   DRY_RUN=1 bash examples/mirroropd/run_math_opd_ablation_fsdp2.sh E84
#   bash examples/mirroropd/run_math_opd_ablation_fsdp2.sh E84 -- trainer.test_freq=40
# 优先级：编号预设 < 环境变量 < --method/--kappa/--eta 等选项 < 最后的 Hydra 覆盖。
# DRY_RUN=1 仅打印最终命令；-- 后的参数原样传给 Hydra。custom 使用公共默认值，不加载编号预算。
# 本注释与 --overview 共用 launchers/math_experiment.py 的 preset_overview()；修改源后同步生成区。
# 以下均为当前预设默认值；已启动实验以其 resolved config 为准。
#
# 一、E81–E90 / E92–E93：Qwen Base 的聚合与 future reward 对照
# 学生 Qwen3-1.7B-Base ← 教师 Qwen3-30B-A3B-Instruct-2507。
# 共同训练：full-vocab MDPO kappa=1/eta=0.1，LR=1e-6；shared_sync。
# 每 outer 抽取 256 题，rollout n=1，生成 256 条新回答；mini-batch=256，同批 reuse=16。
# E81 为 25 outer/400 次更新；E82–E90、E92/E93 为 100 outer/1600 次更新。
# 执行：precompute_fused/chunk512、bf16；TP1，每引擎并发 64、显存利用率 0.8。
# 长度：训练 prompt/response=1024/16384，验证 response cap=31744。
# Preset  GPUs  Future           Gamma  Aggregation              Valid every  Val n M/rest  Save every
# E81     4     none             1.0    token-mean               5            16/16         -1
# E82     4     none             1.0    token-mean               5            16/16         -1
# E83     4     none             1.0    seq-mean-token-mean      5            16/16         25
# E84     8     mean             1.0    token-mean               20           4/16          25
# E85     8     none             1.0    token-sum                20           16/16         -1
# E86     8     none             1.0    seq-mean-token-sum       20           16/16         -1
# E87     8     none             1.0    seq-mean-token-sum-norm  20           4/16          -1
# E88     8     sum              1.0    token-mean               20           4/16          -1
# E89     8     discounted_sum   0.99   token-mean               20           4/16          -1
# E90     8     discounted_mean  0.99   token-mean               20           4/16          -1
# E92     8     discounted_sum   0.95   token-mean               20           4/16          -1
# E93     8     discounted_mean  0.95   token-mean               20           4/16          -1
# 表中 Valid/Save 均以 outer 为单位；Save=-1 表示不保存 checkpoint。
# 验证：MATH-500、AMC23、AIME24/25/26；M/rest 表示 MATH-500/其余四集的每题回答数。
# 16/16 每次 10080 条回答，4/16 每次 4080 条；各组均无初始验证。
# 每 20 outer 验证对应 20/40/60/80/100，共 5 次；每 5 outer 的组也包含终点验证。
# 训练与验证均默认开启 repetition stop：16–128 token 模式重复至少 8 次时提前停止。
# E83/E84 每 25 outer 保存完整训练状态，仅保留最新一份，另保留最终 HF 模型。
# E81/E82、E85–E90、E92/E93 不保存 checkpoint、最终模型或 optimizer。
# E85–E90、E92/E93 resume_mode=disable。
# 本组未配置固定 bank RKL；训练 teacher RKL 读取当前训练前缀。
# 实验解释见 docs/experiments/learning-signal/weighted-targets-active.md。
#
# 二、E1–E53：算法、模型、长度与精度对照
#
# 模型/长度对照：Sampled=legacy current+vanilla；Full=Full-KL（MDPO eta=1）。
# Sampled  Full  Sampled-r16  Full-r16  Student       Teacher              Train cap
# E15      E13   E33          E17       R1-1.5B       JustRL-1.5B          7168
# E23      E24   E34          E35       R1-1.5B       JustRL-1.5B          15360
# E25      E26   E36          E37       R1-1.5B       R1-7B                7168
# E27      E28   E38          E39       Q3-1.7B-Base  Q3-8B                7168
# E29      E30   E40          E41       Q3-1.7B-Base  Q3-4B                7168
# E31      E32   E42          -         Q3-1.7B       Q3-4B-Instruct-2507  4096
# -        -     -            E43       Q3-1.7B       Q3-4B-Instruct-2507  7168
#
# 模型缩写：R1=DeepSeek-R1-Distill-Qwen；Q3=Qwen3；JustRL=JustRL-DeepSeek。
# E14 与 E13 同模型，LR=5e-7；E16 与 E13 同模型，reuse=4。
#
# Sampled 算法对照：Short/Long 的预算见下表。
# Short  Long  Advantage  Policy loss
# E1     E7    rollout    token_ppo
# E2     E8    rollout    token_is
# E3     E9    current    token_ppo
# E4     E10   current    token_is
# E5     E11   current    reinforce
# E6     E12   rollout    reinforce
#
# Mirror 正式对照：直接调用编号即可；prompt/response=4096/7168，验证 cap=31744。
# Preset  Method   Estimator   Kappa  Eta  Student  Teacher      Val n  Backend           Chunk
# E44     full_kl  full_vocab  -      1    R1-1.5B  JustRL-1.5B  16     precompute_fused  128
# E45     mdpo     full_vocab  0      0.3  R1-1.5B  JustRL-1.5B  16     precompute_fused  128
# E46     bpo      full_vocab  0      0.3  R1-1.5B  JustRL-1.5B  16     precompute_fused  128
# E47     mdpo     full_vocab  0.5    0.3  R1-1.5B  JustRL-1.5B  16     precompute_fused  128
# E44–E47：batch=64，seed=42，IO async/prefetch 开启；MDPO/BPO 开启诊断，Full-KL 沿用 endpoint 指标。
#
# E48：R1-1.5B student / Polaris-7B-Preview teacher；沿用 E20 Full-KL、reuse16 与预算，开启初始验证。
# Teacher FSDP token budget 沿用公共默认 32768；支持 E48-valid-student / E48-valid-teacher。
# E49：沿用 E48 模型、预算和验证；改为 full-vocab Forward KL，precompute_fused，teacher 支持显式条件化到 student 输出行。
#
# 精度对照：复用 E44/E45 的模型、256 outer / reuse16 / save128，教师与 rollout 保持原配置。
# Full-KL  MDPO  Actor/anchor precision
# E44      E45   BF16 baseline
# E50      E51   FP16 + GradScaler
# E52      E53   BF16 GEMM / FP32 intermediates
# --precision bf16|fp16|bf16_fp32 可覆盖预设；fp32 是较慢的 packed math attention 参考。
#
# 三、E54–E62：Qwen Base 的 TOP-D 与执行后端对照
#
# E54：Qwen3-1.7B-Base ← Qwen3-30B-A3B-Instruct-2507；沿用 E41 的 100 outer/reuse16，无初始验证。
#
# E55–E58：同一 Base ← 30B-A3B-Instruct-2507 长程对照，从初始权重开始。
# Preset  Method   Kappa  Eta
# E55     mdpo     0      0.1
# E56     mdpo     0.5    0.1
# E57     mdpo     1      0.1
# E58     full_kl  -      1
# 统一 8 卡 shared_sync、batch64、reuse16、100 outer、LR=1e-6、train cap=16384；每 10 outer 验证五个数学 bench，val n=16、cap=31744，无初始验证。
# 原生共同前缀 enable_thinking=null；EOS 合并 [151643,151645]；precompute/chunk512。
# 不保存任何 checkpoint（save_freq=-1，包括最后一步）；E54 保留原 7k/旧前缀协议。
#
# E59–E62：分别沿用 E55–E58 的模型、目标和预算，改用已验收的 EOS precompute_fused/chunk512；actor/anchor 显式 bf16，完整诊断保留。旧 E55–E58 默认不变。
#
# 四、E63–E68 / E78–E80：rollout 来源对照
#
# E63–E68：Qwen3-1.7B non-thinking ← Qwen3-4B-Instruct-2507，六种前缀。
# Preset  Rollout
# E63     current
# E64     initial
# E65     teacher
# E66     logit fusion
# E67     probability fusion
# E68     batch fusion
# 默认每实验 8 卡；single 用 8 个 TP1，fusion 用 4 个 TP2；全局 batch64、40 outer/reuse16。fusion 每副本并发 16/每卡 KV24 GiB，current+ 评估并发 16/KV64 GiB；计算 token budget32768；shared_sync 保持已支持的参数/优化器卸载。
# 采用 main 统一原生评估与连续调度；QWEN_EVAL_ROOT 指定新银行。旧 HF 评估不可混接；八卡性能须实测。
#
# E78–E80：Qwen3-1.7B non-thinking ← Qwen3-4B-Instruct-2507；current/initial/teacher。
# batch=mini-batch128，100 outer/reuse16，1600 次更新；默认 2 卡 TP1，可显式配置 1/2/4/8 卡。
# 原生 bench0/50/100，MATH500 n4、AMC23 n16、AIME24/25/26 n32；source128 和双银行各 128，每 10 轮 RKL。
#
# 五、E71–E77 / E91：教师训练阶段与学生初始化对照
#
# E69/E70 DeepSeekMath 预设已撤销；E71–E73 固定 MiMo-7B-Base student。
# E71=MiMo-7B-RL（SFT+RL）；E72=MiMo-7B-RL-Zero（直接 RL）；E73=MiMo-7B-SFT。
# 默认 4 卡 shared_sync、batch128、LR=1e-6；训练题按顺序取用，无重复题目。
# E71：25 outer/reuse32/800 次更新/3200 题；E72/E73：20 outer/reuse16/320 次更新/2560 题。
# 原生 32K：训练 prompt/response=2048/7168；公共 prompt8192，Math/Code/Science response24576，IF8192。
# source/history RKL 关闭；四域固定 bank 在 0/1/2 及每 5 outer 评分。
# 完整 bench 在 0 与终点评测；预算超过 25 outer 时增加第 25 轮评测。
# MIMO_QUESTIONS 指定 prepared 题目，MIMO_BOOTSTRAP 指定对应 teacher 的新银行；关闭 MTP 推测解码与辅助训练；最终只导出 HF 模型并重载核对。
# 完整设置见 docs/experiments/teacher-transfer/post-training-stages.md。
#
# E91：MiMo-7B-SFT student ← MiMo-7B-RL-Zero direct-RL teacher。
# 复用 E72/E73 的 20 outer、reuse16、320 updates、batch128 设置，仅改变 student 初始化。
# E72 已覆盖同一 teacher 的 RL→Base。
#
# E74/E75：Olmo3 Base 学习 C3PO（SFT+RL）/IVON-SFT；E76/E77：Qwen2.5-14B Base 学习 Light-R1/R1-Distill。
# E76/E77 阶段不匹配：Light-R1 在 R1-Distill 之后额外做 3K SFT 再 GRPO，缺 RL 直接起点的 SFT-only 权重。
# 四组使用明确登记的共同 SFT tokenizer 协议视图，原始参数不改；Light-R1 记录 7 个特殊 ID 的标签变化。
# 14B 少于 4 卡时自动使用 FSDP2 逐层 CPU offload，保留 FP32 主权重和 AdamW；默认 4 卡使用阶段搬运。
# E74–E77 使用 batch128/reuse32/50 outer、训练 2048+7168、公共推理回答 24576/IF8192，关闭 source/history。
# STAGE_CHAIN_QUESTIONS 指定 prepared 题目，STAGE_CHAIN_BOOTSTRAP 指定独立银行；详见阶段链选型及接入验收。
#
# 六、全部编号的预算与覆盖规则
#
# 训练预算：Eval/Save 均按 outer step；Updates*=默认 batch==mini-batch 时的更新尝试数。
# Presets          Reuse  Outer  Updates*  LR      Eval/Save  Initial val
# E1-E6            4      32     128       shared  32/32      no
# E7-E12           16     200    3200      shared  50/100     no
# E13,E15,E23-E32  1      250    250       1e-6    20/100     yes
# E14              1      250    250       5e-7    20/100     yes
# E16              4      250    1000      1e-6    20/100     yes
# E17,E33-E40,E42  16     250    4000      1e-6    20/100     yes
# E18              64     64     4096      1e-6    8/16       no
# E19              32     128    4096      1e-6    16/32      no
# E20              16     256    4096      1e-6    32/64      no
# E41,E54          16     100    1600      1e-6    10/50      no
# E43              16     100    1600      1e-6    10/50      yes
# E44-E47,E50-E53  16     256    4096      1e-6    32/128     no
# E48-E49          16     256    4096      1e-6    32/64      yes
# E55-E62          16     100    1600      1e-6    10/-1      no
# E63-E68          16     40     640       1e-6    -1/-1      yes
# E71              32     25     800       1e-6    -1/-1      yes
# E72-E73,E91      16     20     320       1e-6    -1/-1      yes
# E74-E77          32     50     1600      1e-6    -1/-1      yes
# E78-E80          16     100    1600      1e-6    -1/-1      yes
# E81              16     25     400       1e-6    5/-1       no
# E82              16     100    1600      1e-6    5/-1       no
# E83              16     100    1600      1e-6    5/25       no
# E84              16     100    1600      1e-6    20/25      no
# E85-E90,E92-E93  16     100    1600      1e-6    20/-1      no
# Eval=-1 关闭常规验证；原生 evaluation 协议的 bank/bench 仍按各组独立日程执行。
# Save=-1 关闭周期 checkpoint；原生协议的最终 HF 导出见各组说明。
#
# shared LR 继承公共 launcher；其余模型、资源、验证长度以最终配置为准。
# 编号别名：E21=E15; E22=E13。
# E44–E47 沿用 E20 预算、每 128 outer 保存；其他探索可用 --method/--kappa/--eta 覆盖；custom 不加载编号预算。
# model 预设可加 -valid-student/-valid-teacher，独立验证原始模型。
#
# E41/E54 Full-KL 与 full-vocab MDPO 默认对齐终止动作 [151643,151645]；FULL_KL_TERMINATION_TOKEN_IDS=[] 可恢复原始词表目标。
# END GENERATED EXPERIMENT OVERVIEW
# 统一训练入口：python3 仅用标准库解析实验配置并转交公共 launcher。
# 训练与 Ray 使用的解释器仍由 PYTHON_BIN 指定。
set -eo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec python3 "$SCRIPT_DIR/launchers/math_experiment.py" "$@"
