#!/usr/bin/env bash
# BEGIN GENERATED EXPERIMENT OVERVIEW
# E 编号速查：以下为默认设置，环境、方法选项和 Hydra 可以覆盖。
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
# E85-E90          16     100    1600      1e-6    20/-1      no
#
# shared LR 继承公共 launcher；其余模型、资源、验证长度以最终配置为准。
# 编号别名：E21=E15; E22=E13。
# E44–E47 沿用 E20 预算、每 128 outer 保存；其他探索可用 --method/--kappa/--eta 覆盖；custom 不加载编号预算。
# model 预设可加 -valid-student/-valid-teacher，独立验证原始模型。
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
# E63–E68：Qwen3-1.7B non-thinking ← Qwen3-4B-Instruct-2507，六种前缀。
# Preset  Rollout
# E63     current
# E64     initial
# E65     teacher
# E66     logit fusion
# E67     probability fusion
# E68     batch fusion
# 默认每实验8卡；single用8个TP1，fusion用4个TP2；全局batch64、40 outer/reuse16。fusion 每副本并发16/每卡KV24 GiB，current+评估并发16/KV64 GiB；计算token budget32768；shared_sync保持已支持的参数/优化器卸载。
# 采用main统一原生评估与连续调度；QWEN_EVAL_ROOT 指定新银行。旧HF评估不可混接；八卡性能须实测。
#
# E69/E70 DeepSeekMath预设已撤销；E71–E73固定MiMo-7B-Base student。
# E71=MiMo-7B-RL（SFT+RL）；E72=MiMo-7B-RL-Zero（直接RL）；E73=MiMo-7B-SFT。
# 默认4卡 shared_sync、batch128、reuse32、50 outer、LR=1e-6；同一6400题，无重复题目。
# 原生32K：训练prompt/response=2048/7168；公共prompt8192，Math/Code/Science response24576，IF8192。
# source/history RKL关闭；四域固定bank在0/1/2及每5 outer评分，current RKL与完整accuracy在0/25/50评分。
# MIMO_QUESTIONS指定prepared题目，MIMO_BOOTSTRAP指定对应teacher的新银行；关闭MTP推测解码与辅助训练；最终只导出HF模型并重载核对；完整设置见teacher-transfer/transfer-mechanism/mimo/。
#
# E91：MiMo-7B-SFT student ← MiMo-7B-RL-Zero direct-RL teacher；复用E72/E73的20 outer、reuse16、320 updates、batch128设置，仅改变student初始化。E72已覆盖同一teacher的RL→Base。
#
# E74/E75：Olmo3 Base学习C3PO（SFT+RL）/IVON-SFT；E76/E77：Qwen2.5-14B Base学习Light-R1/R1-Distill。
# E76/E77阶段不匹配：Light-R1在R1-Distill之后额外做3K SFT再GRPO，缺RL直接起点的SFT-only权重。
# 四组使用明确登记的共同SFT tokenizer协议视图，原始参数不改；Light-R1记录7个特殊ID的标签变化。
# 14B少于4卡时自动使用FSDP2逐层CPU offload，保留FP32主权重和AdamW；默认4卡使用阶段搬运。
# 沿用batch128/reuse32/50 outer、训练2048+7168、公共推理回答24576/IF8192，关闭source/history。
# STAGE_CHAIN_QUESTIONS指定prepared题目，STAGE_CHAIN_BOOTSTRAP指定独立银行；详见阶段链选型及接入验收。
#
# E78–E80：Qwen3-1.7B non-thinking ← Qwen3-4B-Instruct-2507；current/initial/teacher。batch=mini-batch128，100 outer/reuse16，1600更新；默认2卡TP1，可显式配置1/2/4/8卡。原生bench0/50/100，MATH500 n4、AMC23 n16、AIME24/25/26 n32；source128和双银行各128，每10轮RKL。
#
# E81/E82：Qwen3-1.7B-Base ← Qwen3-30B-A3B-Instruct-2507；MDPO kappa1/eta0.1。4卡shared_sync、batch=mini-batch256、n1/reuse16；A=25 outer/400更新，B=100 outer/1600更新。无初始验证，每5 outer及终点验证；不保存checkpoint。每引擎并发64/TP1、显存0.8；precompute_fused/chunk512、bf16；验证avg16/31744；训练与验证默认开启重复提前停止。
#
# E83/E84：继承E82的100 outer/1600更新与LR1e-6；E83=future none/seq-mean-token-mean，E84=future mean(gamma1)/token-mean。E83维持4卡/每5 outer验证；E84改为8卡/每20 outer验证（20/40/60/80/100）。每25 outer同步保存完整训练状态，仅保留最新一份，另保留最终HF模型。
#
# E85–E90：继承E82预算与MDPO设置，改用8卡 shared_sync；E85/E86/E87分别为none下的token-sum、seq-mean-token-sum、seq-mean-token-sum-norm；E88/E89/E90分别为token-mean下的future sum、discounted_sum(gamma=.99)、discounted_mean(gamma=.99)。E87–E90 使用MATH-500 n=4、其余四个数学bench n=16；E85/E86维持原avg@16。每20 outer及终点验证，100 outer内共5次，无初始验证。六组均save_freq=-1、resume_mode=disable，不保存checkpoint、HF模型或optimizer。
#
# E41/E54 Full-KL 与 full-vocab MDPO 默认对齐终止动作 [151643,151645]；FULL_KL_TERMINATION_TOKEN_IDS=[] 可恢复原始词表目标。
# END GENERATED EXPERIMENT OVERVIEW
# Single public training entry. Planning uses Python's standard library only;
# PYTHON_BIN continues to select the training/Ray interpreter in the shared launcher.
set -eo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec python3 "$SCRIPT_DIR/launchers/math_experiment.py" "$@"
