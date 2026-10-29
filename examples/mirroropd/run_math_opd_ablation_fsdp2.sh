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
# E55-E58          16     100    1600      1e-6    10/-1      no
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
# E41/E54 Full-KL 与 full-vocab MDPO 默认对齐终止动作 [151643,151645]；FULL_KL_TERMINATION_TOKEN_IDS=[] 可恢复原始词表目标。
# END GENERATED EXPERIMENT OVERVIEW
# Single public training entry. Planning uses Python's standard library only;
# PYTHON_BIN continues to select the training/Ray interpreter in the shared launcher.
set -eo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec python3 "$SCRIPT_DIR/launchers/math_experiment.py" "$@"
