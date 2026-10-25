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
# E31      E32   E42          E43       Q3-1.7B       Q3-4B-Instruct-2507  4096
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
# 训练预算：Eval/Save 均按 outer step；Updates*=默认 batch==mini-batch 时的更新尝试数。
# Presets          Reuse  Outer  Updates*  LR      Eval/Save  Initial val
# E1-E6            4      32     128       shared  32/32      no
# E7-E12           16     200    3200      shared  50/100     no
# E13,E15,E23-E32  1      250    250       1e-6    20/100     yes
# E14              1      250    250       5e-7    20/100     yes
# E16              4      250    1000      1e-6    20/100     yes
# E17,E33-E43      16     250    4000      1e-6    20/100     yes
# E18              64     64     4096      1e-6    8/16       no
# E19              32     128    4096      1e-6    16/32      no
# E20              16     256    4096      1e-6    32/64      no
# E44-E47          16     256    4096      1e-6    32/128     no
#
# shared LR 继承公共 launcher；其余模型、资源、验证长度以最终配置为准。
# 编号别名：E21=E15; E22=E13。
# E44–E47 沿用 E20 预算、每 128 outer 保存；其他探索可用 --method/--kappa/--eta 覆盖；custom 不加载编号预算。
# model 预设可加 -valid-student/-valid-teacher，独立验证原始模型。
# END GENERATED EXPERIMENT OVERVIEW
# Single public training entry. Planning uses Python's standard library only;
# PYTHON_BIN continues to select the training/Ray interpreter in the shared launcher.
set -eo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec python3 "$SCRIPT_DIR/launchers/math_experiment.py" "$@"
