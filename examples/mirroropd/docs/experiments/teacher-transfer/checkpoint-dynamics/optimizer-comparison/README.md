# SGD 与 Muon 的九时刻权重动力学

## 设置与比较范围

E124–E129 均为 Qwen3-1.7B ← Qwen3-4B-Instruct-2507，40 outer × reuse16，共 640 次优化器更新。E123 只有预备配置，没有训练产物，按确认标记缺失。这里不包含 AdamW 的新测量，也不能用这六组比较 AdamW 的优劣。

复用 [checkpoint-dynamics](../README.md) 的测量实现：outer 1/2/3/4/8/12/20/30/40；每层 q/k/v/o/gate/up/down，共 28×7=196 矩阵。读取经 native reload RKL 验收的 FP32 模型导出，在 GPU 4–7 上执行 FP64 精确 SVD（gesvd），右子空间 rank=16/32/64/128/256。测量与训练分离，不需要 optimizer state。

累计更新为 ΔWₜ=Wₜ−W₀，区间增量为 Wᵦ−Wₐ；完整权重谱 σ(Wₜ)、累计更新谱 σ(ΔWₜ) 与谱差 σ(Wₜ)−σ(W₀) 分开报告。Frame 使用主夹角和随机基线校正 overlap，原图同时报告能量捕获与边界谱隙。这里的 frame 是各配置内部随时间的比较，没有计算两个优化器的直接跨配置主夹角。

E126–E129 复用 E124 的固定 bank；E125 独立生成固定 bank。各组题单相同不等于生成的回答相同，E125 与其余组的绝对 RKL 比较包含 bank 差异。Muon 的矩阵学习率变化，辅助参数仍用 AdamW LR=1e-6、WD=0；不能解释为所有参数均只使用 Muon。

## 覆盖与数据

数据读取时间：2026-11-15 16:05:01 +0800。几何状态：测量进行中，以下几何为部分层预览。共同完整层：[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]。

| 实验 | Optimizer |    LR | 已测矩阵 | 快照/矩阵 |
| ---- | --------- | ----: | -------: | --------: |
| E124 | SGD       |  0.01 |  196/196 |         9 |
| E125 | SGD       |  0.03 |  196/196 |         9 |
| E126 | Muon      | 3e-07 |  196/196 |         9 |
| E127 | Muon      | 1e-06 |  195/196 |         9 |
| E128 | Muon      | 3e-06 |  195/196 |         9 |
| E129 | Muon      | 1e-05 |  195/196 |         9 |

原始测量：`/mnt/tidalfs-bdsz01/dataset/llm_dataset/zkn_data/fht_data/MirrorOPD-verl/examples/mirroropd/research-evidence/optimizer-geometry/measurements`。运行协议、张量 SHA256、逐矩阵原始奇异值、计时和失败回执均保留；[覆盖](data/coverage.json)、[统计](data/summary.json)、[精确 readout](data/readout.json.gz)。每条成功矩阵记录覆盖九时刻；196/196 才是该组全层完成。

## Held-out RKL 与后段变化

all 为 Math/Code/IF 三域等权平均。后段均值和斜率使用 outer31–40；斜率单位为每 outer RKL。这是对后段趋势的描述，单 seed 和十个时刻不足以确定渐近上限。

| 实验 | initial-bank 终点 | teacher-bank 终点 | initial 后 10 均值 | teacher 后 10 均值 | initial 后段斜率 | teacher 后段斜率 |
| ---- | ----------------: | ----------------: | -----------------: | -----------------: | ---------------: | ---------------: |
| E124 |          0.885784 |          0.968343 |           0.882964 |           0.980175 |        +0.000187 |        -0.000859 |
| E125 |          0.894661 |          1.025552 |           0.887030 |           0.995630 |         +0.00174 |         +0.00361 |
| E126 |          0.885773 |          0.901701 |           0.888643 |           0.901251 |        -0.000758 |        -0.000115 |
| E127 |          0.853561 |          0.917587 |           0.857349 |           0.919124 |         -0.00091 |        +9.54e-06 |
| E128 |          0.834639 |          0.944805 |           0.837364 |           0.944663 |         -0.00035 |        +0.000562 |
| E129 |          0.856795 |          1.020498 |           0.855657 |           1.015592 |         +0.00109 |         +0.00169 |

![两套固定 bank 的 RKL](figures/optimizer-rkl.png)

Muon 的 LR 从 3e-7 提至 3e-6 时，initial-bank all 终点改善，但 teacher-bank all 终点变差；到 1e-5 时 teacher-bank all 终点超过初始值。较大的 LR 改变了对两种状态分布的适配取舍。E124 的 teacher-bank all 终点同样高于初始值，SGD 也没有消除这种分布差别。不能只按 initial bank 的下降认定平台改善。

## 谱与 frame 的共同坐标比较

以下逐矩阵指标只使用六组都测齐七类投影的共同层，取矩阵中位数。矩阵间 IQR 是不同层/模块的差异，不是多 seed 置信区间。

| 实验 | 纳入矩阵 | 最终 stable rank | 最终 rank90 | ‖ΔW‖/‖W₀‖ | ΔW frame 16→640 k64 | ΔW frame 480→640 k64 | W frame 0→640 k64 |
| ---- | -------: | ---------------: | ----------: | --------: | ------------------: | -------------------: | ----------------: |
| E124 |      189 |             2.68 |          76 | 0.0003117 |              0.4941 |               0.9135 |            1.0000 |
| E125 |      189 |             3.16 |         113 | 0.0005588 |              0.4759 |               0.8871 |            1.0000 |
| E126 |      189 |              141 |         722 | 0.0003405 |              0.0396 |               0.8885 |            1.0000 |
| E127 |      189 |              229 |         911 |  0.001029 |              0.0301 |               0.8302 |            1.0000 |
| E128 |      189 |              342 |         989 |  0.002796 |              0.0231 |               0.6448 |            1.0000 |
| E129 |      189 |              410 |    1.02e+03 |  0.008484 |              0.0160 |               0.5395 |            0.9999 |

![优化器的共同层几何轨迹](figures/optimizer-geometry.png)

共同层累计更新的最终 stable rank：SGD 为 2.68–3.16，Muon 为 141–410。覆盖尚未完整，数值随层数增加更新，暂不推广到全模型。

Muon 3e-7→1e-5 的最终相对更新范数中位数扩大 24.92 倍。对应的 k64 早期→最终 overlap 为 0.040→0.016，末段 480→640 为 0.888→0.540。

Stable rank=‖ΔW‖²_F/‖ΔW‖²₂ 由最大奇异值主导；rank90 统计累积到 90% 能量所需维数。SGD 即便 stable rank 很低，rank90 仍可能很大，表示存在主导方向及长谱尾，不能直接称为严格低秩更新。

E124：最终 stable rank=2.68，rank90=76；teacher-bank all 后段均值=0.98017，拟合十个 outer 的变化为均值的 -0.88%。

E125：最终 stable rank=3.16，rank90=113；teacher-bank all 后段均值=0.99563，拟合十个 outer 的变化为均值的 +3.63%。

E126：最终 stable rank=141，rank90=722；teacher-bank all 后段均值=0.90125，拟合十个 outer 的变化为均值的 -0.13%。

E127：最终 stable rank=229，rank90=911；teacher-bank all 后段均值=0.91912，拟合十个 outer 的变化为均值的 +0.01%。

E128：最终 stable rank=342，rank90=989；teacher-bank all 后段均值=0.94466，拟合十个 outer 的变化为均值的 +0.59%。

E129：最终 stable rank=410，rank90=1.02e+03；teacher-bank all 后段均值=1.01559，拟合十个 outer 的变化为均值的 +1.67%。

Muon 每步对动量矩阵做近似正交化，弱化梯度奇异值幅度的差别；SGD 沿原梯度/动量的谱幅度更新。这使更新谱是否集中成为可检验差别，但累计 ΔW 是各步矩阵之和，仍会受到方向旋转与相消影响。LR 改变轨迹以后，累计谱形状与 frame 也会变化，不能把两档 LR 视为同一个最终方向的纯缩放。

较低的初始—最终 frame overlap 表示主更新子空间继续变化，不能单独解释为优化更好或泛化更好。应结合 late-increment 被早期 frame 捕获的能量、rank 边界谱隙以及两套 bank 的后段 RKL 判断。完整权重 frame 保持稳定而 ΔW frame 变化也不矛盾：在更新相对 W₀ 很小时，两者描述的对象与尺度不同。

现有数据能描述优化器如何分配更新能量、方向何时形成，以及这些量是否伴随 held-out 改善；不能从相关性推出某种谱造成平台。优先检查同坐标下各 LR 的增量范数是否仍大、480→640 frame 是否稳定、两套 bank 是否同时下降。如果参数持续变化而 teacher bank 不再改善，才进一步区分目标分布取舍与有效迁移方向不足。训练批内拟合的因果检验还需要结合 batch-before/after 指标，当前权重测量没有新增训练干预。

## 全部原协议图

每图同时保存 PNG、PDF 与 audit 文件。完整精确奇异值保留在原始矩阵记录；readout 为绘图用能量 CDF 和角度分位数。

### E124

![E124-early-final](figures/E124-early-final.png)

![E124-frame-angle-time](figures/E124-frame-angle-time.png)

![E124-frame-angles](figures/E124-frame-angles.png)

![E124-frame-increments](figures/E124-frame-increments.png)

![E124-frame-layer-modules](figures/E124-frame-layer-modules.png)

![E124-frame-modules](figures/E124-frame-modules.png)

![E124-frame-outside-energy](figures/E124-frame-outside-energy.png)

![E124-frame-pairwise](figures/E124-frame-pairwise.png)

![E124-frame-rank-time](figures/E124-frame-rank-time.png)

![E124-sigma-dynamics](figures/E124-sigma-dynamics.png)

![E124-sigma-layers](figures/E124-sigma-layers.png)

![E124-sigma-pairwise](figures/E124-sigma-pairwise.png)

![E124-sigma-teacher](figures/E124-sigma-teacher.png)

![E124-singular-energy-rank](figures/E124-singular-energy-rank.png)

![E124-singular-frame-phase](figures/E124-singular-frame-phase.png)

![E124-singular-gap-angles](figures/E124-singular-gap-angles.png)

![E124-singular-layer-modules](figures/E124-singular-layer-modules.png)

![E124-singular-profiles](figures/E124-singular-profiles.png)

![E124-singular-rank-time](figures/E124-singular-rank-time.png)

![E124-singular-scale-shape](figures/E124-singular-scale-shape.png)

![E124-singular-update-profiles](figures/E124-singular-update-profiles.png)

![E124-update-dimension](figures/E124-update-dimension.png)

![E124-weight-spectrum](figures/E124-weight-spectrum.png)

![E124-weight-update-frames](figures/E124-weight-update-frames.png)

### E125

![E125-early-final](figures/E125-early-final.png)

![E125-frame-angle-time](figures/E125-frame-angle-time.png)

![E125-frame-angles](figures/E125-frame-angles.png)

![E125-frame-increments](figures/E125-frame-increments.png)

![E125-frame-layer-modules](figures/E125-frame-layer-modules.png)

![E125-frame-modules](figures/E125-frame-modules.png)

![E125-frame-outside-energy](figures/E125-frame-outside-energy.png)

![E125-frame-pairwise](figures/E125-frame-pairwise.png)

![E125-frame-rank-time](figures/E125-frame-rank-time.png)

![E125-sigma-dynamics](figures/E125-sigma-dynamics.png)

![E125-sigma-layers](figures/E125-sigma-layers.png)

![E125-sigma-pairwise](figures/E125-sigma-pairwise.png)

![E125-sigma-teacher](figures/E125-sigma-teacher.png)

![E125-singular-energy-rank](figures/E125-singular-energy-rank.png)

![E125-singular-frame-phase](figures/E125-singular-frame-phase.png)

![E125-singular-gap-angles](figures/E125-singular-gap-angles.png)

![E125-singular-layer-modules](figures/E125-singular-layer-modules.png)

![E125-singular-profiles](figures/E125-singular-profiles.png)

![E125-singular-rank-time](figures/E125-singular-rank-time.png)

![E125-singular-scale-shape](figures/E125-singular-scale-shape.png)

![E125-singular-update-profiles](figures/E125-singular-update-profiles.png)

![E125-update-dimension](figures/E125-update-dimension.png)

![E125-weight-spectrum](figures/E125-weight-spectrum.png)

![E125-weight-update-frames](figures/E125-weight-update-frames.png)

### E126

![E126-early-final](figures/E126-early-final.png)

![E126-frame-angle-time](figures/E126-frame-angle-time.png)

![E126-frame-angles](figures/E126-frame-angles.png)

![E126-frame-increments](figures/E126-frame-increments.png)

![E126-frame-layer-modules](figures/E126-frame-layer-modules.png)

![E126-frame-modules](figures/E126-frame-modules.png)

![E126-frame-outside-energy](figures/E126-frame-outside-energy.png)

![E126-frame-pairwise](figures/E126-frame-pairwise.png)

![E126-frame-rank-time](figures/E126-frame-rank-time.png)

![E126-sigma-dynamics](figures/E126-sigma-dynamics.png)

![E126-sigma-layers](figures/E126-sigma-layers.png)

![E126-sigma-pairwise](figures/E126-sigma-pairwise.png)

![E126-sigma-teacher](figures/E126-sigma-teacher.png)

![E126-singular-energy-rank](figures/E126-singular-energy-rank.png)

![E126-singular-frame-phase](figures/E126-singular-frame-phase.png)

![E126-singular-gap-angles](figures/E126-singular-gap-angles.png)

![E126-singular-layer-modules](figures/E126-singular-layer-modules.png)

![E126-singular-profiles](figures/E126-singular-profiles.png)

![E126-singular-rank-time](figures/E126-singular-rank-time.png)

![E126-singular-scale-shape](figures/E126-singular-scale-shape.png)

![E126-singular-update-profiles](figures/E126-singular-update-profiles.png)

![E126-update-dimension](figures/E126-update-dimension.png)

![E126-weight-spectrum](figures/E126-weight-spectrum.png)

![E126-weight-update-frames](figures/E126-weight-update-frames.png)

### E127

![E127-early-final](figures/E127-early-final.png)

![E127-frame-angle-time](figures/E127-frame-angle-time.png)

![E127-frame-angles](figures/E127-frame-angles.png)

![E127-frame-increments](figures/E127-frame-increments.png)

![E127-frame-layer-modules](figures/E127-frame-layer-modules.png)

![E127-frame-modules](figures/E127-frame-modules.png)

![E127-frame-outside-energy](figures/E127-frame-outside-energy.png)

![E127-frame-pairwise](figures/E127-frame-pairwise.png)

![E127-frame-rank-time](figures/E127-frame-rank-time.png)

![E127-sigma-dynamics](figures/E127-sigma-dynamics.png)

![E127-sigma-layers](figures/E127-sigma-layers.png)

![E127-sigma-pairwise](figures/E127-sigma-pairwise.png)

![E127-sigma-teacher](figures/E127-sigma-teacher.png)

![E127-singular-energy-rank](figures/E127-singular-energy-rank.png)

![E127-singular-frame-phase](figures/E127-singular-frame-phase.png)

![E127-singular-gap-angles](figures/E127-singular-gap-angles.png)

![E127-singular-layer-modules](figures/E127-singular-layer-modules.png)

![E127-singular-profiles](figures/E127-singular-profiles.png)

![E127-singular-rank-time](figures/E127-singular-rank-time.png)

![E127-singular-scale-shape](figures/E127-singular-scale-shape.png)

![E127-singular-update-profiles](figures/E127-singular-update-profiles.png)

![E127-update-dimension](figures/E127-update-dimension.png)

![E127-weight-spectrum](figures/E127-weight-spectrum.png)

![E127-weight-update-frames](figures/E127-weight-update-frames.png)

### E128

![E128-early-final](figures/E128-early-final.png)

![E128-frame-angle-time](figures/E128-frame-angle-time.png)

![E128-frame-angles](figures/E128-frame-angles.png)

![E128-frame-increments](figures/E128-frame-increments.png)

![E128-frame-layer-modules](figures/E128-frame-layer-modules.png)

![E128-frame-modules](figures/E128-frame-modules.png)

![E128-frame-outside-energy](figures/E128-frame-outside-energy.png)

![E128-frame-pairwise](figures/E128-frame-pairwise.png)

![E128-frame-rank-time](figures/E128-frame-rank-time.png)

![E128-sigma-dynamics](figures/E128-sigma-dynamics.png)

![E128-sigma-layers](figures/E128-sigma-layers.png)

![E128-sigma-pairwise](figures/E128-sigma-pairwise.png)

![E128-sigma-teacher](figures/E128-sigma-teacher.png)

![E128-singular-energy-rank](figures/E128-singular-energy-rank.png)

![E128-singular-frame-phase](figures/E128-singular-frame-phase.png)

![E128-singular-gap-angles](figures/E128-singular-gap-angles.png)

![E128-singular-layer-modules](figures/E128-singular-layer-modules.png)

![E128-singular-profiles](figures/E128-singular-profiles.png)

![E128-singular-rank-time](figures/E128-singular-rank-time.png)

![E128-singular-scale-shape](figures/E128-singular-scale-shape.png)

![E128-singular-update-profiles](figures/E128-singular-update-profiles.png)

![E128-update-dimension](figures/E128-update-dimension.png)

![E128-weight-spectrum](figures/E128-weight-spectrum.png)

![E128-weight-update-frames](figures/E128-weight-update-frames.png)

### E129

![E129-early-final](figures/E129-early-final.png)

![E129-frame-angle-time](figures/E129-frame-angle-time.png)

![E129-frame-angles](figures/E129-frame-angles.png)

![E129-frame-increments](figures/E129-frame-increments.png)

![E129-frame-layer-modules](figures/E129-frame-layer-modules.png)

![E129-frame-modules](figures/E129-frame-modules.png)

![E129-frame-outside-energy](figures/E129-frame-outside-energy.png)

![E129-frame-pairwise](figures/E129-frame-pairwise.png)

![E129-frame-rank-time](figures/E129-frame-rank-time.png)

![E129-sigma-dynamics](figures/E129-sigma-dynamics.png)

![E129-sigma-layers](figures/E129-sigma-layers.png)

![E129-sigma-pairwise](figures/E129-sigma-pairwise.png)

![E129-sigma-teacher](figures/E129-sigma-teacher.png)

![E129-singular-energy-rank](figures/E129-singular-energy-rank.png)

![E129-singular-frame-phase](figures/E129-singular-frame-phase.png)

![E129-singular-gap-angles](figures/E129-singular-gap-angles.png)

![E129-singular-layer-modules](figures/E129-singular-layer-modules.png)

![E129-singular-profiles](figures/E129-singular-profiles.png)

![E129-singular-rank-time](figures/E129-singular-rank-time.png)

![E129-singular-scale-shape](figures/E129-singular-scale-shape.png)

![E129-singular-update-profiles](figures/E129-singular-update-profiles.png)

![E129-update-dimension](figures/E129-update-dimension.png)

![E129-weight-spectrum](figures/E129-weight-spectrum.png)

![E129-weight-update-frames](figures/E129-weight-update-frames.png)
