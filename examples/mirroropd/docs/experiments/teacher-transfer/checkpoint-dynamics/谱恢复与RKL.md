# 恢复初始奇异值后的完整 bank RKL

对每个 checkpoint 的完整权重执行 $W_t=U_t\Sigma_tV_t^\top \to U_t\Sigma_0V_t^\top$。编辑所有 Transformer 层的 q/k/v/o/gate/up/down；embedding、head、norm 等参数保留该 checkpoint。

固定使用本轮 initial 和 teacher 两种 bank，每个 bank 为 Math128、Code64、IF64，共 512 条轨迹。保留原始 token 与 mask，评分所有有效响应位置，上限 7168；每题先按有效 token 平均，三域平均再等权。

编辑在 FP64 中分解并重构，前向使用原生 BF16 权重、FA2 和主线稳定全词表 RKL 内核。未编辑与编辑条件均在独立 HF 路径重新评分，差值定义为编辑后减编辑前。训练时 FSDP 的逐题分数另作执行对照，不能替代这里的成对基线。每个阶段还对双 bank 三域各一条完整响应重复无编辑前向，要求最大逐题 RKL 差不超过 1e-8。

[曲线数据](data/sigma0-bank/curve-data.json)与相邻 rows 子目录保存逐题及位置分段结果，绘图不依赖运行目录。

| Pair | 完整阶段 | 最新更新数 | initial-bank Math：原始→恢复谱 | teacher-bank Math：原始→恢复谱 | 与训练记录最大逐题偏差 |
| ---- | -------: | ---------: | -----------------------------: | -----------------------------: | ---------------------: |
| E94  |      9/9 |        640 |      0.000256755 → 0.000257374 |      0.000221113 → 0.000221298 |               0.000178 |
| E95  |      9/9 |        640 |        0.00210915 → 0.00211251 |         0.0018515 → 0.00184974 |                0.00141 |
| E96  |      9/9 |        640 |        0.00637979 → 0.00638507 |         0.00819899 → 0.0081885 |                0.00124 |
| E97  |      9/9 |        640 |        0.00355718 → 0.00357146 |        0.00178394 → 0.00178331 |                 0.0037 |
| E98  |      9/9 |        640 |        0.00478123 → 0.00478857 |        0.00468845 → 0.00469994 |                 0.0122 |
| E99  |      9/9 |        640 |      0.000331525 → 0.000327671 |       0.000342272 → 0.00034933 |               0.000534 |
| E100 |      9/9 |        640 |      0.000337136 → 0.000343286 |      0.000343142 → 0.000348521 |               0.000593 |
| E101 |      9/9 |        640 |          0.0170987 → 0.0170169 |         0.00910823 → 0.0091128 |                   0.01 |
| E102 |      9/9 |        640 |            0.252754 → 0.252866 |            0.274415 → 0.274833 |                  0.126 |
| E103 |      9/9 |        640 |        0.00968692 → 0.00965826 |      0.000377339 → 0.000373629 |                0.00424 |
| E104 |      9/9 |        640 |            0.135573 → 0.135604 |          0.0877205 → 0.0877268 |                  0.018 |

## 图与指标

| 图       | 纵轴                       | 用途                                               |
| -------- | -------------------------- | -------------------------------------------------- |
| absolute | 未编辑与恢复谱后的绝对 RKL | 判断已学分布匹配是否保留                           |
| delta    | 恢复谱后 RKL − 未编辑 RKL  | 判断训练阶段对谱变化的依赖；正值表示恢复谱损伤匹配 |

## E94

![E94 的绝对 RKL](figures/sigma0-bank/E94-absolute.png)

![E94 的换谱效应](figures/sigma0-bank/E94-delta.png)

## E95

![E95 的绝对 RKL](figures/sigma0-bank/E95-absolute.png)

![E95 的换谱效应](figures/sigma0-bank/E95-delta.png)

## E96

![E96 的绝对 RKL](figures/sigma0-bank/E96-absolute.png)

![E96 的换谱效应](figures/sigma0-bank/E96-delta.png)

## E97

![E97 的绝对 RKL](figures/sigma0-bank/E97-absolute.png)

![E97 的换谱效应](figures/sigma0-bank/E97-delta.png)

## E98

![E98 的绝对 RKL](figures/sigma0-bank/E98-absolute.png)

![E98 的换谱效应](figures/sigma0-bank/E98-delta.png)

## E99

![E99 的绝对 RKL](figures/sigma0-bank/E99-absolute.png)

![E99 的换谱效应](figures/sigma0-bank/E99-delta.png)

## E100

![E100 的绝对 RKL](figures/sigma0-bank/E100-absolute.png)

![E100 的换谱效应](figures/sigma0-bank/E100-delta.png)

## E101

![E101 的绝对 RKL](figures/sigma0-bank/E101-absolute.png)

![E101 的换谱效应](figures/sigma0-bank/E101-delta.png)

## E102

![E102 的绝对 RKL](figures/sigma0-bank/E102-absolute.png)

![E102 的换谱效应](figures/sigma0-bank/E102-delta.png)

## E103

![E103 的绝对 RKL](figures/sigma0-bank/E103-absolute.png)

![E103 的换谱效应](figures/sigma0-bank/E103-delta.png)

## E104

![E104 的绝对 RKL](figures/sigma0-bank/E104-absolute.png)

![E104 的换谱效应](figures/sigma0-bank/E104-delta.png)
