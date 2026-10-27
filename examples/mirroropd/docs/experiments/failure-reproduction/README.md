# 模型与长度的失败复现

本系列采用 DAPO → MATH-500／AMC23／AIME25，比较 sampled-token 与 full-vocab，包含基础版和同批 reuse16。详细模型、prompt/response 长度、终止协议、采样、训练预算、论文依据和启动命令集中在[正式实验方案](2026-10-25-Sampled-token与Full-vocab失败复现方案.md)。

## 正式训练进度

| 设置 | 基础 sampled / full | Reuse16 sampled / full | 现有证据与待跑项 |
| --- | --- | --- | --- |
| R1-1.5B → JustRL-1.5B，7168 response | E15 / E13 | E33 / E17 | E13/E15/E17 有旧 200-step 结果；本轮 250-step＋初始验证协议待跑，E33 未发现正式 run |
| 同模型对，15360 response | E23 / E24 | E34 / E35 | 四臂均未发现正式 run |
| R1-1.5B → R1-7B，7168 response | E25 / E26 | E36 / E37 | 有 E25 两步短测；四臂正式预算均待跑 |
| Qwen3-1.7B-Base → Qwen3-8B | E27 / E28 | E38 / E39 | [E39终止动作对齐变体](../e39-base-8b/README.md)已完成200 outer/3200更新；原250-step四臂对照仍待补齐 |
| Qwen3-1.7B-Base → Qwen3-4B Non-thinking | E29 / E30 | E40 / E41 | [E41现行100步配置](../e41-e43/README.md)已准备，关闭初始验证；其余原对照待补齐 |
| Qwen3-1.7B Non-thinking → Qwen3-4B-Instruct-2507 | E31 / E32 | E42 / E43 | [E43现行100步配置](../e41-e43/README.md)已准备，训练cap7168且开启初始验证；其余原对照待补齐 |

状态依据 2026-10-25 项目清单与日志提交 `ff8b9f0`。E21/E22 是启动脚本既有的 E15/E13 别名，不重复计算独立实验。每臂计划 250 outer，batch=mini-batch=64、n=1、LR=1e-6；基础版 250 updates、reuse16 版 4000 updates。短测只证明接口经过执行，不证明停滞、退化或失稳已复现。

2026-10-27补充的E39变体采用终止动作对齐、200 outer、每10步验证、初始验证开启，完整结果见上方独立入口；其准确率和重复改善属于该单次运行，不等同于原方案各臂或多seed比较已经完成。

## 原始师生参考与分析条件

各模型对还需要相同验证协议下的原始 student/teacher 评测。现有 1.5B 历史参考见[Full-KL 师生分析](../full-kl-comparison/2026-10-24-Full-KL收敛与师生参考评测.md)；执行／模板不同，不能直接补成新矩阵的真实 step 0。正式比较保持同组 sampled/full 的停止协议与过滤后题目集合一致，并补充相同 seed 集合。

## 研究依据与工程证据

- [原论文设置档案](2026-10-24-OPD失败复现设置与Full-vocab对照.md)记录作者参数与本仓适配差异。
- [论文原图](figures/README.md)与 [原始检索记录](sources/2026-10-24-OPD收敛原始检索报告.md)随实验方案维护。
- [实现与对齐诊断](../../failure-reproduction/2026-10-24-OPD实现验收与对齐诊断.md)保留 EOS、词表、探针和短测证据；数据和脚本仍属于实现验收，不计入正式运行表。
