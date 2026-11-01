# 文档归属与检查

执行[maintain-docs](../.agents/skills/maintain-docs/SKILL.md)和[maintain-prose](../.agents/skills/maintain-prose/SKILL.md)，保留父层规则。

- 研究入口采用五个 RQ：信号/估计器、reuse 与预算、六类 rollout 来源、失败复现与吸收机制、TOP-D 复现与κ/η消融。物理目录保留四条研究线，RQ5 由 learning-signal 内 weighted-targets 承接；编号不作为长期路径身份。
- RQ4 先记录 pair/setting 下的吸收边界，再讨论跨 domain 与跨 batch 原因；SFT/SFT+RL 为阶段对照。实际设置、负结果和证据边界不能省略。“完全蒸馏教师”的定义保留开放，不新增评价门槛。
- 精度、optimizer、数值执行与性能实验统一归 infra，不设 experiments/infra。RQ5 仅引用工程结论，正文聚焦 TOP-D 复现与κ/η。Qwen 修复与 E54 故障诊断仍归 infra。
- 理论定义与完整推导归 theory；公共参数、模型、数据与指标含义归 settings；外部论文和公开来源归 literature。实际 run 的配置由原报告与 resolved config 维护，当前默认值不回写历史。
- progress 只放周报及周报专用附件。实验入口直接引用正式研究正文；同一读者问题只维护一份完整正文，必要局部定义与摘要可以引用。新 run、seed、模型条件或日期通常更新已有专题，不自动新建报告。
- 合并先保留独有条件、推导、失败和复算依赖，再删除重复正文并修复入链、锚点、行内路径与生成器。协议变化注明适用运行与替代关系；历史设计不作为当前接口手册。
- archive、data、sources、原始模型卡及证据源码保持冻结；活动页面进入其中的引用仍检查。同步排障、凭据、GPU 清理与执行流水不进入研究正文。
- 正文自然段不硬换行，公式与表格按含义使用。运行 make docs-format 与 make docs-check；机械检查不替代科学结论及内容归属审阅，入口见[维护流程](infra/maintenance/README.md)。

- 正文先写观察、定义与实际条件；同一限制放在适用的设置或比较处说明一次。清理会话/迁移流水和重复防御，不删负结果、数学假设、失败行为或影响解释的限制。
