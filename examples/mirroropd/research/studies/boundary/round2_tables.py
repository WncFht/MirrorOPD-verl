# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Render complete task-score/GapRecovery tables from audited measurements."""

import os
from pathlib import Path

from examples.mirroropd.research.core.io import read
from examples.mirroropd.research.core.paths import REPO
from examples.mirroropd.research.studies.boundary.round2_common import DATASETS, OLD_OUT, OUT

TEACHERS = {
    "justrl": "JustRL",
    "nemotron": "Nemotron v1",
    "minimath": "MiniMath",
    "deepscaler": "DeepScaleR",
    "brorl": "BroRL",
    "deepcoder": "DeepCoder",
    "math15": "Qwen-Math",
    "polaris": "Polaris",
    "r7": "R1-7B",
    "skywork": "Skywork",
    "light": "Light",
}
ORDER = [
    "TB1-J",
    "CONT-J",
    "TB1-N",
    "CONT-N",
    "TB2-M1",
    "CONT-M",
    "NEW-DS",
    "NEW-BRO",
    "NEW-DC",
    "TB1-R",
    "TB1-P",
    "TB1-S",
    "TB2-L1",
    "TB2-P7",
    "TB2-L7",
]


def table(headers, rows):
    return (
        "\n".join(
            ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
            + ["| " + " | ".join(map(str, row)) + " |" for row in rows]
        )
        + "\n\n"
    )


def main():
    import argparse

    argparse.ArgumentParser(description=__doc__).parse_args()
    page = REPO / (
        "examples/mirroropd/docs/experiments/teacher-transfer/transfer-mechanism/r1/"
        "2026-10-29-同架构教师吸收假说与续训设计.md"
    )
    link = os.path.relpath(OUT / "analysis", page.parent) + "/"
    script_link = os.path.relpath(Path(__file__), page.parent)
    assert read(OUT / "analysis/verification.json")["complete"]
    data = read(OUT / "analysis/summary.json")
    index = {(r["arm"], r["dataset"]): r for r in data["gap_recovery"]}
    accuracy = {(r["name"], r["dataset"]): r for r in data["accuracies"]}
    datasets = list(DATASETS)
    labels = ["MATH500", "AMC23", "AIME24", "AIME25", "AIME26"]

    def label(arm):
        row = index[arm, "macro5"]
        return f"{TEACHERS[row['teacher']]} / {row['step']}"

    def student(arm):
        return "1.5B" if index[arm, "macro5"]["student"] == "r1" else "7B"

    def gap(row):
        v = row["gap_recovery_percent"]
        if v is None:
            return "未定义"
        tags = "‡" if row["interpretation"] == "teacher_below_student" else ""
        tags += "†" if not row["ratio_ci_defined"] else ""
        return f"{v:.2f}{tags}"

    text = (
        "## 8. 完整任务评测与 GapRecovery 结果\n\n四个新教师和十五个学生终点评测"
        "全部完成，共 47880 个回答，无缺失回答槽位。以下准确率为百分数，均是逐题四次采样的平"
        "均；macro5 对五个任务等权。原始逐题答案、采样 seed、截断信息及评分保存在对应 "
        "evaluation 目录。\n\n### 8.1 新教师基线\n\n"
    )
    old = read(OLD_OUT / "baseline/r1-shared/metrics.json")
    keys = ["math500", "amc23", "aime24", "aime25", "aime26"]
    rows = [["Student S0（共同 32K 协议）"] + [f"{100 * old[f'eval/{k}/acc/mean']:.2f}" for k in keys]]
    for name in ["deepscaler", "brorl", "deepcoder"]:
        rows.append([TEACHERS[name]] + [f"{accuracy['teacher-' + name, d]['acc_percent']:.2f}" for d in datasets])
    text += table(["模型"] + labels, rows)
    text += "Qwen-Math 单独使用原生 4K 上下文、响应上限 3072，暂不训练；下表不参与共同 32K 协议的 GapRecovery。\n\n"
    text += table(
        ["独立基线"] + labels,
        [["Qwen2.5-Math-1.5B"] + [f"{accuracy['teacher-math15', d]['acc_percent']:.2f}" for d in datasets]],
    )
    text += "### 8.2 全部学生终点准确率\n\nTeacher 列中的数字是累计训练更新数。"
    text += "7B 两行只有原有 step064 的补充评测，没有新增训练。\n\n"
    text += table(
        ["Student", "Teacher / step"] + labels + ["macro5"],
        [
            [student(a), label(a)]
            + [f"{index[a, d]['after_acc']:.2f}" for d in datasets]
            + [f"{index[a, 'macro5']['after_acc']:.3f}"]
            for a in ORDER
        ],
    )
    text += (
        "### 8.3 全部配对的 GapRecovery\n\n数值单位为 %，保留负值及超过 10"
        "0% 的原值。‡ 表示 teacher 准确率低于初始 student，此时正的公式值可能"
        "来自 student 能力下降；† 表示分母不稳定，无法报告有界比值区间。零分母记为未定义"
        "。每行的 before 都是相应原始 student，128-step 行也不改用 ste"
        "p064 作 before。\n\n"
    )
    text += table(
        ["Student", "Teacher / step"] + labels + ["macro5"],
        [[student(a), label(a)] + [gap(index[a, d]) for d in datasets + ["macro5"]] for a in ORDER],
    )
    text += (
        "### 8.4 最新终点的 macro5 原始量与区间\n\n置信区间采用 5000 次按题配"
        "对 bootstrap，反映当前问题集合的采样不确定性，不是训练 seed 方差。区间跨零"
        "时，单个点估计的正负不能单独作为稳定能力改善的证据。\n\n"
    )
    rows = []
    for arm in ORDER:
        if arm in ["TB1-J", "TB1-N", "TB2-M1"]:
            continue
        r = index[arm, "macro5"]
        ci = f"[{r['ratio_ci_low']:.2f}, {r['ratio_ci_high']:.2f}]" if r["ratio_ci_defined"] else "不报告"
        rows.append(
            [student(arm), label(arm)]
            + [f"{r[k]:.3f}" for k in ["before_acc", "teacher_acc", "after_acc", "gain_pp"]]
            + [gap(r), ci]
        )
    text += table(
        ["Student", "Teacher / step", "Before", "Teacher", "After", "增益 pp", "GapRecovery", "95% 区间"], rows
    )
    text += (
        "完整的75个单任务比值和15个macro5比值，以及三组准确率、分母区间、准确率增益区间、"
        f"无界情况和重采样零分母比例，见[GapRecovery CSV]({link}gap-recovery.csv)。"
        f"各模型的平均回答长度和截断比例见[任务准确率与长度表]({link}accuracies.csv)，"
        f"完整性核验见[验证记录]({link}verification.json)。"
        f"本节由[round2_tables.py]({script_link})从已核验汇总生成。\n\n"
    )
    text += "### 8.5 额外 64 次更新的准确率变化\n\n"
    text += "直接对同一题集的 step064 与 step128 做配对比较。增益单位为百分点。\n\n"
    rows = []
    for row in data["continuation_accuracy"]:
        if row["dataset"] != "macro5":
            continue
        rows.append(
            [
                TEACHERS[row["teacher"]],
                f"{row['acc64_percent']:.3f}",
                f"{row['acc128_percent']:.3f}",
                f"{row['gain_pp']:+.3f}",
                f"[{row['gain_ci_low_pp']:.3f}, {row['gain_ci_high_pp']:.3f}]",
            ]
        )
    text += table(["Teacher", "step064", "step128", "增益 pp", "95% 区间"], rows)
    text += f"五个单任务及 macro5 的完整结果见[续训增量表]({link}continuation-accuracy.csv)。\n\n"
    (OUT / "analysis/task-tables.md").write_text(text)
    content = page.read_text()
    start = content.find("## 8. 完整任务评测")
    if start < 0:
        start = content.index("### 参考文献")
    end = content.find("## 9.", start)
    if end < 0:
        end = content.index("### 参考文献", start)
    page.write_text(content[:start] + text + content[end:])


if __name__ == "__main__":
    main()
