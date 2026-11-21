"""Verified pair identities and a shared visual encoding for analysis figures.

Color = student family + actual Qwen thinking protocol; marker = teacher stage.
A large unfilled ring independently marks cross-size. Stages describe teacher
provenance, not the OPD optimizer; unknown lineage is retained explicitly.
Sources: checkpoint-dynamics README (protocol/roster), training-protocol-19 JSON,
and settings/模型与师生配对.md (Light SFT / Polaris RL provenance).
"""

from matplotlib.lines import Line2D

FAMILIES = {
    "R1-Distill-Qwen": "vteal",
    "Qwen3 thinking": "vorange",
    "Qwen3 non-thinking": "vviolet",
    "Qwen3 Base": "vgold",
    "Llama-3.2 Instruct": "vblue",
    "MiMo": "vcoral",
}
STAGES = {"RL": "o", "SFT": "^", "SFT?": "v", "Instruct/mixed": "s"}
# ID, student, teacher, student family/protocol, teacher provenance, cross-size.
RECORDS = [
    ("E94", "R1-1.5B", "DeepScaleR", "R1-Distill-Qwen", "RL", False),
    ("E95", "R1-1.5B", "JustRL", "R1-Distill-Qwen", "RL", False),
    ("E96", "R1-1.5B", "MiniMath", "R1-Distill-Qwen", "SFT", False),
    ("E97", "R1-1.5B", "Nemotron", "R1-Distill-Qwen", "RL", False),
    ("E98", "R1-1.5B", "BroRL", "R1-Distill-Qwen", "RL", False),
    ("E99", "Qwen3-1.7B", "GRPO-KL-math", "Qwen3 thinking", "RL", False),
    ("E100", "Qwen3-1.7B", "e3-1.7B", "Qwen3 thinking", "RL", False),
    ("E101", "Qwen3-1.7B", "Samsung Math", "Qwen3 thinking", "SFT", False),
    ("E102", "Qwen3-1.7B", "Qwen3-4B-Instruct-2507", "Qwen3 non-thinking", "Instruct/mixed", True),
    ("E103", "Llama3.2-3B-Instruct", "Co-rewarding", "Llama-3.2 Instruct", "RL", False),
    ("E104", "Llama3.2-3B-Instruct", "arithmetic-full", "Llama-3.2 Instruct", "SFT?", False),
    ("E105", "MiMo-7B-Base", "MiMo-7B-SFT", "MiMo", "SFT", False),
    ("E106", "MiMo-7B-Base", "MiMo-7B-RL-Zero", "MiMo", "RL", False),
    ("E107", "MiMo-7B-SFT", "MiMo-7B-RL", "MiMo", "RL", False),
    ("E108", "R1-1.5B", "R1-7B (distill SFT)", "R1-Distill-Qwen", "SFT", True),
    ("E109", "R1-1.5B", "Polaris-7B", "R1-Distill-Qwen", "RL", True),
    ("E110", "R1-7B", "Polaris-7B", "R1-Distill-Qwen", "RL", False),
    ("E111", "R1-7B", "Light-R1-7B-DS", "R1-Distill-Qwen", "SFT", False),
    ("E112", "Qwen3-0.6B", "Qwen3-4B-Instruct-2507", "Qwen3 non-thinking", "Instruct/mixed", True),
    ("E113", "Qwen3-4B", "Qwen3-4B-Instruct-2507", "Qwen3 non-thinking", "Instruct/mixed", False),
    ("E114", "R1-1.5B", "Light-R1-7B-DS", "R1-Distill-Qwen", "SFT", True),
    ("E117", "Qwen3-1.7B-Base", "Thinking-Space SFT", "Qwen3 Base", "SFT", False),
]
IDENTITIES = {
    n: dict(student=s, teacher=t, family=f, teacher_stage=stage, cross_size=c) for n, s, t, f, stage, c in RECORDS
}
GROUPS = {
    "all": "All",
    "RL": "RL",
    "SFT": "SFT",
    "SFT?": "SFT?",
    "Instruct/mixed": "Instruct/mixed",
    "compatible": "Same-size",
    "cross_size": "Cross-size",
    **{f: f for f in FAMILIES},
}


def member(name, group):
    p = IDENTITIES[name]
    return (
        group == "all"
        or (group == "compatible" and not p["cross_size"])
        or (group == "cross_size" and p["cross_size"])
        or group == p["family"]
        or group == p["teacher_stage"]
        or (group == "without_E113" and name != "E113")
    )


def scatter(ax, points, fv, size=48, **kwargs):
    """Plot (pair ID, x, y) rows with consistent, orthogonal identity channels."""
    for family, hue in FAMILIES.items():
        for stage, marker in STAGES.items():
            rows = [
                r for r in points if IDENTITIES[r[0]]["family"] == family and IDENTITIES[r[0]]["teacher_stage"] == stage
            ]
            if rows:
                ax.scatter(
                    [r[1] for r in rows], [r[2] for r in rows], color=fv.HUES[hue], marker=marker, s=size, **kwargs
                )
    cross = [r for r in points if IDENTITIES[r[0]]["cross_size"]]
    if cross:
        ax.scatter(
            [r[1] for r in cross],
            [r[2] for r in cross],
            facecolors="none",
            edgecolors=fv.NEUTRALS["vink"],
            marker="o",
            s=size * 2.4,
            linewidths=0.9,
            zorder=4,
        )


def handles(names, fv):
    ids = [IDENTITIES[n] for n in names]
    result = [
        Line2D([], [], marker="o", color=fv.HUES[hue], linestyle="", label=f)
        for f, hue in FAMILIES.items()
        if any(p["family"] == f for p in ids)
    ]
    result += [
        Line2D([], [], marker=m, color=fv.NEUTRALS["vink"], linestyle="", label="Teacher: " + s)
        for s, m in STAGES.items()
        if any(p["teacher_stage"] == s for p in ids)
    ]
    if any(p["cross_size"] for p in ids):
        result.append(
            Line2D(
                [],
                [],
                marker="o",
                markerfacecolor="none",
                color=fv.NEUTRALS["vink"],
                markersize=10,
                linestyle="",
                label="Outer ring: cross-size",
            )
        )
    return result


def legend(fig, names, fv, extra=(), top=False):
    fig.legend(
        handles=[*handles(names, fv), *extra],
        loc="upper center" if top else "lower center",
        bbox_to_anchor=(0.5, 0.995 if top else 0.045),
        ncol=3,
        fontsize=9,
        columnspacing=1.2,
        handletextpad=0.5,
    )


def curve_label(name):
    p = IDENTITIES[name]
    return f"{name} {p['student']} | {p['teacher_stage']}" + (" | cross-size" if p["cross_size"] else "")


def markdown_table(names):
    lines = [
        "| Pair | Student | Teacher | Family／输入模式 | Teacher 来源 | 尺寸 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for n in names:
        p = IDENTITIES[n]
        lines.append(
            f"| {n} | {p['student']} | {p['teacher']} | {p['family']} | "
            f"{p['teacher_stage']} | {'cross-size' if p['cross_size'] else 'same-size'} |"
        )
    return "\n".join(lines)


def place_labels(fig, ax, points):
    """Keep pair labels unambiguous when nearby points share color and shape."""
    from examples.mirroropd.research.studies.absorption_dynamics.cohort_comparison import place_labels as original

    before = set(ax.texts)
    original(fig, ax, points)
    for text in [t for t in ax.texts if t not in before]:
        label, xy, position = text.get_text(), text.xy, text.get_position()
        ha, va = text.get_ha(), text.get_va()
        text.remove()
        ax.annotate(
            label,
            xy,
            xytext=position,
            textcoords="offset points",
            ha=ha,
            va=va,
            fontsize=8,
            arrowprops=dict(arrowstyle="-", color="#85898F", lw=0.45, shrinkA=2, shrinkB=5),
        )
