# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Plot measured acquisition, behavioral drift and functional retention.

Colors encode state source; line styles encode loss. The synthetic experiment
uses measured full-model outputs on an explicitly constructed one-action task.
No curve in this module uses simulated or illustrative numerical values.
"""

import argparse
from pathlib import Path

from examples.mirroropd.research.core.io import dump, read
from examples.mirroropd.research.core.plotting import load_style


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    a.out.mkdir(parents=True, exist_ok=True)
    fv = load_style()
    fv.apply_rc("paper", fontsize=11)
    gray = fv.NEUTRALS["baseline"]
    colors = dict(student=fv.HUES["vteal"], teacher=fv.HUES["vorange"], online=fv.HUES["vblue"])
    styles = dict(reverse="-", forward="--", sft=":")
    names = dict(student="Frozen student", teacher="Teacher", online="Refreshed student")
    losses = dict(reverse="RKL", forward="FKL", sft="SFT")
    teacher_names = dict(justrl="JustRL", polaris="Polaris")
    risks = read(a.data / "risk-summary.json")
    arms = read(a.data / "protocol.json")["arms"]
    audits = {}

    def save(fig, name, caption):
        fig.text(0.5, 0.015, caption, ha="center", fontsize=9, color=gray)
        fig.canvas.draw()
        # Figures are supplied at their native full width, not a 5.5-inch print slot.
        audits[name] = [str(x) for x in fv.audit(fig, ignore=["print-fit"])]
        for ext in ["png", "pdf", "svg"]:
            fig.savefig(a.out / f"{name}.{ext}", dpi=180)
        plt.close(fig)

    def curve(arm, panel):
        return sorted([r for r in risks if r["name"] == arm and r["panel"] == panel], key=lambda r: r["step"])

    fig, axes = plt.subplots(2, 3, figsize=(13, 8), sharey=True)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.22, top=0.93, wspace=0.22, hspace=0.36)
    for i, teacher in enumerate(["justrl", "polaris"]):
        for j, source in enumerate(["student", "teacher", "online"]):
            ax = axes[i, j]
            for arm in arms:
                if arm["teacher"] != teacher or arm["source"] != source:
                    continue
                rows = curve(arm["name"], "student-test-math")
                if not rows:
                    continue
                ax.plot(
                    [r["step"] for r in rows],
                    [100 * r["js_drop"] for r in rows],
                    color=colors[source],
                    linestyle=styles[arm["direction"]],
                    marker="o",
                    markersize=3,
                )
            ax.axhline(0, color=gray, lw=0.8)
            ax.set(
                title=f"{teacher_names[teacher]} / {names[source]}",
                xlabel="Optimizer updates",
                ylabel="Held math JS reduction (%)",
                xticks=[0, 32, 64, 96, 128],
            )
    fig.legend(
        [Line2D([], [], color=c) for c in colors.values()] + [Line2D([], [], color=gray, ls=styles[d]) for d in styles],
        list(names.values()) + list(losses.values()),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.055),
        ncol=3,
        frameon=False,
    )
    save(
        fig,
        "loss-source-factorial",
        "Same fixed student-prefix test bank: 128 unseen math prompts. Higher is better; negative means worse.",
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), sharey=True)
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.26, top=0.91, wspace=0.25)
    for ax, teacher in zip(axes, ["justrl", "polaris"], strict=True):
        for arm in arms:
            if arm["teacher"] != teacher:
                continue
            rows = curve(arm["name"], "student-test-math")
            if not rows:
                continue
            ax.plot(
                [r["retain_fkl"] for r in rows],
                [100 * r["js_drop"] for r in rows],
                color=colors[arm["source"]],
                ls=styles[arm["direction"]],
                marker="o",
                markersize=3,
            )
        ax.axhline(0, color=gray, lw=0.8)
        ax.set(
            title=teacher_names[teacher],
            xlabel="Initial-policy KL(p0 || p) on held math",
            ylabel="Held math JS reduction (%)",
        )
    handles = [Line2D([], [], color=c, label=names[s]) for s, c in colors.items()]
    handles += [Line2D([], [], color=gray, ls=styles[d], label=losses[d]) for d in styles]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.055), ncol=3, frameon=False)
    save(
        fig,
        "acquisition-versus-drift",
        "Points: updates 0, 32, 64, 96, 128. Behavioral drift measures distribution change, not task forgetting.",
    )

    matched_views = read(a.data / "matched-prefix-views.json")
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), sharey=True)
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.23, top=0.91, wspace=0.25)
    for ax, teacher in zip(axes, ["justrl", "polaris"], strict=True):
        for view in ["student", "teacher"]:
            for direction in ["reverse", "forward", "sft"]:
                rows = sorted(
                    [r for r in matched_views if r["name"] == f"{teacher}-teacher-{direction}" and r["view"] == view],
                    key=lambda r: r["step"],
                )
                if rows:
                    ax.plot(
                        [r["step"] for r in rows],
                        [100 * r["js_drop"] for r in rows],
                        color=colors[view],
                        ls=styles[direction],
                        marker="o",
                        markersize=3,
                    )
        ax.axhline(0, color=gray, lw=0.8)
        ax.set(
            title=teacher_names[teacher],
            xlabel="Optimizer updates",
            ylabel="Held math JS reduction (%)",
            xticks=[0, 32, 64, 96, 128],
        )
    handles = [Line2D([], [], color=colors[v], label=f"{v.title()} test prefixes") for v in ["student", "teacher"]]
    handles += [Line2D([], [], color=gray, ls=styles[d], label=losses[d]) for d in styles]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.055), ncol=3, frameon=False)
    save(
        fig,
        "prefix-view-transfer",
        "All methods train on teacher prefixes; both test views use the same 64 unseen prompts and fixed trajectories.",
    )

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), sharey="col")
    fig.subplots_adjust(left=0.09, right=0.98, top=0.93, bottom=0.17, wspace=0.28, hspace=0.4)
    for i, teacher in enumerate(["justrl", "polaris"]):
        for j, (metric, baseline_metric, ylabel, scale) in enumerate(
            [
                ("hp", "hq", "Student entropy (nats / active token)", 1),
                ("teacher_only_initial_p", "teacher_only_initial_q", "Probability mass on teacher-only set (%)", 100),
            ]
        ):
            ax = axes[i, j]
            for direction in ["reverse", "forward"]:
                rows = curve(f"{teacher}-student-{direction}", "student-test-math")
                if rows:
                    ax.plot(
                        [r["step"] for r in rows],
                        [scale * r[metric] for r in rows],
                        color=colors["student"],
                        ls=styles[direction],
                        marker="o",
                        markersize=3,
                    )
            initial = curve(f"{teacher}-student-reverse", "student-test-math")[0]
            ax.axhline(scale * initial[baseline_metric], color=gray, ls="-.", lw=1)
            ax.axhline(scale * initial[metric], color=gray, ls=":", lw=1)
            ax.set(title=teacher_names[teacher], xlabel="Optimizer updates", ylabel=ylabel, xticks=[0, 32, 64, 96, 128])
    handles = [Line2D([], [], color=colors["student"], ls=styles[d], label=losses[d]) for d in ["reverse", "forward"]]
    handles += [
        Line2D([], [], color=gray, ls="-.", label="Teacher"),
        Line2D([], [], color=gray, ls=":", label="Initial student"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.055), ncol=4, frameon=False)
    save(
        fig,
        "probability-concentration",
        "Frozen student-prefix training and test. "
        "Teacher-only set = teacher top-16 minus initial student top-16, fixed per state.",
    )

    synthetic = read(a.data / "synthetic-summary.json") if (a.data / "synthetic-summary.json").exists() else []
    if synthetic:
        last_step = max(r["step"] for r in synthetic)
        step_ticks = [s for s in [0, 1, 4, 16, 64, 256, 512] if s <= last_step]
        fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharey=True)
        fig.subplots_adjust(left=0.09, right=0.98, top=0.93, bottom=0.16, wspace=0.24, hspace=0.36)
        for i, kind in enumerate(["shared", "random"]):
            for j, direction in enumerate(["reverse", "forward"]):
                ax = axes[i, j]
                for split, col in [("train", fv.HUES["vteal"]), ("test", fv.HUES["vorange"])]:
                    rows = sorted(
                        [
                            r
                            for r in synthetic
                            if r["kind"] == kind and r["direction"] == direction and r["split"] == split
                        ],
                        key=lambda r: r["step"],
                    )
                    ax.plot(
                        [r["step"] for r in rows],
                        [100 * r["top1_correct"] for r in rows],
                        color=col,
                        marker="o",
                        label=split.title(),
                        linestyle="-" if split == "train" else "--",
                    )
                ax.axhline(25, color=gray, ls=":", lw=1)
                ax.set(
                    xscale="symlog",
                    xlim=(-0.15, last_step * 1.15),
                    ylim=(-3, 103),
                    title=f"{kind.title()} rule / {losses[direction]}",
                    xlabel="Optimizer updates",
                    ylabel="Full-vocabulary top-1 accuracy (%)",
                    xticks=step_ticks,
                )
                ax.set_xticklabels([str(s) for s in step_ticks])
        fig.legend(
            *axes[0, 0].get_legend_handles_labels(),
            loc="lower center",
            bbox_to_anchor=(0.5, 0.05),
            ncol=2,
            frameon=False,
        )
        save(
            fig,
            "synthetic-generalization",
            "Constructed one-action task; measured full-model training. "
            "16 train / 64 test. Dotted line: four-label chance.",
        )

    tasks = read(a.data / "task-summary.json")
    ifrows = read(a.data / "if-summary.json") if (a.data / "if-summary.json").exists() else []
    for domain, rows in [("math", tasks), ("if", ifrows)]:
        if not rows:
            continue
        if domain == "if" and len(rows) < 17:
            continue
        base = (
            sum(r["grade"]["acc"] for r in read(a.data / "task-student-baseline.json"))
            if domain == "math"
            else next(r["correct"] for r in rows if r["name"] == "baseline-student")
        )
        index = {r["name"]: r for r in rows}
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.8), sharey=True)
        fig.subplots_adjust(left=0.07, right=0.98, top=0.92, bottom=0.32, wspace=0.15)
        keys = [
            ("student", "reverse"),
            ("student", "forward"),
            ("teacher", "reverse"),
            ("teacher", "forward"),
            ("teacher", "sft"),
            ("online", "reverse"),
            ("online", "forward"),
        ]
        for ax, teacher in zip(axes, ["justrl", "polaris"], strict=True):
            for x, (source, direction) in enumerate(keys):
                row = index.get(f"{teacher}-{source}-{direction}")
                if row:
                    ax.bar(x, row["correct"], color=colors[source], width=0.65)
                    ax.text(x, row["correct"] + 1, str(row["correct"]), ha="center", fontsize=10)
            ax.axhline(base, color=gray, ls="--", lw=1)
            ax.set(
                title=teacher_names[teacher],
                ylabel="Correct prompts (out of 64)",
                ylim=(0, 66),
                xticks=range(7),
                xticklabels=[f"{dict(student='S0', teacher='Q', online='OnP')[s]}\n{losses[d]}" for s, d in keys],
                xlabel="State source: S0 = frozen student, Q = teacher, OnP = refreshed",
            )
        save(
            fig,
            f"{domain}-functional-retention",
            f"Dashed line: initial student ({base}/64). Paired prompts; one sampled answer each. "
            + ("Single-turn IF diagnostic." if domain == "if" else "MATH-500 subset."),
        )
    retention = read(a.data / "old-batch-retention.json") if (a.data / "old-batch-retention.json").exists() else []
    if retention:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
        fig.subplots_adjust(left=0.10, right=0.98, top=0.91, bottom=0.27, wspace=0.22)
        for ax, teacher in zip(axes, ["justrl", "polaris"], strict=True):
            for direction in ["reverse", "forward", "sft"]:
                rows = sorted(
                    [
                        r
                        for r in retention
                        if r["teacher"] == teacher and r["source"] == "teacher" and r["direction"] == direction
                    ],
                    key=lambda r: r["batch"],
                )
                ax.plot(
                    [r["batch"] for r in rows],
                    [r["ratio_after_leaving_batch"] for r in rows],
                    color=colors["teacher"],
                    ls=styles[direction],
                    marker="o",
                    label=losses[direction],
                )
            ax.axhline(1, color=gray, lw=1)
            ax.set(
                title=teacher_names[teacher],
                xlabel="Previously trained batch",
                ylabel="Final JS / JS just after its training",
                xticks=range(1, 8),
                xticklabels=list("ABCDEFG"),
            )
        fig.legend(
            *axes[0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, 0.055), ncol=3, frameon=False
        )
        save(
            fig,
            "old-batch-retention",
            "Teacher-prefix training, same saved states. "
            "Ratio <1: later new batches improved old states. No batch revisits.",
        )
    calibrated = read(a.data / "calibration-summary.json") if (a.data / "calibration-summary.json").exists() else []
    if calibrated:
        index = {r["arm"]: r for r in calibrated}
        order = [
            "initial",
            "polaris-student-reverse",
            "polaris-student-forward",
            "polaris-teacher-reverse",
            "polaris-teacher-forward",
        ]
        fig, ax = plt.subplots(figsize=(9, 5.5))
        fig.subplots_adjust(left=0.12, right=0.98, top=0.9, bottom=0.24)
        for i, key in enumerate(order):
            if key not in index:
                continue
            r = index[key]
            ax.bar(
                i - 0.18,
                100 * r["uncalibrated_drop"],
                width=0.32,
                color=gray,
                label="Original temperature 1" if i == 0 else None,
            )
            ax.bar(
                i + 0.18,
                100 * r["calibrated_drop"],
                width=0.32,
                color=colors["student"],
                label="Diagnostic-selected temperature" if i == 0 else None,
            )
            height = 100 * r["calibrated_drop"]
            ax.text(
                i + 0.18,
                height + (1 if height >= 0 else -1.5),
                f"T={r['temperature']:g}",
                ha="center",
                va="bottom" if height >= 0 else "top",
                fontsize=10,
            )
        ax.axhline(0, color=gray, lw=0.8)
        ax.set(
            xticks=range(5),
            xticklabels=["Initial", "S0 / RKL", "S0 / FKL", "Q / RKL", "Q / FKL"],
            ylabel="Held math JS reduction from original student (%)",
            xlabel="Polaris target; trained endpoints at update 128",
            ylim=(
                min(100 * r["uncalibrated_drop"] for r in calibrated) - 4,
                max(100 * r["calibrated_drop"] for r in calibrated) + 5,
            ),
        )
        fig.legend(
            *ax.get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, 0.06), ncol=2, frameon=False
        )
        save(
            fig,
            "posthoc-temperature",
            "Temperature selected on 64 diagnostic prompts; measured on 128 held-out prompts. No model weight updates.",
        )
    continuations = (
        read(a.data / "continuation-summary.json") if (a.data / "continuation-summary.json").exists() else []
    )
    if len(continuations) == 2:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        fig.subplots_adjust(left=0.13, right=0.95, top=0.9, bottom=0.3)
        index = {r["arm"]: r for r in continuations}
        for i, direction in enumerate(["reverse", "forward"]):
            r = index["polaris-student-" + direction]
            ax.barh(
                i, r["recovered_correct"], height=0.5, color=colors["student"], label="Recovered" if i == 0 else None
            )
            ax.barh(
                i,
                r["selected_cases"] - r["recovered_correct"],
                left=r["recovered_correct"],
                height=0.5,
                color=gray,
                label="Unresolved" if i == 0 else None,
            )
            ax.text(
                r["selected_cases"] + 0.15,
                i,
                f"{r['recovered_correct']}/{r['selected_cases']} recovered",
                va="center",
                fontsize=11,
            )
        ax.set(
            yticks=[0, 1],
            yticklabels=["S0 / RKL", "S0 / FKL"],
            xlabel="Selected capped responses (count)",
            ylabel="Polaris training objective",
            xlim=(0, 9.5),
            xticks=[0, 2, 4, 6, 8],
        )
        ax.invert_yaxis()
        fig.legend(
            *ax.get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, 0.08), ncol=2, frameon=False
        )
        save(
            fig,
            "capped-answer-recovery",
            "Saved 8192-token prefixes + up to 8192 extra tokens. "
            "Selected-case diagnostic, not full benchmark accuracy.",
        )
    dump(a.out / "figure-audit.json", audits)
    print("Figures", list(audits))


if __name__ == "__main__":
    main()
