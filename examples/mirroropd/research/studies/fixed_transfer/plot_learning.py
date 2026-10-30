# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Equal-budget learning curves: fixed training pool, old batches and three held-out domains."""

import argparse
import re
import textwrap
from pathlib import Path

from examples.mirroropd.research.core.artifacts import file_hash
from examples.mirroropd.research.core.io import dump, read, write_csv
from examples.mirroropd.research.core.plotting import load_style
from examples.mirroropd.research.studies.fixed_transfer.display_policy import MAX_DISPLAY_UPDATES, in_main_comparison
from examples.mirroropd.research.studies.fixed_transfer.overview import MODEL_LABELS
from examples.mirroropd.research.studies.fixed_transfer.spec import Study

PANELS = {
    "train_all": ("Fixed training pool A-T (320)", list("ABCDEFGHIJKLMNOPQRST")),
    "old_train_ad": ("Old training batches A-D (64)", list("ABCD")),
    "held_all": ("All held-out (128)", ["M01", "M02", "M03", "M04", "Code01", "Code02", "IF01", "IF02"]),
    "held_math": ("Held-out Math (64)", ["M01", "M02", "M03", "M04"]),
    "held_code": ("Held-out Code (32)", ["Code01", "Code02"]),
    "held_if": ("Held-out IF (32)", ["IF01", "IF02"]),
}


def panel_means(rows, horizon):
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate scored prompts")
    result = {}
    for name, (_, groups) in PANELS.items():
        selected = [r for r in rows if r["group"] in groups]
        if len(selected) != 16 * len(groups):
            raise ValueError(f"Incomplete fixed panel: {name}")
        result[name] = sum(r["stats"]["all"]["rkl"] for r in selected) / (len(selected) * horizon)
    return result


def collect(study, budget):
    if budget > MAX_DISPLAY_UPDATES:
        raise ValueError("Current presentation is restricted to at most 320 updates")
    if budget <= 0 or budget % study.training.score_interval:
        raise ValueError("Budget must be a complete scoring interval")
    result, evidence = [], []
    common = None
    for arm in study.arms:
        if not in_main_comparison(arm.teacher):
            continue
        root = study.directory / "arms" / arm.name
        marker = read(root / "complete.json")
        if marker["updates"] < budget:
            raise ValueError(f"{arm.name} has not completed the requested budget")
        baseline = None
        for step in range(0, budget + 1, study.training.score_interval):
            path = root / f"fixed-{step:03d}.json"
            rows = read(path)
            if any(r["step"] != step for r in rows):
                raise ValueError("Snapshot step differs from its filename")
            identity = [(r["id"], r["source"], r["length"], r["group"]) for r in rows]
            if common is None:
                common = identity
            if identity != common:
                raise ValueError("Teachers or updates use different frozen states")
            values = panel_means(rows, study.training.horizon)
            baseline = baseline or values
            for panel, value in values.items():
                result.append(
                    dict(
                        arm=arm.name,
                        teacher=arm.teacher,
                        step=step,
                        panel=panel,
                        rkl=value,
                        initial=baseline[panel],
                        drop_percent=100 * (1 - value / baseline[panel]) if baseline[panel] > 0 else None,
                    )
                )
            evidence.append(dict(arm=arm.name, step=step, path=str(path), sha256=file_hash(path)))
    return result, evidence


def plot(study, rows, output, budget):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fv = load_style()
    fv.apply_rc("paper", fontsize=10)
    colors = [fv.HUES[k] for k in ["vteal", "vorange", "vviolet"]]
    student_size = float(re.search(r"(\d+(?:\.\d+)?)B", study.models[study.student].repo, re.IGNORECASE).group(1))
    markers = ["o", "s", "^", "D", "v", "P", "X"]
    for metric, axis_label in [("rkl", "Absolute RKL"), ("drop_percent", "RKL reduction from initial (%)")]:
        fig, axes = plt.subplots(2, 3, figsize=(14, 9))
        fig.subplots_adjust(left=0.075, right=0.98, top=0.88, bottom=0.27, hspace=0.43, wspace=0.32)
        handles, labels = [], []
        for ax, (panel, (title, _)) in zip(axes.flat, PANELS.items(), strict=True):
            for index, arm in enumerate(a for a in study.arms if in_main_comparison(a.teacher)):
                chosen = [r for r in rows if r["arm"] == arm.name and r["panel"] == panel]
                label_key = (
                    "coreward-i-llama"
                    if study.models[study.student].repo.lower().find("llama") >= 0 and arm.teacher == "coreward-i"
                    else arm.teacher
                )
                label = MODEL_LABELS[label_key][0]
                if arm.teacher == "L3":
                    label += " / final RLVR"
                if arm.teacher == "L4":
                    label += " / input-mismatch control"
                (line,) = ax.plot(
                    [r["step"] for r in chosen],
                    [r[metric] for r in chosen],
                    color=colors[index % len(colors)],
                    linestyle="-" if MODEL_LABELS[label_key][1] == student_size else "--",
                    marker=markers[index % len(markers)],
                    markevery=4,
                    markersize=3,
                    linewidth=1.7,
                )
                if panel == "train_all":
                    handles.append(line)
                    labels.append("\n".join(textwrap.wrap(label, 52)))
            ax.set(title=title, xlabel="Optimizer updates", ylabel=axis_label, xlim=(0, budget))
            ax.axvline(64, color=fv.NEUTRALS["baseline"], linewidth=0.5, alpha=0.4)
            ax.spines[["top", "right"]].set_visible(False)
        mode = "thinking" if study.evaluation.thinking else "non-thinking / shared off input"
        fig.suptitle(f"Student: {study.models[study.student].repo} | {mode}\nEqual budget: 0-{budget} updates", y=0.97)
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.095), ncol=2, fontsize=9)
        fig.text(
            0.5,
            0.025,
            "Fixed S0; H1024; one training seed. "
            "Full training pool includes not-yet-visited batches early in training.\n"
            "Solid: same size; dashed: different size. Old A-D is a subset of the training pool. "
            "Held-out pooling weights Math/Code/IF by 64/32/32 prompts.",
            ha="center",
            fontsize=9,
            color=fv.NEUTRALS["baseline"],
        )
        fig.canvas.draw()
        problems = fv.audit(fig, print_slot=fig.get_figwidth())
        dump(output / f"{metric}-audit.json", [str(p) for p in problems])
        if any(fv.problem_severity(p) == "error" for p in problems):
            plt.close(fig)
            raise RuntimeError(f"Learning-curve figure audit failed: {problems}")
        for extension in ["png", "pdf"]:
            fig.savefig(output / f"equal-budget-{metric}.{extension}", dpi=180)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=320)
    args = parser.parse_args()
    study = Study.load(args.plan)
    rows, evidence = collect(study, args.budget)
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "learning-curves.csv", rows)
    dump(args.output / "sources.json", dict(study=study.identity, budget=args.budget, snapshots=evidence))
    plot(study, rows, args.output, args.budget)


if __name__ == "__main__":
    main()
