"""Teacher-cluster error comparison of fixed leave-one-teacher-out predictions.

Bootstrap resamples teacher errors from already fixed predictions; it does not
refit training, retune ridge, or estimate variation over training seeds. Color
identifies the feature addition; both panels use the same MAE scale in dex.
"""

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "spectrum-expanded-v4"
sys.path.insert(0, str(ROOT.parents[2] / "dotfiles/.agents/skills/figure-viz/assets/mpl"))
import fvstyle as fv  # noqa: E402


def main():
    rows = list(csv.DictReader((OUT / "teacher-heldout-prediction.csv").open()))
    rng = np.random.default_rng(20261110)
    comparisons = []
    for step in ["64", "320"]:
        for scope in ["compatible", "published_direct_parent"]:
            subset = [r for r in rows if r["step"] == step and r["scope"] == scope]
            models = sorted({r["model"] for r in subset})
            teachers = sorted({r["teacher_path"] for r in subset})
            if not teachers:
                continue
            errors = {}
            for model in models:
                errors[model] = np.array(
                    [
                        np.mean(
                            [
                                abs(float(r["predicted_log10_K"]) - float(r["actual_log10_K"]))
                                for r in subset
                                if r["model"] == model and r["teacher_path"] == teacher
                            ]
                        )
                        for teacher in teachers
                    ]
                )
            indices = rng.integers(0, len(teachers), size=(10000, len(teachers)))
            for model, values in errors.items():
                diff = values - errors["initial_KL+group"]
                ci = np.quantile(diff[indices].mean(1), [0.025, 0.975])
                comparisons.append(
                    dict(
                        step=int(step),
                        scope=scope,
                        model=model,
                        n_teacher=len(teachers),
                        MAE_difference_dex=float(diff.mean()),
                        conditional_teacher_bootstrap95=ci.tolist(),
                    )
                )
    (OUT / "prediction-error-bootstrap.json").write_text(
        json.dumps(
            dict(
                comparisons=comparisons,
                protocol="10000 paired teacher-cluster resamples of fixed CV errors, not refit or seed uncertainty",
            ),
            indent=2,
        )
    )
    summary = json.loads((OUT / "prediction-summary.json").read_text())
    fv.apply_rc("paper", fontsize=10)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.2, right=0.96, bottom=0.23, top=0.85, wspace=0.22)
    models = ["initial_KL+group", "+norm_over_base", "+top1pct", "+all_geometry"]
    labels = ["Initial KL + student/mode", "+ Update magnitude", "+ Top-1% energy", "+ All seven geometry features"]
    palette = fv.series_colors("paper", 4)
    for ax, step in zip(axes, [64, 320], strict=True):
        selected = [
            next(r for r in summary if r["step"] == step and r["scope"] == "compatible" and r["model"] == model)
            for model in models
        ]
        ax.barh(range(4), [r["teacher_balanced_MAE_dex"] for r in selected], color=palette)
        ax.set_yticks(range(4), labels)
        ax.set(
            xlim=(0, 0.31),
            xlabel="Teacher-balanced MAE (dex; lower is better)",
            title=f"Step{step}: {selected[0]['n_teacher']} held-out teachers",
        )
        for i, r in enumerate(selected):
            ax.text(r["teacher_balanced_MAE_dex"] + 0.005, i, f"{r['teacher_balanced_MAE_dex']:.3f}", va="center")
    axes[0].invert_yaxis()
    fig.text(
        0.5,
        0.025,
        "Leave one teacher out, grouping input modes; fixed ridge alpha=1; exploratory comparison",
        ha="center",
        fontsize=8,
        color=fv.NEUTRALS["baseline"],
    )
    fig.canvas.draw()
    audit = [str(x) for x in fv.audit(fig, print_slot=11)]
    (OUT / "prediction-comparison.audit.json").write_text(json.dumps(audit, indent=2))
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(OUT / f"prediction-comparison.{ext}", dpi=180, bbox_inches="tight")
    print("prediction figure audit", audit)
    for r in comparisons:
        if r["model"] == "+norm_over_base":
            print(r)


if __name__ == "__main__":
    main()
