"""Rebuild the E57/E81/E82 training-reward clock figures from archived metrics."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = next(parent for parent in HERE.parents if (parent / ".git").exists())
DATA_CSV = HERE / "training-clocks.csv"


def import_figure_style():
    for parent in REPO.parents:
        for scripts in (
            parent / ".agents/skills/figure-viz/scripts",
            parent / "dotfiles/.agents/skills/figure-viz/scripts",
        ):
            if not scripts.is_dir() or not (scripts / "fvlocate.py").is_file():
                continue
            sys.path.insert(0, str(scripts))
            import fvlocate

            fvlocate.ensure()
            import fvstyle as fv

            return fv
    raise RuntimeError("Could not locate the figure-viz matplotlib style assets")


def load_series() -> dict[str, list[dict[str, str]]]:
    series = {run: [] for run in ("E57", "E81", "E82")}
    with DATA_CSV.open(newline="") as stream:
        for row in csv.DictReader(stream):
            series[row["run"]].append(row)
    if not all(series.values()):
        raise ValueError("The published CSV must contain data for E57, E81 and E82")
    return series


def plot(series: dict[str, list[dict[str, str]]], x_key: str, x_label: str, stem: str) -> None:
    import matplotlib.pyplot as plt

    fv = import_figure_style()
    theme = fv.apply_rc("paper")
    colors = fv.series_colors("paper", 3)
    markers = fv.series_markers("paper", 3)
    labels = {
        "E57": "E57 · 64 responses/outer",
        "E81": "E81 (A) · 256 responses/outer",
        "E82": "E82 (B) · 256 responses/outer",
    }
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for index, (run, rows) in enumerate(series.items()):
        ax.plot(
            [float(row[x_key]) for row in rows],
            [float(row["rollout_reward"]) for row in rows],
            color=colors[index],
            marker=markers[index],
            markevery=max(1, len(rows) // 16),
            markersize=4.0,
            linewidth=1.9,
            label=labels[run],
        )
    ax.set_xlabel(x_label)
    ax.set_ylabel("Training rollout reward")
    ax.set_ylim(0, 1)
    ax.set_xlim(left=0)
    fv.apply_frame(ax, theme.frame)
    fv.apply_grid(ax, theme.grid)
    ax.legend(frameon=False, loc="best")
    fig.text(
        0.5,
        0.012,
        "E82 ends at the latest recorded outer.",
        ha="center",
        color=fv.NEUTRALS["baseline"],
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.045, 1, 1))

    errors = []
    for problem in fv.audit(fig):
        rule, _, message = problem.partition(" ")
        severity = "ERR" if fv.RENDER_RULES.get(rule, "error") == "error" else "warn"
        print(f"{severity} {rule}: {message}")
        if severity == "ERR":
            errors.append(problem)
    if errors:
        plt.close(fig)
        raise RuntimeError(f"Figure audit found {len(errors)} error(s)")

    fig.savefig(
        HERE / f"{stem}.png",
        dpi=180,
        metadata={"Description": "Training rollout reward by training clock; see provenance.json"},
    )
    fig.savefig(
        HERE / f"{stem}.pdf", metadata={"Subject": "Training rollout reward by training clock; see provenance.json"}
    )
    plt.close(fig)


def main() -> None:
    series = load_series()
    plot(series, "outer", "Outer step", "reward-by-outer")
    plot(series, "optimizer_update", "Optimizer update", "reward-by-optimizer-update")
    plot(
        series,
        "response_presentations",
        "Cumulative response presentations × optimizer updates",
        "reward-by-sample-update",
    )


if __name__ == "__main__":
    main()
