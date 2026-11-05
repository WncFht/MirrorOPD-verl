"""Rebuild the E57/E81/E82 training-reward clock figures from archived metrics."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = next(parent for parent in HERE.parents if (parent / ".git").exists())
RUN_DIR = REPO / "runs" / "topd-large-batch-20261105"
E57_DATA = REPO / "runs" / "e55-e57-refresh-20261102-0900" / "data.json"
E81_METRICS = RUN_DIR / "formal" / "E81" / "metrics.jsonl"
E82_METRICS = RUN_DIR / "formal" / "E82" / "metrics.jsonl"
DATA_CSV = HERE / "training-clocks.csv"
PROVENANCE = HERE / "provenance.json"
UPDATE_PER_OUTER = 16
RESPONSES_PER_OUTER = {"E57": 64, "E81": 256, "E82": 256}


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


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_row(run: str, outer: float, update: float, reward: float) -> dict[str, float | str]:
    return {
        "run": run,
        "outer": outer,
        "optimizer_update": update,
        "response_presentations": outer * RESPONSES_PER_OUTER[run] * UPDATE_PER_OUTER,
        "rollout_reward": reward,
    }


def load_e57() -> list[dict[str, float | str]]:
    payload = json.loads(E57_DATA.read_text())
    for run in payload["runs"]:
        if run["name"] == "E57":
            return [
                make_row("E57", float(row["step"]), float(row["step"]) * UPDATE_PER_OUTER, float(row["score"]))
                for row in run["outer"]
            ]
    raise ValueError("E57 data is missing from the archived refresh")


def load_metrics(run: str, path: Path) -> list[dict[str, float | str]]:
    # Keep the final reward-bearing record for each outer. Optimizer-update records
    # often repeat step/outer but do not contain rollout/score/mean.
    by_outer: dict[int, dict[str, float | str]] = {}
    for line in path.read_text().splitlines():
        record = json.loads(line)
        data = record.get("data", {})
        reward = data.get("rollout/score/mean")
        outer = data.get("step/outer")
        update = data.get("step/optimizer")
        if reward is None or outer is None or update is None:
            continue
        by_outer[int(outer)] = make_row(run, float(outer), float(update), float(reward))
    return [by_outer[outer] for outer in sorted(by_outer)]


def archive_data() -> tuple[dict[str, list[dict[str, float | str]]], dict[str, object]]:
    series = {
        "E57": load_e57(),
        "E81": load_metrics("E81", E81_METRICS),
        "E82": load_metrics("E82", E82_METRICS),
    }
    if not all(series.values()):
        raise ValueError("At least one experiment has no reward-bearing outer records")

    fields = ("run", "outer", "optimizer_update", "response_presentations", "rollout_reward")
    with DATA_CSV.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for rows in series.values():
            writer.writerows(rows)

    details = {}
    for run, rows in series.items():
        last = rows[-1]
        details[run] = {
            "point_count": len(rows),
            "last_reward_outer": last["outer"],
            "last_optimizer_update": last["optimizer_update"],
            "last_response_presentations": last["response_presentations"],
            "last_rollout_reward": last["rollout_reward"],
        }
    provenance = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sources": {
            "E57": {"path": str(E57_DATA.relative_to(REPO)), "sha256": sha256(E57_DATA)},
            "E81": {"path": str(E81_METRICS.relative_to(REPO)), "sha256": sha256(E81_METRICS)},
            "E82": {"path": str(E82_METRICS.relative_to(REPO)), "sha256": sha256(E82_METRICS)},
        },
        "series": details,
        "axes": {
            "outer": "completed outer step",
            "optimizer_update": "completed optimizer update recorded by the trainer",
            "response_presentations": (
                "outer × responses per outer × 16 optimizer updates; repeated answer presentations count again"
            ),
            "rollout_reward": "mean task reward of the training rollouts generated for that outer",
        },
        "limitations": [
            (
                "E81 outer 25 reward is not yet logged while final evaluation is pending; "
                "its latest logged reward point in this snapshot is outer 24."
            ),
            "E82 is an in-progress run; the curve ends at its latest reward-bearing outer in the source snapshot.",
            "These are training rollout rewards, not held-out benchmark accuracy.",
        ],
    }
    PROVENANCE.write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n")
    return series, provenance


def plot(series: dict[str, list[dict[str, float | str]]], x_key: str, x_label: str, stem: str) -> None:
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
    series, _ = archive_data()
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
