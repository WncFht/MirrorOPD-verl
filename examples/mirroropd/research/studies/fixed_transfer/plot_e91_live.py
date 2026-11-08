# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Render a read-only progress snapshot for the live E91 training run."""

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
RUN = ROOT / "runs/e91-mimo-rlzero-sft-20261107"
METRICS = RUN / "formal/E91/metrics.jsonl"
EVENTS = RUN / "formal/E91/evaluation/events"
OUTPUT = ROOT / "examples/mirroropd/docs/experiments/teacher-transfer/figures/e91-live-20261108"


def read_training():
    rows = []
    for line in METRICS.read_text().splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        data = item["data"]
        if "optim/actor/loss" not in data:
            continue
        rows.append(
            {
                "step": int(item["step"]),
                "outer": int(data["step/outer"]),
                "loss": float(data["optim/actor/loss"]),
                "full_kl": float(data["distill/full_kl"]),
                "grad_norm": float(data["optim/actor/grad_norm_preclip"]),
            }
        )
    return rows


def read_events():
    rows = []
    for path in sorted(EVENTS.glob("outer-*/manifest.json")):
        item = json.loads(path.read_text())
        metadata = item.get("metadata", {})
        metrics = metadata.get("metrics", {})
        row = {
            "outer": int(metadata.get("outer", 0)),
            "step": int(metadata.get("optimizer_steps", 0)),
            "status": item.get("status", "unknown"),
            "initial_rkl": metrics.get("eval/rkl/public_initial/all/question_mean"),
            "teacher_rkl": metrics.get("eval/rkl/public_teacher/all/question_mean"),
        }
        rows.append(row)
    return rows


def render(training, events):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from examples.mirroropd.research.core.plotting import load_style, save_figure

    style = load_style()
    style.apply_rc("paper", fontsize=10)
    colors = style.series_colors("paper", 3)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.22, wspace=0.32)

    ax = axes[0]
    ax.plot(
        [r["step"] for r in training],
        [r["loss"] for r in training],
        color=colors[0],
        linewidth=1.8,
        label="Training loss",
    )
    ax.scatter([training[-1]["step"]], [training[-1]["loss"]], color=colors[0], zorder=3)
    ax.set(xlabel="Optimizer updates", ylabel="Full-vocabulary RKL loss", xlim=(-5, 325))
    ax.axvline(320, color=style.NEUTRALS["baseline"], linestyle=":", linewidth=0.8)
    ax.grid(alpha=0.18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper right")

    ax = axes[1]
    complete = [r for r in events if r["status"] == "complete" and r["initial_rkl"] is not None]
    for key, label, color in [
        ("initial_rkl", "Initial-student bank", colors[1]),
        ("teacher_rkl", "Teacher bank", colors[2]),
    ]:
        selected = [r for r in complete if r[key] is not None]
        ax.plot(
            [r["step"] for r in selected],
            [r[key] for r in selected],
            "o-",
            label=label,
            color=color,
            linewidth=1.6,
            markersize=4,
        )
    ax.axvline(320, color=style.NEUTRALS["baseline"], linestyle=":", linewidth=0.8, label="Outer-020 evaluation active")
    ax.set(xlabel="Optimizer updates", ylabel="Evaluated question-mean RKL", xlim=(-5, 325), yscale="log")
    ax.grid(alpha=0.18, which="both")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="best")
    caption = (
        "E91 MiMo-7B-SFT student · live snapshot 2026-11-08 03:47 (UTC+8) · outer-020 evaluation remains incomplete"
    )
    save_figure(fig, OUTPUT / "e91-live-progress", caption)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    training = read_training()
    events = read_events()
    with (OUTPUT / "training.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=training[0].keys())
        writer.writeheader()
        writer.writerows(training)
    with (OUTPUT / "evaluation.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=events[0].keys())
        writer.writeheader()
        writer.writerows(events)
    render(training, events)
    provenance = {
        "snapshot": "2026-11-08T03:47:00+08:00",
        "training_rows": len(training),
        "evaluation_rows": len(events),
        "training_last_step": training[-1]["step"],
        "evaluation_last_complete_step": max(r["step"] for r in events if r["status"] == "complete"),
        "sources": {
            str(METRICS.relative_to(ROOT)): hashlib.sha256(METRICS.read_bytes()).hexdigest(),
            str((EVENTS / "outer-015/manifest.json").relative_to(ROOT)): hashlib.sha256(
                (EVENTS / "outer-015/manifest.json").read_bytes()
            ).hexdigest(),
            str((EVENTS / "outer-020/manifest.json").relative_to(ROOT)): hashlib.sha256(
                (EVENTS / "outer-020/manifest.json").read_bytes()
            ).hexdigest(),
        },
    }
    (OUTPUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
