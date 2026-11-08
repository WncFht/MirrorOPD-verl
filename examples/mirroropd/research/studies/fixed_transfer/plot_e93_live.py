# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Render an auditable live status snapshot for E93 without touching the run."""

import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
RUN = ROOT / "runs/e93-mimo-rl-sft-20261108"
QUEUE = RUN / "queue-state.json"
TRAIN_LOG = RUN / "logs/E93-train.log"
METRICS = RUN / "formal/E93/metrics.jsonl"
OUTPUT = ROOT / "examples/mirroropd/docs/experiments/teacher-transfer/figures/e93-live-20261108"


def snapshot():
    queue = json.loads(QUEUE.read_text())
    metric_rows = [json.loads(line) for line in METRICS.read_text().splitlines() if line.strip()] if METRICS.exists() else []
    updates = [r for r in metric_rows if r.get("data", {}).get("optim/actor/update_applied") == 1.0]
    try:
        gpu_lines = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total", "--format=csv,noheader"],
            text=True,
        ).strip().splitlines()
    except (OSError, subprocess.CalledProcessError):
        gpu_lines = []
    gpus = []
    for line in gpu_lines:
        index, util, used, total = [part.strip() for part in line.split(",")]
        gpus.append({"gpu": int(index), "utilization": util, "memory_used": used, "memory_total": total})
    return {
        "snapshot": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "phase": queue.get("phase"),
        "queue_status": queue.get("status"),
        "completed_stages": ",".join(queue.get("finished", [])),
        "optimizer_updates": len(updates),
        "planned_updates": 20,
        "train_log_bytes": TRAIN_LOG.stat().st_size if TRAIN_LOG.exists() else 0,
        "gpus": gpus,
    }


def render(data):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from examples.mirroropd.research.core.plotting import load_style, save_figure

    style = load_style()
    style.apply_rc("paper", fontsize=10)
    colors = style.series_colors("paper", 3)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.84, bottom=0.21, wspace=0.3)

    ax = axes[0]
    ax.barh([50], [data["optimizer_updates"]], color=colors[0], height=20, label="completed")
    ax.barh([50], [data["planned_updates"] - data["optimizer_updates"]], left=[data["optimizer_updates"]], color=style.NEUTRALS["vmist"], height=20, label="remaining")
    ax.set_xlim(0, data["planned_updates"])
    ax.set(xlabel="Updates", ylabel="Progress", ylim=(0, 100), title=f'{data["optimizer_updates"]}/{data["planned_updates"]} completed')
    ax.set_yticks([])
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", alpha=0.18)
    ax.legend(frameon=False, fontsize=8, loc="upper right")

    ax = axes[1]
    gpu_ids = [r["gpu"] for r in data["gpus"]]
    usage = [float(r["utilization"].split()[0]) for r in data["gpus"]]
    bars = ax.bar([str(g) for g in gpu_ids], usage, color=[colors[1] if g in {1, 2, 3, 4} else colors[2] for g in gpu_ids])
    ax.set(xlabel="Physical GPU", ylabel="Utilization (%)", ylim=(0, 100))
    ax.grid(axis="y", alpha=0.18)
    ax.spines[["top", "right"]].set_visible(False)
    for bar, value in zip(bars, usage):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 3, f"{value:.0f}", ha="center", fontsize=8)
    caption = f'E93 MiMo-7B-RL → MiMo-7B-SFT · {data["snapshot"]} · phase={data["phase"]}'
    save_figure(fig, OUTPUT / "e93-live-progress", caption)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    data = snapshot()
    fields = ["snapshot", "phase", "queue_status", "completed_stages", "optimizer_updates", "planned_updates", "train_log_bytes"]
    with (OUTPUT / "status.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow({key: data[key] for key in fields})
    render(data)
    provenance = {
        "snapshot": data["snapshot"],
        "sources": {
            str(QUEUE.relative_to(ROOT)): hashlib.sha256(QUEUE.read_bytes()).hexdigest(),
            str(TRAIN_LOG.relative_to(ROOT)): hashlib.sha256(TRAIN_LOG.read_bytes()).hexdigest(),
            str(METRICS.relative_to(ROOT)): hashlib.sha256(METRICS.read_bytes()).hexdigest() if METRICS.exists() else None,
        },
        "data": data,
    }
    (OUTPUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
