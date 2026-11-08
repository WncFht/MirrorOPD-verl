# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Export fixed-bank absorption metrics and an offline MiMo trajectory explorer."""

import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
DOCS = ROOT / "examples/mirroropd/docs/experiments/teacher-transfer"
INPUTS = DOCS / "data"
OUTPUT = DOCS / "fixed-state-results/figures/mimo-absorption-20261107"
RUNS = ("E71", "E72", "E73")
TEACHERS = {"E71": "MiMo-7B-RL", "E72": "MiMo-7B-RL-Zero", "E73": "MiMo-7B-SFT"}
DOMAINS = ("math", "code", "science", "if", "all")
BANKS = ("public_initial", "public_teacher")


def load_training():
    path = INPUTS / "mimo-position-kl-20261106/loss-before-update.csv"
    latest = {}
    first = {}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            key = row["run"], int(row["outer"])
            first.setdefault(row["run"], float(row["loss"]))
            latest[key] = {
                "update": int(row["update"]),
                "rkl": float(row["loss"]),
            }
    return latest, first


def build_rows():
    training, train_initial = load_training()
    manifest = json.loads((INPUTS / "mimo-position-kl-20261106/manifest.json").read_text())
    rows = []
    sources = []
    for run in RUNS:
        path = INPUTS / f"mimo-stage-20261104-final/{run}/fixed-bank-rkl.csv"
        expected = manifest["sources"][path.relative_to(ROOT).as_posix()]
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen bank input changed: {run}")
        sources.append(path)
        with path.open(newline="") as stream:
            for source in csv.DictReader(stream):
                step = int(source["optimizer_updates"])
                initial_key = run, source["bank"], source["domain"]
                value = float(source["rkl"])
                rows.append(
                    dict(
                        run=run,
                        teacher=TEACHERS[run],
                        bank=source["bank"],
                        domain=source["domain"],
                        step=step,
                        outer=int(source["outer"]),
                        rkl=value,
                        questions=int(source["questions"]),
                        _initial_key=initial_key,
                    )
                )
    initial = {}
    for row in rows:
        if row["step"] == 0:
            initial[row["_initial_key"]] = row["rkl"]
    for row in rows:
        start = initial[row.pop("_initial_key")]
        if not start > 0 or not row["rkl"] > 0:
            raise ValueError("RKL must be positive for logarithmic absorption")
        row["log_absorption"] = math.log10(start / row["rkl"])
        row["relative_decrease"] = 100 * (1 - row["rkl"] / start)
        row["absolute_decrease"] = start - row["rkl"]
        row["initial_rkl"] = start
        observation = training.get((row["run"], row["outer"]))
        row["train_rollout_rkl_before_update"] = observation["rkl"] if observation else None
        row["train_rollout_update"] = observation["update"] if observation else None
        # The final forward in outer t observes theta_(step-1); bank scoring
        # observes theta_step. Keep both parameter times in the export.
        train_value = observation["rkl"] if observation else train_initial[row["run"]]
        row["train_parameter_step"] = observation["update"] - 1 if observation else 0
        row["train_initial_rkl"] = train_initial[row["run"]]
        row["train_rkl_proxy"] = train_value
        row["train_absorption_proxy"] = math.log10(train_initial[row["run"]] / train_value)
        progress = row["train_absorption_proxy"]
        row["gap_proxy"] = progress - row["log_absorption"]
        row["efficiency_proxy"] = row["log_absorption"] / progress if progress > 0.01 else None
    keys = {(r["run"], r["bank"], r["domain"], r["step"]) for r in rows}
    if len(keys) != len(rows) or len(rows) != 220:
        raise ValueError("Incomplete or duplicated MiMo snapshots")
    for run in RUNS:
        steps = [0, 32, 64, 160, 320, 480, 640, 800] if run == "E71" else [0, 16, 32, 80, 160, 240, 320]
        for bank in BANKS:
            for domain in DOMAINS:
                if {r["step"] for r in rows if (r["run"], r["bank"], r["domain"]) == (run, bank, domain)} != set(steps):
                    raise ValueError("Unexpected bank checkpoint grid")
            for step in steps:
                panel = {r["domain"]: r for r in rows if (r["run"], r["bank"], r["step"]) == (run, bank, step)}
                average = sum(panel[d]["rkl"] for d in DOMAINS[:-1]) / 4
                if not math.isclose(average, panel["all"]["rkl"], rel_tol=1e-8):
                    raise ValueError("All-domain RKL must match the four-domain mean")
    return rows, sources


def render(rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from examples.mirroropd.research.core.plotting import load_style, save_figure

    fv = load_style()
    fv.apply_rc("paper", fontsize=10)
    colors = fv.series_colors("paper", 3)
    for bank in BANKS:
        for metric, label in (
            ("rkl", "RKL (nat; log scale)"),
            ("log_absorption", "Absorption A (dex)"),
            ("phase", "Held-out absorption A (dex)"),
            ("gap_proxy", "Absorption gap (rollout proxy, dex)"),
            ("efficiency_proxy", "Efficiency (rollout proxy)"),
        ):
            fig, axes = plt.subplots(2, 3, figsize=(14, 9))
            for ax, domain in zip(axes.flat, DOMAINS, strict=False):
                for run, color in zip(RUNS, colors, strict=True):
                    chosen = sorted(
                        [r for r in rows if (r["run"], r["bank"], r["domain"]) == (run, bank, domain)],
                        key=lambda r: r["step"],
                    )
                    x = [r["train_absorption_proxy"] if metric == "phase" else r["step"] for r in chosen]
                    y = [r["log_absorption"] if metric == "phase" else r[metric] for r in chosen]
                    ax.plot(x, y, color=color, marker="o", markersize=4, label=f"{run}: {TEACHERS[run]}")
                ax.set_title("Avg (four domains)" if domain == "all" else domain.capitalize())
                ax.set_xlabel("Rollout absorption (proxy, dex)" if metric == "phase" else "Optimizer updates")
                ax.set_ylabel(label)
                if metric == "rkl":
                    ax.set_yscale("log")
                if metric == "phase":
                    low, high = min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])
                    ax.plot(
                        [low, high],
                        [low, high],
                        color=fv.NEUTRALS["baseline"],
                        linestyle=":",
                        linewidth=1,
                        label="Equal absorption",
                    )
                    ax.set_xlim(low, high)
                    ax.set_ylim(low, high)
                ax.grid(alpha=0.18)
            panels = list(axes.flat)[:5]
            low = min(ax.get_ylim()[0] for ax in panels)
            high = max(ax.get_ylim()[1] for ax in panels)
            for ax in panels:
                ax.set_ylim(low, high)
                if metric == "phase":
                    ax.set_xlim(low, high)
            legend_ax = axes.flat[-1]
            legend_ax.axis("off")
            handles, labels = axes.flat[0].get_legend_handles_labels()
            legend_ax.legend(handles, labels, loc="center", frameon=False)
            fig.suptitle(f"MiMo-7B-Base | {bank} | {metric}")
            fig.subplots_adjust(left=0.075, right=0.98, top=0.9, bottom=0.11, wspace=0.32, hspace=0.4)
            caption = (
                "Fixed-bank question-mean RKL; each domain/bank uses its own initial value. "
                "E71: 800 updates; E72/E73: 320."
            )
            if metric in ("phase", "gap_proxy", "efficiency_proxy"):
                caption = (
                    "Rollout: before the outer's final update; bank: after it. "
                    "Changing rollout states confound transfer efficiency."
                )
            save_figure(fig, OUTPUT / f"{bank}-{metric}", caption)


def main():
    from plotly.offline import get_plotlyjs

    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows, sources = build_rows()
    render(rows)
    fields = [
        "run",
        "teacher",
        "bank",
        "domain",
        "outer",
        "step",
        "questions",
        "rkl",
        "initial_rkl",
        "log_absorption",
        "relative_decrease",
        "absolute_decrease",
        "train_rollout_update",
        "train_rollout_rkl_before_update",
        "train_parameter_step",
        "train_initial_rkl",
        "train_rkl_proxy",
        "train_absorption_proxy",
        "gap_proxy",
        "efficiency_proxy",
    ]
    with (OUTPUT / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in rows)

    payload = json.dumps(
        {"rows": rows, "domains": DOMAINS, "banks": BANKS, "teachers": TEACHERS},
        ensure_ascii=False,
        separators=(",", ":"),
    ).replace("</", "<\\/")
    template = Path(__file__).with_name("mimo_explorer.html").read_text()
    script = Path(__file__).with_name("mimo_explorer.js").read_text()
    html = template.replace("__DATA__", payload).replace("__PLOTLY__", get_plotlyjs()).replace("__APP__", script)
    target = OUTPUT / "explorer.html"
    target.write_text(html)
    provenance = {
        "rows": len(rows),
        "runs": RUNS,
        "domains": DOMAINS,
        "banks": BANKS,
        "source_sha256": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "loss_source_sha256": hashlib.sha256(
            (INPUTS / "mimo-position-kl-20261106/loss-before-update.csv").read_bytes()
        ).hexdigest(),
        "metrics_sha256": hashlib.sha256((OUTPUT / "metrics.csv").read_bytes()).hexdigest(),
        "html_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "generator_sha256": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("mimo_absorption.py", "mimo_explorer.html", "mimo_explorer.js")
        },
    }
    (OUTPUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
