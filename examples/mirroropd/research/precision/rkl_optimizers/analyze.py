#!/usr/bin/env python3
# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Recompute summaries and figures from completed optimizer mechanism runs.

Semantic ledger: one series is one optimizer under one prefix-refresh rule.
The x axis counts completed optimizer updates; y is conditional reverse KL on
four fixed, independent initial-student rollouts. Every vocabulary row is used.
There are no uncertainty bands for a single seed, and no task-accuracy claim.
Geometry ledger: both BF16 panels share axes, scales and optimizer colors.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

LABELS = {"adamw": "AdamW", "adam0": "AdamW (beta1=0)", "sgd": "SGD", "muon": "Muon + AdamW"}


def read(p):
    return json.loads(p.read_text())


def summarize(root):
    rows = []
    curves = []
    calibration = {}
    for run, mode in [("frozen-v2", "frozen"), ("online-v2", "online"), ("fp32", "fp32")]:
        base = root / run
        if not base.exists():
            continue
        for cal in sorted(base.glob("calibration-*.json")):
            calibration[f"{run}/{cal.name}"] = read(cal)
        for p in sorted(base.glob("*-seed*/complete.json")):
            done = read(p)
            history = read(p.parent / "history.json")
            probe = read(p.parent / "probe.json")
            assert len(history) == done["steps"] and history[-1]["step"] == done["steps"]
            assert probe[-1]["step"] == done["steps"]
            first, last = probe[0], probe[-1]
            cos = [h["momentum_cosine"] for h in history if h["momentum_cosine"] is not None]
            rows.append(
                {
                    "condition": mode,
                    "arm": done["arm"],
                    "seed": done["seed"],
                    "steps": done["steps"],
                    "lr": done["lr"],
                    "initial_probe_kl": first["teacher_kl"],
                    "final_probe_kl": last["teacher_kl"],
                    "probe_reduction_pct": 100 * (1 - last["teacher_kl"] / first["teacher_kl"]),
                    "initial_student_kl": last["initial_student_kl"],
                    "final_entropy": last["entropy"],
                    "same_batch_increases": sum(h["post_kl"] > h["pre_kl"] + 1e-8 for h in history),
                    "mean_momentum_cosine": sum(cos) / len(cos) if cos else None,
                    "negative_gradient_dot_steps": sum(h["gradient_dot_step"] < 0 for h in history),
                    "first_step_l2": history[0]["step_l2"],
                    "last_step_l2": history[-1]["step_l2"],
                    "peak_allocated_gib": max(h["peak_allocated_gib"] for h in history),
                    "elapsed_seconds": done["elapsed_seconds"],
                }
            )
            curves.extend({"condition": mode, "arm": done["arm"], "seed": done["seed"], **r} for r in probe)
    checks = {}
    for mode in ["frozen-v2", "online-v2", "fp32"]:
        p = root / mode / "initial-rollouts-42.json"
        if p.exists():
            checks[mode] = [(r["id"], r["prompt_ids"], r["response_ids"]) for r in read(p)]
    if len(checks) > 1:
        assert all(v == next(iter(checks.values())) for v in checks.values()), "initial prefix banks differ"
    precision = read(root / "precision/precision.json") if (root / "precision/precision.json").exists() else None
    return {
        "rows": rows,
        "curves": curves,
        "calibration": calibration,
        "precision": precision,
        "initial_prefix_banks_identical": True,
        "audit": read(root / "audit.json"),
        "scope": "Short conditional KL mechanism study; shared GPU; no task-quality ranking.",
    }


def plot(result, out, style):
    sys.path.insert(0, str(style))
    import fvstyle as fv
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fv.apply_rc("paper")
    colors = dict(zip(LABELS, fv.series_colors("paper", 4), strict=False))
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4), sharex=True, sharey=True)
    handles = []
    for ax, condition, title in zip(
        axes, ["frozen", "online"], ["Frozen initial rollouts", "Fresh student rollouts"], strict=False
    ):
        for arm, label in LABELS.items():
            rs = [r for r in result["curves"] if r["condition"] == condition and r["arm"] == arm]
            if not rs:
                continue
            (line,) = ax.plot(
                [r["step"] for r in rs], [r["teacher_kl"] for r in rs], marker="o", color=colors[arm], label=label
            )
            if condition == "frozen":
                handles.append(line)
        ax.set_xlabel("Completed optimizer updates")
        ax.set_title(title)
        ax.set_xticks([0, 4, 8])
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Fixed-probe reverse KL (nats/token)")
    fig.legend(
        handles, [h.get_label() for h in handles], loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.0), frameon=False
    )
    fig.subplots_adjust(top=0.77, bottom=0.24, wspace=0.15)
    gray = fv.NEUTRALS
    color = gray.get("vgray", next(iter(gray.values()))) if isinstance(gray, dict) else "gray"
    fig.text(
        0.5,
        0.035,
        "1.5B full-parameter probe; one seed; 64-token cap; no accuracy claim",
        ha="center",
        color=color,
        fontsize=8,
    )
    fig.canvas.draw()
    issues = fv.audit(fig)
    (out / "figure-audit.txt").write_text(str(issues) + "\n")
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out / f"probe-kl.{ext}", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("--data", type=Path, required=True)
    a.add_argument("--figures", type=Path)
    a.add_argument("--style", type=Path)
    args = a.parse_args()
    result = summarize(args.data)
    (args.data / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    if result["rows"]:
        with (args.data / "summary.csv").open("w") as f:
            w = csv.DictWriter(f, fieldnames=list(result["rows"][0]))
            w.writeheader()
            w.writerows(result["rows"])
    for r in result["rows"]:
        print(json.dumps(r), flush=True)
    if args.figures:
        if not args.style:
            a.error("--style is required for figures")
        required = {(mode, arm) for mode in ["frozen", "online"] for arm in LABELS}
        assert required <= {(r["condition"], r["arm"]) for r in result["rows"]}, "BF16 comparison incomplete"
        args.figures.mkdir(parents=True, exist_ok=True)
        plot(result, args.figures, args.style)
