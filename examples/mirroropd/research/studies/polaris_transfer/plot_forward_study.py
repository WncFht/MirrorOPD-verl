#!/usr/bin/env python3
# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Ledger: first/last curves are pre-update batch observations; bars compare
objectives on the same held-out states. Fold ranges are not seed confidence
intervals. All plots read frozen data; no synthetic training curves.
"""

import argparse
import json
import statistics as st
from pathlib import Path

from examples.mirroropd.research.core.plotting import load_style


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--data", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    args = a.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fv = load_style()
    args.out.mkdir(parents=True, exist_ok=True)
    fv.apply_rc("paper", fontsize=10)
    colors = fv.series_colors("paper", 3)
    audit = {}

    def save(fig, name, note):
        fig.text(0.5, 0.025, note, ha="center", fontsize=8, color=fv.NEUTRALS["baseline"])
        fig.canvas.draw()
        audit[name] = [str(x) for x in fv.audit(fig)]
        for ext in ["png", "pdf"]:
            fig.savefig(args.out / f"{name}.{ext}", dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.3, 4.3))
    fig.subplots_adjust(left=0.14, right=0.97, bottom=0.27, top=0.96)
    rows = [json.loads(s) for s in (args.data / "run/history.jsonl").open()]
    for reuse, ls, color in [(1, "-", colors[0]), (16, "-", colors[1])]:
        rs = [r for r in rows if "distill/full_kl" in r and r["rollout/data_reuse/update_index"] == reuse]
        ax.plot(
            [r["step/outer"] for r in rs],
            [r["distill/full_kl"] for r in rs],
            ls,
            color=color,
            lw=1,
            label=f"E48 reuse {reuse}",
        )
    r20 = [json.loads(s) for s in (args.data / "E20/history.jsonl").open()]
    r20 = [r for r in r20 if "distill/full_kl" in r and r["rollout/data_reuse/update_index"] == 1]
    ax.plot([r["step/outer"] for r in r20], [r["distill/full_kl"] for r in r20], color=colors[2], label="E20 reuse 1")
    ax.set(xlabel="Outer step", ylabel="Reverse KL (nats/token)", ylim=(0, 0.23))
    fig.legend(
        *ax.get_legend_handles_labels(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.09),
        ncol=3,
        frameon=False,
        fontsize=8,
    )
    save(fig, "complete-run", "Full run: 256 batches / 4,096 updates. Curves use changing prefixes.")
    summary = json.loads((args.data / "direction-summary.json").read_text())
    for tag, targets, title in [
        ("initial", ["justrl", "polaris"], "Initial student"),
        ("warm64", ["math", "code", "if"], "Polaris local checkpoint after 64 updates"),
    ]:
        fig, axes = plt.subplots(2, 1, figsize=(6.3, 7.3))
        fig.subplots_adjust(left=0.14, right=0.97, bottom=0.16, top=0.94, hspace=0.42)
        for ax, metric in zip(axes, ["reverse", "forward"], strict=False):
            for direction, offset, color in [("reverse", -0.18, colors[0]), ("forward", 0.18, colors[1])]:
                means = []
                lo = []
                hi = []
                for t in targets:
                    rs = [
                        r
                        for r in summary
                        if r["arm"].startswith(tag + "-")
                        and r["direction"] == direction
                        and r["step"] == 16
                        and r["partition"] == "heldout"
                        and r["source"] == "student"
                        and r["region"] == "early"
                        and (r["teacher"] == t and r["domain"] == "math" if tag == "initial" else r["domain"] == t)
                    ]
                    v = [100 * r[metric + "_relative_drop"] for r in rs]
                    assert len(v) == 2
                    means.append(st.mean(v))
                    lo.append(st.mean(v) - min(v))
                    hi.append(max(v) - st.mean(v))
                bars = ax.bar(
                    np.arange(len(targets)) + offset,
                    means,
                    width=0.32,
                    color=color,
                    label=direction.capitalize() + " KL training",
                    yerr=[lo, hi],
                    capsize=3,
                    error_kw={"ecolor": color},
                )
                for cap in bars.errorbar.lines[1]:
                    cap.set_color(color)
                    cap.set_markeredgecolor(color)
            ax.set(
                xticks=range(len(targets)),
                xticklabels=["JustRL", "Polaris"] if tag == "initial" else ["Math", "Code", "IF"],
                xlabel="Target teacher" if tag == "initial" else "Held-out domain",
                ylabel=metric.capitalize() + " KL decrease (%)",
            )
        fig.legend(
            *axes[0].get_legend_handles_labels(),
            loc="lower center",
            bbox_to_anchor=(0.5, 0.065),
            ncol=2,
            frameon=False,
            fontsize=8,
        )
        save(fig, "direction-" + tag, "16 updates; bars average two training folds, ranges span the folds.")
    (args.out / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
