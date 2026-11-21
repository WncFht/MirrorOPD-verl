# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Live, update-aligned reuse curves and their exact numerical inputs.

Semantic ledger: color identifies a reuse arm in every panel; solid/dashed RKL
lines identify initial-student/teacher prefix banks. Each RKL point is the
question mean for the exact named held-out dataset, never the domain aggregate
that also includes development questions. Training loss is observed before the
logged update, so its x coordinate is applied_updates - 1. S is plotted when its
completed-batch decision becomes available. Online S is undefined, never zero.
Geometry ledger: all x axes count successful optimizer updates; no outer-index
or wall-time rescaling. Faint training lines are raw observations, bold lines
are nonoverlapping 16-update means. No held-out interpolation or extrapolation.
"""

import argparse
import hashlib
import html
import importlib.util
import json
import math
from pathlib import Path

from examples.mirroropd.research.core.io import dump, read, write_csv, write_text


def read_live_records(path):
    if not path.exists():
        return [], None
    payload = path.read_bytes()
    lines = payload.splitlines(keepends=True)
    rows = []
    for index, line in enumerate(lines):
        try:
            rows.append(json.loads(line)["data"])
        except (ValueError, KeyError):
            if index == len(lines) - 1 and not line.endswith(b"\n"):
                break
            raise ValueError(f"Malformed complete metric record: {path}:{index + 1}") from None
    return rows, hashlib.sha256(payload).hexdigest()


def extract(rows, name):
    result = dict(training=[], heldout=[], signals=[], schedule=[])
    seen = {key: set() for key in result}

    def append(kind, key, row):
        if key in seen[kind]:
            raise ValueError(f"Duplicate {kind} observation for {name}: {key}")
        seen[kind].add(key)
        result[kind].append(dict(run=name, **row))

    for values in rows:
        completed = values.get("step/optimizer")
        if completed is None:
            continue
        if "optim/actor/update_applied" in values:
            if values["optim/actor/update_applied"] != 1:
                raise ValueError("Cannot align a run with skipped updates")
            loss = values.get("optim/actor/loss")
            if loss is not None and math.isfinite(loss):
                append(
                    "training", completed, dict(update=completed - 1, loss=loss, rkl=values.get("distill/teacher_rkl"))
                )
        for bank in ("public_initial", "public_teacher"):
            for domain in ("math", "code", "if"):
                key = f"eval/rkl/{bank}/dataset/{domain}/question_mean"
                if key in values and math.isfinite(values[key]):
                    append(
                        "heldout",
                        (completed, bank, domain),
                        dict(update=completed, bank=bank, domain=domain, rkl=values[key]),
                    )
        if "reuse/gap_over_drop" in values:
            append(
                "signals",
                completed,
                dict(
                    update=completed,
                    s=values["reuse/gap_over_drop"],
                    drop=values.get("reuse/drop"),
                    gap=values.get("reuse/refresh_gap"),
                ),
            )
        if "reuse/used" in values:
            append(
                "schedule",
                completed,
                dict(
                    update=completed,
                    used=values["reuse/used"],
                    next=values["reuse/next"],
                    switched=values["reuse/switched"],
                ),
            )
    return result


def block_means(rows, width=16):
    return [
        dict(update=sum(r["update"] for r in block) / len(block), loss=sum(r["loss"] for r in block) / len(block))
        for start in range(0, len(rows), width)
        if (block := rows[start : start + width])
    ]


def render(root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plan, manifest = read(root / "plan.json"), read(root / "prepared.json")
    output = root / "figures"
    output.mkdir(exist_ok=True)
    style_path = Path(manifest["repo"]).parent / "dotfiles/.agents/skills/figure-viz/assets/mpl/fvstyle.py"
    # Use the skill's exact palette and audit; pin its source hash in the report.
    spec = importlib.util.spec_from_file_location("reuse_fvstyle", style_path)
    import sys

    fv = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = fv
    spec.loader.exec_module(fv)
    fv.apply_rc("paper")
    colors = {
        arm: fv.HUES[hue]
        for arm, hue in zip(
            ("adaptive", "fixed16", "fixed8", "fixed4", "online"),
            ("vteal", "vorange", "vblue", "vviolet", "vgray"),
            strict=True,
        )
    }
    series, inputs = {}, {}
    combined = dict(training=[], heldout=[], signals=[], schedule=[])
    for run in plan["runs"]:
        path = root / "training" / run["name"] / "metrics.jsonl"
        rows, digest = read_live_records(path)
        inputs[str(path)] = digest
        values = extract(rows, run["name"])
        series[run["name"]] = values
        for key in combined:
            combined[key].extend(values[key])
    for key, rows in combined.items():
        if rows:
            write_csv(output / f"{key}.csv", rows)
    summary = dict(
        inputs=inputs,
        style_sha256=hashlib.sha256(style_path.read_bytes()).hexdigest(),
        points={key: len(rows) for key, rows in combined.items()},
        online_s="undefined: only one pre-update observation per fresh rollout batch",
        training_x="completed successful updates before the observed loss (logged step/optimizer minus one)",
        signal_x="completed successful updates when the observation becomes available",
    )
    if not combined["training"] and not combined["heldout"]:
        dump(output / "audit.json", summary | dict(state="waiting_for_measurements"))
        return summary

    active = [run for run in plan["runs"] if any(series[run["name"]].values())]
    handles = [Line2D([], [], color=colors[r["arm"]], label=r["arm"]) for r in active]
    bank_handles = [
        Line2D([], [], color=fv.NEUTRALS["vink"], linestyle=ls, label=label)
        for ls, label in (("-", "initial-student prefixes"), ("--", "teacher prefixes"))
    ]

    def frame(fig, axes, caption):
        fig.subplots_adjust(left=0.09, right=0.98, top=0.82, bottom=0.15, hspace=0.45, wspace=0.3)
        fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False, bbox_to_anchor=(0.5, 1.0))
        fig.legend(handles=bank_handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.94))
        for ax in axes:
            ax.set_xlabel("Successful optimizer updates")
            ax.set_xlim(-plan["updates"] * 0.015, plan["updates"] * 1.015)
        fig.text(0.5, 0.025, caption, ha="center", fontsize=9, color=fv.NEUTRALS["baseline"])

    def heldout(ax, domain):
        for run in active:
            for bank, style in (("public_initial", "-"), ("public_teacher", "--")):
                rows = sorted(
                    (r for r in series[run["name"]]["heldout"] if r["domain"] == domain and r["bank"] == bank),
                    key=lambda r: r["update"],
                )
                if rows:
                    ax.plot(
                        [r["update"] for r in rows],
                        [r["rkl"] for r in rows],
                        color=colors[run["arm"]],
                        linestyle=style,
                        marker="o",
                        markersize=3,
                    )
        ax.set_ylabel(f"{domain.upper()} held-out RKL (nats/token)")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    loss_ax, math_ax, s_ax, reuse_ax = axes.flat
    for run in active:
        values, color = series[run["name"]], colors[run["arm"]]
        raw = values["training"]
        if raw:
            loss_ax.plot([r["update"] for r in raw], [r["loss"] for r in raw], color=color, alpha=0.2, linewidth=0.5)
            means = block_means(raw)
            loss_ax.plot([r["update"] for r in means], [r["loss"] for r in means], color=color, linewidth=1.5)
        signals = values["signals"]
        if signals:
            s_ax.plot([r["update"] for r in signals], [r["s"] for r in signals], color=color, marker=".", markersize=3)
        schedule = values["schedule"]
        if schedule:
            reuse_ax.step(
                [0] + [r["update"] for r in schedule],
                [run["reuse"]] + [r["next"] for r in schedule],
                where="post",
                color=color,
            )
    loss_ax.set_ylabel("Training loss (nats/token)")
    heldout(math_ax, "math")
    s_ax.set_ylabel("S = window refresh gap / within-batch drop")
    s_ax.axhline(0.9, color=fv.NEUTRALS["baseline"], linestyle=":", linewidth=1)
    reuse_ax.set_ylabel("Updates per fresh rollout batch")
    reuse_ax.set_yticks([1, 4, 8, 16])
    frame(
        fig,
        list(axes.flat),
        "Training: raw + 16-update means. Online S is undefined. Only measured held-out points are shown.",
    )
    figures = [("update-aligned", fig)]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    for ax, domain in zip(axes, ("math", "code", "if"), strict=True):
        heldout(ax, domain)
    frame(fig, axes, "Exact held-out dataset question means; development questions are excluded.")
    figures.append(("heldout-rkl", fig))
    summary["render_audit"] = {}
    for name, fig in figures:
        fig.canvas.draw()
        # These are standalone wide figures, not reduced single-column inserts.
        summary["render_audit"][name] = fv.audit(fig, print_slot=fig.get_figwidth())
        for extension in ("png", "pdf"):
            temporary = output / f".{name}.{extension}"
            fig.savefig(temporary, dpi=180)
            temporary.replace(output / f"{name}.{extension}")
        plt.close(fig)
    dump(output / "audit.json", summary)
    status = read(root / "status.json") if (root / "status.json").exists() else dict(state="prepared")
    write_text(
        output / "index.html",
        '<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="60">'
        "<title>OPD reuse ablations</title><h1>按成功 update 对齐的 OPD 对照</h1>"
        f"<p>{html.escape(str(status.get('state')))} · {html.escape(str(status.get('current', '')))}</p>"
        "<p>online 的 S 未定义；虚线为 teacher prefixes，实线为 initial-student prefixes。</p>"
        '<img src="update-aligned.png" style="max-width:100%"><img src="heldout-rkl.png" style="max-width:100%">',
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    render(args.root.resolve())


if __name__ == "__main__":
    main()
