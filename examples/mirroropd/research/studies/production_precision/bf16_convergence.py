# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Compare historical BF16 baselines with E50/E52 using local Full-RKL logs.

Ledger: one color per run; thin lines show every observed outer, thick lines
show nonoverlapping 16-outer means (terminal short block retained). First and
last values are pre-update observations. Applied-update coordinates count only
updates BEFORE the corresponding last observation. Timing uses cumulative
logged outer-stage time associated with each batch, not exact loss timestamps
or total experiment wall-clock. Historical teacher/code/backend/hardware
differences prohibit interpreting the curves as a pure dtype intervention.
"""

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

from examples.mirroropd.research.core.io import dump, write_csv
from examples.mirroropd.research.core.plotting import load_style
from examples.mirroropd.research.studies.production_precision.analyze import flatten

RUNS = {"E17": "cix5i2x2", "E20": "lwgb5eex", "E50": "rsi5krqd", "E52": "7rwc1ioz"}
LABELS = {
    "E17": "E17: historical BF16 / A100",
    "E20": "E20: historical BF16 / A800",
    "E50": "E50: FP16 / A100",
    "E52": "E52: BF16 GEMM + FP32 / A100",
}


def collect(logs, out):
    out.mkdir(parents=True, exist_ok=True)
    rows, windows, evaluations, summary, configs = [], [], [], {}, {}
    for label, rid in RUNS.items():
        (source,) = (logs / "derived/MirrorOPD-verl").glob("*" + rid)
        meta = json.loads((source / "run.json").read_text())
        raw = Path(meta["cache_key"]["source"])
        stat = raw.stat()
        if meta["status"] != "finished" or meta["exit_code"] != 0:
            raise ValueError(f"Expected a finished baseline: {label}")
        if (stat.st_size, stat.st_mtime_ns) != (meta["cache_key"]["bytes"], meta["cache_key"]["mtime_ns"]):
            raise ValueError(f"Stale decode: {label}")
        c = json.loads((source / "config.json").read_text())
        configs[label] = flatten(c)
        loss = c["distillation"]["distillation_loss"]
        if loss["loss_mode"] != "full_reverse_kl" and (loss.get("mirror") or {}).get("eta") != 1:
            raise ValueError(f"Not ordinary Full-RKL: {label}")
        history = [json.loads(line) for line in (source / "history.jsonl").read_text().splitlines()]
        groups, outer = {}, {}
        for row in history:
            if "optim/actor/loss" in row:
                groups.setdefault(int(row["step/outer"]), []).append(row)
            if "rollout/response_tokens" in row:
                outer[int(row["step/outer"])] = row
            for dataset in ("math500", "amc23", "aime25"):
                key = f"eval/{dataset}/acc/mean"
                if key in row:
                    evaluations.append({"run": label, "outer": row["step/outer"], "dataset": dataset, "mean": row[key]})
        if any(len(g) != 16 for g in groups.values()) or set(groups) != set(outer):
            raise ValueError(f"Missing outer/update records: {label}")
        if sorted(groups) != list(range(1, max(groups) + 1)):
            raise ValueError(f"Noncontiguous history: {label}")
        initial = groups[1][0]["distill/full_kl"]
        applied, tokens, times = 0, [], []
        for step, g in groups.items():
            o = outer[step]
            if [r["step/update"] for r in g] != list(range(16 * (step - 1) + 1, 16 * step + 1)):
                raise ValueError(f"Update-attempt sequence mismatch: {label}/{step}")
            if not all(math.isfinite(r["distill/full_kl"]) and r["distill/full_kl"] > 0 for r in g):
                raise ValueError(f"Nonpositive/nonfinite KL: {label}/{step}")
            attempts = [int(r["optim/actor/update_applied"]) for r in g]
            before_last = applied + sum(attempts[:-1])
            applied += sum(attempts)
            if applied != g[-1]["step/optimizer"]:
                raise ValueError(f"Optimizer coordinate mismatch: {label}/{step}")
            tokens.append(o["rollout/response_tokens"])
            times.append(o["perf/timing/outer_step_seconds"])
            first, last = g[0]["distill/full_kl"], g[-1]["distill/full_kl"]
            rows.append(
                {
                    "run": label,
                    "outer": step,
                    "first": first,
                    "last": last,
                    "first_normalized": first / initial,
                    "last_normalized": last / initial,
                    "observed_applied_updates": before_last,
                    "completed_updates": applied,
                    "cumulative_outer_hours": math.fsum(times) / 3600,
                    "cumulative_response_mtokens": math.fsum(tokens) / 1e6,
                    "actor_seconds": o["perf/timing/update_actor_seconds"],
                    "outer_seconds": o["perf/timing/outer_step_seconds"],
                    "actor_peak_gib": o["perf/actor/allocated_peak_gib"],
                    "first_rise": g[1]["distill/full_kl"] > first + 1e-10,
                    "end_worse": last > first + 1e-10,
                    "end_improved": last < first - 1e-10,
                    "adjacent_rises": sum(
                        b["distill/full_kl"] > a["distill/full_kl"] + 1e-10 for a, b in zip(g, g[1:], strict=False)
                    ),
                    "skipped": sum(int(r["optim/actor/update_skipped"]) for r in g),
                }
            )
        own = [r for r in rows if r["run"] == label]
        environment = [json.loads(line) for line in (source / "environment.jsonl").read_text().splitlines()]
        summary[label] = {
            "run_id": rid,
            "source": str(raw.relative_to(logs)),
            "sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
            "recorded_git": meta["run"].get("git"),
            "initial_kl": initial,
            "outers": len(own),
            "updates": applied,
            "skipped": sum(r["skipped"] for r in own),
            "first_rises": sum(r["first_rise"] for r in own),
            "end_worse": sum(r["end_worse"] for r in own),
            "end_improved": sum(r["end_improved"] for r in own),
            "adjacent_rises": sum(r["adjacent_rises"] for r in own),
            "hardware": {k: env[k] for env in environment for k in ("gpu_type", "gpu_count", "cpu_count") if k in env},
            "backend": loss["full_vocab"]["backend"],
            "chunk": loss["full_vocab"]["token_chunk_size"],
            "precision": c["actor_rollout_ref"]["actor"]["fsdp_config"].get("forward_precision") or "historical BF16",
            "test_freq": c["trainer"]["test_freq"],
            "save_freq": c["trainer"]["save_freq"],
            "recorded_outer_hours": own[-1]["cumulative_outer_hours"],
            "actor_hours": math.fsum(r["actor_seconds"] for r in own) / 3600,
            "response_mtokens": own[-1]["cumulative_response_mtokens"],
            "actor_peak_gib": max(r["actor_peak_gib"] for r in own),
            "settings": {
                "student": Path(c["actor_rollout_ref"]["model"]["path"]).name,
                "teacher": Path(c["distillation"]["teacher_models"]["teacher_model"]["model_path"]).name,
                "optimizer": c["actor_rollout_ref"]["actor"]["optim"],
                "reuse": c["actor_rollout_ref"]["actor"]["ppo_epochs"],
                "batch": c["data"]["train_batch_size"],
                "response_cap": c["actor_rollout_ref"]["rollout"]["response_length"],
                "seed": c["actor_rollout_ref"]["actor"]["data_loader_seed"],
            },
        }
        for start, end in ((1, 16), (17, 32), (33, 64), (65, 96), (169, 200), (225, 256)):
            selected = [r for r in own if start <= r["outer"] <= end]
            if len(selected) != end - start + 1:
                continue
            f = statistics.mean(r["first"] for r in selected)
            last_mean = statistics.mean(r["last"] for r in selected)
            windows.append(
                {
                    "run": label,
                    "start": start,
                    "end": end,
                    "first_mean": f,
                    "last_mean": last_mean,
                    "within_batch_reduction_pct": 100 * (1 - last_mean / f),
                }
            )
    blocks = []
    keys = (
        "first",
        "last",
        "first_normalized",
        "last_normalized",
        "observed_applied_updates",
        "cumulative_outer_hours",
        "cumulative_response_mtokens",
    )
    for label in RUNS:
        own = [r for r in rows if r["run"] == label]
        for start in range(0, len(own), 16):
            group = own[start : start + 16]
            blocks.append(
                {
                    "run": label,
                    "start": group[0]["outer"],
                    "end": group[-1]["outer"],
                    "count": len(group),
                    "outer": statistics.mean(r["outer"] for r in group),
                    **{key: statistics.mean(r[key] for r in group) for key in keys},
                }
            )
    differences = {
        k: {label: c.get(k) for label, c in configs.items()}
        for k in sorted(set().union(*(set(c) for c in configs.values())))
        if len({json.dumps(c.get(k), sort_keys=True) for c in configs.values()}) > 1
    }
    dump(out / "summary.json", summary)
    dump(out / "config-differences.json", differences)
    for name, data in [("outer", rows), ("blocks", blocks), ("windows", windows), ("evaluations", evaluations)]:
        write_csv(out / (name + ".csv"), data)
    dump(out / "plot-input.json", {"outer": rows, "blocks": blocks, "summary": summary})
    return {"outer": rows, "blocks": blocks, "summary": summary}


def plot(data, destination, style):
    import matplotlib.pyplot as plt

    fv = load_style(style)
    fv.apply_rc("paper")
    destination.mkdir(parents=True, exist_ok=True)
    colors = {
        "E17": fv.HUES["vviolet"],
        "E20": fv.NEUTRALS["baseline"],
        "E50": fv.HUES["vteal"],
        "E52": fv.HUES["vcoral"],
    }
    audits = {}

    def draw(ax, x, y, scale="log"):
        for label in RUNS:
            raw = [r for r in data["outer"] if r["run"] == label]
            smooth = [r for r in data["blocks"] if r["run"] == label]
            ax.plot([r[x] for r in raw], [r[y] for r in raw], color=colors[label], lw=0.5, alpha=0.22)
            ax.plot([r[x] for r in smooth], [r[y] for r in smooth], color=colors[label], lw=1.7, label=LABELS[label])
        ax.set_yscale(scale)

    def save(fig, name, caption):
        handles = [plt.Line2D([], [], color=colors[k], label=LABELS[k]) for k in RUNS]
        fig.legend(handles=handles, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 0.995), frameon=False, fontsize=10)
        fig.text(0.5, 0.02, caption, ha="center", fontsize=8, color=fv.NEUTRALS["baseline"])
        audits[name] = fv.audit(fig, print_slot=fig.get_figwidth())
        fv.save(fig, name, outdir=destination, formats=["png", "pdf"])

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 8))
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.12, top=0.83, hspace=0.5, wspace=0.3)
    for row, fields in enumerate((("first", "last"), ("first_normalized", "last_normalized"))):
        for col, field in enumerate(fields):
            ax = axes[row, col]
            draw(ax, "outer", field)
            ax.set(
                xlabel="Outer step",
                ylabel="KL (nats/token)" if row == 0 else "KL / initial KL",
                title=("New-batch first KL" if col == 0 else "Last observed KL") + (" (relative)" if row else ""),
            )
        low = min(a.get_ylim()[0] for a in axes[row])
        high = max(a.get_ylim()[1] for a in axes[row])
        for a in axes[row]:
            a.set_ylim(low, high)
    save(
        fig,
        "kl-convergence",
        "Thin: every outer; thick: 16-outer means. Historical BF16 differs in code, teacher path and backend.",
    )
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.5), sharey=True)
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.22, top=0.75, wspace=0.25)
    for ax, x, title in zip(
        axes,
        ("observed_applied_updates", "cumulative_response_mtokens", "cumulative_outer_hours"),
        (
            "Successful updates before observation",
            "Cumulative new response tokens (M)",
            "Cumulative logged outer time (h)",
        ),
        strict=True,
    ):
        draw(ax, x, "last")
        ax.set(xlabel=title, ylabel="Last observed KL (nats/token)")
    save(
        fig,
        "kl-budget-speed",
        "Batch-associated logged time, not total wall-clock. Hardware, backend and save schedules differ.",
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.5))
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.22, top=0.75, wspace=0.3)
    for ax, field, title in zip(
        axes, ("first", "last"), ("New-batch first KL: outer 65+", "Last observed KL: outer 65+"), strict=True
    ):
        for label in RUNS:
            selected = [r for r in data["blocks"] if r["run"] == label and r["start"] >= 65]
            ax.plot(
                [r["outer"] for r in selected],
                [r[field] * 1e4 for r in selected],
                color=colors[label],
                label=LABELS[label],
                lw=1.7,
            )
        ax.set(xlabel="Outer step", ylabel="KL (× 1e−4)", title=title)
    low = min(a.get_ylim()[0] for a in axes)
    high = max(a.get_ylim()[1] for a in axes)
    for a in axes:
        a.set_ylim(low, high)
    save(
        fig,
        "kl-late-zoom",
        "Late-phase linear zoom of 16-outer means; the E50 excursion is retained, not removed as an outlier.",
    )
    dump(destination / "figure-audit.json", audits)
    dump(
        destination / "figure-design.json",
        {
            "ledger": __doc__,
            "series": LABELS,
            "data_sha256": hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest(),
        },
    )
    print(audits)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs-repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--figures", type=Path)
    parser.add_argument("--style", type=Path)
    args = parser.parse_args()
    data = collect(args.logs_repo.resolve(), args.out.resolve())
    if args.figures:
        plot(data, args.figures, args.style)


if __name__ == "__main__":
    main()
