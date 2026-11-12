# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Read-only scalar mirror and repeatable live figures for the eighteen-pair study.

Ledger: a point is one completed outer's domain-weighted question-mean RKL.
Color denotes teacher pair, columns denote fixed prefix bank; x is outer index.
Raw optimizer records retain their source step and are never coalesced for upload.
"""

import argparse
import fcntl
import hashlib
import json
import math
import netrc
import os
import statistics
import subprocess
import time
from pathlib import Path

from examples.mirroropd.research.core.io import dump, read


def records(path):
    result = []
    with path.open() as stream:
        for line in stream:
            if not line.endswith("\n"):
                break
            result.append(json.loads(line))
    return result


def collect(study):
    series = {}
    for lane in sorted(study.glob("absorption-*")):
        for path in sorted((lane / "training").glob("E*/metrics.jsonl")):
            series[path.parent.name] = (path, records(path))
    return series


def render(study, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from examples.mirroropd.research.core.plotting import load_style, save_figure

    fv = load_style()
    theme = fv.apply_rc("paper")
    series = collect(study)
    output.mkdir(parents=True, exist_ok=True)
    points, timing = [], []
    for name, (_, rows) in series.items():
        for row in rows:
            d = row["data"]
            if "eval/rkl/public_initial/all/question_mean" in d:
                points.append(dict(experiment=name, outer=d["step/outer"], metrics=d))
            if "perf/timing/outer_step_seconds" in d:
                timing.append(dict(experiment=name, outer=d["step/outer"], metrics=d))
    dump(output / "curve-data.json", dict(points=points, timing=timing, captured_at=time.time()))
    colors = fv.series_colors("paper", len(series))
    for relative in (False, True):
        fig, axes = plt.subplots(4, 2, figsize=(10, 10), sharex=True, layout="constrained")
        fig.get_layout_engine().set(rect=(0, 0.065, 1, 0.935))
        for col, bank in enumerate(("public_initial", "public_teacher")):
            for row, domain in enumerate(("all", "math", "code", "if")):
                ax = axes[row, col]
                key = f"eval/rkl/{bank}/{domain}/question_mean"
                for color, name in zip(colors, series, strict=True):
                    data = [p for p in points if p["experiment"] == name]
                    values = [(p["outer"], p["metrics"][key]) for p in data if key in p["metrics"]]
                    if not values:
                        continue
                    base = values[0][1]
                    ax.plot(
                        [v[0] for v in values],
                        [v[1] / base if relative else v[1] for v in values],
                        label=name,
                        color=color,
                        marker="o",
                        markersize=3,
                    )
                ax.set_title(f"{bank.removeprefix('public_')} prefix / {domain}")
                ax.set_ylabel("RKL / initial RKL" if relative else "Absolute RKL (nats/token)")
                ax.set_ylim(bottom=0)
                fv.apply_frame(ax, theme.frame)
                fv.apply_grid(ax, theme.grid)
                if row == 3:
                    ax.set_xlabel("Completed outer (16 optimizer updates each)")
                if row == 0:
                    ax.legend(frameon=False)
        for row in range(4):
            ceiling = max(ax.get_ylim()[1] for ax in axes[row])
            for ax in axes[row]:
                ax.set_ylim(0, ceiling * 1.03)
        save_figure(
            fig,
            output / ("rkl-relative" if relative else "rkl-absolute"),
            "Fixed long banks; Math128 / Code64 / IF64; cap7168. Partial training curves.",
        )
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    fig.get_layout_engine().set(rect=(0, 0.13, 1, 0.87))
    for ax, key, label in zip(
        axes,
        ("outer_step_seconds", "evaluation_seconds"),
        ("Total outer time (s)", "Bank readout + optional export (s)"),
        strict=True,
    ):
        for color, name in zip(colors, series, strict=True):
            values = [p for p in timing if p["experiment"] == name]
            ax.plot(
                [p["outer"] for p in values],
                [p["metrics"][f"perf/timing/{key}"] for p in values],
                color=color,
                marker="o",
                markersize=3,
                label=name,
            )
        ax.set_xlabel("Completed outer")
        ax.set_ylabel(label)
        ax.set_ylim(bottom=0)
        ax.legend(frameon=False)
        fv.apply_frame(ax, theme.frame)
        fv.apply_grid(ax, theme.grid)
    save_figure(fig, output / "training-speed", "Snapshot outers: 1,2,3,4,8,12,20,30,40. First outer includes warmup.")


def mirror(study, credential):
    import wandb

    authentication = netrc.netrc(str(credential)).authenticators("api.wandb.ai")
    if authentication is None:
        raise ValueError("W&B authentication is absent")
    os.environ["WANDB_API_KEY"] = authentication[2]
    os.environ["WANDB_MODE"] = "online"
    runs, positions, finished = {}, {}, set()
    state = study / "wandb-live"
    state.mkdir(exist_ok=True)
    receipt = read(state / "runs.json") if (state / "runs.json").exists() else {}
    try:
        while True:
            for name, (path, rows) in collect(study).items():
                if name in finished:
                    continue
                if name not in runs:
                    run_id = "abs-" + hashlib.sha256(f"{study}:{name}".encode()).hexdigest()[:16]
                    run = wandb.init(
                        project="MirrorOPD-verl",
                        id=run_id,
                        name=f"{name}-long-bank-live",
                        group=study.name,
                        job_type="scalar-mirror",
                        resume="allow",
                        reinit="create_new",
                        dir=str(state),
                        config=dict(
                            experiment=name,
                            batch=64,
                            reuse=16,
                            outer=40,
                            response_cap=7168,
                            accuracy=False,
                            scalar_mirror=True,
                        ),
                        settings=wandb.Settings(init_timeout=45, x_disable_stats=True),
                    )
                    run.define_metric("step/outer")
                    run.define_metric("step/optimizer")
                    run.define_metric("eval/*", step_metric="step/outer")
                    run.define_metric("perf/*", step_metric="step/outer")
                    run.define_metric("optim/*", step_metric="step/optimizer")
                    run.define_metric("distill/*", step_metric="step/optimizer")
                    runs[name] = run
                    positions[name] = int(run.summary.get("mirror/source_lines", 0))
                    receipt[name] = dict(id=run_id, url=run.url, source=str(path))
                    dump(state / "runs.json", receipt)
                for index in range(positions[name], len(rows)):
                    record = rows[index]
                    data = {k: v for k, v in record["data"].items() if isinstance(v, int | float) and math.isfinite(v)}
                    data.update({"mirror/source_step": record["step"], "mirror/source_lines": index + 1})
                    runs[name].log(data, step=index, commit=True)
                positions[name] = len(rows)
                lane_status = read(path.parents[2] / "status.json")
                if any(
                    stage.get("label") == f"{name}-train" and stage.get("state") == "complete"
                    for stage in lane_status.get("stages", [])
                ):
                    runs.pop(name).finish()
                    finished.add(name)
            dump(state / "status.json", dict(updated_at=time.time(), source_lines=positions))
            statuses = [read(lane / "status.json").get("state") for lane in study.glob("absorption-*")]
            if statuses and all(s in {"complete", "failed", "stopped"} for s in statuses):
                break
            time.sleep(20)
    finally:
        for run in runs.values():
            run.finish()


def watch(study, output):
    """CPU-only telemetry and figures; cannot acquire, stop or restart GPU work."""
    destination = study / "readout-watch"
    destination.mkdir(exist_ok=True)
    with (destination / "lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            groups = []
            for lane in sorted(study.glob("absorption-*")):
                status = read(lane / "status.json")
                groups.append(
                    dict(
                        lane=lane.name,
                        state=status.get("state"),
                        current=status.get("current"),
                        stages=status.get("stages", []),
                        error=status.get("error"),
                    )
                )
            performance = {}
            for name, (path, rows) in collect(study).items():
                boundaries = [r["data"] for r in rows if "perf/timing/outer_step_seconds" in r["data"]]
                steady = [d for d in boundaries if d["step/outer"] > 1]
                performance[name] = dict(
                    latest_optimizer_step=max((r["data"].get("step/optimizer", 0) for r in rows), default=0),
                    completed_outer=max((d["step/outer"] for d in boundaries), default=0),
                    seconds_since_metrics=time.time() - path.stat().st_mtime,
                    median_outer_seconds=statistics.median(d["perf/timing/outer_step_seconds"] for d in steady)
                    if steady
                    else None,
                    nonfinite_records=sum(
                        any(
                            d.get(k, 0)
                            for k in ("optim/actor/loss_nonfinite", "optim/actor/grad_norm_preclip_nonfinite")
                        )
                        for d in [r["data"] for r in rows]
                    ),
                )
            gpu = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=15,
            )
            report = dict(at=time.time(), groups=groups, performance=performance, gpu=gpu.stdout)
            dump(destination / f"inspection-{int(time.time())}.json", report)
            dump(destination / "latest.json", report)
            try:
                render(study, output)
                dump(destination / "figure-status.json", dict(at=time.time(), state="complete"))
            except Exception as error:
                dump(destination / "figure-status.json", dict(at=time.time(), state="failed", error=repr(error)))
            if groups and all(g["state"] in {"complete", "failed", "stopped"} for g in groups):
                break
            time.sleep(60)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("render", "mirror", "watch"))
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--credential", type=Path)
    args = parser.parse_args()
    if args.action == "render":
        render(args.study.resolve(), args.output.resolve())
    elif args.action == "watch":
        watch(args.study.resolve(), args.output.resolve())
    else:
        mirror(args.study.resolve(), args.credential)


if __name__ == "__main__":
    main()
