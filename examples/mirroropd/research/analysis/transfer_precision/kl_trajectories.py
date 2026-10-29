# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Freeze and render every recorded outer/inner KL observation.

Ledger: metric colors are teacher=teal, anchor=orange, target=violet. Each
atlas panel is exactly one outer and its 16 pre-update observations, without
smoothing or interpolation across batches. First/last outer curves connect
different rollout populations, not repeated evaluation of fixed states.
Eta=1 target KL is a declared alias of teacher KL; missing anchor is never zero.
Heatmaps encode log10(target/first target) for each batch; missing cells remain
masked. Kappa comparisons fix eta within a figure and use the coarsened EOS
action space for E54. Source inputs are read-only; no training is launched.
"""

import argparse
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from examples.mirroropd.research.analysis.transfer_precision.analyze import write_csv
from examples.mirroropd.research.analysis.transfer_precision.precision_comparison import RUNS, TOL

HISTORICAL = {"E45": "625vw4ob", **RUNS}
METRICS = ("teacher", "anchor", "target")


def read_records(path):
    """Read one byte snapshot, allowing only an unfinished final JSON line."""
    data = path.read_bytes()
    lines = data.splitlines()
    rows = []
    for i, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if i != len(lines) - 1 or data.endswith(b"\n"):
                raise
    return rows, hashlib.sha256(data).hexdigest()


def trajectory(rows, eta):
    groups = {}
    for row in rows:
        if "distill/target_kl" not in row and "distill/full_kl" not in row:
            continue
        if "step/outer" not in row:
            continue
        teacher = row.get("distill/teacher_kl", row.get("distill/full_kl"))
        target = row.get("distill/target_kl", teacher if eta == 1 else None)
        if teacher is None or target is None or not all(math.isfinite(x) and x > 0 for x in (teacher, target)):
            raise ValueError("Nonpositive/nonfinite or missing KL requires separate diagnosis")
        outer = int(row["step/outer"])
        group = groups.setdefault(outer, [])
        group.append(
            {
                "outer": outer,
                "inner": len(group) + 1,
                "teacher": teacher,
                "anchor": row.get("distill/anchor_kl"),
                "target": target,
                "log_z": row.get("distill/target_log_z"),
                "objective": row.get("distill/mdpo/objective", teacher if eta == 1 else None),
                "update": row.get("step/update"),
                "skipped": row.get("optim/actor/update_skipped", 0),
                "target_alias": "distill/target_kl" not in row,
            }
        )
    if any(len(g) != 16 for g in groups.values()):
        raise ValueError("Expected 16 complete pre-update observations per outer")
    return groups


def collect(repo, logs, out):
    out.mkdir(parents=True, exist_ok=True)
    snapshot = {"observed_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "runs": {}, "evaluations": []}
    for label, rid in HISTORICAL.items():
        (source,) = (logs / "derived/MirrorOPD-verl").glob("*" + rid)
        meta = json.loads((source / "run.json").read_text())
        if meta["status"] not in ("finished", "snapshot", "partial_snapshot"):
            raise ValueError(f"Unusable decoded snapshot: {label}")
        raw = Path(meta["cache_key"]["source"])
        stat = raw.stat()
        if (stat.st_size, stat.st_mtime_ns) != (meta["cache_key"]["bytes"], meta["cache_key"]["mtime_ns"]):
            raise ValueError(f"Stale decode: {label}")
        config = json.loads((source / "config.json").read_text())
        mirror = config["distillation"]["distillation_loss"]["mirror"]
        eta, kappa = mirror["eta"], mirror["kappa"]
        rows, _ = read_records(source / "history.jsonl")
        snapshot["runs"][label] = {
            "family": "R1/JustRL",
            "eta": eta,
            "kappa": kappa,
            "precision": config["actor_rollout_ref"]["actor"]["fsdp_config"].get("forward_precision")
            or "historical BF16",
            "source": str(raw.relative_to(logs)),
            "sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
            "exit_code": meta["exit_code"],
            "lr": config["actor_rollout_ref"]["actor"]["optim"]["lr"],
            "groups": trajectory(rows, eta),
        }
    for dirname in ("e54-topd-mdpo-20261029", "e54-mdpo-sweep-20261029"):
        root = repo / "outputs" / dirname
        protocol = json.loads((root / "protocol.json").read_text())
        for arm in protocol["arms"]:
            path = root / arm["label"]
            rows, digest = read_records(path / "metrics.jsonl")
            exit_path = path / "exit.json"
            snapshot["runs"]["E54/" + arm["label"]] = {
                "family": "E54/Qwen",
                "eta": arm["eta"],
                "kappa": arm["kappa"],
                "precision": "pilot production path",
                "lr": protocol["lr"],
                "source": str((path / "metrics.jsonl").relative_to(repo)),
                "sha256": digest,
                "exit_code": json.loads(exit_path.read_text())["exit_code"] if exit_path.exists() else None,
                "groups": trajectory([r["data"] for r in rows], arm["eta"]),
                "action_space": protocol["diagnostics"],
                "shared_checkpoint": protocol["shared_checkpoint"],
            }
        summary = root / "evaluation/summary.json"
        if summary.exists():
            d = json.loads(summary.read_text())
            snapshot["evaluations"].extend(d["evaluation"])
    traces, summaries = [], []
    for label, run in snapshot["runs"].items():
        for outer, rows in run["groups"].items():
            traces.extend({"run": label, **row} for row in rows)
            first, second, last = rows[0], rows[1], rows[-1]
            item = {
                "run": label,
                "outer": outer,
                "kappa": run["kappa"],
                "eta": run["eta"],
                "target_first": first["target"],
                "target_second": second["target"],
                "target_last": last["target"],
                "teacher_first": first["teacher"],
                "teacher_second": second["teacher"],
                "teacher_last": last["teacher"],
                "anchor_first": first["anchor"],
                "anchor_second": second["anchor"],
                "anchor_last": last["anchor"],
                "target_reduction_pct": 100 * (1 - last["target"] / first["target"]),
                "target_first_rise": second["target"] > first["target"] + TOL,
                "target_net_worse": last["target"] > first["target"] + TOL,
                "objective_reduction_pct": 100 * (1 - last["objective"] / first["objective"]),
                "teacher_reduction_pct": 100 * (1 - last["teacher"] / first["teacher"]),
            }
            if run["kappa"] == 0 and run["eta"] != 1:
                lam = (1 - run["eta"]) / run["eta"]
                gain = first["teacher"] - second["teacher"]
                displacement = second["anchor"] - first["anchor"]
                item.update(
                    first_teacher_gain=gain,
                    first_anchor_cost=lam * displacement,
                    first_lambda_critical=gain / displacement if displacement > 0 else None,
                )
                error = max(
                    abs(
                        (r["target"] - first["target"])
                        - run["eta"] * (r["teacher"] - first["teacher"])
                        - (1 - run["eta"]) * (r["anchor"] - first["anchor"])
                    )
                    for r in rows
                )
                item["delta_identity_max_error"] = error
                if error > 1e-7:
                    raise ValueError(f"KL decomposition inconsistent: {label}/{outer}: {error}")
            summaries.append(item)
    snapshot["counterexamples"] = []
    for q, teacher in (([0.9, 0.1], [0.1, 0.9]), ([0.1, 0.9], [0.999, 0.001])):
        for k in (0, 0.5, 1):
            e = 0.1
            b = [
                math.exp((1 - e) * math.log(a) + e * math.log(t)) if k == 0 else ((1 - e) * a**k + e * t**k) ** (1 / k)
                for a, t in zip(q, teacher, strict=True)
            ]
            m = [a / math.fsum(b) for a in b]
            snapshot["counterexamples"].append(
                {
                    "q": q,
                    "teacher": teacher,
                    "eta": e,
                    "kappa": k,
                    "target": m,
                    "target_to_teacher_rkl": math.fsum(a * math.log(a / t) for a, t in zip(m, teacher, strict=True)),
                }
            )
    (out / "snapshot.json").write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n")
    write_csv(out / "traces.csv", traces)
    write_csv(out / "outer-summary.csv", summaries)
    write_csv(out / "evaluations.csv", snapshot["evaluations"])
    print(snapshot["observed_at"], len(snapshot["runs"]), "runs;", len(summaries), "complete outers")


def plot(snapshot, out, skill, atlas=False, source_sha256=None):
    import sys

    sys.path.insert(0, str(skill / "assets/mpl"))
    import fvstyle as fv
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.colors import TwoSlopeNorm

    fv.apply_rc("paper")
    out.mkdir(parents=True, exist_ok=True)
    colors = {"teacher": fv.HUES["vteal"], "anchor": fv.HUES["vorange"], "target": fv.HUES["vviolet"]}
    audit = {}
    for run in snapshot["runs"].values():
        run["groups"] = {int(k): v for k, v in run["groups"].items()}

    def caption(fig, text):
        fig.text(0.5, 0.02, text, ha="center", fontsize=9, color=fv.NEUTRALS["baseline"])

    def save(fig, name, ignore=()):
        issues = fv.audit(fig, ignore=ignore, print_slot=fig.get_figwidth())
        audit[name] = issues
        fv.save(fig, name, outdir=out, formats=["png", "pdf"])

    def kl_lines(ax, rows):
        for metric in METRICS:
            if metric == "target" and rows[0]["target_alias"]:
                continue
            if rows[0][metric] is not None:
                ax.plot(
                    [r["inner"] for r in rows], [r[metric] for r in rows], color=colors[metric], label=metric, lw=1.2
                )
        ax.set(xlabel="Inner observation", ylabel="KL (nats/token)", xticks=[1, 4, 8, 12, 16])
        ax.set_yscale("symlog", linthresh=1e-8)
        ax.set_ylim(bottom=0)

    for label in HISTORICAL:
        run = snapshot["runs"][label]
        groups = run["groups"]
        fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.6))
        fig.subplots_adjust(left=0.07, right=0.98, bottom=0.23, top=0.72, wspace=0.35)
        for ax, metric in zip(axes, METRICS, strict=True):
            ax.set(title=metric.title() + " KL", xlabel="Outer step", ylabel="KL (nats/token)")
            if next(iter(groups.values()))[0][metric] is None:
                ax.text(0.5, 0.5, "Not recorded", transform=ax.transAxes, ha="center")
                continue
            for index, style, name in ((0, "--", "First observation"), (-1, "-", "16th observation")):
                ax.plot(
                    list(groups),
                    [g[index][metric] for g in groups.values()],
                    color=colors[metric],
                    ls=style,
                    lw=0.9,
                    label=f"{metric}: {name}",
                )
            ax.set_yscale("symlog", linthresh=1e-8)
            ax.set_ylim(bottom=0)
        fig.suptitle(f"{label}: {run['precision']}, kappa={run['kappa']:g}, eta={run['eta']:g}", y=0.97)
        handles = [line for ax in axes for line in ax.lines]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.92), ncol=3, frameon=False)
        suffix = "; eta=1: target is teacher alias, anchor unavailable" if run["eta"] == 1 else ""
        caption(
            fig, "Every outer, no smoothing; changing rollout populations; 15 attempts between observations" + suffix
        )
        save(fig, "outer-" + label, ignore=("facet-ylim-mismatch",))

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 6.4), sharey=True)
    fig.subplots_adjust(left=0.08, right=0.86, bottom=0.14, top=0.88, wspace=0.22)
    labels = ("E45", "E51", "E53")
    matrices = []
    for label in labels:
        m = np.full((256, 16), np.nan)
        for outer, rows in snapshot["runs"][label]["groups"].items():
            m[outer - 1, :] = np.log10([r["target"] / rows[0]["target"] for r in rows])
        matrices.append(m)
    bound = max(float(np.nanmax(np.abs(m))) for m in matrices)
    cmap = fv.ramp_cmap("div").reversed()
    cmap.set_bad(fv.NEUTRALS["vmist"])
    for ax, label, m in zip(axes, labels, matrices, strict=True):
        im = ax.imshow(
            m,
            aspect="auto",
            origin="lower",
            extent=(0.5, 16.5, 0.5, 256.5),
            cmap=cmap,
            norm=TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound),
            interpolation="nearest",
        )
        ax.set(title=label, xlabel="Inner observation", ylabel="Outer step", xticks=[1, 4, 8, 12, 16])
    cax = fig.add_axes([0.89, 0.2, 0.018, 0.6])
    fig.colorbar(im, cax=cax, label="log10(target KL / first target KL)")
    caption(fig, "Every recorded cell; green = improvement, red = increase; gray = no observation; no smoothing")
    save(fig, "inner-heatmaps")

    fig, axes = plt.subplots(2, 3, figsize=(12.6, 7.6))
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.13, top=0.88, wspace=0.35, hspace=0.55)
    for col, metric in enumerate(METRICS):
        values = [r[metric] for label in ("E51", "E53") for r in snapshot["runs"][label]["groups"][80]]
        low = -0.05 * max(values) if metric == "anchor" else min(values) * 0.9
        high = max(values) * 1.1
        for row, label in enumerate(("E51", "E53")):
            rs = snapshot["runs"][label]["groups"][80]
            ax = axes[row, col]
            ax.plot(
                [r["inner"] for r in rs],
                [r[metric] for r in rs],
                marker="o",
                markersize=3,
                color=colors[metric],
                label=metric,
            )
            ax.set(
                title=f"{label}: {metric} KL",
                xlabel="Inner observation",
                ylabel="KL (nats/token)",
                xticks=[1, 4, 8, 12, 16],
                ylim=(low, high),
            )
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    fig.legend(
        handles=[plt.Line2D([], [], color=colors[m], label=m) for m in METRICS],
        loc="upper center",
        ncol=3,
        frameon=False,
    )
    caption(fig, "Actual outer 80; linear zoom with per-metric scales shared across rows; different rollout states")
    save(fig, "inner-outer80", ignore=("facet-ylim-mismatch",))

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.4))
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.22, top=0.83, wspace=0.3)
    balance = {}
    for label in labels:
        g = []
        c = []
        for rows in snapshot["runs"][label]["groups"].values():
            g.append(rows[0]["teacher"] - rows[1]["teacher"])
            c.append((7 / 3) * (rows[1]["anchor"] - rows[0]["anchor"]))
        balance[label] = (g, c)
    low = min(0, min(x for pair in balance.values() for series in pair for x in series)) * 1.2
    high = max(x for pair in balance.values() for series in pair for x in series) * 1.2
    run_colors = {"E45": fv.NEUTRALS["baseline"], "E51": fv.HUES["vorange"], "E53": fv.HUES["vviolet"]}
    for ax, label in zip(axes, labels, strict=True):
        g, c = balance[label]
        ax.scatter(g, c, s=12, alpha=0.55, color=run_colors[label], label=label)
        fv.mark_neutral(ax.plot([low, high], [low, high], ls=":", color=fv.NEUTRALS["baseline"], lw=0.8)[0])
        ax.set_xscale("symlog", linthresh=1e-6)
        ax.set_yscale("symlog", linthresh=1e-6)
        ax.set(
            title=label,
            xlabel="First-step teacher gain",
            ylabel="Weighted anchor cost (7/3)",
            xlim=(low, high),
            ylim=(low, high),
        )
    caption(
        fig, "Every first step at eta=.3, kappa=0; above diagonal: anchor cost exceeds teacher gain, target KL rises"
    )
    save(fig, "first-step-balance")

    for eta in (0.1, 0.2):
        fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.3))
        fig.subplots_adjust(left=0.08, right=0.98, bottom=0.2, top=0.79, wspace=0.3)
        for ax, k in zip(axes, (0, 0.5, 1), strict=True):
            run = next(
                r
                for r in snapshot["runs"].values()
                if r["family"] == "E54/Qwen" and r["eta"] == eta and r["kappa"] == k
            )
            for outer, rows in run["groups"].items():
                for metric in METRICS:
                    ax.plot(
                        [r["update"] for r in rows],
                        [r[metric] for r in rows],
                        color=colors[metric],
                        lw=1.3,
                        label=metric if outer == 3 else None,
                    )
            for x in (48.5, 64.5):
                fv.mark_neutral(ax.axvline(x, color=fv.NEUTRALS["baseline"], ls=":", lw=0.8))
            ax.set(
                title=f"kappa={k:g}",
                xlabel="Pre-update index",
                ylabel="KL (nats/token)",
                xticks=[33, 48, 64, 80],
                ylim=(0, 0.4),
            )
            ax.set_yscale("symlog", linthresh=1e-6)
        fig.suptitle(f"E54: fixed eta={eta:g}; three separate outer batches", y=0.98)
        fig.legend(
            handles=[plt.Line2D([], [], color=colors[m], label=m) for m in METRICS],
            loc="upper center",
            bbox_to_anchor=(0.5, 0.92),
            ncol=3,
            frameon=False,
        )
        caption(fig, "Boundaries reset anchor/target and change input states; curves are NOT joined across boundaries")
        save(fig, f"e54-kappa-eta{str(eta).replace('.', '')}")

    if atlas:
        collections = {
            label: [(f"{label} / outer {o}", g) for o, g in snapshot["runs"][label]["groups"].items()]
            for label in HISTORICAL
        }
        collections["E54"] = [
            (f"{label[4:]} / outer {o}", g)
            for label, run in snapshot["runs"].items()
            if run["family"] == "E54/Qwen"
            for o, g in run["groups"].items()
        ]
        for label, panels in collections.items():
            with PdfPages(out / f"inner-atlas-{label}.pdf") as pdf:
                for start in range(0, len(panels), 8):
                    fig, axes = plt.subplots(4, 2, figsize=(10.4, 13.6))
                    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.07, top=0.92, wspace=0.3, hspace=0.7)
                    active = []
                    for ax, (title, rows) in zip(axes.flat, panels[start : start + 8], strict=False):
                        active.append(ax)
                        kl_lines(ax, rows)
                        ax.set_title(title, fontsize=11)
                    for ax in list(axes.flat)[len(active) :]:
                        ax.set_visible(False)
                    fig.suptitle(f"{label} — all recorded inner trajectories — page {start // 8 + 1}", y=0.99)
                    metrics = ("teacher",) if label in ("E50", "E52") else METRICS
                    fig.legend(
                        handles=[plt.Line2D([], [], color=colors[m], label=m) for m in metrics],
                        loc="upper center",
                        bbox_to_anchor=(0.5, 0.965),
                        ncol=3,
                        frameon=False,
                    )
                    caption(
                        fig,
                        "Pre-update observations; no 16th-post-update score. "
                        "Per-panel scale; eta=1: teacher=target, anchor not logged.",
                    )
                    issues = fv.audit(fig, ignore=("facet-ylim-mismatch",), print_slot=fig.get_figwidth())
                    audit[f"atlas-{label}-{start // 8 + 1}"] = issues
                    pdf.savefig(fig)
                    if start == 0:
                        fig.savefig(out / f"inner-atlas-{label}-preview.png", dpi=160)
                    plt.close(fig)
            print(label, len(panels), "outer panels; atlas complete", flush=True)
    overview_audit = {k: v for k, v in audit.items() if not k.startswith("atlas-")}
    (out / "figure-audit.json").write_text(json.dumps(overview_audit, indent=2) + "\n")
    if atlas:
        atlas_audit = {k: v for k, v in audit.items() if k.startswith("atlas-")}
        (out / "atlas-audit.json").write_text(json.dumps(atlas_audit, indent=2) + "\n")
    (out / "figure-design.json").write_text(
        json.dumps(
            {
                "ledger": __doc__,
                "exceptions": {
                    "facet-ylim-mismatch": "Outer metric panels and atlas panels use explicit independent scales"
                },
                "observed_at": snapshot["observed_at"],
                "snapshot_sha256": source_sha256,
                "groups": {label: len(r["groups"]) for label, r in snapshot["runs"].items()},
            },
            indent=2,
        )
        + "\n"
    )
    print("Audit issues:", {k: v for k, v in audit.items() if v}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--logs-repo", type=Path)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--figures", type=Path)
    parser.add_argument("--figure-skill", type=Path)
    parser.add_argument("--atlas", action="store_true")
    args = parser.parse_args()
    if args.collect:
        if not args.logs_repo:
            parser.error("--collect requires --logs-repo")
        collect(args.repo.resolve(), args.logs_repo.resolve(), args.data.resolve())
    if args.figures:
        if not args.figure_skill:
            parser.error("--figures requires --figure-skill")
        source = (args.data / "snapshot.json").read_bytes()
        plot(json.loads(source), args.figures, args.figure_skill, args.atlas, hashlib.sha256(source).hexdigest())


if __name__ == "__main__":
    main()
