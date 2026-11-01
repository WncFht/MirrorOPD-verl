# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Snapshot partial Qwen arms; render only observations available at capture time.

Semantic ledger: a series is one learner trained on a specified rollout source.
Loss observations are pre-update, on that arm's own states; reuse 1 and 16 span
15 updates. Fixed risk is mean(sum RKL / 1024) across identical frozen questions,
not valid-token normalization. A completed update block is not a completed eval.
Geometry ledger: all training panels share outer 1..40 and one loss scale.
Risk facets share outer 0..40, with matching y scales for the same domain.
Timing excludes outer 1 and periodic readouts; it is not an equal-work speedup.
"""

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

from .report_qwen import dump, sha256, table

ARMS = ("S0", "S", "T", "LM", "PM", "TM")
LABELS = dict(
    S0="Initial rollout",
    S="Current rollout",
    T="Teacher rollout",
    LM="Logit fusion",
    PM="Prob. fusion",
    TM="Batch fusion",
)


class Snapshot:
    """Hash the exact bytes read, including the prefix length of growing JSONL files."""

    def __init__(self, repo):
        self.repo = repo
        self.sources = {}

    def read(self, path, lines=False):
        with path.open("rb") as stream:
            size = path.stat().st_size
            raw = stream.read(size)
        ignored = 0
        if lines and raw and not raw.endswith(b"\n"):
            end = raw.rfind(b"\n") + 1
            ignored = len(raw) - end
            raw = raw[:end]
        self.sources[str(path.relative_to(self.repo))] = dict(
            bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(), ignored_incomplete_tail_bytes=ignored
        )
        return [json.loads(line) for line in raw.splitlines() if line.strip()] if lines else json.loads(raw)


def collect(root, out):
    evidence = Snapshot(root.parents[1])
    started = datetime.now().astimezone().isoformat(timespec="seconds")
    recovery = root / "recovery-20261101"
    manifest = evidence.read(recovery / "manifest.json")
    result = dict(capture_started_at=started, revision=manifest["revision"], arms={}, fixed_denominator=1024)
    updates, outers, risks = [], [], []
    common_keys = None
    for arm in ARMS:
        run = recovery / "training" / f"prefix-science-Q-{arm}-qwenfix-v2"
        if not (run / "metrics.jsonl").exists():
            result["arms"][arm] = dict(updates=0, training_outers=0, status="not_started")
            continue
        records = [r["data"] for r in evidence.read(run / "metrics.jsonl", lines=True)]
        own = [r for r in records if "optim/actor/update_applied" in r]
        assert [r["step/optimizer"] for r in own] == list(range(1, len(own) + 1)), arm
        assert len(own) <= 640
        for r in own:
            step = int(r["step/optimizer"])
            assert r["step/outer"] == (step - 1) // 16 + 1
            assert r["optim/actor/update_applied"] == 1 and r["optim/actor/update_skipped"] == 0
            assert r["optim/actor/loss_nonfinite"] == 0
            loss, grad = r["optim/actor/distillation_loss"], r["optim/actor/grad_norm_preclip"]
            assert math.isfinite(loss) and math.isfinite(grad)
            updates.append(
                dict(
                    arm=arm,
                    outer=r["step/outer"],
                    update=step,
                    reuse=(step - 1) % 16 + 1,
                    loss=loss,
                    grad_norm_preclip=grad,
                )
            )
        own_outers = []
        for r in records:
            if "perf/timing/outer_step_seconds" not in r:
                continue
            row = dict(
                arm=arm,
                outer=int(r["step/outer"]),
                seconds=r["perf/timing/outer_step_seconds"],
                mean_length=r["rollout/response_length/mean"],
                response_tokens=r["rollout/response_tokens"],
                rollout_seconds=r["perf/timing/rollout_policy_batch_seconds"],
                update_seconds=r["perf/timing/update_actor_seconds"],
                teacher_seconds=r["perf/timing/teacher_transfer_seconds"] + r["perf/timing/teacher_forward_seconds"],
            )
            assert all(math.isfinite(v) for k, v in row.items() if k != "arm")
            assert r["rollout/policy/requests"] == 64
            own_outers.append(row)
        assert len({r["outer"] for r in own_outers}) == len(own_outers)
        outers.extend(own_outers)
        for directory in sorted((run / "science").glob("outer-*/fixed")):
            if not (directory / "complete.json").exists():
                continue
            receipt = evidence.read(directory / "complete.json")
            values = evidence.read(directory / "per-question.json")
            source = evidence.sources[str((directory / "per-question.json").relative_to(evidence.repo))]
            assert source["sha256"] == receipt["metrics_sha256"]
            assert receipt["inputs"]["horizon"] == 1024 and receipt["questions"] == len(values) == 256
            keys = {(v["source"], v["domain"], v["id"], v["length"]) for v in values}
            assert len(keys) == 256
            if common_keys is None:
                common_keys = keys
            assert keys == common_keys, (arm, directory, "Fixed-state membership changed")
            groups = defaultdict(list)
            for v in values:
                risk = v["stats"]["all"]["rkl"] / 1024
                assert math.isfinite(risk)
                groups[v["source"], v["domain"]].append(risk)
            outer = int(directory.parent.name.split("-")[1])
            parent = directory.parent / "complete.json"
            logged = evidence.read(parent)["metrics"] if parent.exists() else None
            for (bank, domain), values in groups.items():
                assert bank in {"S0", "Q"} and domain in {"math", "code", "if"}
                assert len(values) == (64 if domain == "math" else 32)
                risk = float(np.mean(values))
                if logged is not None:
                    assert abs(risk - logged[f"science/fixed/{bank}/{domain}/rkl"]) < 1e-10
                risks.append(dict(arm=arm, outer=outer, bank=bank, domain=domain, rkl=risk))
        audit = recovery / "stages" / f"science-{arm}-audit.json"
        audit_passed = evidence.read(audit)["passed"] if audit.exists() else False
        final = run / "science/outer-040/accuracy/complete.json"
        final_complete = final.exists()
        if final_complete:
            receipt = evidence.read(final)
            assert receipt["questions"] == 630 and receipt["records"] == 2520
        ordinary = [r for r in own_outers if r["outer"] > 1 and r["outer"] % 5]
        result["arms"][arm] = dict(
            updates=len(own),
            training_outers=len(own) // 16,
            completed_outer_logs=len(own_outers),
            status="audited_complete" if audit_passed else "finalizing" if len(own) == 640 else "training",
            final_accuracy_complete=final_complete,
            final_audit_passed=audit_passed,
            last_loss=own[-1]["optim/actor/distillation_loss"] if own else None,
            ordinary_outer_median_seconds=float(np.median([r["seconds"] for r in ordinary])) if ordinary else None,
            ordinary_mean_response_tokens=float(np.mean([r["mean_length"] for r in ordinary])) if ordinary else None,
            fixed_rkl_outers=sorted({r["outer"] for r in risks if r["arm"] == arm}),
            update_health="all recorded updates applied and finite",
        )
    result["capture_finished_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    table(out / "updates.csv", updates)
    table(out / "outers.csv", outers)
    table(out / "fixed-risk.csv", risks)
    dump(out / "summary.json", result)
    dump(out / "sources.json", dict(files=evidence.sources, script_sha256=sha256(Path(__file__))))
    return result, updates, outers, risks


def figures(summary, updates, outers, risks, destination, style):
    import matplotlib

    matplotlib.use("Agg")
    sys.path.insert(0, str(style))
    import fvstyle as fv
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fv.apply_rc("paper")
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10, "legend.fontsize": 9})
    colors = {
        k: fv.HUES[v]
        for k, v in dict(S0="vviolet", S="vblue", T="vgray", LM="vteal", PM="vorange", TM="vviolet").items()
    }
    audits = {}
    captured = summary["capture_finished_at"].replace("T", " ")
    present = [a for a in ARMS if summary["arms"][a]["updates"]]

    def finish(fig, name, caption):
        fig.text(0.5, 0.02, caption, ha="center", color=fv.NEUTRALS["baseline"], fontsize=9)
        fig.canvas.draw()
        problems = fv.audit(fig, print_slot=fig.get_figwidth())
        audits[name] = fv.format_audit_lines(problems)
        for ext in ["png", "pdf", "svg"]:
            fig.savefig(destination / f"{name}.{ext}", dpi=180, facecolor=fv.NEUTRALS["paper"])
        plt.close(fig)
        if any(fv.problem_severity(p) == "error" for p in problems):
            raise ValueError(audits[name])

    fig, axes = plt.subplots(2, 3, figsize=(12, 7.6))
    upper = max(r["loss"] for r in updates) * 1.12
    for ax, arm in zip(axes.flat, ARMS, strict=True):
        own = [r for r in updates if r["arm"] == arm]
        n = summary["arms"][arm]["training_outers"]
        ax.set(
            xlim=(0, 41),
            ylim=(0, upper),
            xlabel="Training outer",
            ylabel="Recorded RKL loss",
            title=f"{arm}: {LABELS[arm]} ({n}/40)",
        )
        ax.grid(axis="y", alpha=0.2)
        for reuse, ls in [(1, "-"), (16, "--")]:
            rows = [r for r in own if r["reuse"] == reuse]
            if rows:
                ax.plot(
                    [r["outer"] for r in rows],
                    [r["loss"] for r in rows],
                    color=colors[arm],
                    ls=ls,
                    lw=1.7,
                    marker="o",
                    ms=2.7,
                    label="_nolegend_",
                )
        if not own:
            ax.text(0.5, 0.5, "Not started at snapshot", transform=ax.transAxes, ha="center", color=colors[arm])
    handles = [
        Line2D([], [], color=fv.NEUTRALS["vink"], ls=ls, label=f"Before reuse update {n}")
        for n, ls in [(1, "-"), (16, "--")]
    ]
    handles += [Line2D([], [], color=colors[a], label=f"{a}: {LABELS[a]}") for a in present]
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.99), frameon=False)
    fig.subplots_adjust(top=0.85, bottom=0.13, hspace=0.55, wspace=0.3)
    finish(fig, "loss", f"Own rollout states; no cross-arm loss ranking. Snapshot: {captured}")

    fig, axes = plt.subplots(2, 3, figsize=(12, 7.6))
    for row, bank in enumerate(["S0", "Q"]):
        for col, domain in enumerate(["math", "code", "if"]):
            ax = axes[row, col]
            for arm in present:
                rs = sorted(
                    [r for r in risks if (r["arm"], r["bank"], r["domain"]) == (arm, bank, domain)],
                    key=lambda r: r["outer"],
                )
                if rs:
                    ax.plot(
                        [r["outer"] for r in rs],
                        [r["rkl"] for r in rs],
                        color=colors[arm],
                        marker="o",
                        ms=3,
                        lw=1.5,
                        ls=":" if arm == "TM" else "-",
                        label=f"{arm}: {LABELS[arm]}",
                    )
            maximum = max(r["rkl"] for r in risks)
            ax.set(
                xlim=(-1, 41),
                ylim=(0.1, maximum * 1.2),
                yscale="log",
                xlabel="Training outer",
                ylabel="Fixed-H RKL (log scale)",
                title=f"{'Initial' if bank == 'S0' else 'Teacher'} bank / {domain.upper()}",
            )
            ax.set_yticks([0.1, 0.2, 0.5, 1.0], labels=["0.1", "0.2", "0.5", "1.0"])
            ax.minorticks_off()
            ax.grid(axis="y", alpha=0.2)
    handles = [
        Line2D([], [], color=colors[a], marker="o", ls=":" if a == "TM" else "-", label=f"{a}: {LABELS[a]}")
        for a in present
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.99), frameon=False)
    fig.subplots_adjust(top=0.83, bottom=0.13, hspace=0.55, wspace=0.3)
    finish(
        fig, "fixed-risk", f"Same frozen states; H1024; shared log scale; completed readouts only. Snapshot: {captured}"
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for arm in present:
        rs = [r for r in outers if r["arm"] == arm]
        axes[0].plot(
            [r["outer"] for r in rs],
            [r["mean_length"] for r in rs],
            color=colors[arm],
            lw=1.5,
            label=f"{arm}: {LABELS[arm]}",
        )
        rs = [r for r in rs if r["outer"] > 1 and r["outer"] % 5]
        axes[1].plot(
            [r["outer"] for r in rs],
            [r["seconds"] for r in rs],
            color=colors[arm],
            marker="o",
            ms=3,
            lw=1,
            label=f"{arm}: {LABELS[arm]}",
        )
    for ax, ylabel, title in zip(
        axes,
        ["Mean response tokens", "Ordinary outer seconds"],
        ["Training response length", "Excludes first outer and readouts"],
        strict=True,
    ):
        ax.set(xlim=(0, 41), ylim=(0, None), xlabel="Training outer", ylabel=ylabel, title=title)
        ax.grid(axis="y", alpha=0.2)
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0), frameon=False)
    fig.subplots_adjust(top=0.74, bottom=0.19, wspace=0.3)
    finish(fig, "workload", "Different generated workloads; timing is not an equal-work backend comparison.")
    dump(destination / "figure-audit.json", audits)


def load_rows(path):
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    return [{k: v if k in {"arm", "bank", "domain"} else float(v) for k, v in r.items()} for r in rows]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--figures", type=Path, required=True)
    parser.add_argument("--style", type=Path, required=True)
    parser.add_argument("--replot", action="store_true", help="Render saved CSVs without reading live runs")
    args = parser.parse_args()
    args.figures.mkdir(parents=True, exist_ok=False)
    if args.replot:
        values = (
            json.loads((args.out / "summary.json").read_text()),
            *(load_rows(args.out / name) for name in ["updates.csv", "outers.csv", "fixed-risk.csv"]),
        )
    else:
        if args.root is None:
            parser.error("--root is required for a new snapshot")
        args.out.mkdir(parents=True, exist_ok=False)
        values = collect(args.root.resolve(), args.out)
    figures(*values, args.figures, args.style)
    print(json.dumps(values[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
