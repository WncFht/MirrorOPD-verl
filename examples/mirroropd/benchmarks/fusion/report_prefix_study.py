# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Export observed formal-study updates and render only real, available learning curves.

Semantic ledger: a row is one recorded optimizer attempt; a curve is one prefix arm.
Geometry ledger: x is applied optimizer step (or actual cumulative loss-token count),
y is observed pre-update RKL. Missing arms produce no synthetic points or zero curves.
Teacher pairs use separate panels; colors identify arms consistently in every figure.
The outer-level panel compares observations 1 and 16 of the same fixed batch;
it does not compare different batches or claim a post-sixteenth-update evaluation.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ARMS = {"S0": "Initial student", "S": "Current student", "LM": "Logit mix", "TM": "Trajectory mix", "T": "Teacher"}
STUDY_COMMIT = "91674e68941c1be634cf62231711a1df753e28a9"


def completion_matches(audit, pair, arm, exit_code, source_hashes):
    """Require a successful formal audit of the exact metrics/config being exported."""
    witnesses = {item["path"]: item["sha256"] for item in audit.get("witnesses", [])}
    return (
        exit_code == 0
        and audit.get("controller_exit_code") == 0
        and audit.get("passed") is True
        and audit.get("formal_complete") is True
        and audit.get("mode") == "formal"
        and (audit.get("pair"), audit.get("arm")) == (pair, arm)
        and audit.get("source_commit") == STUDY_COMMIT
        and audit.get("applied_updates") == 640
        and audit.get("errors") == []
        and len(source_hashes) == 2
        and all(witnesses.get(path) == digest for path, digest in source_hashes.items())
    )


def summarize_outers(rows):
    """Compare observations 1 and 16 on one fixed batch, spanning 15 updates."""
    groups = {}
    for row in rows:
        groups.setdefault((row["pair"], row["arm"], row["outer"]), []).append(row)
    result = []
    for (pair, arm, outer), batch in groups.items():
        complete = (
            len(batch) == 16
            and [r["reuse_index"] for r in batch] == list(range(1, 17))
            and all(r["valid_update"] for r in batch)
        )
        first, last = (batch[0]["rkl"], batch[-1]["rkl"]) if complete else (None, None)
        counts = [r["loss_tokens"] for r in batch]
        result.append(
            {
                "pair": pair,
                "arm": arm,
                "outer": outer,
                "recorded_updates": len(batch),
                "complete_reuse16": complete,
                "first_rkl": first,
                "sixteenth_rkl": last,
                "within_batch_reduction_pct": 100 * (first - last) / first if first and first > 0 else None,
                "reused_loss_tokens": sum(counts) if all(c is not None for c in counts) else None,
                "batch_loss_tokens": counts[0] if complete and len(set(counts)) == 1 else None,
            }
        )
    return result


def load_snapshot(directory):
    """Read a frozen export without touching its live experiment paths."""
    summary = json.loads((directory / "status.json").read_text())
    raw = (directory / "updates.csv").read_bytes()
    expected = summary.get("updates_csv_sha256")
    if expected is not None and hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("Frozen update CSV does not match its recorded hash")
    rows = []
    for record in csv.DictReader(raw.decode().splitlines()):
        if record["valid_update"] not in ("True", "False"):
            raise ValueError("Invalid update-validity marker in snapshot")
        row = {}
        for key, value in record.items():
            row[key] = (
                value
                if key in ("pair", "arm", "run")
                else value == "True"
                if key == "valid_update"
                else float(value)
                if value
                else None
            )
        rows.append(row)
    if len(rows) != summary["recorded_update_rows"]:
        raise ValueError("Frozen snapshot row count differs from status")
    return rows, summary


def snapshot(workspace, out):
    out.mkdir(parents=True, exist_ok=False)
    rows, states, sources = [], [], []
    repo = workspace / "MirrorOPD-verl-vllm-fusion"
    root = workspace / "runs/vllm-fusion-20261029"
    for pair in ("J", "P"):
        control = root / f"formal-prefix-study-{pair}-01"
        for arm in ARMS:
            name = f"prefix-formal-{pair}-{arm}-study40x16-v1"
            run = repo / "outputs" / name
            path = run / "metrics.jsonl"
            observations = []
            source_hashes = {}
            if path.exists():
                raw = path.read_bytes()
                source_hashes[str(path.resolve())] = hashlib.sha256(raw).hexdigest()
                complete = raw if not raw or raw.endswith(b"\n") else raw[: raw.rfind(b"\n") + 1]
                observations = [json.loads(line)["data"] for line in complete.splitlines()]
                sources.append(
                    {
                        "path": str(path),
                        "snapshot_sha256": hashlib.sha256(complete).hexdigest(),
                        "bytes": len(complete),
                        "incomplete_tail_omitted": len(complete) != len(raw),
                    }
                )
            updates = [r for r in observations if "optim/actor/update_applied" in r]
            if updates:
                config_path = run / "resolved_config.yaml"
                raw_config = config_path.read_bytes()
                source_hashes[str(config_path.resolve())] = hashlib.sha256(raw_config).hexdigest()
                config = yaml.safe_load(raw_config)
                trainer, actor = config["trainer"], config["actor_rollout_ref"]["actor"]
                policy = config["rollout_policy"]
                expected_policy = {
                    "S0": ("single", "student", "initial", 0),
                    "S": ("trajectory_mixture", "student", "current", 0),
                    "LM": ("token_mixture", "student", "current", 0.5),
                    "TM": ("trajectory_mixture", "student", "current", 0.5),
                    "T": ("single", "teacher", "current", 1),
                }[arm]
                if tuple(policy[k] for k in ["mode", "source", "student", "teacher_weight"]) != expected_policy:
                    raise ValueError(f"Wrong prefix arm: {name}")
                mirror = config["distillation"]["distillation_loss"]["mirror"]
                if (
                    trainer["total_training_steps"] != 40
                    or trainer["val_before_train"]
                    or trainer["test_freq"] != -1
                    or trainer["save_freq"] != -1
                    or config["data"]["train_batch_size"] != 64
                    or actor["ppo_epochs"] != 16
                    or actor["ppo_mini_batch_size"] != 64
                    or (mirror["objective"], mirror["estimator"], mirror["kappa"], mirror["eta"])
                    != ("mdpo", "full_vocab", 0, 1)
                ):
                    raise ValueError(f"Refusing nonformal or mismatched learning settings: {name}")
                sources.append({"path": str(config_path), "sha256": hashlib.sha256(raw_config).hexdigest()})
            cumulative, valid = 0, True
            for index, row in enumerate(updates, 1):
                applied = row.get("optim/actor/update_applied") == 1
                loss = row.get("optim/actor/loss")
                rkl = row.get("optim/actor/distillation_loss")
                grad = row.get("optim/actor/grad_norm_preclip")
                finite = all(isinstance(v, int | float) and math.isfinite(v) for v in (loss, rkl, grad))
                good = (
                    applied
                    and finite
                    and row.get("step/optimizer") == index
                    and row.get("step/update") == index
                    and row.get("step/outer") == (index - 1) // 16 + 1
                    and row.get("rollout/data_reuse/update_index") == (index - 1) % 16 + 1
                    and row.get("optim/actor/execution_known") == 1
                    and row.get("optim/actor/update_skipped") == 0
                )
                valid &= good
                counts = [
                    v
                    for k, v in row.items()
                    if k.startswith("distill/full_kl_position/") and k.endswith("/token_count")
                ]
                tokens = sum(counts) if len(counts) == 10 and all(math.isfinite(x) and x >= 0 for x in counts) else None
                if cumulative is not None:
                    cumulative = cumulative + tokens if tokens is not None else None
                rows.append(
                    {
                        "pair": pair,
                        "arm": arm,
                        "run": name,
                        "outer": row.get("step/outer"),
                        "optimizer_step": row.get("step/optimizer"),
                        "reuse_index": row.get("rollout/data_reuse/update_index"),
                        "loss": loss if finite else None,
                        "rkl": rkl if finite else None,
                        "grad_norm_preclip": grad if finite else None,
                        "valid_update": good,
                        "loss_tokens": tokens,
                        "cumulative_loss_tokens": cumulative,
                    }
                )
            exit_file, audit_file = control / f"{arm}-exit.json", control / f"{arm}-audit.json"
            exit_record = json.loads(exit_file.read_text()) if exit_file.exists() else {}
            exit_code = exit_record.get("exit_code")
            audit = json.loads(audit_file.read_text()) if audit_file.exists() else None
            for evidence in (exit_file, audit_file):
                if evidence.exists():
                    sources.append({"path": str(evidence), "sha256": hashlib.sha256(evidence.read_bytes()).hexdigest()})
            audited = (
                completion_matches(audit, pair, arm, exit_code, source_hashes)
                and exit_record.get("run") == name
                and valid
                and len(updates) == 640
                if audit is not None
                else False
            )
            status = (
                "accepted"
                if audited
                else "failed"
                if exit_code not in (None, 0) or not valid
                else "audit_mismatch"
                if audit is not None
                else "finished_pending_audit"
                if exit_code == 0
                else "running"
                if run.exists()
                else "queued"
            )
            states.append(
                {
                    "pair": pair,
                    "arm": arm,
                    "run": name,
                    "status": status,
                    "recorded_updates": sum(r.get("optim/actor/update_applied") == 1 for r in updates),
                    "target_updates": 640,
                    "recorded_outers": max((r.get("step/outer", 0) for r in updates), default=0),
                    "exit_code": exit_code,
                    "formal_complete": audited,
                }
            )
    fields = [
        "pair",
        "arm",
        "run",
        "outer",
        "optimizer_step",
        "reuse_index",
        "loss",
        "rkl",
        "grad_norm_preclip",
        "valid_update",
        "loss_tokens",
        "cumulative_loss_tokens",
    ]
    with (out / "updates.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Formal 40x16 study only; live partial snapshot, no extrapolation",
        "states": states,
        "sources": sources,
        "all_ten_complete": len(states) == 10 and all(s["formal_complete"] for s in states),
        "recorded_update_rows": len(rows),
        "invalid_update_rows": sum(not r["valid_update"] for r in rows),
        "updates_csv_sha256": hashlib.sha256((out / "updates.csv").read_bytes()).hexdigest(),
    }
    (out / "status.json").write_text(json.dumps(summary, indent=2) + "\n")
    return rows, summary


def render(rows, summary, directory, style_dir):
    if not rows:
        return []
    sys.path.insert(0, str(style_dir))
    import fvstyle as fv
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    fv.apply_rc("paper", fontsize=10)
    palette = fv.series_colors("paper", 4)
    colors = dict(zip(["S", "LM", "TM", "T"], palette, strict=True))
    colors["S0"] = fv.NEUTRALS["baseline"]
    directory.mkdir(parents=True, exist_ok=False)
    outer_rows = summarize_outers(rows)
    with (directory / "outers.csv").open("w") as stream:
        writer = csv.DictWriter(stream, list(outer_rows[0]))
        writer.writeheader()
        writer.writerows(outer_rows)
    outputs = []
    for name, field, label in [
        ("formal-rkl-by-update", "rkl", "Full-vocab RKL (pre-update)"),
        ("formal-gradient-by-update", "grad_norm_preclip", "Preclip gradient norm"),
        ("formal-within-batch-reduction", "within_batch_reduction_pct", "Within-batch RKL reduction (%)"),
    ]:
        by_outer = field == "within_batch_reduction_pct"
        plot_rows = [r for r in outer_rows if r["complete_reuse16"]] if by_outer else rows
        if not plot_rows:
            continue
        xfield = "outer" if by_outer else "optimizer_step"
        fig, axes = plt.subplots(1, 2, figsize=(9, 4.3), sharex=True, sharey=True)
        handles = {}
        for ax, pair, title in zip(axes, ("J", "P"), ("JustRL-1.5B", "Polaris-7B-Preview"), strict=True):
            for arm in ARMS:
                data = [r for r in plot_rows if r["pair"] == pair and r["arm"] == arm]
                if not data:
                    continue
                (line,) = ax.plot(
                    [r[xfield] for r in data],
                    [r[field] if r.get("valid_update", True) else math.nan for r in data],
                    color=colors[arm],
                    label=ARMS[arm],
                    linewidth=1.4,
                    marker="o" if by_outer else None,
                    markersize=3,
                )
                handles.setdefault(arm, line)
            ax.set_title(title)
            ax.set_xlabel("Outer (observations 1 to 16)" if by_outer else "Successful optimizer step")
            ax.grid(axis="y", alpha=0.18)
        axes[0].set_ylabel(label)
        maximum = max(r[xfield] or 0 for r in plot_rows)
        axes[0].set_xlim((0.5, maximum + 0.5) if by_outer else (0, max(1, maximum)))
        if by_outer:
            if maximum == 1:
                axes[0].set_xticks([1])
            else:
                axes[0].xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
        if field == "rkl":
            axes[0].set_ylim(bottom=0)
        ordered = [arm for arm in ARMS if arm in handles]
        fig.legend(
            [handles[a] for a in ordered],
            [ARMS[a] for a in ordered],
            loc="lower center",
            bbox_to_anchor=(0.5, 0.09),
            ncol=min(3, len(ordered)),
            frameon=False,
        )
        fig.text(
            0.5,
            0.025,
            "Observed formal-run updates only; target 640/run; pending arms omitted",
            ha="center",
            color=fv.NEUTRALS["baseline"],
            fontsize=8,
        )
        fig.tight_layout(rect=(0, 0.2, 1, 1))
        fig.canvas.draw()
        problems = fv.audit(fig, print_slot=fig.get_figwidth())
        audit = list(fv.format_audit_lines(problems))
        (directory / f"{name}.audit.txt").write_text(
            "\n".join(audit) + "\n" if audit else "PASS: fv.audit(fig) returned no issues.\n"
        )
        if any(fv.problem_severity(p) == "error" for p in problems):
            raise RuntimeError(f"Figure audit failed: {name}: {audit}")
        for suffix in ("png", "pdf", "svg"):
            path = directory / f"{name}.{suffix}"
            fig.savefig(path, dpi=180, facecolor=fv.NEUTRALS["paper"])
            outputs.append(str(path))
        plt.close(fig)
    (directory / "figure-data.json").write_text(
        json.dumps(
            {
                "snapshot": summary["captured_at_utc"],
                "rows": rows,
                "scope": "One seed per arm; partial curves are not final comparisons",
            },
            indent=2,
        )
        + "\n"
    )
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--workspace", type=Path)
    source.add_argument("--replot", action="store_true", help="Read --snapshot without consulting live runs")
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--figures", type=Path, required=True)
    parser.add_argument("--style-dir", type=Path, default=os.environ.get("FIGURE_VIZ_MPL"))
    args = parser.parse_args()
    rows, summary = load_snapshot(args.snapshot) if args.replot else snapshot(args.workspace.resolve(), args.snapshot)
    if rows and args.style_dir is None:
        parser.error("Set --style-dir or FIGURE_VIZ_MPL to render actual observations")
    outputs = render(rows, summary, args.figures, args.style_dir) if rows else []
    print(json.dumps({"update_rows": len(rows), "all_complete": summary["all_ten_complete"], "figures": outputs}))


if __name__ == "__main__":
    main()
