# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Analyze fixed E55/E56/E57 log snapshots without models or cloud access.

Semantic ledger: curves show complete history records, never an inferred live
state. Each rollout point is a new training batch; each loss endpoint is a
pre-update observation in one reuse block. Validation bars compare the latest
shared checkpoint, use observed mean accuracy over n=16, and do not substitute pass@16.
Geometry ledger: common outer axes, shared cap-rate scales, masked future
points. MDPO targets differ across kappa; their absolute losses are not ranks.
Validation curves retain every observed checkpoint without smoothing; benchmark
macro averages weight the five benchmarks equally, not individual responses.
"""

import argparse
import hashlib
import json
import math
import statistics
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from examples.mirroropd.assets.paths import asset_name
from examples.mirroropd.research.core.plotting import load_style

RUNS = {"E55": "j5uaal3j", "E56": "zsnhqzts", "E57": "6j5knkmk"}
DATASETS = ["math500", "amc23", "aime24", "aime25", "aime26"]


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def analyze(logs):
    logs = logs.resolve()
    result = {
        "date": datetime.now(ZoneInfo("Asia/Shanghai")).date().isoformat(),
        "logs_head": subprocess.check_output(["git", "-C", str(logs), "rev-parse", "HEAD"], text=True).strip(),
        "analysis_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "runs": [],
        "scope": "Received log snapshots; not live training state",
    }
    for name, rid in RUNS.items():
        directories = list((logs / "derived/MirrorOPD-verl").glob("*-" + rid))
        if len(directories) != 1:
            raise ValueError(f"Expected one decoded directory for {rid}, got {len(directories)}")
        directory = directories[0]
        meta = json.loads((directory / "run.json").read_text())
        key = meta["cache_key"]
        raw = Path(key["source"])
        stamp = raw.stat()
        if (stamp.st_size, stamp.st_mtime_ns) != (key["bytes"], key["mtime_ns"]):
            raise ValueError(f"Stale decode for {rid}")
        if meta["status"] not in ("finished", "snapshot", "partial_snapshot"):
            raise ValueError(f"Invalid decode for {rid}: {meta['status']}")
        cfg = json.loads((directory / "config.json").read_text())
        history = read(directory / "history.jsonl")
        updates = [d for d in history if "distill/mdpo/objective" in d]
        outer = [d for d in history if "rollout/response_length/mean" in d]
        validation = [d for d in history if "eval/math500/acc/observed_mean" in d]
        assert [d["step/update"] for d in updates] == list(range(1, len(updates) + 1))
        assert len({d["_step"] for d in history}) == len(history)
        actor, rollout = cfg["actor_rollout_ref"]["actor"], cfg["actor_rollout_ref"]["rollout"]
        loss = cfg["distillation"]["distillation_loss"]
        pair = (
            asset_name(cfg["actor_rollout_ref"]["model"]["path"])
            + " <- "
            + asset_name(cfg["distillation"]["teacher_models"]["teacher_model"]["model_path"])
        )
        assert pair == "Qwen3-1.7B-Base <- Qwen3-30B-A3B-Instruct-2507"
        result["models"] = pair
        assert cfg["trainer"]["save_freq"] == -1 and cfg["trainer"]["test_freq"] == 10
        assert cfg["trainer"]["total_training_steps"] == 100 and cfg["trainer"]["resume_mode"] == "disable"
        assert cfg["data"]["apply_chat_template_kwargs"]["enable_thinking"] is None
        assert cfg["data"]["train_batch_size"] == actor["ppo_mini_batch_size"] == 64 and actor["ppo_epochs"] == 16
        assert loss["mirror"]["eta"] == 0.1 and not loss["use_policy_gradient"]
        assert loss["mirror"]["kappa"] == {"E55": 0, "E56": 0.5, "E57": 1}[name]
        assert cfg["data"]["max_response_length"] == 16384 and rollout["val_kwargs"]["max_tokens"] == 31744
        assert rollout["val_kwargs"]["n"] == 16
        assert loss["full_vocab"]["termination_token_ids"] == [151643, 151645]
        environment = read(directory / "environment.jsonl")
        gpu = next(r for r in environment if "gpu_type" in r)
        row = {
            "name": name,
            "run_id": rid,
            "url": f"https://wandb.ai/wncfht/MirrorOPD-verl/runs/{rid}",
            "source_status": meta["status"],
            "exit_code": meta["exit_code"],
            "source_bytes": meta["cache_key"]["bytes"],
            "raw_source": str(raw.relative_to(logs)),
            "sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
            "code_commit": meta["run"]["git"]["commit"],
            "snapshot_end_shanghai": datetime.fromtimestamp(
                max(d["_timestamp"] for d in history), ZoneInfo("Asia/Shanghai")
            ).isoformat(),
            "kappa": loss["mirror"]["kappa"],
            "eta": loss["mirror"]["eta"],
            "gpu_type": gpu["gpu_type"],
            "gpu_count": gpu["gpu_count"],
            "history_rows": len(history),
            "update_count": len(updates),
            "update_outer": max(d["step/outer"] for d in updates),
            "complete_outer": max(d["step/outer"] for d in outer),
            "validation_outers": [d["step/outer"] for d in validation],
            "config": {
                "seed": cfg["data"]["seed"],
                "optimizer": actor["optim"]["optimizer"],
                "fsdp_model_dtype": actor["fsdp_config"]["model_dtype"],
                "fsdp_dtype": actor["fsdp_config"]["dtype"],
                "forward_precision": actor["fsdp_config"]["forward_precision"],
                "train_cap": cfg["data"]["max_response_length"],
                "val_cap": rollout["val_kwargs"]["max_tokens"],
                "val_n": rollout["val_kwargs"]["n"],
                "lr": actor["optim"]["lr"],
                "train_temperature": rollout["temperature"],
                "train_top_p": rollout["top_p"],
                "val_temperature": rollout["val_kwargs"]["temperature"],
                "val_top_p": rollout["val_kwargs"]["top_p"],
                "concurrency": rollout["max_num_seqs"],
                "memory_utilization": rollout["gpu_memory_utilization"],
            },
            "bad_flags": {
                k: sum(d.get(k, 0) for d in updates)
                for k in (
                    "optim/actor/update_skipped",
                    "optim/actor/loss_nonfinite",
                    "optim/actor/grad_norm_preclip_nonfinite",
                )
            },
            "grad_max": max(d["optim/actor/grad_norm_preclip"] for d in updates),
            "grad_over_clip_count": sum(d["optim/actor/grad_norm_preclip"] > actor["grad_clip"] for d in updates),
            "updates": [],
            "outer": [],
            "fitting": [],
            "validation": [],
            "validation_timing": [
                {"outer": d["step/outer"], "seconds": d.get("perf/timing/validation_rollout_seconds")}
                for d in validation
            ],
        }
        for d in updates:
            assert all(
                math.isfinite(d[k])
                for k in ("distill/mdpo/objective", "distill/teacher_kl", "optim/actor/grad_norm_preclip")
            )
            row["updates"].append(
                {
                    k: v
                    for k, v in d.items()
                    if k
                    in (
                        "step/outer",
                        "step/update",
                        "distill/mdpo/objective",
                        "distill/target_kl",
                        "distill/target_log_z",
                        "distill/teacher_kl",
                        "distill/anchor_kl",
                        "optim/actor/grad_norm_preclip",
                    )
                    or "objective_position/" in k
                    and k.endswith("/mean")
                }
            )
        for d in outer:
            row["outer"].append(
                {
                    "step": d["step/outer"],
                    "length": d["rollout/response_length/mean"],
                    "cap_count": d["rollout/finish_reason/length_count"],
                    "responses": d["rollout/response_count"],
                    "cap_rate": d["rollout/finish_reason/length_count"] / d["rollout/response_count"],
                    "stop_count": d["rollout/finish_reason/stop_count"],
                    "score": d["rollout/score/mean"],
                    "actor_seconds": d["perf/timing/update_actor_seconds"],
                    "rollout_seconds": d["perf/timing/rollout_batch_seconds"],
                    "allocated_peak_gib": d["perf/actor/allocated_peak_gib"],
                    "outer_seconds": d["perf/timing/outer_step_seconds"],
                }
            )
        for o in sorted({d["step/outer"] for d in updates}):
            block = [d for d in updates if d["step/outer"] == o]
            if len(block) != 16:
                continue
            a, b = block[0], block[-1]
            row["fitting"].append(
                {
                    "outer": o,
                    "objective": [a["distill/mdpo/objective"], b["distill/mdpo/objective"]],
                    "teacher_kl": [a["distill/teacher_kl"], b["distill/teacher_kl"]],
                    "target_kl_drop_fraction": 1 - b["distill/target_kl"] / a["distill/target_kl"],
                    "anchor_first": a["distill/anchor_kl"],
                    "anchor_last": b["distill/anchor_kl"],
                    "position_first": {
                        k.split("/")[-2]: v for k, v in a.items() if "objective_position/" in k and k.endswith("/mean")
                    },
                    "position_last": {
                        k.split("/")[-2]: v for k, v in b.items() if "objective_position/" in k and k.endswith("/mean")
                    },
                }
            )
        for d in validation:
            for ds in DATASETS:
                prefix = f"eval/{ds}/"

                def get(key, prefix=prefix, record=d):
                    return record[prefix + key]

                n = get("generation/response_length/observed_count")
                assert n == (8000 if ds == "math500" else 640 if ds == "amc23" else 480)
                assert 0 <= get("acc/observed_mean") <= 1
                stop = get("generation/finish_reason/stop_count")
                cap = get("generation/finish_reason/length_count")
                assert stop + cap == n
                row["validation"].append(
                    {
                        "outer": d["step/outer"],
                        "dataset": ds,
                        "responses": n,
                        "accuracy": get("acc/observed_mean"),
                        "accuracy_ci95": [get("acc/mean_ci_low"), get("acc/mean_ci_high")],
                        "length": get("generation/response_length/mean"),
                        "median_length": get("generation/response_length/p50"),
                        "cap_count": cap,
                        "cap_rate": cap / n,
                        "stop_count": stop,
                    }
                )
        result["runs"].append(row)
    return result


def summarize(data):
    """Aggregate descriptive windows and validation macro/response-weighted means."""
    summaries = {}
    for r in data["runs"]:
        s = {
            k: r[k]
            for k in [
                "run_id",
                "source_status",
                "exit_code",
                "source_bytes",
                "sha256",
                "code_commit",
                "snapshot_end_shanghai",
                "kappa",
                "gpu_type",
                "history_rows",
                "update_count",
                "update_outer",
                "complete_outer",
                "validation_outers",
                "bad_flags",
                "grad_max",
                "config",
            ]
        }
        for name, rows in [("first7", r["outer"][:7]), ("last10", r["outer"][-10:]), ("all", r["outer"])]:
            s[name] = {
                "range": [rows[0]["step"], rows[-1]["step"]],
                "length": statistics.mean(x["length"] for x in rows),
                "cap_rate": sum(x["cap_count"] for x in rows) / sum(x["responses"] for x in rows),
                "score": statistics.mean(x["score"] for x in rows),
                "actor_hours": sum(x["actor_seconds"] for x in rows) / 3600,
                "rollout_hours": sum(x["rollout_seconds"] for x in rows) / 3600,
                "outer_hours": sum(x["outer_seconds"] for x in rows) / 3600,
            }
        s["last_fitting"] = r["fitting"][-1]
        s["fitting_summary"] = {
            "blocks": len(r["fitting"]),
            "objective_end_worse": sum(x["objective"][-1] > x["objective"][0] + 1e-10 for x in r["fitting"]),
            "teacher_end_worse": sum(x["teacher_kl"][-1] > x["teacher_kl"][0] + 1e-10 for x in r["fitting"]),
            "first_teacher": r["fitting"][0]["teacher_kl"],
            "last_teacher": r["fitting"][-1]["teacher_kl"],
        }
        s["validation_timing"] = r["validation_timing"]
        s["validation"] = {}
        for step in r["validation_outers"]:
            a = [v for v in r["validation"] if v["outer"] == step]
            s["validation"][step] = {
                "macro_acc": statistics.mean(x["accuracy"] for x in a),
                "macro_length": statistics.mean(x["length"] for x in a),
                "macro_cap_rate": statistics.mean(x["cap_rate"] for x in a),
                "response_weighted_acc": sum(x["accuracy"] * x["responses"] for x in a)
                / sum(x["responses"] for x in a),
                "response_weighted_cap_rate": sum(x["cap_count"] for x in a) / sum(x["responses"] for x in a),
                "datasets": {x["dataset"]: {k: v for k, v in x.items() if k not in ("outer", "dataset")} for x in a},
            }
        summaries[r["name"]] = s
    return summaries


def plots(data, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fv = load_style()
    fv.apply_rc("paper", fontsize=12)
    colors = {"E55": fv.HUES["vteal"], "E56": fv.HUES["vorange"], "E57": fv.HUES["vviolet"]}

    def save(fig, name):
        fig.canvas.draw()
        issues = fv.audit(fig, ignore=["facet-ylim-mismatch"])
        (out / f"{name}-audit.json").write_text(json.dumps(issues) + "\n")
        if any(fv.problem_severity(issue) == "error" for issue in issues):
            raise RuntimeError(f"Figure audit failed: {name}: {issues}")
        for ext in ("png", "pdf"):
            fig.savefig(out / f"{name}.{ext}", dpi=200)
        print(name, issues)
        plt.close(fig)

    max_outer = max(r["complete_outer"] for r in data["runs"])
    max_update = max(r["update_count"] for r in data["runs"])
    outer_padding = max(1, max_outer * 0.025)

    fig, axes = plt.subplots(2, 2, figsize=(9, 6.9))
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.25, top=0.83, wspace=0.35, hspace=0.50)
    specs = [
        ("length", 0.001, "Training length (k)", (0, 16.384)),
        ("cap_rate", 100, "Cap hits (%)", (-3, 100)),
        ("score", 100, "Training accuracy (%)", (0, 100)),
        ("actor_seconds", 1 / 60, "Actor update time (min)", None),
    ]
    for ax, (key, scale, label, limits) in zip(axes.flat, specs, strict=True):
        for run in data["runs"]:
            ax.plot(
                [r["step"] for r in run["outer"]],
                [r[key] * scale for r in run["outer"]],
                color=colors[run["name"]],
                marker="o",
                ms=3,
                label=f"{run['name']} (k={run['kappa']:g})",
            )
        ax.set(xlabel="Outer step", ylabel=label, xlim=(1 - outer_padding, max_outer + outer_padding))
        if limits:
            ax.set_ylim(*limits)
        ax.grid(alpha=0.2)
    fig.legend(
        *axes[0, 0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, 0.11), ncol=3, frameon=False
    )
    fig.suptitle(data["models"] + "\nE55–E57: received long-run snapshots", y=0.96)
    fig.text(
        0.5,
        0.035,
        "New training batches each outer; E56 runs on A800, E55/E57 on A100.\n"
        "Only received complete outer records are shown; hardware and generated token counts differ.",
        ha="center",
        fontsize=10,
    )
    save(fig, "training")

    fig, axes = plt.subplots(2, 3, figsize=(10, 6.7))
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.26, top=0.82, wspace=0.50, hspace=0.55)
    for col, run in enumerate(data["runs"]):
        x, y = [], []
        for block in run["fitting"]:
            u = [d for d in run["updates"] if d["step/outer"] == block["outer"]]
            x.extend([d["step/update"] for d in u] + [float("nan")])
            y.extend([d["distill/mdpo/objective"] for d in u] + [float("nan")])
        axes[0, col].plot(x, y, color=colors[run["name"]])
        axes[0, col].set(
            title=f"{run['name']}: MDPO objective",
            xlabel="Optimizer update",
            ylabel="Token mean",
            xlim=(0, max_update),
        )
        latest = max(b["outer"] for b in run["fitting"])
        for outer, style in sorted({1: ":", 7: "--", latest: "-"}.items()):
            b = next(x for x in run["fitting"] if x["outer"] == outer)
            keys = sorted(b["position_last"])
            axes[1, col].plot(
                range(5, 100, 10),
                [b["position_last"][k] for k in keys],
                ls=style,
                color=fv.NEUTRALS["vink"],
                label="Latest shown" if outer == latest else f"Outer {outer}",
            )
        axes[1, col].set(
            title=f"{run['name']}: latest complete block {latest}",
            xlabel="Response position (%)",
            ylabel="MDPO token mean",
        )
        for ax in axes[:, col]:
            ax.grid(alpha=0.2)
    fig.suptitle(data["models"] + "\nOptimization and MDPO position profiles", y=0.96)
    fig.legend(
        *axes[1, 1].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, 0.11), ncol=3, frameon=False
    )
    fig.text(
        0.5,
        0.035,
        "Each segment is one fixed batch; targets differ across kappa, so raw losses are not ranks.\n"
        "Bottom: before each selected outer's 16th update; these are not teacher-KL position curves.",
        ha="center",
        fontsize=10,
    )
    save(fig, "optimization")

    evaluated = [r for r in data["runs"] if r["validation"]]
    if not evaluated:
        return
    common = set.intersection(*(set(v["outer"] for v in r["validation"]) for r in evaluated))
    if not common:
        raise ValueError("No common validation checkpoint")
    checkpoint = max(common)
    fig, axes = plt.subplots(1, 3, figsize=(10, 4.3))
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.32, top=0.77, wspace=0.40)
    for ax, key, scale, ylabel, limit in zip(
        axes,
        ["accuracy", "length", "cap_rate"],
        [100, 0.001, 100],
        ["Accuracy (%)", "Mean length (k)", "Cap hits (%)"],
        [100, 32, 100],
        strict=True,
    ):
        width = 0.8 / len(evaluated)
        for index, run in enumerate(evaluated):
            name = run["name"]
            offset = (index - (len(evaluated) - 1) / 2) * width
            vals = {v["dataset"]: v for v in run["validation"] if v["outer"] == checkpoint}
            ax.bar(
                [i + offset for i in range(5)],
                [vals[ds][key] * scale for ds in DATASETS],
                width=width * 0.9,
                color=colors[name],
                label=name,
            )
        ax.set(
            xticks=range(5),
            xticklabels=["MATH", "AMC", "A24", "A25", "A26"],
            ylabel=ylabel,
            ylim=(0, limit),
            xlabel="Benchmark",
        )
        ax.tick_params(axis="x", labelsize=10)
        ax.grid(axis="y", alpha=0.2)
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.12),
        ncol=len(evaluated),
        frameon=False,
    )
    fig.suptitle(data["models"] + f"\nOuter-{checkpoint} full validation", y=0.96)
    fig.text(
        0.5,
        0.035,
        "630 questions x 16 samples; response cap 31,744; latest shared validation checkpoint.",
        ha="center",
        fontsize=10,
    )
    save(fig, "validation")

    labels = ["MATH-500", "AMC23", "AIME24", "AIME25", "AIME26", "Macro mean (5 benchmarks)"]
    for key, scale, ylabel, upper in (
        ("accuracy", 100, "Accuracy (%)", 100),
        ("length", 0.001, "Mean length (k tokens)", 32),
        ("cap_rate", 100, "Cap hits (%)", 100),
    ):
        fig, axes = plt.subplots(2, 3, figsize=(10.5, 7.6))
        fig.subplots_adjust(left=0.08, right=0.98, bottom=0.22, top=0.9, wspace=0.35, hspace=0.5)
        for index, ax in enumerate(axes.flat):
            maximum = 0
            for run in evaluated:
                by_step = {}
                for v in run["validation"]:
                    if index == 5 or v["dataset"] == DATASETS[index]:
                        by_step.setdefault(v["outer"], []).append(v[key])
                expected = 5 if index == 5 else 1
                if any(len(values) != expected for values in by_step.values()):
                    raise ValueError("Incomplete or duplicated validation benchmarks")
                steps = sorted(by_step)
                values = [sum(by_step[s]) / expected * scale for s in steps]
                maximum = max(maximum, *values)
                ax.plot(
                    steps,
                    values,
                    marker="o",
                    ms=3,
                    color=colors[run["name"]],
                    label=run["name"],
                )
            limit = max(10, math.ceil(maximum / 10) * 10 + 5) if key == "accuracy" else upper
            ax.set(title=labels[index], xlabel="Outer step", ylabel=ylabel, ylim=(-0.5, limit))
            ax.grid(alpha=0.2)
        fig.legend(
            *axes[0, 0].get_legend_handles_labels(),
            loc="lower center",
            bbox_to_anchor=(0.5, 0.1),
            ncol=3,
            frameon=False,
        )
        fig.text(
            0.5,
            0.035,
            "Recorded checkpoints only; macro means weight datasets equally.\nOne training seed per arm.",
            ha="center",
            fontsize=12,
            color=fv.NEUTRALS["vink"],
        )
        save(fig, "validation-" + key)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data-out", type=Path, help="Full input JSON; defaults to the log repository derived cache")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    data = analyze(args.logs)
    data_out = args.data_out or args.logs.resolve() / "derived/MirrorOPD-verl/e55-e57" / data["date"] / "data.json"
    data_out.parent.mkdir(parents=True, exist_ok=True)
    data_out.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    (args.out / "summary.json").write_text(json.dumps(summarize(data), ensure_ascii=False, indent=2) + "\n")
    plots(data, args.out)


if __name__ == "__main__":
    main()
