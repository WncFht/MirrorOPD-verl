# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Analyze fixed E55/E56/E57 log snapshots without models or cloud access.

Semantic ledger: curves show complete history records, never an inferred live
state. Each rollout point is a new training batch; each loss endpoint is a
pre-update observation in one reuse block. Validation bars compare outer 10
only, use observed mean accuracy over n=16, and do not substitute pass@16.
Geometry ledger: common outer axes, shared cap-rate scales, masked future
points. MDPO targets differ across kappa; their absolute losses are not ranks.
"""

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from examples.mirroropd.research.core.plotting import load_style

RUNS = {"E55": "j5uaal3j", "E56": "zsnhqzts", "E57": "6j5knkmk"}
DATASETS = ["math500", "amc23", "aime24", "aime25", "aime26"]


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def analyze(logs):
    result = {"date": "2026-10-30", "runs": [], "scope": "Received partial snapshots; not live training state"}
    for name, rid in RUNS.items():
        directory = next((logs / "derived/MirrorOPD-verl").glob("*-" + rid))
        meta = json.loads((directory / "run.json").read_text())
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
            Path(cfg["actor_rollout_ref"]["model"]["path"]).name
            + " <- "
            + Path(cfg["distillation"]["teacher_models"]["teacher_model"]["model_path"]).name
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
                    "stop_count": d["rollout/finish_reason/stop_count"],
                    "score": d["rollout/score/mean"],
                    "actor_seconds": d["perf/timing/update_actor_seconds"],
                    "rollout_seconds": d["perf/timing/rollout_batch_seconds"],
                    "allocated_peak_gib": d["perf/actor/allocated_peak_gib"],
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
        for ext in ("png", "pdf"):
            fig.savefig(out / f"{name}.{ext}", dpi=200)
        print(name, issues)
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(9, 6.9))
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.25, top=0.83, wspace=0.35, hspace=0.50)
    specs = [
        ("length", 0.001, "Training length (k)", (0, 16.384)),
        ("cap_count", 100 / 64, "Cap hits (%)", (-3, 100)),
        ("score", 100, "Training accuracy (%)", (0, 40)),
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
        ax.set(xlabel="Outer step", ylabel=label, xlim=(0.6, 20.4), xticks=[1, 5, 10, 15, 20])
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
        "E56 outer-20 optimizer records exist; its outer aggregate/validation is absent.",
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
            xlim=(0, 320),
            xticks=[0, 160, 320],
        )
        for outer, style in sorted({1: ":", 7: "--", run["update_outer"]: "-"}.items()):
            b = next(x for x in run["fitting"] if x["outer"] == outer)
            keys = sorted(b["position_last"])
            axes[1, col].plot(
                range(5, 100, 10),
                [b["position_last"][k] for k in keys],
                ls=style,
                color=fv.NEUTRALS["vink"],
                label="Latest shown" if outer == run["update_outer"] else f"Outer {outer}",
            )
        axes[1, col].set(
            title=f"{run['name']}: latest outer {run['update_outer']}",
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
        for offset, name in [(-0.18, "E56"), (0.18, "E57")]:
            run = next(r for r in data["runs"] if r["name"] == name)
            vals = {v["dataset"]: v for v in run["validation"] if v["outer"] == 10}
            ax.bar(
                [i + offset for i in range(5)],
                [vals[ds][key] * scale for ds in DATASETS],
                width=0.34,
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
        *axes[0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, 0.12), ncol=2, frameon=False
    )
    fig.suptitle(data["models"] + "\nOuter-10 full validation: E56 versus E57", y=0.96)
    fig.text(
        0.5,
        0.035,
        "630 questions x 16 samples; response cap 31,744. E55 has no validation record yet.",
        ha="center",
        fontsize=10,
    )
    save(fig, "validation")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    data = analyze(args.logs)
    (args.out / "data.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    plots(data, args.out)


if __name__ == "__main__":
    main()
