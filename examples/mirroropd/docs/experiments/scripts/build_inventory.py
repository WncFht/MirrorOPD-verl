#!/usr/bin/env python3
"""Build the current experiment register directly from decoded logs and a cloud index.

Only formal Math/Coding/IF runs enter the register. Binary decoding is delegated
to wandb-logs before reading its cache; this does not contact W&B, alter report
snapshots, or infer an evaluation at the last update.
"""

import argparse
import csv
import json
import re
import subprocess
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def select(config, key):
    for part in key.split("."):
        if not isinstance(config, dict):
            return None
        config = config.get(part)
    return config


def is_full_kl(config):
    mode = select(config, "distillation.distillation_loss.loss_mode")
    return mode == "full_reverse_kl" or (
        mode == "mirror"
        and select(config, "distillation.distillation_loss.mirror.objective") == "mdpo"
        and select(config, "distillation.distillation_loss.mirror.estimator") == "full_vocab"
        and select(config, "distillation.distillation_loss.mirror.eta") == 1
    )


def method_label(config):
    if is_full_kl(config):
        return "Full-KL"
    if select(config, "distillation.distillation_loss.loss_mode") == "mirror":
        m = select(config, "distillation.distillation_loss.mirror") or {}
        return f"{str(m.get('objective')).upper()}/{m.get('estimator')} (κ={m.get('kappa')}, η={m.get('eta')})"
    return "/".join(
        str(select(config, "distillation.distillation_loss." + key)) for key in ("advantage_source", "policy_loss_mode")
    )


def family(name):
    if re.match(r"math-opd-E(44|45|46|47)-", name):
        return "mirror-comparison"
    if name.startswith("math-opd-custom-") or re.match(r"math-opd-E\d+-(mdpo|bpo|sampled|full_kl)-", name):
        return "method-overrides"
    match = re.match(r"math-opd-E(\d+)-", name)
    if match:
        number = int(match[1])
        return (
            "sampled-ablation"
            if number <= 12
            else "full-kl-comparison"
            if number <= 17
            else "equal-update-budget"
            if number <= 20
            else "failure-reproduction"
        )
    if name.startswith(("math-full-", "code-full-", "if-full-")):
        return "other-runs"
    return None


CONFIG_FIELDS = {
    "student": "actor_rollout_ref.model.path",
    "teacher": "distillation.teacher_models.teacher_model.model_path",
    "train_files": "data.train_files",
    "validation_files": "data.val_files",
    "seed": "data.seed",
    "loss_mode": "distillation.distillation_loss.loss_mode",
    "advantage_source": "distillation.distillation_loss.advantage_source",
    "policy_loss_mode": "distillation.distillation_loss.policy_loss_mode",
    "backend": "distillation.distillation_loss.full_vocab.backend",
    "lr": "actor_rollout_ref.actor.optim.lr",
    "reuse_epochs": "actor_rollout_ref.actor.ppo_epochs",
    "batch_size": "data.train_batch_size",
    "mini_batch_size": "actor_rollout_ref.actor.ppo_mini_batch_size",
    "rollout_n": "actor_rollout_ref.rollout.n",
    "planned_outer": "trainer.total_training_steps",
    "prompt_cap": "data.max_prompt_length",
    "response_cap": "data.max_response_length",
    "eval_n": "actor_rollout_ref.rollout.val_kwargs.n",
    "eval_response_cap": "actor_rollout_ref.rollout.val_kwargs.max_tokens",
    "eval_temperature": "actor_rollout_ref.rollout.val_kwargs.temperature",
    "eval_top_p": "actor_rollout_ref.rollout.val_kwargs.top_p",
    "eval_freq": "trainer.test_freq",
    "eval_before_train": "trainer.val_before_train",
    "metrics_format": "trainer.metrics_format",
    "execution_mode": "distillation.execution_mode",
    "gpus": "trainer.n_gpus_per_node",
}
DATASETS = {"math500": "MATH-500", "amc23": "AMC23", "aime25": "AIME25", "livecodebench": "livecodebench"}


def inspect(directory, cloud):
    meta = read(directory / "run.json")
    config = read(directory / "config.json")
    identity = meta["run"]
    run_id = identity["run_id"]
    name = identity.get("display_name", cloud.get("name", ""))
    result = {
        "run_id": run_id,
        "name": name,
        "family": family(name),
        "method": method_label(config),
        "cloud_state": cloud.get("state"),
        "local_status": meta["status"],
        "exit_code": meta.get("exit_code"),
        "start_time_utc": identity.get("start_time"),
        "history_rows": meta["history_rows"],
        "source_directory": directory.name,
        "recorded_git": identity.get("git"),
        "max_outer": None,
        "completed_outer": None,
        "optimizer_updates": None,
        "applied_update_records": None,
        "skipped_update_records": None,
        "last_eval_outer": None,
        "last_eval_optimizer": None,
        **{key: select(config, value) for key, value in CONFIG_FIELDS.items()},
        **{f"{key}_mean_pct": None for key in DATASETS},
    }
    updates = {}
    outer = optimizer = None
    with (directory / "history.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            outer = row.get("step/outer", row.get("training/global_step", outer))
            optimizer = row.get("step/optimizer", row.get("training/actor_optimizer_step", optimizer))
            if outer is not None:
                result["max_outer"] = max(result["max_outer"] or 0, outer)
            if optimizer is not None:
                result["optimizer_updates"] = max(result["optimizer_updates"] or 0, optimizer)
            if "perf/timing/outer_step_seconds" in row or "timing_s/step" in row:
                result["completed_outer"] = outer
            if "optim/actor/update_applied" in row:
                update = row["step/update"]
                signature = (row["optim/actor/update_applied"], row.get("optim/actor/update_skipped"))
                if update in updates and updates[update] != signature:
                    raise ValueError(f"Conflicting update {run_id}/{update}")
                updates[update] = signature
            metrics = {}
            for slug, old in DATASETS.items():
                keys = [f"eval/{slug}/acc/mean", f"eval/{slug}/acc_mean", f"val-core/{old}/acc/mean@16"]
                for key in keys:
                    if key in row:
                        metrics[slug] = {"key": key, "value": row[key]}
                        result[f"{slug}_mean_pct"] = row[key] * 100
                        break
            if metrics:
                result["last_eval_outer"] = outer
                result["last_eval_optimizer"] = optimizer
    if updates:
        result["applied_update_records"] = sum(v[0] for v in updates.values())
        result["skipped_update_records"] = sum(v[1] for v in updates.values() if v[1] is not None)
    return result


def render(inventory):
    labels = {
        "sampled-ablation": "Sampled 消融 E1–E12",
        "full-kl-comparison": "Full-KL 对照 E13–E17",
        "equal-update-budget": "等更新预算 E18–E20",
        "failure-reproduction": "失败复现正式训练",
        "other-runs": "历史 Math / Coding / IF",
        "method-overrides": "统一入口的方法变体与 custom 实验",
        "mirror-comparison": "Full-KL / MDPO / BPO 正式对照 E44–E47",
    }
    lines = [
        "# 实际运行记录",
        "",
        f"数据冻结时间：{inventory['observed_at']}；日志提交 `{inventory['logs_commit']}`。",
        "",
        "此表由 `scripts/build_inventory.py` 直接读取 `wandb-logs` 解码结果生成；"
        "完整 config/history 留在日志仓库，实验设计和解释维护在各系列 README。",
        "",
        "分数为最后一次已记录验证的 mean accuracy（%）；M/A/I 分别为 MATH-500、AMC23、AIME25。"
        "更新计数与验证坐标分别记录。`—` 表示缺记录。"
        "云端 failed 与本地缺退出记录分别保留，不能据此定位失败原因。",
        "",
    ]

    def fmt(value):
        return "—" if value is None else str(int(value))

    for group, title in labels.items():
        selected = [r for r in inventory["runs"] if r["family"] == group]
        if not selected:
            continue
        lines += [
            f"## {title}",
            "",
            "| 实验 / run | 实际 setting | 云端 / 本地 | 完整 outer / 更新 | 验证 outer / 更新 | M / A / I | LCB |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for r in selected:
            experiment = re.search(r"E\d+", r["name"])
            label = experiment[0] if experiment else r["name"].split("-2026")[0]
            legacy_method = " / ".join(str(r.get(k)) for k in ("loss_mode", "advantage_source", "policy_loss_mode"))
            setting = f"{r.get('method', legacy_method)}；reuse={r['reuse_epochs']}；LR={r['lr']}"
            score = " / ".join(
                "—" if r[f"{ds}_mean_pct"] is None else f"{r[f'{ds}_mean_pct']:.2f}"
                for ds in ["math500", "amc23", "aime25"]
            )
            lcb = "—" if r["livecodebench_mean_pct"] is None else f"{r['livecodebench_mean_pct']:.2f}"
            lines.append(
                f"| {label} · `{r['run_id']}`[^run-{r['run_id']}] | {setting} | "
                f"{r['cloud_state'] or '未核对'} / {r['local_status']} | "
                f"{fmt(r['completed_outer'])} / {fmt(r['optimizer_updates'])} | "
                f"{fmt(r['last_eval_outer'])} / {fmt(r['last_eval_optimizer'])} | {score} | {lcb} |"
            )
        lines.append("")
    lines += [
        "## 早期运行的补充说明",
        "",
        "`y5y8519s` 无训练 history，本地 finished 表示读到退出记录，exit=1 表示执行失败。"
        "其他早期 Math/Coding run 的本地日志为 partial_snapshot；云端 failed 不能定位具体失败原因。",
        "",
        "IF 当前没有正式运行记录。历史运行的完整预算、重复 seed 与失败原因仍待补齐；"
        "后续重跑登记新 run ID，已有记录保留。",
        "",
        "配置与全量历史在 `wandb-logs/derived/MirrorOPD-verl/<run目录>/`；"
        "按 run ID 查找对应目录。来源提交和查看方法见[复现与维护](复现与维护.md)。",
        "",
        "### 参考文献",
        "",
    ]
    lines += [
        f"[^run-{r['run_id']}]: W&B. {r['name']}. [运行页面](https://forge.coreweave.com/wandb/wncfht/MirrorOPD-verl/runs/{r['run_id']})。"
        for r in inventory["runs"]
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs-repo", type=Path, required=True)
    parser.add_argument(
        "--cloud-index", type=Path, required=True, help="Frozen {observed_at, runs: [{id,name,state}]} JSON"
    )
    parser.add_argument("--register", type=Path, required=True, help="Current generated Markdown register")
    args = parser.parse_args()
    args.logs_repo = args.logs_repo.expanduser().resolve()
    python = args.logs_repo / ".venv/bin/python"
    decoder = args.logs_repo / "tools/wandb_local.py"
    if not python.is_file() or not decoder.is_file():
        raise SystemExit(f"Initialize the log reader first: make -C {args.logs_repo} setup")
    subprocess.run([str(python), str(decoder), "decode", "--project", "MirrorOPD-verl"], check=True)
    output = args.logs_repo / "derived/MirrorOPD-verl/inventory"
    cloud = read(args.cloud_index)
    indexed = {r["id"]: r for r in cloud["runs"]}
    decoded = args.logs_repo / "derived/MirrorOPD-verl"
    runs = []
    seen = set()
    for path in sorted(decoded.glob("*/run.json")):
        meta = read(path)
        identity = meta["run"]
        if not family(identity.get("display_name", "")):
            continue
        run_id = identity["run_id"]
        if run_id in seen:
            raise ValueError(f"Duplicate local directories for {run_id}; choose the intended snapshot explicitly")
        if meta["status"] not in {"finished", "snapshot", "partial_snapshot"}:
            raise ValueError(f"Unusable decode: {run_id}: {meta['status']}")
        seen.add(run_id)
        row = inspect(path.parent, indexed.get(run_id, {}))
        runs.append(row)
    missing = [r["id"] for r in cloud["runs"] if family(r["name"]) and r["id"] not in seen]
    if missing:
        raise ValueError(f"Formal cloud runs lack decoded local evidence: {missing}")
    commit = subprocess.check_output(["git", "-C", str(args.logs_repo), "rev-parse", "HEAD"], text=True).strip()
    inventory = {
        "observed_at": cloud["observed_at"],
        "logs_commit": commit,
        "project": "wncfht/MirrorOPD-verl",
        "runs": runs,
        "excluded_cloud_runs": [r for r in cloud["runs"] if not family(r["name"])],
    }
    write(output / "inventory.json", inventory)
    with (output / "runs.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(runs[0]))
        writer.writeheader()
        writer.writerows(
            {k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k, v in row.items()}
            for row in runs
        )
    args.register.write_text(render(inventory))
    print(f"Recorded {len(runs)} formal runs; excluded {len(inventory['excluded_cloud_runs'])} engineering runs")


if __name__ == "__main__":
    main()
