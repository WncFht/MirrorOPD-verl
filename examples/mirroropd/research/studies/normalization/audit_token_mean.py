# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Audit recorded length scaling on CPU, without replaying optimizer updates.

The online audit selects pure full-RKL runs by resolved configuration. Every
gradient rescaling compares the same recorded parameters and batch; it is not
an alternative training trajectory. Frozen-state runs have their own explicit
denominator and are not classified from actual_tokens alone.
"""

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np


def read_config(path):
    value = json.loads(path.read_text())
    return {
        k: v.get("value") if isinstance(v, dict) and set(v).issubset({"value", "desc"}) else v for k, v in value.items()
    }


def statistics(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {}
    return {
        "n": len(values),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "cv": float(values.std() / values.mean()) if values.mean() else None,
        **{
            name: float(np.quantile(values, q))
            for name, q in (("min", 0), ("p05", 0.05), ("median", 0.5), ("p95", 0.95), ("max", 1))
        },
    }


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pure_reverse_kl(objective):
    mirror = objective.get("mirror") or {}
    return objective.get("loss_mode") == "full_reverse_kl" or (
        objective.get("loss_mode") == "mirror"
        and mirror.get("objective") == "mdpo"
        and mirror.get("estimator") == "full_vocab"
        and mirror.get("eta") == 1
        and mirror.get("kappa") == 0
    )


def online_runs(logs):
    results, series = [], []
    for path in sorted((logs / "derived/MirrorOPD-verl").glob("*/config.json")):
        config = read_config(path)
        objective = config.get("distillation", {}).get("distillation_loss", {})
        actor = config.get("actor_rollout_ref", {}).get("actor", {})
        name = config.get("trainer", {}).get("experiment_name", "")
        if not pure_reverse_kl(objective) or actor.get("loss_agg_mode") != "token-mean" or name.startswith("smoke-"):
            continue
        history = path.with_name("history.jsonl")
        if not history.exists():
            continue
        outer, updates, evaluations = {}, {}, {}
        history_hash = hashlib.sha256()
        with history.open("rb") as stream:
            for line in stream:
                history_hash.update(line)
                item = json.loads(line)
                if item.get("step/outer") is None:
                    continue
                step = int(item["step/outer"])
                if "rollout/response_length/mean" in item and step > 0:
                    outer.setdefault(step, {}).update(item)
                if "optim/actor/grad_norm_preclip" in item:
                    update = item.get("step/update", item.get("step/optimizer"))
                    if update is not None:
                        updates[int(update)] = item
                values = {k: v for k, v in item.items() if re.fullmatch(r"eval/[^/]+/acc/mean", k)}
                if values:
                    evaluations.setdefault(step, {}).update(values)
        if len(outer) < 20:
            continue
        outer = dict(sorted(outer.items()))
        label = re.search(r"(?:^|-)[Ee](\d+)(?:-|$)", name)
        label = "E" + label[1] if label else path.parent.name.rsplit("-", 1)[-1]
        lengths = np.array([v["rollout/response_length/mean"] for v in outer.values()])
        constant = float(np.median(lengths))
        batch = config["data"]["train_batch_size"]
        mini = actor["ppo_mini_batch_size"]
        repetition = config["actor_rollout_ref"]["rollout"]["n"]
        empty = [v.get("rollout/empty_loss_mask_fraction") for v in outer.values()]
        implied = [v["rollout/response_tokens"] / v["rollout/response_length/mean"] for v in outer.values()]
        # These recorded runs have one full-rollout mini-batch, no filtered
        # responses, entropy term or reference KL. Otherwise raw norm scaling
        # would not reconstruct the whole objective's gradient.
        assert batch == mini and repetition == 1
        assert all(x == 0 for x in empty)
        assert all(abs(x - batch) < 1e-8 for x in implied)
        assert actor.get("entropy_coeff") == 0 and actor.get("use_kl_loss") is False
        clip = float(actor["grad_clip"])
        by_outer, paired, nonfinite = {}, [], 0
        for update, item in sorted(updates.items()):
            step = int(item["step/outer"])
            norm = float(item["optim/actor/grad_norm_preclip"])
            nonfinite += int(not math.isfinite(norm))
            by_outer.setdefault(step, []).append(norm)
            if step in outer and math.isfinite(norm):
                length = outer[step]["rollout/response_length/mean"]
                paired.append((step, norm, norm * length / constant))
        old = np.array([x[1] for x in paired])
        fixed = np.array([x[2] for x in paired])
        # For a positive common scalar, global norm clipping preserves the
        # gradient direction. Norms therefore determine the clipped-vector
        # relative difference; omit the implementation's tiny clip epsilon.
        used = np.minimum(old, clip)
        fixed_used = np.minimum(fixed, clip)
        used_ratio = np.divide(fixed_used, used, out=np.ones_like(used), where=used > 0)
        postclip_change = np.abs(used_ratio - 1)
        after10_mask = np.array([x[0] > 10 for x in paired])
        after10 = [o["rollout/response_length/mean"] for step, o in outer.items() if step > 10]
        results.append(
            {
                "label": label,
                "run_id": path.parent.name.rsplit("-", 1)[-1],
                "name": name,
                "config_source": str(path.relative_to(logs)),
                "history_source": str(history.relative_to(logs)),
                "config_sha256": digest(path),
                "history_sha256": history_hash.hexdigest(),
                "objective": objective,
                "optimizer": actor.get("optim"),
                "batch_size": batch,
                "reuse": actor["ppo_epochs"],
                "cap": config["data"]["max_response_length"],
                "n_outer": len(outer),
                "outer_range": [min(outer), max(outer)],
                "n_recorded_updates": len(updates),
                "n_length_matched_updates": len(paired),
                "nonfinite_recorded_norms": nonfinite,
                "length": statistics(lengths),
                "reference_C": constant,
                "relative_scale": statistics(constant / lengths),
                "relative_scale_after_outer10": statistics(constant / np.array(after10)),
                "first20_length": float(lengths[:20].mean()),
                "last20_length": float(lengths[-20:].mean()),
                "late_vs_early_scale": float(lengths[:20].mean() / lengths[-20:].mean()),
                "grad_norm": statistics(old),
                "clip": clip,
                "actual_clipped_fraction_matched": float(np.mean(old > clip)),
                "fixed_C_raw_counterfactual_clipped_fraction": float(np.mean(fixed > clip)),
                "clip_trigger_changed_fraction": float(np.mean((old > clip) != (fixed > clip))),
                "same_parameter_postclip_relative_change": statistics(postclip_change),
                "same_parameter_postclip_relative_change_after_outer10": statistics(postclip_change[after10_mask]),
                "evals": [{"outer": s, **v} for s, v in sorted(evaluations.items())],
            }
        )
        for step, item in outer.items():
            norms = by_outer.get(step, [])
            series.append(
                {
                    "label": label,
                    "outer": step,
                    "response_tokens": item["rollout/response_tokens"],
                    "length_mean": item["rollout/response_length/mean"],
                    "relative_scale": constant / item["rollout/response_length/mean"],
                    "grad_norm_first": norms[0] if norms else None,
                    "grad_norm_mean": float(np.mean(norms)) if norms else None,
                    "actual_clipped_fraction": float(np.mean(np.array(norms) > clip)) if norms else None,
                    "recorded_updates": len(norms),
                }
            )
    results.sort(key=lambda r: int(r["label"][1:]))
    return results, series


def fixed_runs(logs):
    definitions = [
        ("T-E3", "qwen3-thinking-e3-stage1280-20261030", "T-E3"),
        ("L3-arithmetic-full", "llama-reasoning-off-stage1280-20261030", "L3-arithmetic-full"),
    ]
    results = []
    for label, study, arm in definitions:
        base = logs / "MirrorOPD-verl/analysis" / study
        history = base / "arms" / arm / "training.json"
        receipt = history.with_name("complete.json")
        config = json.loads(receipt.read_text())["inputs"]["training"]
        snapshots = list((base / "control").glob("*/source/examples/mirroropd/research/studies/fixed_transfer/spec.py"))
        assert len(snapshots) == 1
        text = snapshots[0].read_text()
        assert "return self.batch_size * self.horizon" in text
        denominator = config["batch_size"] * config["horizon"]
        rows = json.loads(history.read_text())
        tokens = [r["actual_tokens"] for r in rows]
        results.append(
            {
                "label": label,
                "history_source": str(history.relative_to(logs)),
                "history_sha256": digest(history),
                "receipt_source": str(receipt.relative_to(logs)),
                "receipt_sha256": digest(receipt),
                "runtime_spec_source": str(snapshots[0].relative_to(logs)),
                "runtime_spec_sha256": digest(snapshots[0]),
                "normalization": "sum_KL / (batch_size * fixed_horizon)",
                "fixed_denominator": denominator,
                "recorded_updates": len(rows),
                "actual_tokens": statistics(tokens),
                "hypothetical_token_mean_over_actual_fixed_loss": statistics(denominator / np.array(tokens)),
            }
        )
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    online, series = online_runs(args.logs.resolve())
    fixed = fixed_runs(args.logs.resolve())
    result = {
        "date": "2026-10-31",
        "scope": "local decoded run snapshots; no optimizer replay or new GPU experiment",
        "scale_definition": "C / batch_mean_response_length; C = each run's median batch mean length",
        "counterfactual": (
            "raw gradients at the same recorded parameters; excludes changed Adam states and future rollouts"
        ),
        "online": online,
        "fixed": fixed,
    }
    (args.output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    with (args.output / "outer-series.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(series[0]))
        writer.writeheader()
        writer.writerows(series)
    print(
        json.dumps(
            {
                "runs": len(online),
                "outer_batches": len(series),
                "recorded_updates": sum(r["n_recorded_updates"] for r in online),
                "fixed_audits": len(fixed),
            }
        )
    )


if __name__ == "__main__":
    main()
