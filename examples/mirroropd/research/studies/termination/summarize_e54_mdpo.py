# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Audit completed MDPO sweeps from archived logs and scored token sequences.

No checkpoints, model forward passes, or GPU work are required. Bootstrap
intervals resample paired questions separately within MATH and DAPO; they do
not capture training-seed variation or adjust for parameter selection.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

RUNS = ("e54-topd-mdpo-20261029", "e54-mdpo-sweep-20261029")
METRICS = ("accuracy", "correct_stopped", "cap_rate", "mean_length", "repeat8_mean")


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def vector(row):
    return [
        float(row["acc"]),
        float(row["acc"] and row["finish_reason"] == "stop"),
        float(row["finish_reason"] == "length"),
        row["response_length"],
        row["repeat8"],
    ]


def paired_comparison(records, candidate, control):
    rng = np.random.default_rng(20261029)
    deltas, boot = [], []
    transitions = {}
    for dataset in ("math", "dapo"):
        keys = sorted(k for k in records[candidate] if k[0] == dataset)
        delta = np.array([np.subtract(vector(records[candidate][k]), vector(records[control][k])) for k in keys])
        assert delta.shape == (64, len(METRICS))
        deltas.append(delta)
        boot.append(delta[rng.integers(0, 64, size=(10000, 64))].mean(axis=1))
        a = [records[candidate][k] for k in keys]
        b = [records[control][k] for k in keys]
        transitions[dataset] = {
            "correct_gains": sum(x["acc"] and not y["acc"] for x, y in zip(a, b, strict=True)),
            "correct_losses": sum(y["acc"] and not x["acc"] for x, y in zip(a, b, strict=True)),
            "cap_removed": sum(
                y["finish_reason"] == "length" and x["finish_reason"] == "stop" for x, y in zip(a, b, strict=True)
            ),
            "cap_added": sum(
                x["finish_reason"] == "length" and y["finish_reason"] == "stop" for x, y in zip(a, b, strict=True)
            ),
        }
    point = np.concatenate(deltas).mean(axis=0)
    interval = np.quantile((boot[0] + boot[1]) / 2, [0.025, 0.975], axis=0)
    return {
        "candidate": candidate,
        "control": control,
        "transitions": transitions,
        "metrics": {
            key: {"difference": float(point[i]), "ci95": interval[:, i].tolist()} for i, key in enumerate(METRICS)
        },
    }


def build(runs):
    result = {
        "models": "Qwen3-1.7B-Base <- Qwen3-30B-A3B-Instruct-2507",
        "batch": 64,
        "reuse": 16,
        "train_cap": 16384,
        "eval_cap": 31744,
        "arms": [],
        "evaluation": [],
    }
    records, training_batches = {}, {}
    cap_root = runs / "e54-cap-controlled4-20261028-hppk13"
    warmup_prompts = set().union(
        *[{tuple(r["prompt_ids"]) for r in read(cap_root / f"warmup/debug/rollout-{s}.jsonl")} for s in (1, 2)]
    )
    for name in RUNS:
        root = runs / name
        assert json.loads((root / "status.json").read_text())["phase"] == "complete"
        protocol = json.loads((root / "protocol.json").read_text())
        for arm in protocol["arms"]:
            label = arm["label"]
            directory = root / label
            assert json.loads((directory / "exit.json").read_text())["exit_code"] == 0
            state = json.loads((directory / "training/checkpoints/global_step_5/trainer_state.json").read_text())
            assert state["actor_optimizer_step"] == state["actor_optimizer_step_attempts"] == 80
            metrics = [r["data"] for r in read(directory / "metrics.jsonl")]
            updates = [r for r in metrics if "distill/mdpo/objective" in r]
            assert [r["step/optimizer"] for r in updates] == list(range(33, 81))
            entry = arm | {"training_complete": True, "updates": [], "rollouts": [], "fitting": []}
            for d in updates:
                assert all(
                    math.isfinite(d[k])
                    for k in ("distill/mdpo/objective", "distill/teacher_kl", "optim/actor/grad_norm_preclip")
                )
                assert (
                    d["optim/actor/update_skipped"]
                    == d["optim/actor/loss_nonfinite"]
                    == d["optim/actor/grad_norm_preclip_nonfinite"]
                    == 0
                )
                entry["updates"].append(
                    {
                        k: v
                        for k, v in d.items()
                        if k.startswith("step/")
                        or k
                        in (
                            "distill/mdpo/objective",
                            "distill/target_kl",
                            "distill/teacher_kl",
                            "distill/anchor_kl",
                            "distill/target_log_z",
                            "optim/actor/grad_norm_preclip",
                        )
                        or "objective_position" in k
                        and k.endswith("/mean")
                    }
                )
            for outer in (3, 4, 5):
                u = [d for d in updates if d["step/outer"] == outer]
                assert len(u) == 16
                a, b = u[0], u[-1]
                assert abs(a["distill/anchor_kl"]) < 1e-8
                assert max(d["distill/target_log_z"] for d in u) - min(d["distill/target_log_z"] for d in u) < 1e-7
                entry["fitting"].append(
                    {
                        "outer": outer,
                        "objective": [a["distill/mdpo/objective"], b["distill/mdpo/objective"]],
                        "target_kl": [a["distill/target_kl"], b["distill/target_kl"]],
                        "teacher_kl": [a["distill/teacher_kl"], b["distill/teacher_kl"]],
                        "anchor_kl_end": b["distill/anchor_kl"],
                        "constant_floor": -a["distill/target_log_z"] / arm["eta"],
                    }
                )
            for d in metrics:
                if "rollout/response_length/mean" in d:
                    entry["rollouts"].append(
                        {
                            "outer": d["step/outer"],
                            "length": d["rollout/response_length/mean"],
                            "cap_count": d["rollout/finish_reason/length_count"],
                            "stop_count": d["rollout/finish_reason/stop_count"],
                            "score": d["rollout/score/mean"],
                        }
                    )
            training_batches[label] = [
                {tuple(r["prompt_ids"]) for r in read(directory / f"debug/rollout-{s}.jsonl")} for s in (3, 4, 5)
            ]
            records[label] = {}
            for dataset in ("math", "dapo"):
                path = root / "evaluation" / f"{label}-{dataset}"
                scored = read(path / "scored.jsonl")
                assert len(scored) == len({r["prompt_id"] for r in scored}) == 64
                for r in scored:
                    ids = r["response_ids"]
                    assert r["status"] == "ok" and len(ids) == r["response_length"]
                    assert r["finish_reason"] in ("stop", "length")
                    if r["finish_reason"] == "length":
                        assert len(ids) == 31744
                    else:
                        assert ids[-1] in (151643, 151645)
                    windows = max(len(ids) - 7, 0)
                    repeat = 1 - len(set(zip(*(ids[i:] for i in range(8)), strict=False))) / windows if windows else 0
                    assert abs(repeat - r["repeat8"]) < 1e-12
                    records[label][(dataset, r["prompt_id"])] = r
                matrix = np.array([vector(r) for r in scored])
                means = matrix.mean(axis=0)
                published = json.loads((path / "metrics.json").read_text())
                for key, value in [("accuracy", means[0]), ("mean_length", means[3]), ("repeat8_mean", means[4])]:
                    assert abs(published[key] - value) < 1e-8
                assert published["cap_count"] == matrix[:, 2].sum()
                result["evaluation"].append(
                    arm
                    | {
                        "arm": label,
                        "dataset": dataset,
                        "responses": 64,
                        "accuracy": means[0],
                        "correct_and_stopped": int(matrix[:, 1].sum()),
                        "cap_count": int(matrix[:, 2].sum()),
                        "mean_length": means[3],
                        "median_length": float(np.median(matrix[:, 3])),
                        "repeat8_mean": means[4],
                        "empty_count": sum(not r["output"].strip() for r in scored),
                        "role_start_count": sum(151644 in r["response_ids"] for r in scored),
                    }
                )
            result["arms"].append(entry)
    reference = records["k0-eta01"]
    for label, recs in records.items():
        assert recs.keys() == reference.keys()
        assert training_batches[label] == training_batches["k0-eta01"]
        for key, r in recs.items():
            assert (r["prompt_ids"], r["seed"], r["gts"]) == (
                reference[key]["prompt_ids"],
                reference[key]["seed"],
                reference[key]["gts"],
            )
            if key[0] == "dapo":
                assert tuple(r["prompt_ids"]) not in warmup_prompts | set().union(*training_batches[label])
    pairs = [
        ("k1-eta01", "k0-eta01"),
        ("k05-eta01", "k0-eta01"),
        ("k1-eta02", "k0-eta02"),
        ("k05-eta02", "k0-eta02"),
        ("k1-eta005", "k1-eta01"),
        ("k05-eta005", "k05-eta01"),
    ]
    result["paired_comparisons"] = [paired_comparison(records, *p) for p in pairs]
    common = set.intersection(
        *[{k for k, r in rs.items() if r["finish_reason"] == "length"} for rs in records.values()]
    )
    result["audit"] = {
        "training_arms": len(result["arms"]),
        "new_updates": sum(len(a["updates"]) for a in result["arms"]),
        "evaluation_responses": sum(len(r) for r in records.values()),
        "same_training_question_batches": True,
        "same_evaluation_prompt_ids_and_seeds": True,
        "all_stop_records_end_on_eos": True,
        "capped_by_all_ten": [list(k) for k in sorted(common)],
        "bootstrap_resamples": 10000,
        "interval_scope": (
            "Paired questions, stratified by dataset; unadjusted exploratory intervals, no training-seed uncertainty"
        ),
    }
    examples = []
    for key, reason in [
        (("math", "MATH-500:249"), "cap-to-correct-stop"),
        (("math", "MATH-500:108"), "cap-to-correct-stop"),
        (("math", "MATH-500:470"), "correct-to-incorrect"),
        (("dapo", "train-3217"), "persistent-repetition"),
    ]:
        row = {"dataset": key[0], "prompt_id": key[1], "reason": reason, "arms": {}}
        for label in ("k0-eta01", "k05-eta01", "k1-eta01"):
            r = records[label][key]
            row["arms"][label] = {k: r[k] for k in ("response_length", "finish_reason", "acc", "pred", "repeat8")}
            row["arms"][label].update(
                boxed_count=r["output"].count("\\boxed"), head=r["output"][:320], tail=r["output"][-700:]
            )
        examples.append(row)
    result["examples"] = examples
    reference = json.loads((cap_root / "evaluation/summary.json").read_text())
    result["historical_full_rkl"] = {
        "evaluation": [r for r in reference["evaluation"] if r["train_cap"] == 16384],
        "scope": (
            "Historical reference: same warmup and question sets, but older runtime/numerics; not a new matched control"
        ),
    }
    result["selection_scope"] = (
        "Same pilot questions reused for screening and eta selection; no confirmatory winner claim"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result["audit"], ensure_ascii=False, indent=2))
    for comparison in result["paired_comparisons"]:
        print(json.dumps(comparison, ensure_ascii=False))


if __name__ == "__main__":
    main()
