#!/usr/bin/env python3
"""Check completed inputs, support partitions, evaluation measures and budgets."""

import argparse
import json
import math
import re
from pathlib import Path


def read(p):
    return json.loads(p.read_text())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    a = p.parse_args()
    root = a.data
    prompts = read(root / "prompts.json")
    assert len(prompts) == 224
    normalize = lambda t: re.sub(r"\s+", " ", t).strip()
    train = {r["id"] for r in prompts if r["split"] == "train"}
    held = {r["id"] for r in prompts if r["split"] == "heldout"}
    assert len(train) == 128 and len(held) == 96 and train.isdisjoint(held)
    assert len({r["prompt"] for r in prompts}) == 224
    assert len({normalize(r["prompt"]) for r in prompts}) == 224
    assert {normalize(r["prompt"]) for r in prompts if r["split"] == "train"}.isdisjoint(
        {normalize(r["prompt"]) for r in prompts if r["split"] == "heldout"}
    )
    for src in ["student", "justrl", "polaris"]:
        refs = read(root / f"reference-{src}.json")
        assert len(refs) == 96 and {r["id"] for r in refs} == held
    max_partition_error = 0.0
    budgets = []
    for arm in [
        "justrl-student-k16",
        "polaris-student-k16",
        "polaris-teacher-k16",
        "polaris-intersection-k16",
        "polaris-topk-k16",
    ]:
        folder = root / arm
        assert read(folder / "complete.json")["updates"] == 64
        updates = read(folder / "training.json")
        assert [r["step"] for r in updates] == list(range(1, 65))
        assert all(math.isfinite(r["pre_update_kl"]) and math.isfinite(r["grad_norm"]) for r in updates)
        visited = []
        sampled_tokens = 0
        for outer in range(1, 5):
            rows = read(folder / f"rollout-{outer}.json")
            assert len(rows) == 32
            visited.extend(r["id"] for r in rows)
            for r in rows:
                n = len(r["response_ids"])
                sampled_tokens += n
                assert 1 <= n <= 1024
                for k in [8, 16, 64]:
                    cats = [r["stats"]["all"][f"category{k}_{c}"] for c in range(4)]
                    assert sum(cats) == n
                    assert len(r["token_categories"][str(k)]) == n
                    for dist in ["p", "q"]:
                        mass = sum(r["stats"]["all"][f"mass_{dist}{k}_{c}"] for c in range(4))
                        max_partition_error = max(max_partition_error, abs(mass / n - 1))
                if "intersection" in arm:
                    assert sum(c != 3 for c in r["token_categories"]["16"]) == r["empty_intersections"]
            tokens = sum(len(r["response_ids"]) for r in rows)
            assert all(r["valid_tokens"] == tokens for r in updates if r["outer"] == outer)
        assert len(visited) == 128 and set(visited) == train
        for step in [16, 32, 48, 64]:
            rows = read(folder / f"fixed-{step}.json")
            assert len(rows) == 288
            assert {(r["prompt_id"], r["source"]) for r in rows} == {
                (i, s) for i in held for s in ["student", "justrl", "polaris"]
            }
            for r in rows:
                v = r["stats"]["all"]
                assert v["alive"] == r["length"]
                assert v["rkl"] >= -1e-4 and -1e-4 <= v["js"] <= math.log(2) * r["length"] + 1e-3
                assert -1e-4 <= v["tv"] <= r["length"] + 1e-3
        for step in [32, 64]:
            rows = read(folder / f"fresh-{step}.json")
            assert len(rows) == 64
            assert {(r["id"], r["seed"]) for r in rows} == {(i, s) for i in range(128, 160) for s in [91730, 91731]}
            assert all(r["mode"] == "student" for r in rows)
        budgets.append(
            dict(
                arm=arm,
                updates=64,
                prompts=128,
                new_tokens=sampled_tokens,
                training_token_visits=sum(r["valid_tokens"] for r in updates),
            )
        )
    assert max_partition_error < 1e-5
    masks = {}
    for mode in ["high", "random", "low"]:
        f = root / f"selection-{mode}"
        assert read(f / "complete.json")["updates"] == 16
        selected = read(f / "masks.json")
        masks[mode] = selected
        assert len(selected) == 32 and all(len(r["positions"]) == 256 for r in selected)
        tr = read(f / "training.json")
        assert len(tr) == 16 and all(r["tokens"] == 8192 and math.isfinite(r["parameter_step_norm"]) for r in tr)
        assert len(read(f / "fresh-16.json")) == 64 and len(read(f / "fixed-16.json")) == 288
    for h, l in zip(masks["high"], masks["low"]):
        assert h["overlap16"] >= l["overlap16"]
    assert read(root / "decomposition/complete.json")["finished"]
    decomposition = read(root / "summary-decomposition.json")
    assert len(decomposition) == 5
    assert all(
        abs(r["parameter_gain_on_old_states"] - r["visitation_cost_at_new_model"] - r["net_gain"]) < 1e-8
        for r in decomposition
    )
    taskprompts = read(root / "task-prompts.json")
    assert len(taskprompts) == 32
    normalize = lambda t: re.sub(r"\s+", " ", t).strip()
    assert {normalize(r["prompt"]) for r in taskprompts}.isdisjoint({normalize(r["prompt"]) for r in prompts})
    tasks = []
    for name in ["initial"] + [r["arm"] for r in budgets]:
        rows = read(root / f"task-{name}.json")
        assert len(rows) == 32 and all("grade" in r and r["mode"] == "student" and r["cap"] == 4096 for r in rows)
        tasks.append(dict(arm=name, responses=32))
    assert len(read(root / "task-paired-contrasts.json")) == 5
    assert len(read(root / "summary-task.json")) == 6
    result = dict(
        prompt_split_disjoint=True,
        normalized_prompts_disjoint=True,
        reference_panels_complete=True,
        fresh_evaluation_unrestricted=True,
        max_probability_partition_error=max_partition_error,
        main_training=budgets,
        selection_updates=48,
        total_updates=368,
        selection_equal_position_budget=True,
        decomposition_identity=True,
        task_panels=tasks,
        smoke=read(root / "smoke.json"),
        divergence_reference_check=read(root / "divergence-reference-check.json"),
    )
    (root / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
