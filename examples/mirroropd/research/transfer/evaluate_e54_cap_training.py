# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Evaluate both cap arms on identical MATH and held-out training-domain prompts."""

import argparse
import concurrent.futures
import json
import os
import statistics
import sys
from pathlib import Path

from .run_e54_recovery_eval import execute


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    repo = Path(__file__).resolve().parents[4]
    out = root / "evaluation"
    cpu = os.environ | {"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2"}
    trained = {}
    for cap in [7168, 16384]:
        arm = root / f"cap{cap}"
        assert json.loads((arm / "exit.json").read_text())["exit_code"] == 0
        state = json.loads((arm / "training/checkpoints/global_step_5/trainer_state.json").read_text())
        assert state["actor_optimizer_step"] == state["actor_optimizer_step_attempts"] == 80
        trained[cap] = []
        for step in range(1, 6):
            owner = root / "warmup" if step <= 2 else arm
            trained[cap].append({tuple(r["prompt_ids"]) for r in read(owner / f"debug/rollout-{step}.jsonl")})
        command = [
            sys.executable,
            "-m",
            "verl.model_merger",
            "merge",
            "--backend",
            "fsdp",
            "--local_dir",
            str(arm / "training/checkpoints/global_step_5/actor"),
            "--target_dir",
            str(out / f"model-cap{cap}"),
        ]
        execute(command, out / f"merge-cap{cap}.log", cpu, 300)
    assert trained[7168] == trained[16384], "Training question batches differ across cap arms"
    all_train = set().union(*trained[7168])
    math_path = repo / "outputs/e54-recovery-controls-r3-20261028-hppk13/evaluation/prompts.json"
    math = json.loads(math_path.read_text())
    candidates = json.loads((repo / "outputs/e54-train-cap-probe-20261028-hppk13/teacher-requests.json").read_text())
    dapo = [
        dict(prompt_id=r["request_id"], data_source="DAPO", prompt_ids=r["prompt_ids"], gts=r["gts"], seed=r["seed"])
        for r in candidates
        if r["cohort"] == "random_training" and tuple(r["prompt_ids"]) not in all_train
    ][:64]
    assert len(math) == len(dapo) == 64
    for name, rows in [("math", math), ("dapo", dapo)]:
        (out / f"prompts-{name}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
    (out / "protocol.json").write_text(
        json.dumps(
            {
                "train_batches_equal": True,
                "dapo_excludes_all_training_prompts": True,
                "models": ["cap7168", "cap16384"],
                "checkpoint_outer": 5,
                "optimizer_updates": 80,
                "shared_warmup_outer_steps": 2,
                "math": {"n": 64, "temperature": 0.7, "top_p": 0.9},
                "dapo": {"n": 64, "temperature": 1.0, "top_p": 1.0},
                "response_cap": 31744,
                "samples_per_prompt": 1,
                "max_num_seqs": 16,
            },
            indent=2,
        )
        + "\n"
    )
    jobs = [
        (0, 7168, "math", 0.7, 0.9),
        (1, 7168, "dapo", 1.0, 1.0),
        (2, 16384, "math", 0.7, 0.9),
        (3, 16384, "dapo", 1.0, 1.0),
    ]

    def one(job):
        gpu, cap, dataset, temperature, top_p = job
        directory = out / f"cap{cap}-{dataset}"
        directory.mkdir(exist_ok=False)
        base = [sys.executable, "-u", "-m", "examples.mirroropd.research.transfer.evaluate_e54_recovery"]
        command = base + [
            "generate",
            "--model",
            str(out / f"model-cap{cap}"),
            "--prompts",
            str(out / f"prompts-{dataset}.json"),
            "--output",
            str(directory),
            "--temperature",
            str(temperature),
            "--top-p",
            str(top_p),
        ]
        execute(
            command,
            directory / "generate.log",
            os.environ
            | {"CUDA_VISIBLE_DEVICES": str(gpu), "OMP_NUM_THREADS": "2", "VLLM_WORKER_MULTIPROC_METHOD": "spawn"},
            3600,
        )
        execute(base + ["grade", "--output", str(directory)], directory / "grade.log", cpu, 600)
        rows = read(directory / "scored.jsonl")
        assert len(rows) == 64 and len({r["prompt_id"] for r in rows}) == 64
        metrics = json.loads((directory / "metrics.json").read_text())
        metrics["correct_and_stopped"] = sum(bool(r["acc"]) and r["finish_reason"] == "stop" for r in rows)
        return {"train_cap": cap, "dataset": dataset, **metrics}

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(one, jobs))
    training = {}
    for cap in [7168, 16384]:
        batches = read(root / f"cap{cap}/debug/rollout-summary.jsonl")
        batches = read(root / "warmup/debug/rollout-summary.jsonl") + batches
        training[str(cap)] = {"batches": batches, "mean_length": statistics.mean(r["mean_length"] for r in batches)}
    summary = {
        "evaluation": results,
        "training": training,
        "scope": "Single-seed pilot; paired questions, not a full benchmark",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    report = [
        "# E54 四卡 response cap 对照",
        "",
        "两组共用前 2 outer 的模型、Adam 和数据状态；第 3 步起改变 cap，最终均为 80 次更新。",
        "每个评测子集 64 题，每题一次。",
        "",
        "| 训练 cap | 题集 | 正确率 | 平均长度 | 触顶 | 判对且停止 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in results:
        report.append(
            f"| {row['train_cap']:,} | {row['dataset']} | {100 * row['accuracy']:.2f}% | "
            f"{row['mean_length'] / 1000:.2f}k | {row['cap_count']}/64 | {row['correct_and_stopped']}/64 |"
        )
    (out / "README.md").write_text("\n".join(report) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
