# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Prepare or evaluate full-vocabulary MDPO reward-shape controls on GPUs 0-3.

Preparation is CPU-only. The copied supervisor must be explicitly started; its
independent recovery guard restores front-four GPU occupation on every exit.
"""

import argparse
import concurrent.futures
import json
import os
import shutil
import sys
import uuid
from pathlib import Path

from examples.mirroropd.research.core.io import dump as write
from examples.mirroropd.research.core.paths import REPO
from examples.mirroropd.research.studies.termination.run_e54_recovery_eval import execute

ARMS = [("k1-eta01", 1, 0.1), ("k0-eta01", 0, 0.1), ("k1-eta02", 1, 0.2), ("k0-eta02", 0, 0.2)]


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def prepare(repo, root):
    old = repo / "outputs/e54-cap-controlled4-20261028-hppk13"
    checkpoint = old / "warmup/training/checkpoints/global_step_2"
    assert (checkpoint / "actor").is_dir()
    root.mkdir(parents=True, exist_ok=False)
    source = root / "source"
    source.mkdir()
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", "docs", ".agents", "wandb", "outputs", "node_modules")
    for name in ("verl", "examples/mirroropd", "tests/trainer/distillation"):
        shutil.copytree(repo / name, source / name, ignore=ignore)
    for name in ("pyproject.toml", "setup.py"):
        shutil.copy2(repo / name, source / name)
    shutil.copy2(Path(__file__).with_name("e54_dynamic_debug.py"), source / "e54_dynamic_debug.py")
    # Read-only rollout IDs permit independent stop/repeat and prompt-pairing audits.
    trainer = source / "verl/trainer/ppo/v1/trainer_base.py"
    text = trainer.read_text()
    marker = (
        "    def _update_actor(self, batch: KVBatchMeta, metrics: dict) -> KVBatchMeta:\n"
        '        """Update the actor network."""\n'
    )
    assert text.count(marker) == 1
    trainer.write_text(
        text.replace(
            marker, marker + "        from e54_dynamic_debug import dump_rollout\n\n        dump_rollout(self, batch)\n"
        )
    )
    engine = source / "verl/workers/engine/fsdp/transformer_impl.py"
    text = engine.read_text()
    marker = "            cast_module_parameters(module, torch_dtype)\n"
    assert text.count(marker) == 1
    engine.write_text(
        text.replace(
            marker, marker + "            from e54_dynamic_debug import rope_check\n\n            rope_check(module)\n"
        )
    )
    shutil.copy2(old / "controller.py", root / "controller.py")
    original = json.loads((old / "cap16384/launch.json").read_text())
    jobs = []
    preflight = root / "gpu-check"
    preflight.mkdir()
    write(
        preflight / "launch.json",
        {
            "cwd": str(source),
            "environment": {"CUDA_VISIBLE_DEVICES": "0", "PYTHONPATH": str(source)},
            "command": [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "tests/trainer/distillation/test_mdpo_termination_on_gpu.py",
            ],
        },
    )
    jobs.append({"preset": "gpu-check", "directory": str(preflight)})
    for label, kappa, eta in ARMS:
        arm = root / label
        arm.mkdir()
        run_id = "t" + uuid.uuid4().hex[:7]

        def translate(value, arm=arm, run_id=run_id, label=label):
            return (
                value.replace(str(old / "source"), str(source))
                .replace(str(old / "cap16384"), str(arm))
                .replace(original["environment"]["WANDB_RUN_ID"], run_id)
                .replace(old.name + "-cap16384", root.name + "-" + label)
                .replace("/tmp/ecap-", "/tmp/etmd-")
            )

        env = {k: translate(v) for k, v in original["environment"].items() if not k.startswith("E54_DEBUG_")}
        env.update(MIRROR_DIAGNOSTICS="True", E54_DEBUG_DIR=str(arm / "debug"), MIRROROPD_LONG_RUN_ID=root.name)
        command = [translate(x) for x in original["command"] if "E54_DEBUG_" not in x]
        command[3:3] = ["--method", "mdpo", "--estimator", "full_vocab", "--kappa", str(kappa), "--eta", str(eta)]
        replacements = {
            "++ray_kwargs.ray_init.object_store_memory=": str(16 * 1024**3),
            "++ray_kwargs.ray_init.runtime_env.env_vars.MIRROROPD_LONG_RUN_ID=": root.name,
        }
        command = [next((p + v for p, v in replacements.items() if x.startswith(p)), x) for x in command]
        command.append(f"++ray_kwargs.ray_init.runtime_env.env_vars.E54_DEBUG_DIR={arm / 'debug'}")
        launch = {"environment": env, "command": command, "cwd": str(source)}
        write(arm / "launch.json", launch)
        # Resolve Hydra before any GPU release, including Ray string-valued env vars.
        execute(command, arm / "config.log", os.environ | env | {"CONFIG_ONLY": "1", "CUDA_VISIBLE_DEVICES": ""}, 120)
        jobs.append({"preset": label, "directory": str(arm), "wandb_run_id": run_id})
    evaluation = root / "evaluation"
    evaluation.mkdir()
    # Freeze the exact previous held-out sets before looking at any new outcome.
    for dataset in ("math", "dapo"):
        shutil.copy2(old / f"evaluation/prompts-{dataset}.json", evaluation / f"prompts-{dataset}.json")
    write(
        evaluation / "launch.json",
        {
            "cwd": str(source),
            "environment": {"PYTHONPATH": str(source), "CUDA_VISIBLE_DEVICES": "0,1,2,3", "OMP_NUM_THREADS": "2"},
            "command": [
                sys.executable,
                "-u",
                "-m",
                "examples.mirroropd.research.studies.termination.run_e54_topd_mdpo",
                "evaluate",
                "--root",
                str(root),
            ],
        },
    )
    jobs.append({"preset": "evaluation", "directory": str(evaluation)})
    write(root / "plan.json", {"jobs": jobs, "restore_physical_gpus": [0, 1, 2, 3]})
    write(
        root / "protocol.json",
        {
            "models": "Qwen3-1.7B-Base <- Qwen3-30B-A3B-Instruct-2507",
            "objective": "Full-vocabulary MDPO; TOP-D-shaped immediate signal, not full TOP-D reproduction",
            "arms": [{"label": label, "kappa": k, "eta": e} for label, k, e in ARMS],
            "shared_checkpoint": str(checkpoint),
            "restore": ["model", "Adam", "loader"],
            "new_outer_steps": 3,
            "final_outer": 5,
            "optimizer_updates": 80,
            "batch": 64,
            "reuse": 16,
            "lr": 1e-6,
            "seed": 42,
            "train_cap": 16384,
            "termination_token_ids": [151643, 151645],
            "rope": "canonical FP32 buffers",
            "prompt": "unchanged Base common empty-think",
            "rollout_policy": "current student",
            "eval": {"math_n": 64, "dapo_n": 64, "n": 1, "cap": 31744},
            "prior_full_rkl": "historical reference only; primary causal contrast is kappa 0 vs 1 at fixed eta",
            "diagnostics": "MDPO teacher_kl/anchor_kl are on the coarsened action space",
        },
    )
    print(json.dumps({"root": str(root), "jobs": jobs}, indent=2))


def evaluate(repo, root):
    out = root / "evaluation"
    cpu = os.environ | {"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2"}
    protocol = json.loads((root / "protocol.json").read_text())
    arms = [(a["label"], a["kappa"], a["eta"]) for a in protocol["arms"]]
    old = Path(protocol["shared_checkpoint"]).parents[3]
    training, prompt_batches = {}, {}
    for label, _, _ in arms:
        arm = root / label
        assert json.loads((arm / "exit.json").read_text())["exit_code"] == 0
        state = json.loads((arm / "training/checkpoints/global_step_5/trainer_state.json").read_text())
        assert state["actor_optimizer_step"] == state["actor_optimizer_step_attempts"] == 80
        prompt_batches[label] = [
            {
                tuple(r["prompt_ids"])
                for r in rows((old / "warmup" if step <= 2 else arm) / f"debug/rollout-{step}.jsonl")
            }
            for step in range(1, 6)
        ]
        training[label] = rows(arm / "debug/rollout-summary.jsonl")
        execute(
            [
                sys.executable,
                "-m",
                "verl.model_merger",
                "merge",
                "--backend",
                "fsdp",
                "--local_dir",
                str(arm / "training/checkpoints/global_step_5/actor"),
                "--target_dir",
                str(out / f"model-{label}"),
            ],
            out / f"merge-{label}.log",
            cpu,
            300,
        )
    first = prompt_batches[arms[0][0]]
    assert all(batches == first for batches in prompt_batches.values()), "Training prompt batches differ"
    held_out = json.loads((out / "prompts-dapo.json").read_text())
    assert not ({tuple(r["prompt_ids"]) for r in held_out} & set().union(*first)), "DAPO evaluation overlaps training"

    def one(job):
        gpu, label, dataset, temperature, top_p = job
        directory = out / f"{label}-{dataset}"
        directory.mkdir()
        base = [sys.executable, "-u", "-m", "examples.mirroropd.research.studies.termination.evaluate_e54_recovery"]
        execute(
            base
            + [
                "generate",
                "--model",
                str(out / f"model-{label}"),
                "--prompts",
                str(out / f"prompts-{dataset}.json"),
                "--output",
                str(directory),
                "--temperature",
                str(temperature),
                "--top-p",
                str(top_p),
            ],
            directory / "generate.log",
            os.environ
            | {"CUDA_VISIBLE_DEVICES": str(gpu), "OMP_NUM_THREADS": "2", "VLLM_WORKER_MULTIPROC_METHOD": "spawn"},
            3600,
        )
        execute(base + ["grade", "--output", str(directory)], directory / "grade.log", cpu, 600)
        scored = rows(directory / "scored.jsonl")
        assert len(scored) == len({r["prompt_id"] for r in scored}) == 64
        metrics = json.loads((directory / "metrics.json").read_text())
        return {
            "arm": label,
            "dataset": dataset,
            **metrics,
            "correct_and_stopped": sum(bool(r["acc"]) and r["finish_reason"] == "stop" for r in scored),
        }

    results = []
    for dataset, temperature, top_p in [("math", 0.7, 0.9), ("dapo", 1.0, 1.0)]:
        # Finish each wave before reusing its physical GPU slots.
        for start in range(0, len(arms), 4):
            jobs = [
                (gpu, label, dataset, temperature, top_p) for gpu, (label, _, _) in enumerate(arms[start : start + 4])
            ]
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                results.extend(pool.map(one, jobs))
    summary = {
        "evaluation": results,
        "training": training,
        "paired_training_prompts": True,
        "scope": "Single-seed short-horizon pilot, 64 questions per dataset; not a full TOP-D reproduction",
    }
    write(out / "summary.json", summary)
    if protocol.get("conditional_parent"):
        parent = Path(protocol["conditional_parent"])
        previous = json.loads((parent / "evaluation/summary.json").read_text())
        previous_protocol = json.loads((parent / "protocol.json").read_text())
        parameters = {a["label"]: a for a in previous_protocol["arms"] + protocol["arms"]}
        combined = [
            dict(row, kappa=parameters[row["arm"]]["kappa"], eta=parameters[row["arm"]]["eta"])
            for row in previous["evaluation"] + results
        ]
        write(
            out / "sweep-summary.json",
            {
                "evaluation": combined,
                "training": previous["training"] | training,
                "scope": "Exploratory grid with shared pilot questions; independent confirmation required",
            },
        )
        report = [
            "# Qwen3-1.7B-Base ← Qwen3-30B-A3B-Instruct-2507：MDPO 参数对照",
            "",
            "共同第 2 步模型、Adam 和数据状态；各组续训 3 outer、48 次更新。",
            "每个题集 64 题，每题一次；同一小题集参与筛选，结果属于探索性比较。",
            "",
            "| κ | η | 题集 | 正确率 | 判对且停止 | 平均长度 | 触顶 | 重复 8-gram 占比 |",
            "| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in sorted(combined, key=lambda r: (r["kappa"], r["eta"], r["dataset"])):
            report.append(
                f"| {row['kappa']:g} | {row['eta']:g} | {row['dataset']} | {100 * row['accuracy']:.2f}% | "
                f"{row['correct_and_stopped']}/64 | {row['mean_length'] / 1000:.2f}k | {row['cap_count']}/64 | "
                f"{100 * row['repeat8_mean']:.2f}% |"
            )
        (out / "README.md").write_text("\n".join(report) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "evaluate"])
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    repo = REPO
    (prepare if args.action == "prepare" else evaluate)(repo, args.root.resolve())


if __name__ == "__main__":
    main()
