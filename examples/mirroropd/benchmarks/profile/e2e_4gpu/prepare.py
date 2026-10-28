# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Prepare an immutable-source, idle-gated whole-training profile plan."""

import argparse
import importlib.util
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

parser = argparse.ArgumentParser(description="Prepare three profiling jobs; CPU config export only.")
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--root", type=Path, required=True)
parser.add_argument("--model-root", type=Path, required=True)
parser.add_argument("--data-root", type=Path, required=True)
args = parser.parse_args()
source = args.source.resolve()
repo = source
root = args.root.resolve()
python = sys.executable
if (root / "plan.json").exists():
    raise SystemExit("Refusing to overwrite an existing plan; use a new evidence directory.")
subprocess.run(["git", "diff", "--exit-code", "HEAD", "--"], cwd=source, check=True, stdout=subprocess.DEVNULL)
spec = importlib.util.spec_from_file_location("q", repo / "examples/mirroropd/tools/queue_full_vocab_ablations.py")
q = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q)
plan = {
    "python": python,
    "model_root": str(args.model_root.resolve()),
    "data_root": str(args.data_root.resolve()),
    "name": "reasoning-e2e",
}
jobs = []
for name, steps, tool in [("baseline", 14, None), ("torch-profile", 5, "torch"), ("nsys-profile", 2, "nsys")]:
    job = {
        "id": name,
        "kind": "engineering",
        "method": "full",
        "seed": 42,
        "steps": steps,
        "wandb_id": "unused",
        "async_write": True,
        "prefetch": True,
    }
    cmd, env, _ = q.public_launch(plan, job, source, root, eager=False)
    cmd = [
        a for a in cmd if not a.startswith("++actor_rollout_ref.rollout.engine_kwargs.vllm.attention_config.backend=")
    ]
    env.update(
        WANDB_MODE="disabled",
        FULL_KL_BACKEND="precompute",
        FULL_KL_TOKEN_CHUNK="512",
        VERL_E2E_TIMELINE_DIR=str(root / name / "events"),
        # Ray's Nsight plugin launches bare `python`, overriding py_executable.
        PATH=str(Path(python).parent) + os.pathsep + os.environ.get("PATH", ""),
    )
    cmd += [
        "trainer.logger=[console,file]",
        "actor_rollout_ref.rollout.disable_log_stats=False",
        f"++ray_kwargs.ray_init.runtime_env.env_vars.VERL_E2E_TIMELINE_DIR={root / name / 'events'}",
    ]
    if name == "baseline":
        cmd += [
            "trainer.save_freq=14",
            "trainer.test_freq=14",
            "data.val_max_samples=8",
            "data.val_files=['" + str(args.data_root.resolve() / "math/MATH-500/test.parquet") + "']",
            "actor_rollout_ref.rollout.val_kwargs.n=2",
            "actor_rollout_ref.rollout.val_kwargs.max_tokens=1024",
        ]
    if tool:
        cmd += [
            f"global_profiler.tool={tool}",
            f"global_profiler.steps=[{4 if tool == 'torch' else 2}]",
            f"global_profiler.save_path={root / name / 'traces'}",
            "actor_rollout_ref.actor.profiler.enable=True",
            "actor_rollout_ref.actor.profiler.all_ranks=True",
            "global_profiler.relocate_results=True",
        ]
    if tool == "torch":
        cmd += [
            "actor_rollout_ref.actor.profiler.tool_config.torch.contents=[cuda]",
            "actor_rollout_ref.rollout.profiler.tool_config.torch.discrete=True",
            "actor_rollout_ref.rollout.profiler.tool_config.torch.profile_token_start=0",
            "actor_rollout_ref.rollout.profiler.tool_config.torch.profile_token_end=128",
        ]
    if tool == "nsys":
        env["VERL_FULL_KL_NVTX"] = "1"
        cmd += [
            "++ray_kwargs.ray_init.runtime_env.env_vars.VERL_FULL_KL_NVTX='1'",
            "global_profiler.global_tool_config.nsys.worker_nsight_options.trace='cuda,nvtx'",
            "global_profiler.global_tool_config.nsys.controller_nsight_options.trace=none",
            "++global_profiler.global_tool_config.nsys.controller_nsight_options.sample=none",
            "++global_profiler.global_tool_config.nsys.controller_nsight_options.cpuctxsw=none",
            "global_profiler.global_tool_config.nsys.worker_nsight_options.capture-range-end=stop-shutdown",
            "global_profiler.finish_hook_all_ranks=True",
            "global_profiler.finish_hook_cmd="
            + json.dumps(
                shlex.join([python, str(source / "examples/mirroropd/benchmarks/profile/e2e_4gpu/finish_nsys.py")])
            ),
        ]
    (root / name).mkdir(parents=True, exist_ok=True)
    # Config export is CPU only and does not allocate Ray resources.
    cfg = subprocess.run(
        cmd,
        cwd=source,
        env=os.environ | env | {"CONFIG_ONLY": "1", "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
    )
    (root / name / "resolved_config.yaml").write_text(cfg.stdout)
    (root / name / "config-export.stderr").write_text(cfg.stderr)
    assert cfg.returncode == 0, cfg.stderr[-2000:]
    wrapped = [
        python,
        str(source / "examples/mirroropd/benchmarks/profile/e2e_4gpu/run_observed.py"),
        "--directory",
        str(root / name),
        "--",
        *cmd,
    ]
    jobs.append({"name": name, "command": wrapped, "environment": env, "timeout": 5400})
commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
(root / "plan.json").write_text(
    json.dumps({"source": str(source), "source_commit": commit, "jobs": jobs}, indent=2) + "\n"
)
print("Prepared", len(jobs), "jobs at", commit)
