#!/usr/bin/env python3
# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Prepare and run a detached, GPU-idle-gated full-vocab ablation queue.

Plans contain public configuration only. Read credentials at launch, never serialize
them. A pinned checkout and absolute asset paths isolate runs from later edits.
"""

import argparse
import csv
import fcntl
import json
import netrc
import os
import signal
import statistics
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import psutil

GPUS = [0, 1, 2, 3]
BASE_URL = "https://api.forge.coreweave.com"


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def make_jobs():
    jobs = [{"id": "preflight", "kind": "preflight", "method": "full", "seed": 42, "steps": 2}]
    for chunk in (512, 256, 1024):
        jobs.append({"id": f"kernel-c{chunk}", "kind": "kernel", "chunk": chunk})
    # Deliver the primary algorithm comparison before the engineering and repeat-seed jobs.
    for method, seed in (("vanilla", 42), ("full", 42), ("token_is", 42)):
        jobs.append(
            {"id": f"quality-{method}-s{seed}", "kind": "quality", "method": method, "seed": seed, "steps": 100}
        )
    for async_write, prefetch in ((False, False), (True, False), (False, True), (True, True)):
        jobs.append(
            {
                "id": f"io-write{int(async_write)}-prefetch{int(prefetch)}",
                "kind": "engineering",
                "method": "full",
                "seed": 42,
                "steps": 8,
                "async_write": async_write,
                "prefetch": prefetch,
            }
        )
    for method in ("full", "vanilla"):
        jobs.append({"id": f"quality-{method}-s43", "kind": "quality", "method": method, "seed": 43, "steps": 100})
    for job in jobs:
        job["wandb_id"] = uuid.uuid4().hex[:8] if job["kind"] != "kernel" else None
    return jobs


def gpu_snapshot():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    )
    rows = {}
    for row in csv.reader(result.stdout.splitlines(), skipinitialspace=True):
        idx = int(row[0])
        if idx in GPUS:
            rows[idx] = {"uuid": row[1], "memory_mib": int(row[2]), "utilization": int(row[3])}
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    )
    occupied = {row[0] for row in csv.reader(result.stdout.splitlines(), skipinitialspace=True) if row}
    idle = len(rows) == len(GPUS) and all(
        row["memory_mib"] < 1024 and row["utilization"] < 10 and row["uuid"] not in occupied for row in rows.values()
    )
    return idle, rows


def public_launch(plan, job, source, root, *, eager=False, attempt=1):
    directory = root / job["id"] / f"attempt-{attempt}"
    if job["kind"] == "kernel":
        backends = ["recompute", "cached", "streaming", "precompute"]
        if job["chunk"] == 512:
            backends.append("liger")
        command = [
            plan["python"],
            str(source / "examples/mirroropd/docs/full-vocab/scripts/benchmark_full_kl.py"),
            "--tokens",
            "4096",
            "32768",
            "--chunk",
            str(job["chunk"]),
            "--backends",
            *backends,
            "--reference-backend",
            "streaming",
            "--output",
            str(directory / "kernels.json"),
        ]
        return command, {"CUDA_VISIBLE_DEVICES": "0", "PYTHONPATH": str(source)}, directory
    preflight = job["kind"] == "preflight"
    quality = job["kind"] == "quality"
    env = {
        "CUDA_VISIBLE_DEVICES": "0,1,2,3",
        "STUDENT_GPUS": "4",
        "TEACHER_GPUS": "0",
        "DOMAIN": "math",
        "DISTILLATION_EXECUTION_MODE": "shared_sync",
        "VERL_USE_UV": "0",
        "PYTHON_BIN": plan["python"],
        "PYTHONPATH": str(source),
        "MODEL_ROOT": plan["model_root"],
        "DATA_ROOT": plan["data_root"],
        "STUDENT_MODEL": str(Path(plan["model_root"]) / "DeepSeek-R1-Distill-Qwen-1.5B"),
        "TEACHER_MODEL": str(Path(plan["model_root"]) / "JustRL-DeepSeek-1.5B"),
        "TRAIN_FILE": str(Path(plan["data_root"]) / "math/DAPO/train.parquet"),
        "MATH_FILE": str(Path(plan["data_root"]) / "math/MATH-500/test.parquet"),
        "AMC_FILE": str(Path(plan["data_root"]) / "math/AMC23/test.parquet"),
        "AIME_FILE": str(Path(plan["data_root"]) / "math/AIME25/test.parquet"),
        "TOTAL_EPOCHS": "1",
        "ROLLOUT_TP": "1",
        "TEACHER_TP": "1",
        "TEACHER_FSDP_DTYPE": "bfloat16",
        "OPD_CLIP_EPS": "0.2",
        "DRY_RUN": "0",
        "CONFIG_ONLY": "0",
        "TRAIN_BATCH_SIZE": "8" if preflight else "64",
        "PPO_EPOCHS": "1",
        "SEED": str(job["seed"]),
        "LEARNING_RATE": "1e-6",
        "MAX_PROMPT_LENGTH": "4096",
        "MAX_RESPONSE_LENGTH": "128" if preflight else "7168",
        "MAX_VAL_RESPONSE_LENGTH": "31744",
        "VAL_N": "16",
        "VAL_BATCH_SIZE": "null",
        "PPO_MAX_TOKEN_LEN_PER_GPU": "32768",
        "TEACHER_FSDP_TOKEN_BUDGET": "32768",
        "ROLLOUT_MAX_NUM_SEQS": "128",
        "ROLLOUT_GPU_MEMORY_UTILIZATION": "0.65",
        "ROLLOUT_TOKEN_BUDGET": "8192",
        "ROLLOUT_ENFORCE_EAGER": str(eager),
        "PARAM_OFFLOAD": "True",
        "OPTIMIZER_OFFLOAD": "True",
        "GRADIENT_CHECKPOINTING": "True",
        "FULL_KL_BACKEND": "precompute",
        "FULL_KL_TOKEN_CHUNK": "512",
        "FULL_KL_VOCAB_CHUNK": "8192",
        "FULL_KL_ASYNC_WRITE": str(job.get("async_write", True)),
        "FULL_KL_PREFETCH": str(job.get("prefetch", True)),
        "OPD_ADVANTAGE_SOURCE": "current",
        "OPD_POLICY_LOSS_MODE": job["method"] if job["method"] == "token_is" else "vanilla",
        "TEST_FREQ": "20" if quality else "-1",
        "SAVE_FREQ": "50" if quality else "-1",
        "EXPERIMENT_NAME": f"{plan['name']}-{job['id']}-a{attempt}",
        "RUN_DIR": str(directory / "training"),
        "WANDB_MODE": "online",
        "WANDB_ENTITY": "wncfht",
        "WANDB_BASE_URL": BASE_URL,
        "WANDB_RUN_ID": job["wandb_id"] if attempt == 1 else job["wandb_id"] + str(attempt),
        "WANDB_RUN_GROUP": plan["name"],
        "WANDB_TAGS": "full-vocab-ablation,math,b38hw8v0-reference",
        "RAY_ADDRESS": "local",
        "RAY_USAGE_STATS_ENABLED": "0",
        "NCCL_DEBUG": "WARN",
        "OMP_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "NVTX_INJECTION64_PATH": "/nonexistent/verl-disabled-nvtx-injection.so",
    }
    entry = "run_full_reverse_kl.sh" if job["method"] == "full" else "run_math_deepseek_r1_1_5b_fsdp2.sh"
    command = [
        "bash",
        str(source / "examples/mirroropd" / entry),
        f"trainer.total_training_steps={job['steps']}",
        "trainer.val_before_train=False",
        "trainer.max_actor_ckpt_to_keep=1",
        "trainer.logger=[console,wandb,file]",
        "actor_rollout_ref.actor.calculate_entropy=False",
        f"actor_rollout_ref.actor.fsdp_config.seed={job['seed']}",
        f"actor_rollout_ref.actor.data_loader_seed={job['seed']}",
        f"actor_rollout_ref.rollout.seed={job['seed']}",
        "++ray_kwargs.ray_init.num_gpus=4",
        "++ray_kwargs.ray_init.num_cpus=64",
        "++ray_kwargs.ray_init.include_dashboard=False",
        "++ray_kwargs.ray_init.object_store_memory=8589934592",
        f"++ray_kwargs.ray_init._temp_dir=/tmp/{plan['name']}-{job['id']}-a{attempt}",
        "++actor_rollout_ref.rollout.engine_kwargs.vllm.attention_config.backend=TRITON_ATTN",
    ]
    safe_ray_env = {
        key: env[key]
        for key in ("WANDB_ENTITY", "WANDB_BASE_URL", "WANDB_RUN_ID", "WANDB_RUN_GROUP", "WANDB_TAGS", "PYTHONPATH")
    }
    safe_ray_env.update(WANDB_DIR=str(directory), VERL_FILE_LOGGER_PATH=str(directory / "metrics.jsonl"))
    command += [f"++ray_kwargs.ray_init.runtime_env.env_vars.{key}='{value}'" for key, value in safe_ray_env.items()]
    return command, env, directory


def add_credentials(env, credentials_path):
    auth = netrc.netrc(credentials_path).authenticators("api.forge.coreweave.com")
    if auth is None:
        auth = netrc.netrc(credentials_path).authenticators("api.wandb.ai")
    if not auth:
        raise RuntimeError("No W&B credentials for the configured host")
    env["WANDB_API_KEY"] = auth[2]
    env["NETRC"] = str(credentials_path)


def cleanup_owned(process, owned):
    # Never ray stop / pkill: terminate only descendants recorded from this launch.
    if process.poll() is None:
        process.terminate()
    alive = []
    for pid, birth in owned.items():
        try:
            child = psutil.Process(pid)
            if child.create_time() == birth and child.status() != psutil.STATUS_ZOMBIE:
                child.terminate()
                alive.append(child)
        except psutil.NoSuchProcess:
            pass
    _, remaining = psutil.wait_procs(alive, timeout=10)
    for child in remaining:
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def validate_result(job, directory):
    if job["kind"] == "kernel":
        return {"rows": len(json.loads((directory / "kernels.json").read_text())["rows"])}
    records = [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines()]
    rows = [record.get("data", record) for record in records]
    updates = max((r.get("step/optimizer", 0) for r in rows), default=0)
    if updates != job["steps"]:
        raise RuntimeError(f"Expected {job['steps']} optimizer updates, found {updates}")
    if any(r.get("optim/actor/update_skipped", 0) or r.get("optim/actor/skipped_updates", 0) for r in rows):
        raise RuntimeError("An optimizer update was skipped")
    evaluations = [row for row in rows if "eval/math500/response_count" in row]
    if job["kind"] == "quality":
        steps = {int(row["step/outer"]) for row in evaluations}
        if not set(range(20, job["steps"] + 1, 20)).issubset(steps):
            raise RuntimeError("Missing a scheduled full validation")
        for row in evaluations:
            for dataset, expected in (("math500", 8000), ("amc23", 640), ("aime25", 480)):
                if (
                    row.get(f"eval/{dataset}/response_count") != expected
                    or row.get(f"eval/{dataset}/failed_count") != 0
                ):
                    raise RuntimeError(f"Incomplete validation for {dataset}")
    steady = [row for row in rows if row.get("step/outer", 0) > 1 and "perf/timing/outer_step_seconds" in row]
    timings = {
        key: statistics.mean(row[key] for row in steady if key in row)
        for key in {key for row in steady for key in row if key.startswith("perf/timing/")}
    }
    return {
        "optimizer_steps": updates,
        "metric_rows": len(rows),
        "evaluation_rows": sum(any(k.startswith("eval/") for k in row) for row in rows),
        "steady_stage_seconds": timings,
        "final_evaluation": {k: v for k, v in evaluations[-1].items() if k.startswith("eval/")} if evaluations else {},
    }


def run_queue(args):
    root, source = args.root.resolve(), args.source.resolve()
    plan = json.loads((root / "plan.json").read_text())
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip() != plan["source_commit"]:
        raise RuntimeError("Source checkout does not match the pinned plan")
    state = {"pid": os.getpid(), "state": "starting", "started_at": now(), "jobs": {}, "eager": False}
    state_path = root / "status.json"

    def status(**updates):
        state.update(updates, updated_at=now())
        write_json(state_path, state)

    def wait_idle():
        since = None
        while True:
            if (root / "STOP").exists():
                raise KeyboardInterrupt
            try:
                idle, snapshot = gpu_snapshot()
                since = (since or time.monotonic()) if idle else None
                status(state="waiting_gpus", gpu_snapshot=snapshot)
                if since is not None and time.monotonic() - since >= args.idle_seconds:
                    return
            except (subprocess.SubprocessError, ValueError):
                since = None
                status(state="waiting_gpu_query")
            time.sleep(args.poll_seconds)

    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    lock_path = Path("/tmp/mirroropd-gpus-0-1-2-3.lock")
    with lock_path.open("a") as lock:
        status(state="waiting_queue_lock")
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            for job in plan["jobs"]:
                state["current_job"] = job["id"]
                attempts = 2 if job["kind"] == "preflight" else 1
                ok = False
                for attempt in range(1, attempts + 1):
                    if attempt == 2:
                        state["eager"] = True
                    wait_idle()
                    command, public_env, directory = public_launch(
                        plan, job, source, root, eager=state["eager"], attempt=attempt
                    )
                    directory.mkdir(parents=True, exist_ok=False)
                    write_json(
                        directory / "launch.json",
                        {
                            "command": command,
                            "environment_overrides": public_env,
                            "source_commit": plan["source_commit"],
                        },
                    )
                    env = os.environ.copy()
                    env.update(public_env)
                    env["PATH"] = str(Path(plan["python"]).parent) + os.pathsep + env.get("PATH", "")
                    if job["kind"] != "kernel":
                        add_credentials(env, args.netrc_file)
                    status(state="running", current_attempt=attempt, current_directory=str(directory))
                    owned = {}
                    begin = time.monotonic()
                    timeout = 12 * 3600 if job["kind"] == "quality" else 3600
                    with (directory / "driver.log").open("w") as log:
                        process = subprocess.Popen(
                            command, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
                        )
                        try:
                            while process.poll() is None:
                                try:
                                    for child in psutil.Process(process.pid).children(recursive=True):
                                        owned[child.pid] = child.create_time()
                                except psutil.NoSuchProcess:
                                    pass
                                status(state="running", child_pid=process.pid, elapsed_seconds=time.monotonic() - begin)
                                if time.monotonic() - begin > timeout:
                                    raise TimeoutError("Experiment exceeded its wall-time limit")
                                time.sleep(10)
                            result = {
                                "exit_code": process.returncode,
                                "wall_seconds": time.monotonic() - begin,
                                "finished_at": now(),
                                "state": "failed",
                            }
                            if process.returncode == 0:
                                try:
                                    result.update(validate_result(job, directory), state="completed")
                                    ok = True
                                except (RuntimeError, OSError, ValueError) as exc:
                                    result["validation_error"] = str(exc)
                        except TimeoutError:
                            result = {"exit_code": None, "state": "timed_out", "finished_at": now()}
                        except KeyboardInterrupt:
                            result = {"exit_code": None, "state": "interrupted", "finished_at": now()}
                            write_json(directory / "exit.json", result)
                            state["jobs"][job["id"]] = {**result, "directory": str(directory)}
                            raise
                        finally:
                            cleanup_owned(process, owned)
                    write_json(directory / "exit.json", result)
                    state["jobs"][job["id"]] = {
                        **result,
                        "directory": str(directory),
                        "attempt": attempt,
                        "wandb_url": f"https://forge.coreweave.com/wandb/wncfht/MirrorOPD-verl/runs/{public_env.get('WANDB_RUN_ID')}"
                        if job["kind"] != "kernel"
                        else None,
                    }
                    status(state="between_jobs")
                    if ok:
                        break
                if job["kind"] == "preflight" and not ok:
                    status(state="blocked_preflight")
                    return
            status(state="finished", current_job=None)
            write_json(root / "results.json", state["jobs"])
        except KeyboardInterrupt:
            status(state="stopped")
        except Exception as exc:
            status(state="failed_setup", error_type=type(exc).__name__)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "run"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--python", default="/root/.local/share/mirroropd/venvs/profile-cu128/bin/python")
    parser.add_argument("--model-root", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--netrc-file", type=Path)
    parser.add_argument("--idle-seconds", type=float, default=60)
    parser.add_argument("--poll-seconds", type=float, default=30)
    args = parser.parse_args()
    if args.action == "prepare":
        if args.root.exists():
            parser.error("Use a new queue root")
        if args.model_root is None or args.data_root is None:
            parser.error("prepare requires absolute asset roots")
        source = args.source.resolve()
        plan = {
            "name": args.root.name,
            "created_at": now(),
            "gpus": GPUS,
            "reference": "wncfht/MirrorOPD-verl/b38hw8v0",
            "python": args.python,
            "model_root": str(args.model_root.resolve()),
            "data_root": str(args.data_root.resolve()),
            "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip(),
            "jobs": make_jobs(),
        }
        write_json(args.root / "plan.json", plan)
        print(json.dumps({"root": str(args.root.resolve()), "jobs": len(plan["jobs"])}))
    else:
        if args.netrc_file is None or not args.netrc_file.is_file():
            parser.error("run requires an existing --netrc-file")
        if args.poll_seconds <= 0 or args.idle_seconds < 0:
            parser.error("invalid polling interval")
        with (args.root / "runner.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if (args.root / "status.json").exists():
                parser.error("This queue has already started; inspect status.json or prepare a new root")
            run_queue(args)


if __name__ == "__main__":
    main()
