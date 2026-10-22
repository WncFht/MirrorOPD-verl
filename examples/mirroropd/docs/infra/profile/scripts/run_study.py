# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Idle-gated sequential study on physical GPUs 0..3; resumable per-job evidence.

Keeps the existing shared GPU queue lock. Never kills unrelated GPU processes.
STOP requests graceful exit after the current child (or immediately while idle).
"""

import argparse
import fcntl
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def select_chunk(result, backend):
    rows = [
        r
        for r in result.get("rows", [])
        if r.get("backend") == backend and r.get("tokens") == 32768 and "wall_ms_median" in r
    ]
    return min(rows, key=lambda r: r["wall_ms_median"])["chunk"] if rows else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--samples", type=Path, required=True)
    p.add_argument("--model-root", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--idle-seconds", type=int, default=120)
    p.add_argument("--poll-seconds", type=int, default=30)
    p.add_argument("--skip-e2e", action="store_true")
    p.add_argument("--publish-dir", type=Path)
    args = p.parse_args()
    source, root = args.source.resolve(), args.root.resolve()
    scripts = source / "examples/mirroropd/docs/infra/profile/scripts"
    helpers = load_module(source / "examples/mirroropd/queue_full_vocab_ablations.py", "queue_helpers")
    state = {
        "pid": os.getpid(),
        "state": "starting",
        "jobs": {},
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip(),
    }

    def status(**changes):
        state.update(changes, updated_at=datetime.now(timezone.utc).isoformat())
        save(root / "status.json", state)

    def wait_idle():
        since = None
        while True:
            if (root / "STOP").exists():
                raise KeyboardInterrupt from None
            try:
                idle, snapshot = helpers.gpu_snapshot()
                since = (since or time.monotonic()) if idle else None
                status(state="waiting_gpus", gpu_snapshot=snapshot)
                if since is not None and time.monotonic() - since >= args.idle_seconds:
                    return
            except (ValueError, subprocess.SubprocessError) as exc:
                since = None
                status(state="waiting_gpu_query", error=str(exc))
            time.sleep(args.poll_seconds)

    def run(name, command, env=None, timeout=3600, gpu=True):
        directory = root / name
        completed = directory / "exit.json"
        if completed.exists():
            old = json.loads(completed.read_text())
            state["jobs"][name] = old
            return old["returncode"] == 0
        if gpu:
            wait_idle()
        elif (root / "STOP").exists():
            raise KeyboardInterrupt from None
        directory.mkdir(parents=True, exist_ok=True)
        public = {
            "CUDA_VISIBLE_DEVICES": "0,1,2,3",
            "PYTHONPATH": str(source),
            "OMP_NUM_THREADS": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "NCCL_DEBUG": "WARN",
        }
        public.update(env or {})
        save(
            directory / "launch.json",
            {"command": command, "environment_overrides": public, "source_commit": state["source_commit"]},
        )
        status(state="running", current_job=name)
        owned = {}
        with (directory / "stdout.log").open("w") as log:
            child = subprocess.Popen(
                command,
                cwd=source,
                env=os.environ | public,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            begin = time.monotonic()
            try:
                while child.poll() is None:
                    try:
                        for proc in psutil.Process(child.pid).children(recursive=True):
                            owned[proc.pid] = proc.create_time()
                    except psutil.NoSuchProcess:
                        pass
                    if time.monotonic() - begin > timeout:
                        raise TimeoutError(name)
                    time.sleep(2)
                code = child.returncode
            except TimeoutError:
                code = 124
            finally:
                helpers.cleanup_owned(child, owned)
        record = {"returncode": code, "wall_seconds": time.monotonic() - begin}
        save(completed, record)
        state["jobs"][name] = record
        status(state="between_jobs")
        return code == 0

    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        with Path("/tmp/mirroropd-gpus-0-1-2-3.lock").open("a") as lock:
            status(state="waiting_queue_lock")
            # Nonblocking lock polling honors STOP while another queue owns the lock.
            while True:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if (root / "STOP").exists():
                        raise KeyboardInterrupt from None
                    time.sleep(args.poll_seconds)
            kernel_ok = run(
                "kernel",
                [
                    sys.executable,
                    str(scripts / "benchmark.py"),
                    "--output",
                    str(root / "kernel/benchmark.json"),
                    "--trace",
                ],
                {"CUDA_VISIBLE_DEVICES": "0"},
            )
            result = (
                json.loads((root / "kernel/benchmark.json").read_text())
                if (root / "kernel/benchmark.json").exists()
                else {}
            )
            if not kernel_ok:
                run(
                    "kernel-baseline",
                    [
                        sys.executable,
                        str(scripts / "benchmark.py"),
                        "--output",
                        str(root / "kernel-baseline/benchmark.json"),
                        "--backends",
                        "precompute",
                        "streaming",
                        "--trace",
                    ],
                    {"CUDA_VISIBLE_DEVICES": "0"},
                )
            run(
                "nsys-head",
                [
                    "nsys",
                    "profile",
                    "--trace=cuda,nvtx",
                    "--sample=none",
                    "--cpuctxsw=none",
                    "--capture-range=cudaProfilerApi",
                    "--capture-range-end=stop",
                    "--force-overwrite=true",
                    "-o",
                    str(root / "nsys-head/head"),
                    sys.executable,
                    str(scripts / "benchmark.py"),
                    "--output",
                    str(root / "nsys-head/benchmark.json"),
                    "--tokens",
                    "4096",
                    "--chunks",
                    "512",
                    "--backends",
                    "precompute",
                    "--repeats",
                    "2",
                    "--capture-nsys",
                ],
                {"CUDA_VISIBLE_DEVICES": "0", "VERL_FULL_KL_NVTX": "1"},
            )

            def pipeline(name, backend, chunk, checkpoint=True):
                cmd = [
                    sys.executable,
                    "-m",
                    "torch.distributed.run",
                    "--standalone",
                    "--nproc-per-node=4",
                    str(scripts / "pipeline.py"),
                    "--student",
                    str(args.model_root / "DeepSeek-R1-Distill-Qwen-1.5B"),
                    "--teacher",
                    str(args.model_root / "JustRL-DeepSeek-1.5B"),
                    "--samples",
                    str(args.samples),
                    "--output",
                    str(root / name / "pipeline.json"),
                    "--backend",
                    backend,
                    "--chunk",
                    str(chunk),
                    "--trace",
                ]
                if name == "pipeline-baseline":
                    cmd.append("--memory")
                if not checkpoint:
                    cmd.append("--no-checkpoint")
                return run(name, cmd)

            base_ok = pipeline("pipeline-baseline", "precompute", 512)
            candidate_chunk = select_chunk(result, "fused_candidate") if kernel_ok else None
            if base_ok and candidate_chunk:
                candidate_ok = pipeline("pipeline-fused", "fused_candidate", candidate_chunk)
                if candidate_ok:
                    run(
                        "pipeline-agreement",
                        [
                            sys.executable,
                            str(scripts / "validate_pipeline.py"),
                            "--baseline",
                            str(root / "pipeline-baseline/pipeline.json"),
                            "--candidate",
                            str(root / "pipeline-fused/pipeline.json"),
                            "--output",
                            str(root / "pipeline-agreement/checks.json"),
                        ],
                        gpu=False,
                    )
            best = select_chunk(result, "precompute") or 512
            if base_ok and best != 512:
                pipeline("pipeline-chunk", "precompute", best)
            if base_ok:
                pipeline("pipeline-no-checkpoint", "precompute", best, False)
            if not args.skip_e2e:
                plan = {
                    "python": sys.executable,
                    "model_root": str(args.model_root),
                    "data_root": str(args.data_root),
                    "name": "full-kl-profile",
                }
                for both in (False, True):
                    name = "e2e-io" + str(int(both))
                    job = {
                        "id": name,
                        "kind": "engineering",
                        "method": "full",
                        "seed": 42,
                        "steps": 4,
                        "wandb_id": "unused",
                        "async_write": both,
                        "prefetch": both,
                    }
                    cmd, env, _ = helpers.public_launch(plan, job, source, root, eager=True)
                    env.update(WANDB_MODE="disabled", FULL_KL_TOKEN_CHUNK=str(best))
                    cmd += [
                        "trainer.logger=[console]",
                        "global_profiler.tool=torch",
                        "global_profiler.steps=[3]",
                        "global_profiler.profile_continuous_steps=True",
                        f"global_profiler.save_path={root / name / 'traces'}",
                        "actor_rollout_ref.actor.profiler.enable=True",
                        "actor_rollout_ref.actor.profiler.all_ranks=True",
                        "actor_rollout_ref.actor.profiler.tool_config.torch.contents=[cuda]",
                    ]
                    run(name, cmd, env, timeout=5400)
            run(
                "analysis",
                [sys.executable, str(scripts / "analyze.py"), "--root", str(root), "--figures", str(root / "figures")],
                gpu=False,
            )
            report_command = [sys.executable, str(scripts / "report.py"), "--root", str(root)]
            if args.publish_dir:
                report_command += ["--publish-dir", str(args.publish_dir)]
            run("report", report_command, gpu=False)
            status(
                state="completed"
                if all(j["returncode"] == 0 for j in state["jobs"].values())
                else "completed_with_failures"
            )
    except KeyboardInterrupt:
        status(state="stopped")
    except Exception as exc:
        status(state="failed", error=repr(exc))
        raise


if __name__ == "__main__":
    main()
