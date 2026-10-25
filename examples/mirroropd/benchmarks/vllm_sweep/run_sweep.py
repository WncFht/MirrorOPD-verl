#!/usr/bin/env python3
"""Queue native vLLM sweeps and original-verl validation on one exclusive GPU pool."""

import argparse
import csv
import fcntl
import gzip
import json
import math
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from prepare_data import prepare, write_json

REPO = Path(__file__).resolve().parents[4]
MODEL_DEFAULT = REPO.parent / "model"
METRIC_NAMES = (
    "vllm:kv_cache_usage_perc",
    "vllm:num_preemptions",
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:generation_tokens",
    "vllm:prompt_tokens",
)


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


def capture_metrics(url, output):
    import requests

    response = requests.get(url + "/metrics", timeout=15)
    response.raise_for_status()
    Path(output).write_text(response.text)


def gpu_snapshot(gpus):
    import pynvml

    pynvml.nvmlInit()
    try:
        rows = []
        for index in gpus:
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            rows.append(
                {
                    "gpu": index,
                    "used_MiB": memory.used / 2**20,
                    "utilization": pynvml.nvmlDeviceGetUtilizationRates(handle).gpu,
                    "power_W": pynvml.nvmlDeviceGetPowerUsage(handle) / 1000,
                    "pids": [p.pid for p in pynvml.nvmlDeviceGetComputeRunningProcesses(handle)],
                }
            )
        return rows
    finally:
        pynvml.nvmlShutdown()


def require_idle(gpus, timeout=45):
    deadline = time.monotonic() + timeout
    while True:
        rows = gpu_snapshot(gpus)
        if all(r["used_MiB"] < 1024 and r["utilization"] < 10 and not r["pids"] for r in rows):
            return rows
        if time.monotonic() >= deadline:
            raise RuntimeError(f"Allocated GPUs are occupied; no processes were terminated: {rows}")
        time.sleep(3)


def capture_sizes(cap):
    values, size = [], 1
    while size < cap:
        values.append(size)
        size *= 2
    return values + [cap]


def build_commands(plan, profile, candidate, model_root, data_file, run_dir, vllm, python, preflight=False):
    seqs, memory, tokens = candidate
    model = str(model_root / profile["directory"])
    port, url = 18180, "http://127.0.0.1:18180"
    serve = [
        str(vllm),
        "serve",
        model,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--dtype",
        "bfloat16",
        "--tensor-parallel-size",
        "1",
        "--data-parallel-size",
        str(len(plan["gpus"])),
        "--data-parallel-size-local",
        str(len(plan["gpus"])),
        "--data-parallel-backend",
        "mp",
        "--max-model-len",
        str(profile["prompt_cap"] + plan["response_cap"]),
        "--seed",
        str(plan["seed"]),
        "--enable-prefix-caching",
        "--worker-cls",
        "vllm_sweep_worker.ProtocolWorker",
        "--override-generation-config",
        json.dumps(
            {
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": -1,
                "repetition_penalty": 1.0,
            }
        ),
        "--compilation-config",
        json.dumps(
            {
                "cudagraph_mode": "FULL_AND_PIECEWISE",
                "cudagraph_capture_sizes": capture_sizes(seqs),
            }
        ),
    ]
    extra = {"add_special_tokens": False}
    if profile["extra_stop_ids"]:
        extra["stop_token_ids"] = profile["extra_stop_ids"]
    bench = [
        str(vllm),
        "bench",
        "serve",
        "--model",
        model,
        "--backend",
        "vllm",
        "--base-url",
        url,
        "--endpoint",
        "/v1/completions",
        "--dataset-name",
        "custom",
        "--dataset-path",
        str(data_file),
        "--skip-chat-template",
        "--disable-shuffle",
        "--no-oversample",
        "--num-prompts",
        "4" if preflight else "0",
        "--request-rate",
        "inf",
        "--max-concurrency",
        str(plan["client_concurrency"]),
        "--temperature",
        str(plan["temperature"]),
        "--top-p",
        str(plan["top_p"]),
        "--top-k",
        "-1",
        "--repetition-penalty",
        "1.0",
        "--seed",
        str(plan["seed"]),
        "--extra-body",
        json.dumps(extra),
        "--num-warmups",
        "0",
        "--save-detailed",
    ]
    if preflight:
        bench += ["--custom-output-len", "64"]
    else:
        # CustomDataset's CLI default is 256, which overrides JSONL output_tokens.
        # -1 is required to honor every record's explicit 31744-token budget.
        bench += ["--custom-output-len", "-1"]
    params = [
        {
            "_benchmark_name": "configured",
            "max_num_seqs": seqs,
            "gpu_memory_utilization": memory,
            "max_num_batched_tokens": tokens,
        }
    ]
    write_json(run_dir / "serve-params.json", params)
    after = [str(python), str(Path(__file__).resolve()), "capture", url, str(run_dir / "final-metrics.txt")]
    command = [
        str(vllm),
        "bench",
        "sweep",
        "serve",
        "--serve-cmd",
        shlex.join(serve),
        "--bench-cmd",
        shlex.join(bench),
        "--serve-params",
        str(run_dir / "serve-params.json"),
        "--after-bench-cmd",
        shlex.join(after),
        "--num-runs",
        "1",
        "--show-stdout",
        "--server-ready-timeout",
        "900",
        "--output-dir",
        str(run_dir / "native"),
        "--experiment-name",
        "measurement",
        "--resume",
    ]
    # Native --resume requires the experiment directory to exist.
    (run_dir / "native/measurement").mkdir(parents=True, exist_ok=True)
    return command


def eligible(result, expected=9120):
    return (
        result.get("completed") == expected
        and result.get("failed", 1) == 0
        and all(
            isinstance(result.get(k), (int, float)) and math.isfinite(result[k]) and result[k] > 0
            for k in ("duration", "output_throughput", "total_output_tokens")
        )
    )


def choose_best(rows):
    groups = {}
    for row in rows:
        if row.get("status") == "complete":
            groups.setdefault(tuple(row["candidate"]), []).append(row)
    if not groups:
        return None
    ranked = []
    for candidate, repeats in groups.items():
        ranked.append(
            {
                "candidate": list(candidate),
                "runs": [r["stage"] for r in repeats],
                "mean_duration": sum(r["duration"] for r in repeats) / len(repeats),
                "pooled_output_tps": sum(r["total_output_tokens"] for r in repeats)
                / sum(r["duration"] for r in repeats),
            }
        )
    return min(ranked, key=lambda x: (x["mean_duration"], -x["pooled_output_tps"]))


class Queue:
    def __init__(self, args):
        self.args = args
        self.root = args.output.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.plan = json.loads(args.plan.read_text())
        if self.plan["gpus"] != [0, 1, 2, 3]:
            raise ValueError("This queue is authorized for physical GPUs 0-3 only")
        unknown = set(args.models or []) - {p["id"] for p in self.plan["models"]}
        if unknown:
            raise ValueError(f"Unknown model IDs: {sorted(unknown)}")
        self.profiles = [p for p in self.plan["models"] if not args.models or p["id"] in args.models]
        if not self.profiles:
            raise ValueError("No matching models")
        # Resolving a venv's python symlink would silently select the base interpreter.
        self.python = args.python.absolute()
        self.vllm = self.python.with_name("vllm")
        self.state_path = self.root / "state.json"
        self.state = (
            json.loads(self.state_path.read_text()) if self.state_path.exists() else {"jobs": {}, "status": "ready"}
        )
        self.state["models_in_scope"] = [p["id"] for p in self.profiles]
        self.env = dict(
            os.environ,
            CUDA_VISIBLE_DEVICES=",".join(map(str, self.plan["gpus"])),
            PYTHONPATH=str(self.root / "source") + os.pathsep + str(REPO),
            VLLM_WORKER_MULTIPROC_METHOD="spawn",
            TOKENIZERS_PARALLELISM="false",
            OMP_NUM_THREADS="1",
            NVTX_INJECTION64_PATH="/nonexistent/verl-disabled-nvtx-injection.so",
            WANDB_MODE="offline",
            VERL_USE_UV="0",
            RAY_ADDRESS="local",
            RAY_USAGE_STATS_ENABLED="0",
        )
        self.process = None
        self.active = None
        source = self.root / "source"
        source.mkdir(exist_ok=True)
        for name in ("run_sweep.py", "prepare_data.py", "plan.json"):
            target = source / name
            if not target.exists() or not any(r.get("kind") == "sweep" for r in self.state["jobs"].values()):
                shutil.copyfile(Path(__file__).with_name(name), target)
        target = source / "vllm_sweep_worker.py"
        if not target.exists() or not any(r.get("kind") == "sweep" for r in self.state["jobs"].values()):
            shutil.copyfile(Path(__file__).with_name("worker.py"), target)
        snapshot = {
            "plan": self.plan,
            "model_root": str(args.model_root.resolve()),
            "data_root": str(args.data_root.resolve()),
        }
        saved = self.root / "resolved_plan.json"
        if saved.exists() and json.loads(saved.read_text()) != snapshot:
            raise ValueError("Existing run uses a different plan or asset roots; choose a fresh run directory")
        write_json(saved, snapshot)

    def save(self):
        self.state["updated"] = now()
        write_json(self.state_path, self.state)
        rows = [r for r in self.state["jobs"].values() if r.get("kind") == "sweep"]
        fields = [
            "model",
            "stage",
            "status",
            "candidate",
            "completed",
            "failed",
            "duration",
            "total_output_tokens",
            "output_throughput",
            "observed_peak_kv",
            "preemptions",
            "log",
        ]
        with (self.root / "summary.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        selected = {p["id"]: choose_best([r for r in rows if r["model"] == p["id"]]) for p in self.profiles}
        write_json(
            self.root / "selected_params.json", {"provisional": self.state["status"] != "complete", "models": selected}
        )

    def stop(self, signum, frame):
        self.state["status"] = "interrupted"
        if self.process is not None and self.process.poll() is None:
            self.process.send_signal(signal.SIGINT)
            try:
                self.process.wait(timeout=45)
            except subprocess.TimeoutExpired:
                import psutil

                parent = psutil.Process(self.process.pid)
                for child in parent.children(recursive=True):
                    child.kill()
                parent.kill()
        self.save()
        raise SystemExit(128 + signum)

    def execute(self, command, directory, url=None, env=None):
        import requests

        directory.mkdir(parents=True, exist_ok=True)
        require_idle(self.plan["gpus"])
        write_json(directory / "command.json", {"argv": command, "gpus": self.plan["gpus"]})
        stop = threading.Event()
        telemetry = {"peak_kv": 0.0}

        def monitor():
            with gzip.open(directory / "telemetry.jsonl.gz", "at") as stream:
                while not stop.is_set():
                    entry = {"time": now(), "gpu": gpu_snapshot(self.plan["gpus"])}
                    if url:
                        try:
                            response = requests.get(url + "/metrics", timeout=2)
                            lines = [x for x in response.text.splitlines() if x.startswith(METRIC_NAMES)]
                            entry["engine_metrics"] = lines
                            for line in lines:
                                if line.startswith("vllm:kv_cache_usage_perc"):
                                    telemetry["peak_kv"] = max(telemetry["peak_kv"], float(line.rsplit(" ", 1)[1]))
                        except (requests.RequestException, ValueError):
                            pass
                    stream.write(json.dumps(entry) + "\n")
                    stream.flush()
                    stop.wait(5)

        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
        try:
            with (directory / "run.log").open("a") as log:
                self.process = subprocess.Popen(command, env=env or self.env, stdout=log, stderr=subprocess.STDOUT)
                self.state["active"]["pid"] = self.process.pid
                self.save()
                code = self.process.wait()
        finally:
            stop.set()
            thread.join(timeout=10)
            self.process = None
        require_idle(self.plan["gpus"])
        return code, telemetry

    def sweep(self, profile, stage, candidate, preflight=False):
        key = profile["id"] + "/" + stage
        if self.state["jobs"].get(key, {}).get("status") == "complete":
            if self.state["jobs"][key]["candidate"] != candidate:
                raise ValueError(f"Completed stage {key} has different parameters")
            return self.state["jobs"][key]
        directory = self.root / "runs" / key
        directory.mkdir(parents=True, exist_ok=True)
        command = build_commands(
            self.plan,
            profile,
            candidate,
            self.args.model_root.resolve(),
            self.root / "data" / f"{profile['id']}.jsonl",
            directory,
            self.vllm,
            self.python,
            preflight,
        )
        if self.args.action == "dry-run":
            command.append("--dry-run")
            subprocess.run(command, env=self.env, check=True)
            return {}
        self.state["active"] = {
            "model": profile["id"],
            "stage": stage,
            "started": now(),
            "log": str(directory / "run.log"),
        }
        self.state["status"] = "running"
        self.save()
        print("START " + json.dumps(self.state["active"]), flush=True)
        code, telemetry = self.execute(command, directory, "http://127.0.0.1:18180")
        row = {
            "model": profile["id"],
            "stage": stage,
            "candidate": candidate,
            "exit_code": code,
            "status": "failed",
            "kind": "preflight" if preflight else "sweep",
            "log": str(directory / "run.log"),
            "observed_peak_kv": telemetry["peak_kv"],
        }
        paths = list(directory.glob("native/measurement/**/run=0.json"))
        if code == 0 and len(paths) == 1:
            result = json.loads(paths[0].read_text())
            for name in ("completed", "failed", "duration", "total_output_tokens", "output_throughput"):
                row[name] = result.get(name)
            row["result"] = str(paths[0])
            expected = 4 if preflight else 570 * self.plan["samples_per_prompt"]
            with (directory / "run.log").open() as stream:
                cap_verified = any(f"custom_output_len={64 if preflight else -1}," in line for line in stream)
            row["response_cap_verified"] = cap_verified
            if eligible(result, expected) and cap_verified:
                row["status"] = "complete"
        metrics = directory / "final-metrics.txt"
        if metrics.exists():
            row["preemptions"] = sum(
                float(x.rsplit(" ", 1)[1])
                for x in metrics.read_text().splitlines()
                if re.match(r"vllm:num_preemptions(_total)?\{", x)
            )
        self.state["jobs"][key] = row
        self.state["active"] = None
        self.save()
        print("FINISH " + json.dumps(row), flush=True)
        return row

    def original_validation(self, profile, selected):
        key = profile["id"] + "/verl-validation"
        if self.state["jobs"].get(key, {}).get("status") == "complete":
            if self.state["jobs"][key]["candidate"] != selected["candidate"]:
                raise ValueError(f"Validated stage {key} has different parameters")
            return
        seqs, memory, tokens = selected["candidate"]
        directory = self.root / "runs" / key
        directory.mkdir(parents=True, exist_ok=True)
        env = dict(
            self.env,
            STUDENT_MODEL=str(self.args.model_root.resolve() / profile["directory"]),
            DATA_ROOT=str(self.args.data_root.resolve()),
            STUDENT_GPUS="4",
            TEACHER_GPUS="0",
            PYTHON_BIN=str(self.python),
            CONTINUOUS_TOKEN_MODEL_FAMILY=profile["family"],
            MAX_PROMPT_LENGTH=str(profile["prompt_cap"]),
            MAX_VAL_RESPONSE_LENGTH="31744",
            VAL_N="16",
            ROLLOUT_MAX_NUM_SEQS=str(seqs),
            ROLLOUT_GPU_MEMORY_UTILIZATION=str(memory),
            ROLLOUT_TOKEN_BUDGET=str(tokens),
            EXPERIMENT_NAME=f"sweep-confirm-{profile['id']}",
            RUN_DIR=str(directory),
        )
        command = [
            "bash",
            str(REPO / "examples/mirroropd/run_math_validation_fsdp2.sh"),
            "student",
            "data.train_max_samples=64",
            "data.filter_overlong_prompts_workers=1",
            "trainer.logger=[console,file]",
            "++ray_kwargs.ray_init.num_gpus=4",
            "++ray_kwargs.ray_init.num_cpus=32",
            "++ray_kwargs.ray_init.include_dashboard=False",
            "++ray_kwargs.ray_init.object_store_memory=8589934592",
            f"++ray_kwargs.ray_init._temp_dir=/tmp/mopd-sweep-{os.getpid()}-{profile['id']}",
            f"++ray_kwargs.ray_init.runtime_env.env_vars.VERL_FILE_LOGGER_PATH={directory / 'metrics.jsonl'}",
        ]
        if profile["family"] == "qwen3":
            command += ["+data.apply_chat_template_kwargs.enable_thinking=False"]
        if profile["extra_stop_ids"]:
            command += ["actor_rollout_ref.rollout.stop_token_ids=" + json.dumps(profile["extra_stop_ids"])]
        self.state["active"] = {
            "model": profile["id"],
            "stage": "verl-validation",
            "started": now(),
            "log": str(directory / "run.log"),
        }
        self.save()
        print("START " + json.dumps(self.state["active"]), flush=True)
        code, _ = self.execute(command, directory, env=env)
        records = directory / "validation/records/0.jsonl"
        count = sum(1 for _ in records.open()) if records.exists() else 0
        row = {
            "model": profile["id"],
            "stage": "verl-validation",
            "kind": "validation",
            "candidate": selected["candidate"],
            "exit_code": code,
            "records": count,
            "status": "complete" if code == 0 and count == 9120 else "failed",
            "log": str(directory / "run.log"),
        }
        self.state["jobs"][key] = row
        self.state["active"] = None
        self.save()
        print("FINISH " + json.dumps(row), flush=True)

    def run(self):
        signal.signal(signal.SIGINT, self.stop)
        signal.signal(signal.SIGTERM, self.stop)
        lock_path = Path("/tmp/mirroropd-gpus-0-1-2-3.lock")
        with lock_path.open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if not (self.root / "data/preparation.json").exists():
                prepare(self.plan, self.args.model_root.resolve(), self.args.data_root.resolve(), self.root / "data")
            if self.args.action in ("preflight", "dry-run"):
                for profile in self.profiles:
                    row = self.sweep(profile, "preflight", profile["candidates"][0], preflight=True)
                    if self.args.action == "preflight" and row.get("status") != "complete":
                        raise RuntimeError("Preflight failed; inspect the recorded log")
                self.state["status"] = "preflight-complete" if self.args.action == "preflight" else "prepared"
                self.save()
                return
            # Every checkpoint gets the first two full measurements before follow-ups.
            for profile in self.profiles:
                for index in (0, 1):
                    self.sweep(profile, f"candidate{index}", profile["candidates"][index])
            for profile in self.profiles:
                first = self.state["jobs"][profile["id"] + "/candidate0"]
                second = self.state["jobs"][profile["id"] + "/candidate1"]
                threshold = self.plan["candidate2_improvement"]
                if second["status"] == "complete" and (
                    first["status"] != "complete"
                    or second["duration"] < first["duration"] * (1 - threshold)
                    or second["output_throughput"] > first["output_throughput"] * (1 + threshold)
                ):
                    self.sweep(profile, "candidate2", profile["candidates"][2])
                rows = [
                    r
                    for r in self.state["jobs"].values()
                    if r["model"] == profile["id"] and r["kind"] == "sweep" and r["stage"].startswith("candidate")
                ]
                selected = choose_best(rows)
                if selected is None:
                    continue
                higher_tokens = selected["candidate"][:2] + [16384]
                self.sweep(profile, "token16384", higher_tokens)
                rows = [
                    r
                    for r in self.state["jobs"].values()
                    if r["model"] == profile["id"] and r["kind"] == "sweep" and r["stage"] != "repeat"
                ]
                selected = choose_best(rows)
                self.sweep(profile, "repeat", selected["candidate"])
                rows = [r for r in self.state["jobs"].values() if r["model"] == profile["id"] and r["kind"] == "sweep"]
                self.original_validation(profile, choose_best(rows))
            all_validated = all(
                self.state["jobs"].get(p["id"] + "/verl-validation", {}).get("status") == "complete"
                for p in self.profiles
            )
            self.state["status"] = "complete" if all_validated else "incomplete"
            self.state["active"] = None
            self.save()
            print("QUEUE_END " + self.state["status"], flush=True)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "capture":
        capture_metrics(sys.argv[2], sys.argv[3])
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["preflight", "dry-run", "run"])
    parser.add_argument("--plan", type=Path, default=Path(__file__).with_name("plan.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, default=MODEL_DEFAULT)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--models", nargs="*")
    args = parser.parse_args()
    queue = Queue(args)
    try:
        queue.run()
    except Exception as exc:
        queue.state["status"] = "error"
        queue.state["last_error"] = repr(exc)
        queue.save()
        raise


if __name__ == "__main__":
    main()
