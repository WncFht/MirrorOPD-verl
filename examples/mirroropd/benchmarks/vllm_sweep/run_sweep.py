#!/usr/bin/env python3
# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
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
from report import SUMMARY_FIELDS, build_report

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
            isinstance(result.get(k), int | float) and math.isfinite(result[k]) and result[k] > 0
            for k in ("duration", "output_throughput", "total_output_tokens")
        )
    )


def rank_candidates(rows):
    groups = {}
    for row in rows:
        if row.get("status") == "complete" and row.get("resource_isolated") is not False:
            groups.setdefault(tuple(row["candidate"]), []).append(row)
    if not groups:
        return []
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
    return sorted(ranked, key=lambda x: (x["mean_duration"], -x["pooled_output_tps"]))


def choose_best(rows):
    ranked = rank_candidates(rows)
    return ranked[0] if ranked else None


def candidate_slug(candidate):
    seqs, memory, tokens = candidate
    return f"c{seqs}-u{round(memory * 100):03d}-t{tokens}"


def validate_plan_extension(old, new):
    """Permit new search points without changing the workload or historical points."""
    mutable = {"schema_version", "priority_aggressive", "memory_headroom_budget", "models"}
    for key in (old.keys() | new.keys()) - mutable:
        if old.get(key) != new.get(key):
            raise ValueError(f"Cannot change workload field {key} in an existing campaign")
    old_models = {p["id"]: p for p in old["models"]}
    new_models = {p["id"]: p for p in new["models"]}
    if old_models.keys() != new_models.keys():
        raise ValueError("An extension must retain the same model collection")
    for name, previous in old_models.items():
        current = new_models[name]
        for key in (previous.keys() | current.keys()) - {"candidates", "aggressive"}:
            if previous.get(key) != current.get(key):
                raise ValueError(f"Cannot change model protocol {name}.{key}")
        for key in ("candidates", "aggressive"):
            before, after = previous.get(key, []), current.get(key, [])
            if after[: len(before)] != before:
                raise ValueError(f"Cannot replace historical candidates {name}.{key}")
            for cap, memory, tokens in after:
                if type(cap) is not int or cap <= 0 or not 0 < memory < 1 or type(tokens) is not int or tokens <= 0:
                    raise ValueError(f"Invalid resource candidate for {name}")


def is_search_stage(stage):
    return stage.startswith(("candidate", "aggressive-", "memory-"))


def validate_original_outputs(directory):
    records = directory / "validation/records/0.jsonl"
    count = sum(1 for _ in records.open()) if records.exists() else 0
    metrics = {}
    history = directory / "metrics.jsonl"
    if history.exists():
        with history.open() as stream:
            for line in stream:
                metrics.update(json.loads(line).get("data", {}))
    expected = {"math500": 8000, "amc23": 640, "aime25": 480}
    checks = {}
    for name, responses in expected.items():
        prefix = f"eval/{name}/"
        checks[name] = {
            "response_count": metrics.get(prefix + "response_count"),
            "failed_count": metrics.get(prefix + "failed_count"),
            "acc_mean": metrics.get(prefix + "acc/mean"),
        }
    valid = count == 9120 and all(
        checks[name]["response_count"] == responses
        and checks[name]["failed_count"] == 0
        and isinstance(checks[name]["acc_mean"], int | float)
        and math.isfinite(checks[name]["acc_mean"])
        for name, responses in expected.items()
    )
    report = {"valid": valid, "record_count": count, "datasets": checks}
    write_json(directory / "validation_summary.json", report)
    return report


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
        for name in ("run_sweep.py", "prepare_data.py", "report.py", "plan.json"):
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
            previous = json.loads(saved.read_text())
            if not getattr(args, "extend_plan", False):
                raise ValueError("Plan changed; use --extend-plan for additive resource candidates")
            if any(previous[key] != snapshot[key] for key in ("model_root", "data_root")):
                raise ValueError("Cannot change asset roots in an existing campaign")
            validate_plan_extension(previous["plan"], snapshot["plan"])
            stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d-%H%M%S-%f")
            write_json(self.root / "plan_history" / f"resolved-{stamp}.json", previous)
        write_json(saved, snapshot)

    def save(self):
        self.state["updated"] = now()
        write_json(self.state_path, self.state)
        rows = [r for r in self.state["jobs"].values() if r.get("kind") == "sweep"]
        with (self.root / "summary.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        selected = {}
        for profile in self.profiles:
            model = profile["id"]
            ranked = rank_candidates([r for r in rows if r["model"] == model])
            native = ranked[0] if ranked else None
            verification = self.state["jobs"].get(model + "/verl-validation", {})
            usable = next(
                (
                    r
                    for r in ranked
                    if verification.get("status") == "complete" and r["candidate"] == verification.get("candidate")
                ),
                None,
            )
            if native is not None:
                choice = dict(usable or native)
                choice.update(native_best=native, original_verl_verified=usable is not None)
                choice.update(
                    zip(
                        ("max_num_seqs", "gpu_memory_utilization", "max_num_batched_tokens"),
                        choice["candidate"],
                        strict=True,
                    )
                )
                selected[model] = choice
            else:
                selected[model] = None
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
                    entry = {"time": now(), "monotonic_time": time.monotonic(), "gpu": gpu_snapshot(self.plan["gpus"])}
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
        if (
            self.state["jobs"].get(key, {}).get("status") == "complete"
            and self.state["jobs"][key].get("resource_isolated") is not False
        ):
            if self.state["jobs"][key]["candidate"] != candidate:
                raise ValueError(f"Completed stage {key} has different parameters")
            return self.state["jobs"][key]
        previous = self.state["jobs"].get(key, {})
        if previous.get("status") == "failed" and not getattr(self.args, "retry_failed", False):
            return previous
        directory = self.root / "runs" / key
        directory.mkdir(parents=True, exist_ok=True)
        if self.args.action != "dry-run" and (directory / "run.log").exists():
            attempt = directory / "attempts" / datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d-%H%M%S-%f")
            attempt.mkdir(parents=True)
            for name in (
                "native",
                "run.log",
                "telemetry.jsonl.gz",
                "final-metrics.txt",
                "command.json",
                "derived_metrics.json",
            ):
                path = directory / name
                if path.exists():
                    path.rename(attempt / name)
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
        result = None
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
        if not preflight:
            row.update(build_report(row, result, self.plan["response_cap"]))
            write_json(directory / "derived_metrics.json", row)
        self.state["jobs"][key] = row
        self.state["active"] = None
        self.save()
        print("FINISH " + json.dumps(row), flush=True)
        return row

    def original_validation(self, profile, selected):
        stage = "verl-validation-" + candidate_slug(selected["candidate"])
        key = profile["id"] + "/" + stage
        if self.state["jobs"].get(key, {}).get("status") == "complete":
            if self.state["jobs"][key]["candidate"] != selected["candidate"]:
                raise ValueError(f"Validated stage {key} has different parameters")
            return self.state["jobs"][key]
        if self.state["jobs"].get(key, {}).get("status") == "failed" and not getattr(self.args, "retry_failed", False):
            return self.state["jobs"][key]
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
            # Match the native benchmark's unmodified stopping protocol.
            VAL_REPETITION_DETECTION="false",
            VAL_BATCH_SIZE="null",
            ROLLOUT_TP="1",
            ROLLOUT_ENFORCE_EAGER="False",
            SEED=str(self.plan["seed"]),
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
            f"++ray_kwargs.ray_init._temp_dir=/tmp/mopd-sweep-{os.getpid()}-{profile['id']}-{stage}",
            f"++ray_kwargs.ray_init.runtime_env.env_vars.VERL_FILE_LOGGER_PATH={directory / 'metrics.jsonl'}",
        ]
        if profile["family"] == "qwen3":
            command += ["+data.apply_chat_template_kwargs.enable_thinking=False"]
        if profile["extra_stop_ids"]:
            command += ["actor_rollout_ref.rollout.stop_token_ids=" + json.dumps(profile["extra_stop_ids"])]
        self.state["active"] = {
            "model": profile["id"],
            "stage": stage,
            "started": now(),
            "log": str(directory / "run.log"),
        }
        self.state["status"] = "running"
        self.save()
        print("START " + json.dumps(self.state["active"]), flush=True)
        code, _ = self.execute(command, directory, env=env)
        report = validate_original_outputs(directory)
        row = {
            "model": profile["id"],
            "stage": stage,
            "kind": "validation",
            "candidate": selected["candidate"],
            "exit_code": code,
            "records": report["record_count"],
            "status": "complete" if code == 0 and report["valid"] else "failed",
            "log": str(directory / "run.log"),
        }
        self.state["jobs"][key] = row
        self.state["active"] = None
        self.save()
        print("FINISH " + json.dumps(row), flush=True)
        return row

    def model_rows(self, profile):
        return [r for r in self.state["jobs"].values() if r.get("model") == profile["id"] and r.get("kind") == "sweep"]

    def search(self, profile, stage, candidate):
        for row in self.model_rows(profile):
            if row["candidate"] == candidate and is_search_stage(row["stage"]):
                if row.get("resource_isolated") is False:
                    continue
                if row["status"] == "complete" or not getattr(self.args, "retry_failed", False):
                    return row
        return self.sweep(profile, stage, candidate)

    def expand(self, profile):
        first = self.state["jobs"].get(profile["id"] + "/candidate0", {})
        second = self.state["jobs"].get(profile["id"] + "/candidate1", {})
        budget = self.plan.get("memory_headroom_budget", 0.9)
        if (
            second.get("status") == "complete"
            and second["candidate"][1] < budget
            and (second.get("preemptions", 0) > 0 or second.get("observed_peak_kv", 0) > 0.85)
        ):
            candidate = [second["candidate"][0], budget, second["candidate"][2]]
            self.search(profile, "memory-" + candidate_slug(candidate), candidate)
        threshold = self.plan["candidate2_improvement"]
        if self.plan.get("priority_aggressive") or (
            second.get("status") == "complete"
            and (
                first.get("status") != "complete"
                or second["duration"] < first["duration"] * (1 - threshold)
                or second["output_throughput"] > first["output_throughput"] * (1 + threshold)
            )
        ):
            self.search(profile, "candidate2", profile["candidates"][2])
        for candidate in profile.get("aggressive", []):
            self.search(profile, "aggressive-" + candidate_slug(candidate), candidate)

    def search_all(self):
        # Snapshot eligibility before new jobs: finish the ready models' extensions
        # before waiting for the remaining models' first full validations.
        if self.plan.get("priority_aggressive"):
            ready = [
                p
                for p in self.profiles
                if all(
                    self.state["jobs"].get(f"{p['id']}/candidate{i}", {}).get("status") == "complete"
                    and self.state["jobs"].get(f"{p['id']}/candidate{i}", {}).get("resource_isolated") is not False
                    for i in (0, 1)
                )
            ]
            for profile in ready:
                self.expand(profile)
        for profile in self.profiles:
            for index in (0, 1):
                self.sweep(profile, f"candidate{index}", profile["candidates"][index])
            self.expand(profile)

    def refresh_reports(self):
        for row in self.state["jobs"].values():
            if row.get("kind") != "sweep":
                continue
            # Existing length/latency columns need no expensive reread of token
            # traces when only resource interpretation has changed.
            result = (
                json.loads(Path(row["result"]).read_text())
                if row.get("result") and not row.get("report_schema")
                else None
            )
            row.update(build_report(row, result, self.plan["response_cap"]))
            del result
            write_json(Path(row["log"]).parent / "derived_metrics.json", row)
            print(f"REPORT {row['model']}/{row['stage']}", flush=True)
        self.save()

    def confirm(self, profile):
        for selected in rank_candidates(self.model_rows(profile)):
            row = self.original_validation(profile, selected)
            if row["status"] == "complete":
                self.state["jobs"][profile["id"] + "/verl-validation"] = dict(row)
                self.save()
                return

    def run(self):
        signal.signal(signal.SIGINT, self.stop)
        signal.signal(signal.SIGTERM, self.stop)
        lock_path = Path("/tmp/mirroropd-gpus-0-1-2-3.lock")
        with lock_path.open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if self.args.action == "report":
                self.refresh_reports()
                return
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
            self.search_all()
            for profile in self.profiles:
                rows = [r for r in self.model_rows(profile) if is_search_stage(r["stage"])]
                selected = choose_best(rows)
                if selected is None:
                    continue
                higher_tokens = selected["candidate"][:2] + [16384]
                self.sweep(profile, "token16384-" + candidate_slug(higher_tokens), higher_tokens)
                rows = [r for r in self.model_rows(profile) if not r["stage"].startswith("repeat")]
                selected = choose_best(rows)
                self.sweep(profile, "repeat-" + candidate_slug(selected["candidate"]), selected["candidate"])
                self.confirm(profile)
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
    parser.add_argument("action", choices=["preflight", "dry-run", "run", "report"])
    parser.add_argument("--plan", type=Path, default=Path(__file__).with_name("plan.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, default=MODEL_DEFAULT)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--models", nargs="*")
    parser.add_argument("--extend-plan", action="store_true", help="Accept additive resource candidates only")
    parser.add_argument("--retry-failed", action="store_true", help="Retry previously failed configurations")
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
