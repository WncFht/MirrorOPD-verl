# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Resumable avg@4 evaluation with explicit prompts, stops, identities and grading."""

import asyncio
import json
import multiprocessing
import os
import random
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

from examples.mirroropd.research.backends.evaluation import aggregate, request
from examples.mirroropd.research.core.artifacts import complete, receipt, records
from examples.mirroropd.research.core.io import dump as write
from examples.mirroropd.research.core.io import read
from examples.mirroropd.research.core.math_grading import grade, init_grader
from examples.mirroropd.research.studies.fixed_transfer.spec import fingerprint


def validate_records(rows, prompts, profile):
    prompts_by_id = {p["prompt_id"]: p for p in prompts}
    expected = {(p["prompt_id"], i) for p in prompts for i in range(profile.samples)}
    done = {(r["prompt_id"], r["sample_index"]) for r in rows}
    if len(done) != len(rows) or not done <= expected:
        raise ValueError("Duplicate or foreign evaluation response slots")
    for row in rows:
        prompt = prompts_by_id[row["prompt_id"]]
        seed = profile.seed + 100 * prompt["seed_index"] + row["sample_index"]
        if row["seed"] != seed or any(row[k] != prompt[k] for k in ["data_source", "gts", "input"]):
            raise ValueError(f"Response identity changed: {row['prompt_id']}/{row['sample_index']}")
        if row["response_length"] != len(row["response_ids"]) or not 0 < row["response_length"] <= profile.response_cap:
            raise ValueError("Invalid recorded response length")
        if row["status"] != "ok":
            raise ValueError("Unresolved grading failure in recorded evaluation")
    return done, expected


async def collect(prompts, profile, port, served_name, dest, concurrency):
    import aiohttp

    rows = records(dest / "records.jsonl", repair=True)
    done, expected = validate_records(rows, prompts, profile)
    semaphore = asyncio.Semaphore(concurrency)
    loop = asyncio.get_running_loop()
    started = time.time()
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=8, mp_context=context, initializer=init_grader) as pool:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=7200, connect=30),
            connector=aiohttp.TCPConnector(limit=concurrency, force_close=True),
            trust_env=False,
        ) as session:
            url = f"http://127.0.0.1:{port}/v1/completions"
            for eos in profile.stop_ids:
                smoke = dict(
                    model=served_name,
                    prompt=prompts[0]["prompt_ids"],
                    max_tokens=2,
                    n=1,
                    temperature=0,
                    allowed_token_ids=[eos],
                    stop_token_ids=list(profile.stop_ids),
                    ignore_eos=False,
                    return_token_ids=True,
                    add_special_tokens=False,
                )
                value, _ = await request(session, url, smoke, dest / "request-retries.jsonl", "smoke", eos)
                choice = value["choices"][0]
                assert choice["finish_reason"] == "stop" and choice["token_ids"] == [eos], choice
                write(dest / f"stop-{eos}.json", choice)
            with (dest / "records.jsonl").open("a") as stream:

                async def one(prompt, index):
                    async with semaphore:
                        body = dict(
                            model=served_name,
                            prompt=prompt["prompt_ids"],
                            n=1,
                            max_tokens=profile.response_cap,
                            temperature=profile.temperature,
                            top_p=profile.top_p,
                            top_k=-1,
                            seed=profile.seed + 100 * prompt["seed_index"] + index,
                            stop_token_ids=list(profile.stop_ids),
                            ignore_eos=False,
                            return_token_ids=True,
                            add_special_tokens=False,
                        )
                        value, attempt = await request(
                            session, url, body, dest / "request-retries.jsonl", prompt["prompt_id"], index
                        )
                        choice = value["choices"][0]
                        row = dict(
                            data_source=prompt["data_source"],
                            prompt_id=prompt["prompt_id"],
                            sample_index=index,
                            input=prompt["input"],
                            gts=prompt["gts"],
                            output=choice["text"],
                            response_ids=choice["token_ids"],
                            response_length=len(choice["token_ids"]),
                            finish_reason=choice["finish_reason"],
                            stop_reason=choice.get("stop_reason"),
                            seed=body["seed"],
                            request_attempt=attempt,
                        )
                        assert 0 < len(row["response_ids"]) <= profile.response_cap
                        row = await loop.run_in_executor(pool, grade, row)
                        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                        stream.flush()
                        rows.append(row)
                        if len(rows) % 32 == 0:
                            write(
                                dest / "progress.json",
                                dict(completed=len(rows), expected=len(expected), elapsed=time.time() - started),
                            )
                            print("EVAL", dest.name, len(rows), len(expected), flush=True)

                pending = [(p, i) for p in prompts for i in range(profile.samples) if (p["prompt_id"], i) not in done]
                random.Random(42).shuffle(pending)
                await asyncio.gather(*(one(p, i) for p, i in pending))
    assert len(rows) == len(expected) and {(r["prompt_id"], r["sample_index"]) for r in rows} == expected
    write(dest / "metrics.json", aggregate(rows, profile.samples))
    write(dest / "progress.json", dict(completed=len(rows), expected=len(expected), elapsed=time.time() - started))
    return len(rows)


def evaluation_inputs(study, model_name, weight_identity=None):
    """The same identity definition is used by workers, the queue and analysis."""
    model = study.models[model_name]
    prepared = read(study.directory / "prepared/complete.json")
    return dict(
        weights=weight_identity or model.identity,
        prompts=prepared["evaluation_key"],
        profile=asdict(study.evaluation),
        execution_dtype="bfloat16",
        grader="verl-one-shot-opd-math-v1",
    )


def evaluate(study, model_name, *, weights=None, weight_identity=None, label=None):
    if (weights is None) != (weight_identity is None):
        raise ValueError("Endpoint weights need their own immutable content identity")
    model = study.models[model_name]
    inputs = evaluation_inputs(study, model_name, weight_identity)
    key = fingerprint(inputs)
    dest = study.cache_directory / "evaluations" / key
    link = study.directory / "evaluations" / (label or model_name)
    dest.mkdir(parents=True, exist_ok=True)
    link.parent.mkdir(parents=True, exist_ok=True)
    if not link.exists():
        link.symlink_to(dest.resolve(), target_is_directory=True)
    if link.resolve() != dest.resolve():
        raise ValueError(f"Evaluation alias identity changed: {link}")
    if complete(dest, inputs):
        return
    if (dest / "inputs.json").exists() and fingerprint(read(dest / "inputs.json")) != fingerprint(inputs):
        raise ValueError("Partial evaluation belongs to different inputs")
    if (dest / "records.jsonl").exists() and not (dest / "inputs.json").exists():
        raise ValueError("Recorded answers lack their input identity receipt")
    write(dest / "inputs.json", inputs)
    prompts = read(study.directory / "prepared/evaluation-prompts.json")
    existing = records(dest / "records.jsonl", repair=True)
    done, expected = validate_records(existing, prompts, study.evaluation)
    if done == expected:
        write(dest / "metrics.json", aggregate(existing, study.evaluation.samples))
        write(dest / "progress.json", dict(completed=len(existing), expected=len(expected), recovered_offline=True))
        receipt(dest, inputs, records=len(existing), samples=study.evaluation.samples, recovered_offline=True)
        return
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    served_name = "study-" + key[:16]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if len(visible) != 1 or visible[0] not in {str(g) for g in study.gpus}:
        raise ValueError("Run evaluation through the GPU4-7 resource owner")
    config = read(model.directory / "config.json")
    total = (
        float(
            subprocess.check_output(
                ["nvidia-smi", "-i", visible[0], "--query-gpu=memory.total", "--format=csv,noheader,nounits"], text=True
            ).strip()
        )
        * 2**20
    )
    weight_bytes = sum(r["bytes"] for r in read(model.directory / "complete.json")["weights"])
    head_dim = config.get("head_dim") or config["hidden_size"] // config["num_attention_heads"]
    kv_bytes = (
        4 * config["num_hidden_layers"] * config.get("num_key_value_heads", config["num_attention_heads"]) * head_dim
    )
    capacity = max(
        1,
        int(
            (total * study.execution.gpu_memory_fraction - weight_bytes - 4 * 2**30)
            / (kv_bytes * study.evaluation.model_length)
        ),
    )
    limit = min(study.execution.evaluation_concurrency, capacity)
    concurrency = 2 ** (limit.bit_length() - 1)
    command = [
        str(Path(sys.executable).with_name("vllm")),
        "serve",
        str(weights or model.directory),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--served-model-name",
        served_name,
        "--dtype",
        "bfloat16",
        "--tensor-parallel-size",
        "1",
        "--max-model-len",
        str(study.evaluation.model_length),
        "--max-num-seqs",
        str(concurrency),
        "--max-num-batched-tokens",
        "8192",
        "--gpu-memory-utilization",
        str(study.execution.gpu_memory_fraction),
        "--enable-prefix-caching",
        "--seed",
        "42",
        "--generation-config",
        "vllm",
    ]
    attempt = dest / "attempts" / str(time.time_ns())
    attempt.mkdir(parents=True)
    write(
        attempt / "protocol.json",
        dict(
            inputs=inputs,
            command=command,
            visible_gpus=os.environ.get("CUDA_VISIBLE_DEVICES"),
            concurrency=concurrency,
            request_order="seed42 shuffle; per-answer seed fixed",
        ),
    )
    with (attempt / "server.log").open("w") as log:
        server = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=dict(os.environ, VLLM_WORKER_MULTIPROC_METHOD="spawn"),
        )
        write(attempt / "pid.json", dict(pid=server.pid))
        try:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            deadline = time.monotonic() + 1200
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    raise RuntimeError(f"vLLM exited {server.returncode}")
                try:
                    with opener.open(f"http://127.0.0.1:{port}/v1/models", timeout=3) as response:
                        listing = json.load(response)
                    if served_name in [r["id"] for r in listing["data"]]:
                        break
                    time.sleep(0.5)
                except Exception:
                    time.sleep(2)
            else:
                raise TimeoutError("vLLM startup")
            count = asyncio.run(collect(prompts, study.evaluation, port, served_name, dest, concurrency))
            receipt(dest, inputs, records=count, samples=study.evaluation.samples)
        finally:
            try:
                os.killpg(server.pid, signal.SIGTERM)
                server.wait(timeout=45)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait()
            except ProcessLookupError:
                pass
