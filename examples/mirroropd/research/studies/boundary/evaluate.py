# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Four samples per question, full five-benchmark generation on assigned GPUs."""

import argparse
import asyncio
import concurrent.futures
import multiprocessing
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from examples.mirroropd.research.backends.evaluation import aggregate, request
from examples.mirroropd.research.core.io import dump, read
from examples.mirroropd.research.core.math_grading import grade, init_grader
from examples.mirroropd.research.studies.boundary.common import MODELS, OUT, model_path


async def evaluate(name, port, template, out, *, prompts=None, eos=None, cap=31744, shuffle=False):
    import aiohttp

    if prompts is None:
        prompts = read(
            OUT / "evaluation-prompts.json" if template == "shared" else OUT / "native-prompts" / f"{name}.json"
        )
    if eos is None:
        eos = read(OUT / "interface-audit.json")[name]["eos_token_id"]
    records = []
    dest = out / "records.jsonl"
    if dest.exists():
        import json

        records = [json.loads(line) for line in dest.read_text().splitlines() if line.strip()]
    done = {(r["prompt_id"], r["sample_index"]) for r in records}
    semaphore = asyncio.Semaphore(64)
    loop = asyncio.get_running_loop()
    context = multiprocessing.get_context("spawn")
    started = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=8, mp_context=context, initializer=init_grader) as pool:
        connector = aiohttp.TCPConnector(limit=64, force_close=True)
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=7200), trust_env=False, connector=connector
        ) as session:
            stop_request = dict(
                model="boundary",
                prompt=prompts[0]["prompt_ids"],
                n=1,
                max_tokens=2,
                temperature=0,
                allowed_token_ids=[eos],
                stop_token_ids=[eos],
                ignore_eos=False,
                return_token_ids=True,
                add_special_tokens=False,
            )
            async with session.post(f"http://127.0.0.1:{port}/v1/completions", json=stop_request) as response:
                stop_result = await response.json()
                assert response.status == 200, stop_result
            stop_choice = stop_result["choices"][0]
            assert stop_choice["finish_reason"] == "stop" and stop_choice["token_ids"] == [eos], stop_result
            dump(out / "stop-smoke.json", stop_choice)
            with dest.open("a") as stream:

                async def one(prompt, index):
                    import json

                    async with semaphore:
                        body = dict(
                            model="boundary",
                            prompt=prompt["prompt_ids"],
                            n=1,
                            max_tokens=cap,
                            temperature=0.7,
                            top_p=0.9,
                            top_k=-1,
                            seed=72000000 + 100 * prompt["seed_index"] + index,
                            stop_token_ids=[eos],
                            ignore_eos=False,
                            return_token_ids=True,
                            add_special_tokens=False,
                        )
                        value, attempt = await request(
                            session,
                            f"http://127.0.0.1:{port}/v1/completions",
                            body,
                            out / "request-retries.jsonl",
                            prompt["prompt_id"],
                            index,
                        )
                        choice = value["choices"][0]
                        ids = choice["token_ids"]
                        record = dict(
                            data_source=prompt["data_source"],
                            prompt_id=prompt["prompt_id"],
                            sample_index=index,
                            input=prompt["input"],
                            gts=prompt["gts"],
                            output=choice["text"],
                            response_ids=ids,
                            response_length=len(ids),
                            finish_reason=choice["finish_reason"],
                            stop_reason=choice.get("stop_reason"),
                            seed=body["seed"],
                            request_attempt=attempt,
                        )
                        record = await loop.run_in_executor(pool, grade, record)
                        records.append(record)
                        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                        stream.flush()
                        if len(records) % 32 == 0:
                            progress = dict(completed=len(records), expected=2520, elapsed=time.time() - started)
                            dump(out / "progress.json", progress)
                            print(name, progress, flush=True)

                tasks = [(p, r) for p in prompts for r in range(4) if (p["prompt_id"], r) not in done]
                if shuffle:
                    import random

                    random.Random(42).shuffle(tasks)
                await asyncio.gather(*(one(p, r) for p, r in tasks))
    assert len(records) == 2520 and len({(r["prompt_id"], r["sample_index"]) for r in records}) == 2520
    metrics = aggregate(records, 4)
    dump(out / "metrics.json", metrics)
    dump(out / "complete.json", dict(records=len(records), samples=4, elapsed=time.time() - started))
    dump(out / "progress.json", dict(completed=len(records), expected=2520, elapsed=time.time() - started))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("name", choices=MODELS)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--template", choices=["shared", "native"], default="shared")
    args = parser.parse_args()
    out = OUT / "baseline" / f"{args.name}-{args.template}"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "complete.json").exists():
        return
    command = [
        str(Path(sys.executable).with_name("vllm")),
        "serve",
        str(model_path(args.name)),
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--served-model-name",
        "boundary",
        "--dtype",
        "bfloat16",
        "--tensor-parallel-size",
        "1",
        "--max-model-len",
        "32768",
        "--max-num-seqs",
        "64",
        "--max-num-batched-tokens",
        "8192",
        "--gpu-memory-utilization",
        "0.88",
        "--enable-prefix-caching",
        "--seed",
        "42",
        "--generation-config",
        "vllm",
    ]
    dump(
        out / "protocol.json",
        dict(
            command=command,
            model=MODELS[args.name],
            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
            samples=4,
            template=args.template,
        ),
    )
    with (out / "server.log").open("a") as stream:
        server = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        dump(out / "server-pid.json", dict(pid=server.pid))
        try:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            deadline = time.monotonic() + 1200
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    raise RuntimeError(f"vLLM exited {server.returncode}")
                try:
                    with opener.open(f"http://127.0.0.1:{args.port}/health", timeout=3) as response:
                        if response.status == 200:
                            break
                except Exception:
                    time.sleep(2)
            else:
                raise TimeoutError("vLLM startup")
            asyncio.run(evaluate(args.name, args.port, args.template, out))
        finally:
            try:
                os.killpg(server.pid, signal.SIGTERM)
                server.wait(timeout=45)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait()
            except ProcessLookupError:
                pass


if __name__ == "__main__":
    main()
