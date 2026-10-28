# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Inference-only Qwen MoE validation with an explicit prompt-template variant.

Uses a local vLLM server, the existing math scorer and evaluation aggregator.
The external supervisor owns exclusive GPUs and post-exit gpu-occupy recovery.
"""

import argparse
import asyncio
import concurrent.futures
import json
import multiprocessing
import os
import random
import subprocess
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

DATASETS = {"MATH-500": 500, "AMC23": 40, "AIME24": 30, "AIME25": 30, "AIME26": 30}
SAMPLES = 16
CAP = 31744


def dump(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def prepare(args):
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    from verl.utils.tokenizer.continuous_token_wiring import create_continuous_token_builder

    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    base = AutoTokenizer.from_pretrained(args.model.parent / "Qwen3-1.7B-Base", local_files_only=True)
    builder = create_continuous_token_builder(tok, model_family="qwen3", hf_model_type="qwen3_moe")
    prompts = []
    offset = 0
    for dataset, expected in DATASETS.items():
        if dataset not in args.datasets:
            offset += expected
            continue
        rows = pq.read_table(args.data / "math" / dataset / "test.parquet").to_pylist()
        assert len(rows) == expected, (dataset, len(rows))
        for index, row in enumerate(rows):
            messages = row["prompt"]
            ids = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=False)
            assert ids == list(builder.build_initial_tokens(messages)), (dataset, index, "builder differs")
            with_flag = tok.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True, return_dict=False, enable_thinking=False
            )
            assert ids == with_flag, "2507 native template unexpectedly uses enable_thinking"
            old = base.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True, return_dict=False, enable_thinking=False
            )
            text = tok.decode(ids, skip_special_tokens=False)
            assert text.endswith("<|im_start|>assistant\n") and len(ids) <= 1024
            assert tok.encode(text, add_special_tokens=False) == ids
            suffix = base.encode("<think>\n\n</think>\n\n", add_special_tokens=False)
            assert old == ids + suffix
            if args.template == "legacy_empty_think":
                ids = old
                text = tok.decode(ids, skip_special_tokens=False)
            prompts.append(
                dict(
                    data_source=dataset,
                    prompt_id=f"{dataset}:{index}",
                    seed_index=offset + index,
                    prompt_ids=ids,
                    input=text,
                    gts=row["reward_model"]["ground_truth"],
                )
            )
        offset += expected
    assert len(prompts) == sum(DATASETS[name] for name in args.datasets)
    dump(args.out / "prompts.json", prompts)
    protocol = dict(
        model=str(args.model),
        model_type="qwen3_moe",
        template=args.template,
        datasets=args.datasets,
        empty_think_block=args.template == "legacy_empty_think",
        builder_matches_native=len(prompts),
        enable_thinking_flag_is_noop=len(prompts),
        base_template_extra_tokens=4,
        tokenizer_width=len(tok),
        prompt_count=len(prompts),
        response_count=len(prompts) * SAMPLES,
        max_prompt_tokens=max(len(r["prompt_ids"]) for r in prompts),
        samples_per_prompt=SAMPLES,
        response_cap=CAP,
        temperature=0.7,
        top_p=0.9,
        top_k=-1,
        repetition_penalty=1.0,
        presence_penalty=0.0,
        stop_ids=[151645, 151643],
        tensor_parallel=2,
        data_parallel=4,
        max_num_seqs_per_replica=64,
        client_concurrency=256,
        gpu_memory_utilization=0.88,
        seed=42,
        scope="Explicit template variant with project evaluation sampling; not the model-card benchmark recipe",
    )
    dump(args.out / "protocol.json", protocol)
    print(json.dumps(protocol), flush=True)
    return prompts, protocol


def init_grader():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = "1"
    # Spawned pool workers use multiprocessing's exit path, which otherwise
    # joins the math verifier's still-idle nested workers before their executor
    # receives shutdown. Close that executor before the child-join phase.
    from multiprocessing.util import Finalize

    Finalize(None, close_grader, exitpriority=20)


def close_grader():
    from verl.utils.reward_score import math_verify

    pool = math_verify._pool
    if pool is not None:
        pool.shutdown(wait=True, cancel_futures=True)
        math_verify._pool = None


def grade(record):
    from verl.utils.reward_score.one_shot_opd.math import reward_func

    result = reward_func(record["data_source"], record["output"], record["gts"])
    result = {key: value.item() if hasattr(value, "item") else value for key, value in result.items()}
    ids = record["response_ids"]
    windows = max(0, len(ids) - 7)
    repeat8 = 1 - len(set(zip(*(ids[i:] for i in range(8)), strict=False))) / windows if windows else 0.0
    return record | result | {"repeat8": repeat8, "status": "ok"}


def aggregate(records):
    import numpy as np

    from verl.utils.metric.evaluation import dataset_slug, evaluation_metrics

    sources = [r["data_source"] for r in records]
    uids = [r["prompt_id"] for r in records]
    infos = {key: [r.get(key) for r in records] for key in ["score", "acc", "format_score", "pred"]}
    infos["__response_length__"] = [r.get("response_length") for r in records]
    infos["__finish_reason__"] = [r.get("finish_reason") for r in records]
    expected = {(r["data_source"], r["prompt_id"]): SAMPLES for r in records}
    metrics = evaluation_metrics(
        sources, uids, infos, expected, sample_indices=[r["sample_index"] for r in records], prompt_ids=uids
    )
    for dataset in DATASETS:
        rows = [r for r in records if r["data_source"] == dataset and r["status"] == "ok"]
        if rows:
            prefix = f"eval/{dataset_slug(dataset)}/generation/"
            metrics[prefix + "repeat8_mean"] = float(np.mean([r["repeat8"] for r in rows]))
            metrics[prefix + "empty_text_fraction"] = float(np.mean([not r["output"].strip() for r in rows]))
    return metrics


async def evaluate(args, prompts, protocol, run):
    import aiohttp

    timeout = aiohttp.ClientTimeout(total=7200, connect=30)
    connector = aiohttp.TCPConnector(limit=256)
    semaphore = asyncio.Semaphore(256)
    records, completed = [], 0
    started = time.time()
    counts = defaultdict(int)
    loop = asyncio.get_running_loop()
    stable_ids = {p["prompt_id"]: p.get("seed_index", i) for i, p in enumerate(prompts)}
    tasks = [(p, index) for p in prompts for index in range(SAMPLES)]
    random.Random(42).shuffle(tasks)
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(max_workers=16, mp_context=context, initializer=init_grader) as pool:
        async with aiohttp.ClientSession(timeout=timeout, connector=connector, trust_env=False) as session:

            async def request(prompt, index, *, cap=CAP, forced=None):
                stable = stable_ids[prompt["prompt_id"]] * SAMPLES + index
                body = dict(
                    model="qwen30b-native",
                    prompt=prompt["prompt_ids"],
                    max_tokens=cap,
                    temperature=0.7,
                    top_p=0.9,
                    top_k=-1,
                    n=1,
                    seed=42 + stable,
                    stop_token_ids=[151645, 151643],
                    ignore_eos=False,
                    repetition_penalty=1.0,
                    presence_penalty=0.0,
                    return_token_ids=True,
                    add_special_tokens=False,
                )
                if forced is not None:
                    body.update(temperature=0, allowed_token_ids=[forced])
                async with session.post(f"http://127.0.0.1:{args.port}/v1/completions", json=body) as response:
                    result = await response.json()
                    if response.status != 200:
                        raise RuntimeError(f"HTTP {response.status}: {result}")
                choice = result["choices"][0]
                ids = choice["token_ids"]
                assert isinstance(ids, list) and 0 < len(ids) <= cap
                assert all(0 <= t < protocol["tokenizer_width"] for t in ids)
                if choice["finish_reason"] == "stop":
                    assert ids[-1] in [151643, 151645], choice
                return dict(
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
                    seed=42 + stable,
                )

            smoke = []
            for stop in [151643, 151645]:
                result = await request(prompts[0], 0, cap=4, forced=stop)
                assert result["response_ids"] == [stop]
                smoke.append(result)
            dump(args.out / "stop-smoke.json", smoke)
            print(f"Both EOS IDs stop correctly; beginning {len(tasks)}-response validation", flush=True)

            with (args.out / "records.jsonl").open("w") as stream:

                async def one(prompt, index):
                    nonlocal completed
                    record = None
                    for attempt in range(3):
                        try:
                            async with semaphore:
                                record = await request(prompt, index)
                            record = await loop.run_in_executor(pool, grade, record)
                            record["attempts"] = attempt + 1
                            break
                        except Exception as exc:
                            if attempt == 2:
                                record = dict(
                                    data_source=prompt["data_source"],
                                    prompt_id=prompt["prompt_id"],
                                    sample_index=index,
                                    input=prompt["input"],
                                    gts=prompt["gts"],
                                    status="failed",
                                    error=repr(exc),
                                    attempts=3,
                                )
                            else:
                                await asyncio.sleep(2)
                    records.append(record)
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                    stream.flush()
                    completed += 1
                    counts[prompt["data_source"]] += 1
                    if completed % 64 == 0 or completed == len(tasks):
                        progress = dict(
                            completed=completed,
                            expected=len(tasks),
                            elapsed_seconds=time.time() - started,
                            by_dataset=dict(counts),
                            failed=sum(r["status"] != "ok" for r in records),
                        )
                        dump(args.out / "progress.json", progress)
                        print(json.dumps(progress), flush=True)
                        if run is not None:
                            run.log(
                                {
                                    "progress/completed": completed,
                                    "progress/failed": progress["failed"],
                                    "progress/elapsed_seconds": progress["elapsed_seconds"],
                                },
                                step=completed,
                            )

                await asyncio.gather(*(one(prompt, index) for prompt, index in tasks))
    assert len(records) == len(tasks)
    assert len({(r["prompt_id"], r["sample_index"]) for r in records}) == len(tasks)
    metrics = aggregate(records)
    metrics["perf/validation_seconds"] = time.time() - started
    dump(args.out / "metrics.json", metrics)
    with (args.out / "metrics.jsonl").open("w") as stream:
        stream.write(json.dumps({"step": 0, "data": metrics}) + "\n")
    if run is not None:
        run.log(metrics | {"eval/round": 0}, step=len(tasks) + 1)
    failed = sum(r["status"] != "ok" for r in records)
    if failed:
        raise RuntimeError(f"Validation produced {failed} failed records; not a complete evaluation")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "run"])
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18654)
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    parser.add_argument("--template", choices=["native", "legacy_empty_think"], default="native")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.action == "prepare":
        prepare(args)
        return
    prompts = json.loads((args.out / "prompts.json").read_text())
    protocol = json.loads((args.out / "protocol.json").read_text())
    run = None
    try:
        import wandb

        run = wandb.init(
            project="MirrorOPD-verl",
            entity="wncfht",
            name=f"Qwen3-30B-A3B-Instruct-2507-{protocol['template']}-{'-'.join(protocol.get('datasets', DATASETS))}",
            group="teacher-reference",
            job_type="validation",
            config=protocol,
            dir=str(args.out),
            mode="online",
        )
        dump(args.out / "wandb.json", dict(id=run.id, url=run.url, mode="online"))
    except Exception as exc:
        dump(args.out / "wandb-error.json", dict(error=repr(exc)))
        print("W&B unavailable; preserving all metrics locally", flush=True)
    command = [
        str(Path(sys.executable).with_name("vllm")),
        "serve",
        str(args.model),
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--served-model-name",
        "qwen30b-native",
        "--dtype",
        "bfloat16",
        "--tensor-parallel-size",
        "2",
        "--data-parallel-size",
        "4",
        "--data-parallel-size-local",
        "4",
        "--data-parallel-backend",
        "mp",
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
        "--compilation-config",
        json.dumps({"cudagraph_mode": "FULL_AND_PIECEWISE", "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32, 64]}),
    ]
    dump(args.out / "server-command.json", command)
    success = False
    with (args.out / "server.log").open("w") as output:
        server = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            deadline = time.monotonic() + 1200
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    raise RuntimeError(f"vLLM server exited with {server.returncode}")
                try:
                    with opener.open(f"http://127.0.0.1:{args.port}/health", timeout=3) as response:
                        if response.status == 200:
                            break
                except Exception:
                    pass
                time.sleep(2)
            else:
                raise TimeoutError("vLLM startup timed out")
            asyncio.run(evaluate(args, prompts, protocol, run))
            success = True
        finally:
            if server.poll() is None:
                server.terminate()
                try:
                    server.wait(timeout=45)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()
            if run is not None:
                run.finish(exit_code=0 if success else 1)


if __name__ == "__main__":
    main()
