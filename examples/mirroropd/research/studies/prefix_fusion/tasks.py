# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Pure-model task accuracy with shared prompts, grading, and per-response seeds."""

import argparse
import json
import multiprocessing
import os
import random
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

from examples.mirroropd.research.backends.evaluation import aggregate
from examples.mirroropd.research.core.artifacts import complete, file_hash, fingerprint, receipt, records
from examples.mirroropd.research.core.io import dump, read
from examples.mirroropd.research.studies.fixed_transfer.evaluation import validate_records
from examples.mirroropd.research.studies.prefix_fusion.grading import grade, init_grader
from examples.mirroropd.research.studies.prefix_fusion.protocol import PROTOCOL, protocol_identity


def validate_raw(rows, prompts, profile):
    """Check response identity without implying that grading has succeeded."""
    for row in rows:
        if "status" in row or row["finish_reason"] not in {"stop", "length"}:
            raise ValueError("Invalid raw generation record")
    return validate_records([dict(row, status="ok") for row in rows], prompts, profile)


def grade_pending(out, prompts, profile):
    """Retry grading the exact saved text; never resample after grader failures."""
    raw = records(out / "raw-responses.jsonl", repair=True)
    raw_done, _ = validate_raw(raw, prompts, profile)
    existing = records(out / "records.jsonl", repair=True)
    done, expected = validate_records(existing, prompts, profile)
    if not done <= raw_done:
        raise ValueError("Graded answers lack their raw generation evidence")
    by_slot = {(r["prompt_id"], r["sample_index"]): r for r in raw}
    for row in existing:
        original = by_slot[(row["prompt_id"], row["sample_index"])]
        if any(row.get(k) != value for k, value in original.items()):
            raise ValueError("Grading changed the saved response")
    pending = [r for r in raw if (r["prompt_id"], r["sample_index"]) not in done]
    if pending:
        with ProcessPoolExecutor(
            max_workers=4, mp_context=multiprocessing.get_context("spawn"), initializer=init_grader
        ) as pool:
            with (out / "records.jsonl").open("a") as stream:
                for row in pool.map(grade, pending):
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                    stream.flush()
                    existing.append(row)
    validate_records(existing, prompts, profile)
    dump(out / "progress.json", dict(completed=len(existing), expected=len(expected)))
    return existing


def evaluate(prepared, assets, weights, out, panel="test", smoke=False):
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {str(gpu) for gpu in range(8)}:
        raise ValueError("Task evaluation requires one explicitly allocated GPU 0 through 7")
    from vllm import LLM, SamplingParams

    from verl.workers.rollout.vllm_fusion.compat import decode_graph_config

    started = time.monotonic()
    prepared, assets, weights, out = map(Path, (prepared, assets, weights, out))
    if panel == "test":
        prompts = read(prepared / "task-prompts.json")
    else:
        rows = [r for r in read(prepared / "heldout-questions.json") if r["domain"] == "math" and r["primary"]]
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(assets / "S0", local_files_only=True)
        prompts = [
            dict(
                data_source="DAPO-heldout",
                prompt_id=f"dev:{r['id']}",
                seed_index=r["id"],
                prompt_ids=r["prompt_ids"],
                input=tokenizer.decode(r["prompt_ids"], skip_special_tokens=False),
                gts=r["ground_truth"],
            )
            for r in rows
        ]
    if smoke:
        prompts = prompts[:2]
    profile = SimpleNamespace(
        samples=4,
        seed=PROTOCOL["accuracy"]["seed"],
        response_cap=64 if smoke else 31744,
        model_length=11264 if smoke else 32768,
    )
    weight_files = sorted(weights.glob("*.safetensors"))
    if not weight_files:
        raise ValueError("Missing evaluation weights")
    config = read(weights / "config.json")
    legal = read(assets / "legal-token-ids.json")
    if config.get("eos_token_id") != 151643 or 151643 not in legal:
        raise ValueError("Unexpected evaluation EOS/support")
    generation_config = weights / "generation_config.json"
    if generation_config.exists() and read(generation_config).get("eos_token_id") != 151643:
        raise ValueError("Generation config changes EOS")
    inputs = dict(
        protocol=protocol_identity(),
        weights=fingerprint({p.name: file_hash(p) for p in [weights / "config.json", *weight_files]}),
        prompts=fingerprint(prompts),
        panel=panel,
        smoke=smoke,
        profile=vars(profile),
        temperature=0.7,
        top_p=0.9,
        dtype="bfloat16",
        attention="TRITON_ATTN",
        legal=fingerprint(legal),
        grader="verl-one-shot-opd-math-direct-worker-v1",
        grader_implementation=file_hash(Path(__file__).with_name("grading.py")),
        implementation=file_hash(__file__),
    )
    if complete(out, inputs):
        return
    out.mkdir(parents=True, exist_ok=True)
    if (out / "inputs.json").exists() and read(out / "inputs.json") != inputs:
        raise ValueError("Partial task evaluation belongs to different inputs")
    dump(out / "inputs.json", inputs)
    existing = records(out / "records.jsonl", repair=True)
    done, expected = validate_records(existing, prompts, profile)
    raw = records(out / "raw-responses.jsonl", repair=True)
    raw_done, _ = validate_raw(raw, prompts, profile)
    if not done <= raw_done:
        raise ValueError("Graded responses have no raw evidence")
    pending = [(p, i) for p in prompts for i in range(4) if (p["prompt_id"], i) not in raw_done]
    random.Random(42).shuffle(pending)
    observer = subprocess.Popen(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[3] / "benchmarks/fusion/monitor.py"),
            "--parent",
            str(os.getpid()),
            "--out",
            str(out / f"resources-{time.time_ns()}.jsonl"),
        ]
    )
    assert observer.pid > 0
    if pending:
        seqs = 16 if config["hidden_size"] > 2048 else 32
        per_token = (
            4
            * config["num_hidden_layers"]
            * config["num_key_value_heads"]
            * (config.get("head_dim") or config["hidden_size"] // config["num_attention_heads"])
        )
        quantum = 256 * 1024**2
        kv = ((per_token * profile.model_length * seqs + quantum - 1) // quantum) * quantum
        llm = LLM(
            model=str(weights),
            dtype="bfloat16",
            generation_config="vllm",
            tensor_parallel_size=1,
            max_model_len=profile.model_length,
            max_num_batched_tokens=profile.model_length,
            max_num_seqs=seqs,
            enforce_eager=False,
            compilation_config=decode_graph_config(seqs),
            enable_prefix_caching=False,
            enable_chunked_prefill=False,
            kv_cache_memory_bytes=kv,
            attention_config={"backend": "TRITON_ATTN"},
            worker_extension_cls="verl.workers.rollout.vllm_rollout.utils.vLLMColocateWorkerExtension",
        )
        llm.collective_rpc("monkey_patch_model", args=(max(legal) + 1, sorted(set(range(max(legal) + 1)) - set(legal))))
        with (out / "raw-responses.jsonl").open("a") as stream:
            for start in range(0, len(pending), 64):
                batch = pending[start : start + 64]
                params = [
                    SamplingParams(
                        temperature=0.7,
                        top_p=0.9,
                        top_k=-1,
                        max_tokens=profile.response_cap,
                        stop_token_ids=[151643],
                        seed=profile.seed + 100 * p["seed_index"] + i,
                    )
                    for p, i in batch
                ]
                outputs = llm.generate(
                    [{"prompt_token_ids": p["prompt_ids"]} for p, _ in batch], params, use_tqdm=False
                )
                if len(outputs) != len(batch):
                    raise ValueError("Incomplete evaluation batch")
                rows = []
                for (prompt, index), value in zip(batch, outputs, strict=True):
                    if list(value.prompt_token_ids) != prompt["prompt_ids"] or len(value.outputs) != 1:
                        raise ValueError("Task identity changed")
                    response = value.outputs[0]
                    if response.finish_reason not in {"stop", "length"}:
                        raise ValueError("Aborted evaluation response")
                    rows.append(
                        dict(
                            data_source=prompt["data_source"],
                            prompt_id=prompt["prompt_id"],
                            sample_index=index,
                            input=prompt["input"],
                            gts=prompt["gts"],
                            output=response.text,
                            response_ids=list(response.token_ids),
                            response_length=len(response.token_ids),
                            finish_reason=response.finish_reason,
                            stop_reason=response.stop_reason,
                            seed=profile.seed + 100 * prompt["seed_index"] + index,
                        )
                    )
                validate_raw(rows, prompts, profile)
                for row in rows:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                raw.extend(rows)
                validate_raw(raw, prompts, profile)
                print("TASK_GENERATE", panel, len(raw), len(expected), flush=True)
        del llm
    existing = grade_pending(out, prompts, profile)
    done, expected = validate_records(existing, prompts, profile)
    if done != expected:
        raise ValueError("Task evaluation has missing responses")
    metrics = aggregate(existing, 4)
    dump(out / "metrics.json", metrics)
    receipt(
        out,
        inputs,
        records=len(existing),
        questions=len(prompts),
        samples=4,
        elapsed_seconds=time.monotonic() - started,
        raw_sha256=file_hash(out / "raw-responses.jsonl"),
        records_sha256=file_hash(out / "records.jsonl"),
        metrics_sha256=file_hash(out / "metrics.json"),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ["prepared", "assets", "weights", "out"]:
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--panel", choices=["test", "dev"], default="test")
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    evaluate(a.prepared, a.assets, a.weights, a.out, a.panel, a.smoke)


if __name__ == "__main__":
    main()
