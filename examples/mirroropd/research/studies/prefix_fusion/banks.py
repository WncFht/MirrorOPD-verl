# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Generate each frozen evaluation bank once; shorter panels are exact prefix views."""

import argparse
import math
import os
import subprocess
import sys
import time
from pathlib import Path

from examples.mirroropd.research.core.artifacts import file_hash, fingerprint, receipt
from examples.mirroropd.research.core.io import dump, read
from examples.mirroropd.research.studies.prefix_fusion.assets import verify_model_assets
from examples.mirroropd.research.studies.prefix_fusion.protocol import PROTOCOL, protocol_identity


def bank_request(row, generator):
    source_index = {"S0": 0, "J": 1, "P": 2}[generator]
    cap = PROTOCOL["banks"]["math_horizon"] if row["domain"] == "math" else PROTOCOL["banks"]["other_horizon"]
    return dict(id=row["id"], seed=PROTOCOL["banks"]["seed"] + 100000 * source_index + 10 * row["id"], cap=cap)


def generate(prepared, assets, generator, out, max_num_seqs=32):
    started = time.monotonic()
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"6", "7"}:
        raise ValueError("Allocate exactly physical GPU 6 or 7 through the local resource wrapper")
    from vllm import LLM, SamplingParams

    from verl.workers.rollout.vllm_fusion.compat import decode_graph_config

    prepared, assets, out = Path(prepared), Path(assets), Path(out)
    data_receipt, asset_receipt = read(prepared / "complete.json"), read(assets / "complete.json")
    if data_receipt["inputs"]["protocol"] != protocol_identity():
        raise ValueError("Prepared data belongs to another scientific protocol")
    # Model identities depend on file contents, not the training/evaluation schedule.
    if set(asset_receipt["models"]) != {"S0", "J", "P"}:
        raise ValueError("Unexpected study model set")
    if type(max_num_seqs) is not int or not 1 <= max_num_seqs <= 32:
        raise ValueError("This bank entry supports 1 through 32 active requests")
    for name, digest in data_receipt["artifacts"].items():
        if file_hash(prepared / name) != digest:
            raise ValueError(f"Changed prepared data: {name}")
    model = asset_receipt["models"][generator]
    verify_model_assets(model)
    rows = [r for r in read(prepared / "heldout-questions.json") if r["primary"]]
    legal = read(assets / "legal-token-ids.json")
    if file_hash(assets / "legal-token-ids.json") != asset_receipt["legal_ids_sha256"]:
        raise ValueError("Changed sampling support")
    out.mkdir(parents=True, exist_ok=False)
    from examples.mirroropd.benchmarks.fusion.fingerprint import source_fingerprint

    dump(out / "source.json", source_fingerprint())
    monitor = Path(__file__).resolve().parents[3] / "benchmarks/fusion/monitor.py"
    observer = subprocess.Popen(
        [
            sys.executable,
            str(monitor),
            "--parent",
            os.environ.get("FUSION_RESOURCE_PARENT", str(os.getpid())),
            "--out",
            str(out / "resources.jsonl"),
        ]
    )
    assert observer.pid > 0
    config = read(Path(model["directory"]) / "config.json")
    head_dim = config.get("head_dim") or config["hidden_size"] // config["num_attention_heads"]
    per_token = 4 * config["num_hidden_layers"] * config["num_key_value_heads"] * head_dim
    granularity = 256 * 1024**2
    kv_bytes = math.ceil(per_token * 11264 * max_num_seqs / granularity) * granularity
    inputs = dict(
        protocol=protocol_identity(),
        prepared=fingerprint(data_receipt["artifacts"]),
        model=model["identity"],
        generator=generator,
        support=asset_receipt["legal_ids_sha256"],
        parameters=PROTOCOL["banks"],
    )
    dump(out / "inputs.json", inputs)
    source_hash = file_hash(__file__)
    llm = LLM(
        model=model["directory"],
        dtype="bfloat16",
        tensor_parallel_size=1,
        max_model_len=11264,
        max_num_batched_tokens=11264,
        max_num_seqs=max_num_seqs,
        enforce_eager=False,
        compilation_config=decode_graph_config(max_num_seqs),
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        kv_cache_memory_bytes=kv_bytes,
        attention_config={"backend": "TRITON_ATTN"},
        logprobs_mode="processed_logprobs",
        worker_extension_cls="verl.workers.rollout.vllm_rollout.utils.vLLMColocateWorkerExtension",
    )
    legal_set = set(legal)
    banned = sorted(set(range(max(legal) + 1)) - legal_set)
    llm.collective_rpc("monkey_patch_model", args=(max(legal) + 1, banned))
    requests = [bank_request(row, generator) for row in rows]
    sampling = [
        SamplingParams(
            temperature=1, top_p=1, top_k=-1, max_tokens=r["cap"], seed=r["seed"], stop_token_ids=[151643], logprobs=0
        )
        for r in requests
    ]
    outputs = llm.generate([{"prompt_token_ids": r["prompt_ids"]} for r in rows], sampling, use_tqdm=True)
    if len(outputs) != len(rows):
        raise ValueError("Incomplete fixed bank")
    bank = []
    for row, request, result in zip(rows, requests, outputs, strict=True):
        if list(result.prompt_token_ids) != row["prompt_ids"] or len(result.outputs) != 1:
            raise ValueError("Bank request identity changed")
        output = result.outputs[0]
        tokens = list(output.token_ids)
        if not 0 < len(tokens) <= request["cap"] or any(t not in legal_set for t in tokens):
            raise ValueError("Invalid frozen response length or token support")
        if output.finish_reason not in {"stop", "length"}:
            raise ValueError("Incomplete frozen response")
        if output.finish_reason == "stop" and tokens[-1] != 151643:
            raise ValueError("Unexpected bank stop event")
        logprobs = [float(p[token].logprob) for token, p in zip(tokens, output.logprobs, strict=True)]
        if any(not math.isfinite(p) for p in logprobs):
            raise ValueError("Nonfinite bank behavior probability")
        bank.append(
            dict(
                row,
                response_ids=tokens,
                behavior_log_probs=logprobs,
                source=generator,
                seed=request["seed"],
                cap=request["cap"],
                finish_reason=output.finish_reason,
            )
        )
    dump(out / "bank.json", bank)
    if file_hash(__file__) != source_hash:
        raise ValueError("Bank generator source changed during execution")
    receipt(
        out,
        inputs,
        rows=len(bank),
        bank_sha256=file_hash(out / "bank.json"),
        response_tokens=sum(len(r["response_ids"]) for r in bank),
        kv_bytes=kv_bytes,
        physical_gpu=os.environ["CUDA_VISIBLE_DEVICES"],
        max_num_seqs=max_num_seqs,
        source_file_sha256=source_hash,
        elapsed_seconds=time.monotonic() - started,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--generator", choices=["S0", "J", "P"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    args = parser.parse_args()
    generate(args.prepared, args.assets, args.generator, args.out, args.max_num_seqs)


if __name__ == "__main__":
    main()
