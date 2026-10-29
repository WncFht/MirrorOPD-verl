# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Replay identical prefixes with independent native models or the composite."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student", required=True)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--component", choices=["student", "teacher", "fusion"], required=True)
    parser.add_argument("--workload", required=True, help="Frozen JSON list of prompt_ids, suffix_ids and checkpoints")
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--graph", action="store_true")
    args = parser.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    workloads = json.loads(Path(args.workload).read_text())
    from fingerprint import source_fingerprint
    from vllm import LLM, SamplingParams

    from verl.workers.rollout.vllm_fusion import register_models
    from verl.workers.rollout.vllm_fusion.compat import fusion_hf_overrides

    register_models()
    model_path = args.teacher if args.component == "teacher" else args.student
    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        enforce_eager=not args.graph,
        hf_overrides=fusion_hf_overrides(args.teacher) if args.component == "fusion" else {},
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_model_len * args.batch,
        max_num_seqs=args.batch,
        kv_cache_memory_bytes=4 * 1024**3,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        attention_config={"backend": "TRITON_ATTN"},
        worker_extension_cls="examples.mirroropd.benchmarks.fusion.probe_worker.PrefixProbe",
    )
    manifest = {"config": vars(args), "source": source_fingerprint(), "cases": []}
    for index, workload in enumerate(workloads):
        prompt, suffix = workload["prompt_ids"], workload["suffix_ids"]
        if not suffix or len(prompt) + len(suffix) > args.max_model_len:
            raise ValueError("Invalid replay length")
        llm.collective_rpc(
            "configure_probe", args=(suffix, workload["checkpoints"], args.component == "fusion", args.batch)
        )
        params = SamplingParams(temperature=0, max_tokens=len(suffix), ignore_eos=True, logprobs=0)
        outputs = llm.generate([{"prompt_token_ids": prompt}] * args.batch, params, use_tqdm=False)
        if any(list(output.outputs[0].token_ids) != suffix for output in outputs):
            raise RuntimeError("Replay changed the prescribed prefix")
        path = out / f"case-{index}.pt"
        audit = llm.collective_rpc("save_probe", args=(str(path),))
        if args.batch > 1 and max(audit[0]["batch_shapes"]) < args.batch:
            raise RuntimeError("The calibration run never exercised the requested batch width")
        manifest["cases"].append(
            {
                "index": index,
                "path": str(path),
                "checkpoints": workload["checkpoints"],
                "audit": audit,
            }
        )
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
