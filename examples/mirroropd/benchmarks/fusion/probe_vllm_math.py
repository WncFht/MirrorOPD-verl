# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Check actual native fusion sampling positions against an independent FP64 oracle."""

import argparse
import json
from pathlib import Path

import torch


def oracle(student, teacher, legal, alpha, temperature, top_k, top_p):
    logits = student.double() * (1 - alpha) + teacher.double() * alpha
    logits[~legal] = -torch.inf
    logits /= temperature
    if top_k > 0:
        threshold = sorted(logits.tolist(), reverse=True)[min(top_k, logits.numel()) - 1]
        logits[logits < threshold] = -torch.inf
    if top_p < 1:
        indices = torch.argsort(logits, descending=True)
        probabilities = logits[indices].softmax(-1)
        running = 0.0
        for index, probability in zip(indices.tolist(), probabilities.tolist(), strict=True):
            if running > top_p:
                logits[index] = -torch.inf
            running += probability
    return logits.log_softmax(-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student", required=True)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from verl.workers.rollout.vllm_fusion import register_models
    from verl.workers.rollout.vllm_fusion.compat import fusion_hf_overrides

    register_models()
    tokenizer = AutoTokenizer.from_pretrained(args.student, local_files_only=True)
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Compute 13 times 17."}], add_generation_prompt=True, return_dict=False
    )
    llm = LLM(
        model=args.student,
        hf_overrides=fusion_hf_overrides(args.teacher),
        dtype="bfloat16",
        enforce_eager=True,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        max_model_len=1024,
        max_num_batched_tokens=1024,
        max_num_seqs=1,
        kv_cache_memory_bytes=1024**3,
        logprobs_mode="processed_logprobs",
        attention_config={"backend": "TRITON_ATTN"},
        worker_extension_cls="examples.mirroropd.benchmarks.fusion.math_probe_worker.MathProbeWorker",
    )
    llm.collective_rpc("fusion_capture", args=(True,))
    results = []
    for index, (alpha, temperature, top_k, top_p) in enumerate(
        [
            (0.0, 1.0, -1, 1.0),
            (0.5, 1.0, -1, 1.0),
            (1.0, 1.0, -1, 1.0),
            (0.5, 0.7, 20, 0.9),
            (0.5, 1.0, 1, 0.5),
        ]
    ):
        llm.collective_rpc("fusion_prepare", args=(alpha,))
        params = SamplingParams(
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            max_tokens=1,
            seed=42,
            logprobs=0,
        )
        response = llm.generate([{"prompt_token_ids": ids}], params, use_tqdm=False)[0].outputs[0]
        path = out / f"case-{index}.pt"
        llm.collective_rpc("fusion_save_logits", args=(str(path),))
        values = torch.load(path, map_location="cpu", weights_only=True)
        expected = oracle(values["student"][0], values["teacher"][0], values["legal"], alpha, temperature, top_k, top_p)
        token = response.token_ids[0]
        measured = response.logprobs[0][token].logprob
        error = abs(measured - expected[token].item())
        actual = values["processed_log_probs"][0].double()
        probability_error = (actual.exp() - expected.exp()).abs().max().item()
        support_equal = torch.equal(torch.isfinite(actual), torch.isfinite(expected))
        results.append(
            {
                "alpha": alpha,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "token": token,
                "actual_logprob": measured,
                "oracle_logprob": expected[token].item(),
                "absolute_error": error,
                "probability_error": probability_error,
                "support_equal": support_equal,
                "passed": error <= 2e-6 and probability_error <= 2e-6 and support_equal,
            }
        )
        (out / "summary.json").write_text(json.dumps({"cases": results, "tolerance": 2e-6}, indent=2) + "\n")
    if not all(row["passed"] for row in results):
        raise RuntimeError("Actual native fusion sampling exceeded the predeclared FP64 oracle tolerance")


if __name__ == "__main__":
    main()
