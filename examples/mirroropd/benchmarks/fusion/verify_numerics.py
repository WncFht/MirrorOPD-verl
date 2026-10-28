# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Real-checkpoint cached/full-prefix and FP64 sampler checks on identical prefixes."""

import argparse
import json
import os
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoModelForCausalLM, AutoTokenizer

from verl.experimental.rollout_policy.math_context import inference_math
from verl.experimental.rollout_policy.sampling import SamplingConfig, fusion_log_probs, validate_tokenizers

# Fixed before measurement: BF16 forward kernels may differ by a few output ULPs.
TOLERANCES = {"logits_abs": 0.25, "probability_abs": 0.01, "selected_logprob_abs": 0.125, "fp64_sampler_abs": 2e-6}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attention-dtype", choices=["bf16", "fp16"], default="fp16")
    parser.add_argument(
        "--precision",
        choices=["bf16", "head", "residual", "linear", "intermediate", "stable", "stable_moe", "stable_moe_attention"],
        default="stable",
    )
    parser.add_argument("--student", required=True)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument("--cache-cap", type=int, default=5)
    parser.add_argument("--prefix-tokens", type=int, default=0)
    parser.add_argument("--replay-dir")
    parser.add_argument("--data")
    parser.add_argument("--request-ids", type=int, nargs=2, default=[5, 24])
    parser.add_argument("--reference-left-padding", type=int, default=0)
    parser.add_argument("--flash-kv", action="store_true")
    parser.add_argument("--flash-kv-splits", type=int, default=0)
    parser.add_argument("--preallocate-kv", action="store_true")
    parser.add_argument("--production-math", action="store_true")
    parser.add_argument("--attention", choices=["sdpa", "flash_attention_2"], default="flash_attention_2")
    parser.add_argument("--sdpa-math", action="store_true")
    parser.add_argument("--training-patch", action="store_true")
    parser.add_argument("--full-precision-reduction", action="store_true")
    parser.add_argument("--device", type=int, choices=range(4), default=0)
    args = parser.parse_args()
    if args.replay_dir and not args.data:
        parser.error("Natural-prefix replay requires the original --data parquet")
    if args.cuda_graph:
        args.flash_kv = True
        args.production_math = True
    if args.flash_kv:
        args.preallocate_kv = True
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0,1,2,3":
        parser.error("Set CUDA_VISIBLE_DEVICES=0,1,2,3")
    torch.cuda.set_device(args.device)
    torch.set_num_threads(1)
    if args.full_precision_reduction:
        torch.backends.cuda.preferred_blas_library("cublaslt")
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (False, False)
    models = [
        AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.bfloat16, attn_implementation=args.attention, local_files_only=True
        )
        .eval()
        .to(args.device)
        for path in [args.student, args.teacher]
    ]
    from verl.experimental.rollout_policy.precision import configure_precision

    for model in models:
        configure_precision(
            model, args.precision, attention_dtype=torch.float16 if args.attention_dtype == "fp16" else None
        )
    if args.training_patch:
        from verl.models.transformers.monkey_patch import apply_monkey_patch

        for model in models:
            apply_monkey_patch(model, use_remove_padding=True)
    tokenizers = [AutoTokenizer.from_pretrained(path, local_files_only=True) for path in [args.student, args.teacher]]
    legal = validate_tokenizers(*tokenizers, models[0].config.vocab_size, models[1].config.vocab_size)
    pad = tokenizers[0].eos_token_id
    prompts = [
        tokenizers[0].encode(text, add_special_tokens=False)
        for text in ["Compute 17 + 29.", "What is 3 times 4? Explain briefly."]
    ]
    if args.replay_dir:
        import pyarrow.parquet as pq

        from verl.utils.tokenizer.continuous_token_wiring import create_continuous_token_builder

        builder = create_continuous_token_builder(
            tokenizers[0], model_family="qwen3", hf_model_type=models[0].config.model_type, chat_template_kwargs={}
        )
        rows = pq.read_table(args.data, columns=["prompt"]).slice(0, max(args.request_ids) + 1).to_pylist()
        responses = {}
        for path in sorted(Path(args.replay_dir).glob("rank-*.jsonl")):
            for line in path.open():
                row = json.loads(line)
                if row["kind"] == "measurement" and row["repeat"] == 0 and row["compact"] == 0:
                    responses.update({int(result["request_id"]): result["token_ids"] for result in row["results"]})
                    break
        prompts = [
            builder.build_initial_tokens(rows[index]["prompt"]) + responses[index][: args.prefix_tokens]
            for index in args.request_ids
        ]
    elif args.prefix_tokens:
        for i, prompt in enumerate(prompts):
            prompt.extend([16 + i] * max(0, args.prefix_tokens - i * 17 - len(prompt)))
    if not args.replay_dir:
        prompts[0].insert(1, pad)  # A real EOS-valued prompt token must remain attended.
    width = max(map(len, prompts))
    ids = torch.full((2, width), pad, device=args.device, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for i, prompt in enumerate(prompts):
        ids[i, -len(prompt) :] = torch.tensor(prompt, device=args.device)
        mask[i, -len(prompt) :] = 1
    prefixes, cached_ids = ids.clone(), ids
    caches = [None, None]
    if args.preallocate_kv:
        from verl.experimental.rollout_policy.kv_cache import AppendCache

        caches = [AppendCache(model.config, width + args.cache_cap) for model in models]
    graphs = [None, None]
    failures = []
    report = {"tolerances": TOLERANCES, "models": [args.student, args.teacher], "rows": [], "pad_equals_eos": True}
    report["resolved_precision"] = [getattr(model, "_fusion_precision", "bf16") for model in models]
    report["prefill_attention"] = [model.config._attn_implementation for model in models]
    from fingerprint import source_fingerprint

    report["source"] = source_fingerprint()
    config = SamplingConfig(legal_ids=legal)
    backends = (
        [SDPBackend.MATH]
        if args.sdpa_math
        else [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
    )
    context = inference_math(cuda=True) if args.production_math else sdpa_kernel(backends)
    with context:
        for step in range(5):
            positions = mask.cumsum(-1).sub(1).clamp_min(0)
            cached, full = [], []
            for i, model in enumerate(models):
                if args.cuda_graph and step > 0:
                    from verl.experimental.rollout_policy.cuda_graph import DecodeGraph

                    if graphs[i] is None:
                        graphs[i] = DecodeGraph(
                            model,
                            cached_ids,
                            positions[:, -1:],
                            mask,
                            caches[i],
                            {"logits_to_keep": 1},
                            splits=args.flash_kv_splits,
                        )
                        out = graphs[i].output
                    else:
                        out = graphs[i].replay(cached_ids, positions[:, -1:], mask)
                else:
                    attention_context = nullcontext()
                    if args.flash_kv and step > 0:
                        from verl.experimental.rollout_policy.flash_kv import flash_kv_context

                        attention_context = flash_kv_context(model, mask, splits=args.flash_kv_splits, cache=caches[i])
                    with attention_context:
                        out = model(
                            input_ids=cached_ids,
                            attention_mask=None if args.flash_kv and step > 0 else mask,
                            position_ids=positions if step == 0 else positions[:, -1:],
                            past_key_values=caches[i],
                            use_cache=True,
                            logits_to_keep=1,
                        )
                caches[i] = out.past_key_values
                cached.append(out.logits[:, -1])
                ref_ids, ref_mask, ref_positions = prefixes, mask, positions
                if args.reference_left_padding:
                    ref_ids = torch.nn.functional.pad(prefixes, (args.reference_left_padding, 0), value=pad)
                    ref_mask = torch.nn.functional.pad(mask, (args.reference_left_padding, 0), value=0)
                    ref_positions = ref_mask.cumsum(-1).sub(1).clamp_min(0)
                full.append(
                    model(
                        input_ids=ref_ids,
                        attention_mask=ref_mask,
                        position_ids=ref_positions,
                        use_cache=False,
                        logits_to_keep=1,
                    ).logits[:, -1]
                )
                error = (cached[-1][:, list(legal)].float() - full[-1][:, list(legal)].float()).abs().max().item()
                report["rows"].append({"step": step, "component": i, "logits_abs": error})
                if error > TOLERANCES["logits_abs"]:
                    failures.append(report["rows"][-1])
            for alpha in [0.0, 0.5, 1.0]:
                actual = fusion_log_probs(*cached, alpha, config)
                baseline = fusion_log_probs(*full, alpha, config)
                probability_error = (actual.exp() - baseline.exp()).abs().max().item()
                selected = actual.argmax(-1, keepdim=True)
                lp_error = (actual.gather(-1, selected) - baseline.gather(-1, selected)).abs().max().item()
                oracle = (1 - alpha) * cached[0].double() + alpha * cached[1].double()
                support = torch.zeros_like(oracle, dtype=torch.bool)
                support[:, list(legal)] = True
                oracle = oracle.masked_fill(~support, -torch.inf).log_softmax(-1)
                sampler_error = (actual.exp().double() - oracle.exp()).abs().max().item()
                row = {
                    "step": step,
                    "alpha": alpha,
                    "probability_abs": probability_error,
                    "selected_logprob_abs": lp_error,
                    "fp64_sampler_abs": sampler_error,
                }
                report["rows"].append(row)
                if not all(
                    row[key] <= TOLERANCES[key]
                    for key in ["probability_abs", "selected_logprob_abs", "fp64_sampler_abs"]
                ):
                    failures.append(row)
            cached_ids = torch.full((2, 1), 16 + step, device=args.device, dtype=torch.long)
            prefixes = torch.cat([prefixes, cached_ids], dim=1)
            mask = torch.cat([mask, torch.ones_like(cached_ids)], dim=1)
    report["passed"] = not failures
    report["arguments"] = vars(args)
    report["failures"] = failures
    report["full_precision_reduction"] = args.full_precision_reduction
    report["attention"] = args.attention
    report["sdpa_math"] = args.sdpa_math
    report["training_patch"] = args.training_patch
    report["production_math"] = args.production_math
    report["preallocate_kv"] = args.preallocate_kv
    report["precision"] = args.precision
    report["cuda_graph"] = args.cuda_graph
    report["prefix_tokens"] = args.prefix_tokens
    report["flash_kv"] = args.flash_kv
    report["flash_kv_splits"] = args.flash_kv_splits
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
