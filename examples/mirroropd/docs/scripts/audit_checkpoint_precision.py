#!/usr/bin/env python3
"""Matched-input forward precision probe on one local checkpoint; no optimizer.

Compares eager FP32/BF16 and a production-like BF16 FlashAttention2 path,
with FP32 output-head logits and FP64 KL measurement. These short teacher-forced math traces are not an accuracy
benchmark, a full-training precision ablation, or a late-checkpoint measurement.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from verl.trainer.distillation.full_vocab import project  # noqa: E402

PROBES = [
    (
        "Solve 3x + 7 = 22. Explain your steps.",
        "<think>Subtracting 7 from both sides gives 3x = 15. "
        "Dividing both sides by 3 gives x = 5. Checking, 3 times 5 plus 7 is 22.</think>\n"
        "The answer is \\boxed{5}.",
    ),
    (
        "Two fair six-sided dice are rolled. What is the probability that their sum is 8?",
        "<think>There are 36 equally likely ordered outcomes. "
        "The pairs summing to 8 are (2,6), (3,5), (4,4), (5,3), and (6,2). "
        "Therefore there are five favorable outcomes out of thirty-six.</think>\n"
        "The probability is \\boxed{\\frac{5}{36}}.",
    ),
]


def summary(x):
    x = x.double()
    return {
        "mean": x.mean().item(),
        "max": x.max().item(),
        "min": x.min().item(),
        "p50": x.quantile(0.5).item(),
        "p95": x.quantile(0.95).item(),
    }


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float32, attn_implementation="eager", local_files_only=True
        )
        .eval()
        .to("cuda")
    )
    examples, effective_probes = [], []
    for question, trace in PROBES:
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": question}], tokenize=False, add_generation_prompt=True
        )
        if prompt.rstrip().endswith("<think>") and trace.startswith("<think>"):
            trace = trace[len("<think>") :]
        effective_probes.append({"question": question, "prompt": prompt, "completion": trace})
        # Trace is a fixed completion, not a sampled output. Keep its precise token boundary.
        prefix = tok(prompt, add_special_tokens=False)["input_ids"]
        completion = tok(trace, add_special_tokens=False)["input_ids"]
        ids = torch.tensor([prefix + completion], device="cuda")
        positions = torch.arange(len(prefix) - 1, ids.shape[1] - 1, device="cuda")
        examples.append((ids, positions))

    def forward(capture_head=False):
        logits, head_only = [], []
        for ids, pos in examples:
            hidden = model.model(input_ids=ids, use_cache=False, return_dict=True).last_hidden_state[0, pos]
            logits.append(project(hidden, model.lm_head.weight).float().cpu())
            if capture_head:
                head_only.append(project(hidden.bfloat16(), model.lm_head.weight.bfloat16()).float().cpu())
        return logits, head_only

    fp32, head_only = forward(capture_head=True)
    # FSDP2 casts compute parameters, not RoPE frequency buffers. Casting the
    # entire module would quantize those buffers and test a different policy.
    buffers_before = {name: buf.dtype for name, buf in model.named_buffers()}
    for parameter in model.parameters():
        parameter.data = parameter.data.to(torch.bfloat16)
    assert buffers_before == {name: buf.dtype for name, buf in model.named_buffers()}
    bf16, _ = forward()
    model.set_attn_implementation("flash_attention_2")
    bf16_flash, _ = forward()
    model.set_attn_implementation("eager")
    for parameter in model.parameters():
        parameter.data = parameter.data.float()
    rounded_weight_fp32, _ = forward()
    rows = []
    for i, (reference, approximate) in enumerate(zip(fp32, bf16, strict=True)):
        lp = reference.double().log_softmax(-1)
        lq = approximate.double().log_softmax(-1)
        q = lq.exp()
        p0 = lp.exp()
        r = lq - lp
        k = (q * r).sum(-1)
        d = approximate.double() - reference.double()
        quadratic = 0.5 * ((p0 * d.square()).sum(-1) - (p0 * d).sum(-1).square())
        kl32 = (approximate.log_softmax(-1).exp() * (approximate.log_softmax(-1) - reference.log_softmax(-1))).sum(-1)

        def reference_kl(logits, reference_lp=lp):
            logq = logits.double().log_softmax(-1)
            return summary((logq.exp() * (logq - reference_lp)).sum(-1))

        rows.append(
            {
                "probe": i,
                "response_tokens": len(reference),
                "kl_bf16_vs_fp32": summary(k),
                "head_only_kl_vs_fp32": reference_kl(head_only[i]),
                "bf16_flash_vs_fp32_eager": reference_kl(bf16_flash[i]),
                "rounded_weights_fp32_compute_kl": reference_kl(rounded_weight_fp32[i]),
                "roundtrip_logits_abs_max": (rounded_weight_fp32[i] - reference).abs().max().item(),
                "quadratic_prediction": summary(quadratic),
                "kl_arithmetic_fp32_abs_error": summary((kl32.double() - k).abs()),
                "reference_entropy": summary(-(p0 * lp).sum(-1)),
                "logit_difference_abs_max": d.abs().max().item(),
                "rows_kl": k.tolist(),
                "rows_quadratic": quadratic.tolist(),
            }
        )
    result = {
        "scope": __doc__,
        "model": str(args.model),
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(),
        "attention": "eager FP32/BF16 plus BF16 FlashAttention2; ordinary unpadded sequences",
        "bf16_reduced_precision_reduction": False,
        "tf32": False,
        "probes": effective_probes,
        "buffer_dtypes": {name: str(dtype) for name, dtype in buffers_before.items()},
        "results": rows,
    }
    all_kl = [x for row in rows for x in row["rows_kl"]]
    result["overall_token_mean"] = sum(all_kl) / len(all_kl)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "overall_token_mean": result["overall_token_mean"],
                "probes": [{"tokens": r["response_tokens"], "kl": r["kl_bf16_vs_fp32"]} for r in rows],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
