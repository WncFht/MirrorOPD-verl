# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Report native BF16 HF/full-prefix differences without imposing a cross-backend gate."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--native-reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible or "," in visible:
        raise ValueError("Allocate one explicit GPU for this HF reference")
    args.out.mkdir(parents=True, exist_ok=False)
    monitor = subprocess.Popen(
        [
            sys.executable,
            str(Path(__file__).with_name("monitor.py")),
            "--parent",
            os.environ.get("FUSION_RESOURCE_PARENT", str(os.getpid())),
            "--out",
            str(args.out / "resources.jsonl"),
        ]
    )
    assert monitor.pid > 0
    import torch
    import transformers
    from compare_native_components import measure
    from fingerprint import source_fingerprint
    from transformers import AutoModelForCausalLM, AutoTokenizer

    raw = args.workload.read_bytes()
    cases = json.loads(raw)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    legal = torch.tensor(sorted(set(tokenizer.get_vocab().values())), dtype=torch.long)
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=torch.bfloat16,
            device_map={"": "cuda:0"},
            attn_implementation="flash_attention_2",
            local_files_only=True,
        )
        .eval()
        .requires_grad_(False)
    )
    config = {
        "model": str(Path(args.model).resolve()),
        "workload": str(args.workload.resolve()),
        "workload_sha256": hashlib.sha256(raw).hexdigest(),
        "native_reference": str(args.native_reference.resolve()),
        "precision": "native HF BF16 weights/activations; FP64 diagnostic normalization",
        "attention": model.config._attn_implementation,
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "source": source_fingerprint(),
    }
    (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    checks = []
    for index, case in enumerate(cases):
        native = torch.load(args.native_reference / f"case-{index}.pt", map_location="cpu", weights_only=True)
        if set(native) != set(case["checkpoints"]):
            raise ValueError("Native reference checkpoint coverage differs")
        captured = {}
        for step in case["checkpoints"]:
            prefix = case["prompt_ids"] + case["suffix_ids"][:step]
            ids = torch.tensor([prefix], dtype=torch.long, device="cuda:0")
            with torch.inference_mode():
                logits = model(input_ids=ids, use_cache=False, logits_to_keep=1).logits[0, -1].float().cpu()
            if not torch.isfinite(logits).all():
                raise RuntimeError("HF reference produced nonfinite logits")
            left = native[step]["raw"]
            if left.shape[0] != 1:
                raise ValueError("This cross-backend diagnostic requires a single-request native reference")
            captured[step] = {"raw": logits.unsqueeze(0)}
            checks.append(
                {
                    "case": index,
                    "step": step,
                    "prefix_tokens": len(prefix),
                    **measure(left[0], logits, legal, case["suffix_ids"][step]),
                }
            )
        torch.save(captured, args.out / f"case-{index}.pt")
        report = {
            "scope": "Cross-backend diagnostic only; native cached vLLM vs HF full-prefix BF16",
            "acceptance_gate": None,
            "checks": checks,
            "maxima": {
                key: max(row[key] for row in checks)
                for key in ["probability_abs", "total_variation", "selected_logprob_abs"]
            },
        }
        (args.out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
