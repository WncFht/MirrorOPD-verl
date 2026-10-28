#!/usr/bin/env python3
# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Audit A-F token contracts and teacher scoring at controlled answer boundaries."""

import argparse
import gc
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from verl.trainer.distillation.full_vocab import project


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--roundtrip-only", action="store_true")
    parser.add_argument("--model-root", type=Path, default=Path(__file__).resolve().parents[5] / "model")
    args = parser.parse_args()
    torch.set_num_threads(1)
    paths = {
        "r1": "DeepSeek-R1-Distill-Qwen-1.5B",
        "just": "JustRL-DeepSeek-1.5B",
        "r7": "DeepSeek-R1-Distill-Qwen-7B",
        "base": "Qwen3-1.7B-Base",
        "q8": "Qwen3-8B",
        "q4": "Qwen3-4B",
        "q17": "Qwen3-1.7B",
        "q4i": "Qwen3-4B-Instruct-2507",
    }
    paths = {key: str(args.model_root / name) for key, name in paths.items()}
    pairs = {
        "A": ("r1", "just"),
        "B": ("r1", "just"),
        "C": ("r1", "r7"),
        "D": ("base", "q8"),
        "E": ("base", "q4"),
        "F": ("q17", "q4i"),
    }
    tokenizers = {k: AutoTokenizer.from_pretrained(v) for k, v in paths.items()}
    if args.roundtrip_only:
        root = args.output.parent
        result = {}
        for setting, (sn, tn) in pairs.items():
            label = "r1" if sn == "r1" else "base" if sn == "base" else "instruct"
            rows = [x for x in json.loads((root / (label + ".json")).read_text())["natural"] if x["policy"] == "native"]
            records = []
            for row in rows:
                ids = row["prompt_ids"] + row["response_ids"]
                preserved = tokenizers[tn].encode(
                    tokenizers[sn].decode(ids, skip_special_tokens=False), add_special_tokens=False
                )
                stripped = tokenizers[tn].encode(
                    tokenizers[sn].decode(ids, skip_special_tokens=True), add_special_tokens=False
                )
                records.append(
                    {
                        "source": row["source"],
                        "index": row["index"],
                        "original_length": len(ids),
                        "preserved_length": len(preserved),
                        "preserved_equal": ids == preserved,
                        "stripped_equal": ids == stripped,
                        "first_mismatch": next(
                            (i for i, (x, y) in enumerate(zip(ids, preserved, strict=False)) if x != y), None
                        ),
                    }
                )
            result[setting] = {
                "count": len(records),
                "preserved_equal": sum(r["preserved_equal"] for r in records),
                "stripped_equal": sum(r["stripped_equal"] for r in records),
                "records": records,
            }
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        return
    result = {"models": {}, "pairs": {}, "boundary_scores": {}}
    for name, path in paths.items():
        t = tokenizers[name]
        c = json.loads((Path(path) / "config.json").read_text())
        g = json.loads((Path(path) / "generation_config.json").read_text())
        result["models"][name] = {
            "path": path,
            "tokenizer_size": len(t),
            "vocab_size": c["vocab_size"],
            "hidden_size": c["hidden_size"],
            "eos_id": t.eos_token_id,
            "eos_token": t.eos_token,
            "pad_id": t.pad_token_id,
            "generation_eos": g.get("eos_token_id"),
        }
    examples = [("What is 1+1?", "2"), ("What is 7*8?", "56"), ("What is 100-37?", "63"), ("Solve x+3=5.", "2")]
    for setting, (sn, tn) in pairs.items():
        st, tt = tokenizers[sn], tokenizers[tn]
        kwargs = {} if sn == "r1" else {"enable_thinking": False}
        probes = []
        for question, answer in examples:
            messages = [{"role": "user", "content": question}]
            sp = st.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True, return_dict=False, **kwargs
            )
            tp = tt.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True, return_dict=False, **kwargs
            )
            prefix = st.decode(sp, skip_special_tokens=False)
            roundtrip = tt.encode(prefix, add_special_tokens=False)
            stripped = tt.encode(st.decode(sp, skip_special_tokens=True), add_special_tokens=False)
            probes.append(
                {
                    "question": question,
                    "student_prompt_ids": sp,
                    "teacher_prompt_ids": tp,
                    "same_prompt": sp == tp,
                    "preserving_roundtrip_equal": roundtrip == sp,
                    "stripping_roundtrip_equal": stripped == sp,
                    "student_prompt_text": prefix,
                    "teacher_prompt_text": tt.decode(tp, skip_special_tokens=False),
                }
            )
        result["pairs"][setting] = {
            "student": sn,
            "teacher": tn,
            "token_to_id_equal": st.get_vocab() == tt.get_vocab(),
            "prompt_probes": probes,
        }
    for setting in ("A", "C", "D", "E", "F"):
        sn, tn = pairs[setting]
        st = tokenizers[sn]
        teacher = AutoModelForCausalLM.from_pretrained(
            paths[tn], dtype=torch.bfloat16, device_map="cuda:0", attn_implementation="sdpa"
        ).eval()
        rows = []
        for index, (_, answer) in enumerate(examples):
            p = result["pairs"][setting]["prompt_probes"][index]
            response = (f"The result is {answer}.\n</think>\n\n" if sn == "r1" else "") + f"\\boxed{{{answer}}}"
            for ending in ("", "\n"):
                response_ids = st.encode(response + ending, add_special_tokens=False)
                logits = {}
                for mode in ("student_prompt", "teacher_prompt"):
                    ids = p[mode + "_ids"] + response_ids
                    tensor = torch.tensor([ids], device="cuda:0")
                    with torch.no_grad():
                        h = teacher.model(input_ids=tensor, use_cache=False).last_hidden_state[:, -1]
                        z = project(h, teacher.get_output_embeddings().weight)[0]
                    logits[mode] = z
                    if index == 0 and ending == "" and mode == "student_prompt":
                        original = teacher.generation_config.eos_token_id
                        teacher.generation_config.eos_token_id = [17, 18]
                        with torch.no_grad():
                            h2 = teacher.model(input_ids=tensor, use_cache=False).last_hidden_state[:, -1]
                            z2 = project(h2, teacher.get_output_embeddings().weight)[0]
                        teacher.generation_config.eos_token_id = original
                        generation_eos_forward_difference = (z2 - z).abs().max().item()
                        assert generation_eos_forward_difference == 0.0
                row = {
                    "example": index,
                    "ending": repr(ending),
                    "generation_eos_forward_difference": generation_eos_forward_difference,
                }
                for mode, z in logits.items():
                    lp = z.double().log_softmax(-1)
                    row[mode] = {
                        "p_151643": lp[151643].exp().item(),
                        "p_151645": lp[151645].exp().item(),
                        "top1_id": z.argmax().item(),
                    }
                row["prompt_change_max_logit_difference"] = (
                    (logits["student_prompt"] - logits["teacher_prompt"]).abs().max().item()
                )
                rows.append(row)
        result["boundary_scores"][setting] = rows
        print(f"{setting} teacher checks complete", flush=True)
        del teacher
        gc.collect()
        torch.cuda.empty_cache()
    result["boundary_scores"]["B"] = result["boundary_scores"]["A"]
    result["scope"] = (
        "Four controlled completed answers, two endings each. Teacher-forced boundaries, not accuracy or training."
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
