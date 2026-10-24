#!/usr/bin/env python3
"""Measure native output support on short student rollouts; no parameter updates."""

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from verl.trainer.distillation.full_vocab import full_reverse_kl, project, teacher_support_correction


def describe(path, tokenizer):
    config = json.loads((path / "config.json").read_text())
    generation = json.loads((path / "generation_config.json").read_text())
    vocab = tokenizer.get_vocab()
    assert set(vocab.values()) == set(range(len(vocab))), "This probe requires contiguous tokenizer IDs"
    return {
        "path": str(path),
        "vocab_size": config["vocab_size"],
        "hidden_size": config["hidden_size"],
        "tie_word_embeddings": config.get("tie_word_embeddings", False),
        "tokenizer_size": len(vocab),
        "tokenizer_eos": tokenizer.eos_token_id,
        "tokenizer_pad": tokenizer.pad_token_id,
        "generation_eos": generation.get("eos_token_id"),
    }


def stats(values):
    x = torch.cat(values).double()
    return {
        "count": len(x),
        "mean": x.mean().item(),
        "min": x.min().item(),
        "median": x.median().item(),
        "p99": torch.quantile(x, 0.99).item(),
        "max": x.max().item(),
    }


def load_model(path):
    return AutoModelForCausalLM.from_pretrained(
        str(path), dtype=torch.bfloat16, device_map="cuda:0", attn_implementation="sdpa"
    ).eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--setting", required=True)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()
    torch.set_num_threads(1)
    set_seed(42)
    st, tt = (AutoTokenizer.from_pretrained(str(p)) for p in (args.student, args.teacher))
    sm, tm = describe(args.student, st), describe(args.teacher, tt)
    assert st.get_vocab() == tt.get_vocab(), "Token-to-ID maps differ; direct comparison is not meaningful"
    vs, vt, named = sm["vocab_size"], tm["vocab_size"], sm["tokenizer_size"]
    assert vt >= vs
    table = pq.read_table(args.dataset, columns=["prompt"])
    indices = np.random.default_rng(42).choice(len(table), size=args.samples, replace=False).tolist()
    result = {
        "setting": args.setting,
        "student": sm,
        "teacher": tm,
        "token_maps_equal": True,
        "seed": 42,
        "dataset_indices": indices,
        "max_new_tokens": args.max_new_tokens,
        "student_temperature": 1.0,
        "teacher_uses_student_input_ids": True,
        "teacher_extra_rows": vt - vs,
        "sequences": [],
        "numeric_checks": [],
    }
    student = load_model(args.student)
    sequences = []
    for index in indices:
        messages = table["prompt"][index].as_py()
        template_args = {"enable_thinking": False} if student.config.model_type == "qwen3" else {}
        batch = st.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
            **template_args,
        ).to("cuda:0")
        assert batch.input_ids.shape[1] <= 1024
        with torch.inference_mode():
            output = student.generate(
                **batch,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=1.0,
                top_p=1.0,
                top_k=0,
                pad_token_id=st.pad_token_id,
                use_cache=True,
            )
        prompt_len = batch.input_ids.shape[1]
        response = output[0, prompt_len:].tolist()
        sequences.append((output, prompt_len))
        result["sequences"].append(
            {
                "dataset_index": index,
                "prompt_tokens": prompt_len,
                "response_tokens": len(response),
                "response_ids": response,
                "last_token_id": response[-1],
                "im_end_positions": [i for i, x in enumerate(response) if x == 151645],
                "text": st.decode(response, skip_special_tokens=False),
            }
        )
        print(f"{args.setting} generated index={index} tokens={len(response)}", flush=True)
    teacher = load_model(args.teacher)
    ws, wt = student.get_output_embeddings().weight, teacher.get_output_embeddings().weight
    masses = {
        k: []
        for k in [
            "teacher_extra_mass",
            "teacher_unnamed_mass",
            "student_unnamed_mass",
            "normalizer_correction_nats",
            "native_kl_nats",
            "cropped_teacher_kl_nats",
        ]
    }
    for seq_id, (ids, prompt_len) in enumerate(sequences):
        with torch.no_grad():
            hs = student.model(input_ids=ids, use_cache=False).last_hidden_state[0, prompt_len - 1 : -1]
            ht = teacher.model(input_ids=ids, use_cache=False).last_hidden_state[0, prompt_len - 1 : -1]
            for start in range(0, len(hs), 32):
                sh, th = hs[start : start + 32], ht[start : start + 32]
                # Match production BF16 GEMM -> FP32 logits, then FP64 normalization
                # so very small tail mass is not lost by subtracting near-equal FP32 sums.
                zs, zt = project(sh, ws).double(), project(th, wt).double()
                lp, lq = zs.log_softmax(-1), zt.log_softmax(-1)
                ps, pt = lp.exp(), lq.exp()
                crop_lq = zt[:, :vs].log_softmax(-1)
                correction = (
                    torch.nn.functional.softplus(zt[:, vs:].logsumexp(-1) - zt[:, :vs].logsumexp(-1))
                    if vt > vs
                    else torch.zeros(len(sh), device=zs.device, dtype=torch.float64)
                )
                native = (ps * (lp - lq[:, :vs])).sum(-1)
                cropped = (ps * (lp - crop_lq)).sum(-1)
                for key, value in [
                    ("teacher_extra_mass", pt[:, vs:].sum(-1)),
                    ("teacher_unnamed_mass", pt[:, named:].sum(-1)),
                    ("student_unnamed_mass", ps[:, named:].sum(-1)),
                    ("normalizer_correction_nats", correction),
                    ("native_kl_nats", native),
                    ("cropped_teacher_kl_nats", cropped),
                ]:
                    masses[key].append(value.cpu())
                torch.testing.assert_close(native, cropped + correction, atol=1e-12, rtol=1e-12)
                if start == 0:
                    reference32 = (
                        (
                            zs.float().log_softmax(-1).exp()
                            * (zs.float().log_softmax(-1) - zt.float().log_softmax(-1)[:, :vs])
                        )
                        .sum()
                        .item()
                    )
                    actual = full_reverse_kl(sh, ws, th, wt, backend="dense", token_chunk=8, vocab_chunk=8192)
                    corrected_error = abs(actual.item() - reference32) / len(sh)
                    correction32 = teacher_support_correction(th, wt, vs, 8, 8192).item() / len(sh) if vt > vs else 0.0
                    check = {
                        "sequence": seq_id,
                        "positions": len(sh),
                        "production_dense_abs_error_per_token": corrected_error,
                        "production_correction_nats_mean": correction32,
                        "fp64_identity_max_error": (native - cropped - correction).abs().max().item(),
                    }
                    with torch.enable_grad():
                        z = zs[:4].detach().requires_grad_()
                        logp = z.log_softmax(-1)
                        raw = (logp.exp() * (logp - lq[:4, :vs].detach())).sum()
                        cropped_loss = (logp.exp() * (logp - crop_lq[:4].detach())).sum()
                        g1 = torch.autograd.grad(raw, z, retain_graph=True)[0]
                        g2 = torch.autograd.grad(cropped_loss + correction[:4].sum().detach(), z)[0]
                    check["logit_grad_max_abs_error"] = (g1 - g2).abs().max().item()
                    assert corrected_error < 1e-4, check
                    assert check["logit_grad_max_abs_error"] < 1e-12, check
                    result["numeric_checks"].append(check)
        print(f"{args.setting} scored sequence={seq_id}", flush=True)
    result["statistics"] = {key: stats(values) for key, values in masses.items()}
    result["scope"] = "Four short initial-student rollouts; no training, no long-context claim."
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result["statistics"], indent=2), flush=True)
    del teacher, student
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
