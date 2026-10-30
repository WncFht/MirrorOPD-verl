# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Read-only OPD numerical audit: archived histories, actual heads, controlled logits.

Run from the repository root with PYTHONPATH=. and one free CUDA device.
FP64 references reuse identical input logits: this isolates loss arithmetic,
not backbone, checkpoint, FSDP, or inference/training discrepancies.
"""

import argparse
import json
import os
import statistics
import subprocess
from pathlib import Path

import torch

from examples.mirroropd.assets.paths import registered_model
from verl.trainer.distillation.full_vocab import precompute_kl_rows, project
from verl.utils.torch_functional import (
    FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE,
    logprobs_from_logits,
)


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def gradient_metrics(actual, reference):
    a, r = actual.double(), reference.double()
    norm = r.norm()
    active = r.abs() > r.abs().amax(-1, keepdim=True) * 1e-6
    flipped = (a * r < 0) & active
    return {
        "relative_l2": ((a - r).norm() / norm.clamp_min(1e-300)).item(),
        "cosine": ((a * r).sum() / (a.norm() * norm).clamp_min(1e-300)).item(),
        "max_abs_error": (a - r).abs().max().item(),
        "reference_l2": norm.item(),
        "actual_l2": a.norm().item(),
        "zero_on_active_fraction": ((a == 0) & active).sum().item() / max(active.sum().item(), 1),
        "sign_flip_active_fraction": flipped.sum().item() / max(active.sum().item(), 1),
        "sign_flip_reference_l1_fraction": (r.abs() * flipped).sum().item() / max(r.abs().sum().item(), 1e-300),
        "relative_sum_residual": a.sum(-1).abs().sum().item() / max(a.abs().sum().item(), 1e-300),
    }


def reverse_rows(zs, zt):
    # Exactly the arithmetic in precompute_kl_rows, before the BF16 VJP cast.
    lp, lt = zs.log_softmax(-1), zt.log_softmax(-1)
    p = lp.exp()
    gap = lp - lt
    kl = (p * gap).sum(-1)
    return kl, p * (gap - kl[:, None])


def centered_gradient(zs, zt):
    # Diagnostic alternative: subtract logit gaps before normalization.
    # Anchor the gap at the largest student logit to protect saturated rows.
    p = zs.softmax(-1)
    gap = zs - zt
    anchor = zs.argmax(-1, keepdim=True)
    gap = gap - gap.gather(-1, anchor)
    return p * (gap - (p * gap).sum(-1, keepdim=True))


def analyze_histories(root):
    results = {}
    for name, rid in [("E15", "krkmq098"), ("E18", "9hx3uegm"), ("E19", "d6jhh378"), ("E20", "lwgb5eex")]:
        directory = next((root / "derived/MirrorOPD-verl").glob("*-" + rid))
        rows = [json.loads(line) for line in (directory / "history.jsonl").open()]
        rows = [r for r in rows if "optim/actor/grad_norm_preclip" in r]
        keys = [
            "distill/full_kl",
            "optim/actor/grad_norm_preclip",
            "advantage/distillation/zero_fraction",
            "advantage/distillation/abs_mean",
            "distill/student_teacher/sampled_log_gap_std",
            "distill/student_teacher/sampled_log_gap_mean",
            "policy/entropy/pre_update",
        ]
        result = {"run": rid, "source": str(directory), "updates": len(rows)}
        for label, subset in [("first20", rows[:20]), ("last20", rows[-20:]), ("last", rows[-1:])]:
            result[label] = {
                k: statistics.mean([r[k] for r in subset if k in r]) for k in keys if any(k in r for r in subset)
            }
        result["nonfinite_loss_updates"] = sum(r.get("optim/actor/loss_nonfinite", 0) != 0 for r in rows)
        result["nonfinite_gradient_updates"] = sum(
            r.get("optim/actor/grad_norm_preclip_nonfinite", 0) != 0 for r in rows
        )
        result["skipped_updates"] = sum(r.get("optim/actor/update_skipped", 0) for r in rows)
        result["zero_gradient_updates"] = sum(r["optim/actor/grad_norm_preclip"] == 0 for r in rows)
        # Compare the last 64->63 or 16->15 observation within each batch, not across batch resets.
        pairs = [(a, b) for a, b in zip(rows, rows[1:], strict=False) if a.get("step/outer") == b.get("step/outer")]
        if name != "E15":
            tail = pairs[-256:]
            result["last256_within_batch_pairs"] = {
                "count": len(tail),
                "decreasing_fraction": statistics.mean(b["distill/full_kl"] < a["distill/full_kl"] for a, b in tail),
                "mean_relative_decrease": statistics.mean(
                    (a["distill/full_kl"] - b["distill/full_kl"]) / a["distill/full_kl"] for a, b in tail
                ),
            }
        results[name] = result
    return results


def capture(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        registered_model("DeepSeek-R1-Distill-Qwen-1.5B", root=args.models), local_files_only=True
    )
    records, seen = [], set()
    for line in args.records.open():
        record = json.loads(line)
        if record["input"] in seen:
            continue
        seen.add(record["input"])
        prompt = tokenizer.encode(record["input"], add_special_tokens=False)
        ids = tokenizer.encode(record["input"] + record["output"], add_special_tokens=False)[: len(prompt) + 512]
        if len(ids) - len(prompt) < 32:
            continue
        positions = torch.linspace(len(prompt) - 1, len(ids) - 2, 32).long()
        records.append({"input": record["input"], "ids": ids, "positions": positions.tolist()})
        if len(records) == 4:
            break
    assert len(records) == 4
    write(
        args.out / "prefixes.json",
        {
            "source": str(args.records),
            "selection": "first four unique prompts; first 512 response tokens; 32 evenly spaced states each",
            "records": records,
        },
    )
    tensors = {}
    for role, name in [("student", "DeepSeek-R1-Distill-Qwen-1.5B"), ("teacher", "JustRL-DeepSeek-1.5B")]:
        print("Loading", role, flush=True)
        model = (
            AutoModelForCausalLM.from_pretrained(
                registered_model(name, root=args.models),
                local_files_only=True,
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
            )
            .to("cuda")
            .eval()
        )
        hidden = []
        with torch.inference_mode():
            for rec in records:
                ids = torch.tensor([rec["ids"]], device="cuda")
                h = model.model(input_ids=ids, use_cache=False).last_hidden_state[0]
                hidden.append(h[rec["positions"]].clone())
            h = torch.cat(hidden)
            w = model.lm_head.weight
            z = project(h, w)
            tensors[role] = z.cpu()
            tensors[role + "_bf16_head"] = (h @ w.T).cpu()
        del model, h, w, hidden, z
        torch.cuda.empty_cache()
    tensors["labels"] = torch.tensor([rec["ids"][p + 1] for rec in records for p in rec["positions"]])
    torch.save(tensors, args.out / "fixed_logits.pt")
    print("Captured", len(tensors["labels"]), "states", flush=True)
    return tensors


def synthetic_checks():
    rows = []
    for margin in [4.0, 6.0, 8.0, 12.0, 16.0, 18.0, 24.0, 32.0]:
        z = torch.tensor([[margin, 0.0]], device="cuda")
        t = torch.tensor([[margin - 2.0, 0.0]], device="cuda")
        ref = centered_gradient(z.double(), t.double())
        k, g = reverse_rows(z, t)
        # Call the repository implementation itself with an identity head.
        actual_k, actual_g = precompute_kl_rows(z, torch.eye(2, device="cuda"), t, torch.eye(2, device="cuda"))
        torch.testing.assert_close(g, actual_g, rtol=0, atol=0)
        row = {
            "margin": margin,
            "full_kl": k.item(),
            "full_gradient": g.tolist(),
            "full_reference": ref.tolist(),
            "full_metrics": gradient_metrics(g, ref),
        }
        for dtype in [torch.bfloat16, torch.float32]:
            a = z.to(dtype).requires_grad_()
            lp = logprobs_from_logits(a, torch.zeros(1, device="cuda", dtype=torch.long), inplace_backward=False)
            ga = torch.autograd.grad(-lp.sum(), a)[0]
            # stable CE reference at exactly the same representable logits
            q = a.detach().double().softmax(-1)
            q[:, 0] = -q[:, 1:].sum(-1)
            row[str(dtype)] = {
                "logprob": lp.item(),
                "gradient": ga.float().tolist(),
                "reference": q.tolist(),
                "metrics": gradient_metrics(ga, q),
            }
        rows.append(row)
    return rows


def real_checks(tensors):
    zs, zt = tensors["student"].cuda(), tensors["teacher"].cuda()
    labels = tensors["labels"].cuda()
    results = {"shape": list(zs.shape), "interpolation": []}
    for alpha in [1.0, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0001, 0.00001, 0.000001, 0.0]:
        # Construct once in FP32; BOTH references use these exact same inputs.
        s = zt + alpha * (zs - zt) if alpha != 1.0 else zs
        k, g = reverse_rows(s, zt)
        kr, _ = reverse_rows(s.double(), zt.double())
        gr = centered_gradient(s.double(), zt.double())
        alternative = centered_gradient(s, zt)
        row = {
            "alpha": alpha,
            "reference_kl_mean": kr.mean().item(),
            "actual_kl_mean": k.mean().item(),
            "kl_abs_error_mean": (k.double() - kr).abs().mean().item(),
            "negative_kl_fraction": (k < 0).float().mean().item(),
            "gradient_fp32": gradient_metrics(g, gr),
            "gradient_after_bf16_cast": gradient_metrics(g.bfloat16(), gr),
            "centered_logit_gradient": gradient_metrics(alternative, gr),
        }
        results["interpolation"].append(row)
        print("alpha", alpha, "KL", kr.mean().item(), "grad relerr", row["gradient_fp32"]["relative_l2"], flush=True)
    # Sampled path: compare native BF16-head logprobs against FP64 on those logits.
    s, t = tensors["student_bf16_head"].cuda(), tensors["teacher_bf16_head"].cuda()
    lp64, lt64 = s.double().log_softmax(-1), t.double().log_softmax(-1)
    pick = lambda x: x.gather(-1, labels[:, None]).squeeze(-1)
    a64 = pick(lt64 - lp64)
    s = s.requires_grad_()
    lp = logprobs_from_logits(s, labels, inplace_backward=False)
    lt = logprobs_from_logits(t, labels, inplace_backward=False)
    adv = (lt - lp).detach().float()
    g = torch.autograd.grad(-(adv * lp).sum(), s)[0]
    q = lp64.exp()
    score = -q
    mask = torch.zeros_like(q, dtype=torch.bool).scatter_(-1, labels[:, None], True)
    score = torch.where(mask, -score.masked_fill(mask, 0).sum(-1, keepdim=True), score)
    gr = -a64[:, None] * score
    max_indices = s.detach().argmax(-1, keepdim=True)
    others = s.detach().double().scatter(-1, max_indices, -torch.inf)
    margin = s.detach().double().gather(-1, max_indices).squeeze(-1) - others.logsumexp(-1)
    results["sampled_native"] = {
        "logprob_dtype": str(lp.dtype),
        "adv_zero_fraction": (adv == 0).float().mean().item(),
        "reference_adv_zero_fraction": (a64 == 0).float().mean().item(),
        "adv_sign_flip_fraction": (adv.double() * a64 < 0).float().mean().item(),
        "lost_adv_reference_abs_mean": a64[adv == 0].abs().mean().item() if (adv == 0).any() else None,
        "adv_absolute_error_mean": (adv.double() - a64).abs().mean().item(),
        "gradient": gradient_metrics(g, gr),
        "sampled_logprob_zero_fraction": (lp == 0).float().mean().item(),
        "sampled_target_gradient_zero_fraction": (pick(g) == 0).float().mean().item(),
        "max_margin": margin.max().item(),
        "margin_gt_5_5_fraction": (margin > 5.5).float().mean().item(),
        "margin_gt_16_6_fraction": (margin > 16.6).float().mean().item(),
        "rows": [
            {
                "label": int(labels[i]),
                "margin": margin[i].item(),
                "adv": adv[i].item(),
                "ref_adv": a64[i].item(),
                "target_grad": pick(g)[i].item(),
                "reference_target_grad": pick(gr)[i].item(),
            }
            for i in range(len(s))
        ],
    }
    results["sampled_native"]["lost_rows_reference_gradient_l2_fraction"] = (gr[adv == 0].norm() / gr.norm()).item()
    results["sampled_native"]["lost_rows_reference_gradient_l1_fraction"] = (
        gr[adv == 0].abs().sum() / gr.abs().sum()
    ).item()
    results["sampled_interpolation"] = []
    # Match both teacher/student extraction kernels to isolate arithmetic alone.
    for alpha in [1.0, 0.03, 0.001, 0.00001, 0.000001, 0.0]:
        a = (zt + alpha * (zs - zt)).detach().requires_grad_()
        lp = logprobs_from_logits(a, labels, inplace_backward=False)
        lt = logprobs_from_logits(zt, labels, inplace_backward=False)
        advantage = (lt - lp).detach()
        ag = torch.autograd.grad(-(advantage * lp).sum(), a)[0]
        lpa, lta = a.detach().double().log_softmax(-1), zt.double().log_softmax(-1)
        ra = pick(lta - lpa)
        score = -lpa.exp()
        score = torch.where(mask, -score.masked_fill(mask, 0).sum(-1, keepdim=True), score)
        rg = -ra[:, None] * score
        results["sampled_interpolation"].append(
            {
                "alpha": alpha,
                "advantage_zero_fraction": (advantage == 0).float().mean().item(),
                "advantage_sign_flip_fraction": (advantage.double() * ra < 0).float().mean().item(),
                "gradient": gradient_metrics(ag, rg),
            }
        )
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--models", type=Path, default=Path("../model"))
    p.add_argument("--logs", type=Path, default=Path("../wandb-logs"))
    p.add_argument(
        "--records",
        type=Path,
        default=Path("outputs/full-validation-20261022/valid-full-wall-s64-20261022-a/training/validation/0.jsonl"),
    )
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    metadata = {
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "logs_commit": subprocess.check_output(["git", "-C", str(args.logs), "rev-parse", "HEAD"], text=True).strip(),
        "flash_ce_available": FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE,
        "scope": (
            "fixed mathematical prefixes, initial local models; "
            "interpolated logits are controlled numerical probes, NOT late checkpoints"
        ),
    }
    write(args.out / "metadata.json", metadata)
    write(args.out / "histories.json", analyze_histories(args.logs))
    write(args.out / "synthetic.json", synthetic_checks())
    path = args.out / "fixed_logits.pt"
    tensors = torch.load(path, weights_only=True) if path.exists() else capture(args)
    write(args.out / "real_logits.json", real_checks(tensors))
    print("Finished", args.out, flush=True)


if __name__ == "__main__":
    main()
