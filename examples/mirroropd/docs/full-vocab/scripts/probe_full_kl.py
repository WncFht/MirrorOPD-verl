# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Standalone full-KL prototypes; no model loading, FSDP, TQ, or training integration.

CUDA measurements use BF16 inputs with FP32 projection outputs/reductions.
Token recomputation is a PyTorch reference, NOT the article's Chunked cuBLAS.
Streaming is a simple Python-loop prototype, NOT the Nitrobrew implementation.
"""

import argparse
import gc
import importlib.metadata
import json
import os
import statistics
import time
from functools import partial
from pathlib import Path

import torch


class ProjectionFP32(torch.autograd.Function):
    """Ordinary linear VJP for torch.mm(out_dtype=...), which lacks autograd here."""

    @staticmethod
    def forward(ctx, hidden, weight):
        ctx.save_for_backward(hidden, weight)
        return torch.mm(hidden, weight.T, out_dtype=torch.float32)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, upstream):
        hidden, weight = ctx.saved_tensors
        dz = upstream.to(hidden.dtype)
        return dz @ weight, dz.T @ hidden


def project(hidden, weight):
    if hidden.is_cuda and hidden.dtype == torch.bfloat16:
        if torch.is_grad_enabled() and (hidden.requires_grad or weight.requires_grad):
            return ProjectionFP32.apply(hidden, weight)
        return torch.mm(hidden, weight.T, out_dtype=torch.float32)
    return hidden @ weight.T


def dense_rows(hs, ws, ht, wt):
    lp = project(hs, ws).log_softmax(-1)
    lt = project(ht.detach(), wt.detach()).log_softmax(-1)
    return (lp.exp() * (lp - lt)).sum(-1)


@torch.no_grad()
def row_statistics(hs, ws, ht, wt, token_chunk, vocab_chunk):
    """Return per-row KL, student log-normalizer, and mean logit gap."""
    n, vocab = hs.shape[0], ws.shape[0]
    dtype = torch.float64 if hs.dtype == torch.float64 else torch.float32
    kl, log_zp, mean_gap = [torch.empty(n, device=hs.device, dtype=dtype) for _ in range(3)]
    for start in range(0, n, token_chunk):
        stop = min(start + token_chunk, n)
        sh, th = hs[start:stop], ht[start:stop]
        if vocab_chunk == 0:
            zp, zt = project(sh, ws), project(th, wt)
            lp, lt = zp.log_softmax(-1), zt.log_softmax(-1)
            p = lp.exp()
            kl[start:stop] = (p * (lp - lt)).sum(-1)
            log_zp[start:stop] = zp.logsumexp(-1)
            mean_gap[start:stop] = (p * (zp - zt)).sum(-1)
        else:
            mp, mt = [torch.full((stop - start,), -torch.inf, device=hs.device, dtype=dtype) for _ in range(2)]
            sp, st, gap = [torch.zeros_like(mp) for _ in range(3)]
            for v in range(0, vocab, vocab_chunk):
                zp = project(sh, ws[v : v + vocab_chunk])
                zt = project(th, wt[v : v + vocab_chunk])
                np, nt = torch.maximum(mp, zp.max(-1).values), torch.maximum(mt, zt.max(-1).values)
                ap, at = (mp - np).exp(), (mt - nt).exp()
                ep, et = (zp - np[:, None]).exp(), (zt - nt[:, None]).exp()
                sp, st = ap * sp + ep.sum(-1), at * st + et.sum(-1)
                gap = ap * gap + (ep * (zp - zt)).sum(-1)
                mp, mt = np, nt
            log_zp[start:stop] = mp + sp.log()
            mean_gap[start:stop] = gap / sp
            kl[start:stop] = gap / sp - log_zp[start:stop] + mt + st.log()
    return kl, log_zp, mean_gap


class RecomputedKL(torch.autograd.Function):
    """First-order custom VJP; teacher frozen, head and hidden both trainable."""

    @staticmethod
    def forward(ctx, hs, ws, ht, wt, token_chunk, vocab_chunk):
        kl, log_zp, mean_gap = row_statistics(hs, ws, ht, wt, token_chunk, vocab_chunk)
        ctx.save_for_backward(hs, ws, ht, wt, log_zp, mean_gap)
        ctx.token_chunk, ctx.vocab_chunk = token_chunk, vocab_chunk or ws.shape[0]
        return kl

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, upstream):
        hs, ws, ht, wt, log_zp, mean_gap = ctx.saved_tensors
        dtype = log_zp.dtype
        dh = torch.zeros(hs.shape, device=hs.device, dtype=dtype)
        dw = torch.zeros(ws.shape, device=ws.device, dtype=dtype)
        for start in range(0, len(hs), ctx.token_chunk):
            stop = min(start + ctx.token_chunk, len(hs))
            for v in range(0, len(ws), ctx.vocab_chunk):
                end = min(v + ctx.vocab_chunk, len(ws))
                zp = project(hs[start:stop], ws[v:end])
                zt = project(ht[start:stop], wt[v:end])
                p = (zp - log_zp[start:stop, None]).exp()
                dz = p * (zp - zt - mean_gap[start:stop, None]) * upstream[start:stop, None]
                # Match low-precision GEMM inputs used for student gradient products.
                dz = dz.to(hs.dtype)
                dh[start:stop] += project(dz, ws[v:end].T)
                dw[v:end] += project(dz.T.contiguous(), hs[start:stop].T)
        return dh.to(hs.dtype), dw.to(ws.dtype), None, None, None, None


def cpu_audit():
    torch.manual_seed(42)
    cases = []
    for seed in range(3):
        torch.manual_seed(seed)
        hs = torch.randn(7, 5, dtype=torch.float64, requires_grad=True)
        ws = torch.randn(19, 5, dtype=torch.float64, requires_grad=True)
        ht = torch.randn(7, 8, dtype=torch.float64, requires_grad=True)
        wt = torch.randn(19, 8, dtype=torch.float64, requires_grad=True)
        for masked in (False, True):
            weights = (
                torch.zeros(7, dtype=torch.float64)
                if masked
                else torch.tensor([0, 0.2, -0.3, 1.2, 0, 0.7, 0.1], dtype=torch.float64)
            )
            expected = dense_rows(hs, ws, ht, wt)
            eg = torch.autograd.grad((expected * weights).sum(), (hs, ws))
            for token_chunk, vocab_chunk in ((3, 0), (3, 4), (20, 30), (1, 1)):
                actual = RecomputedKL.apply(hs, ws, ht, wt, token_chunk, vocab_chunk)
                grads = torch.autograd.grad((actual * weights).sum(), (hs, ws, ht, wt), allow_unused=True)
                errors = {
                    "value": (actual - expected).abs().max().item(),
                    "hidden_grad": (grads[0] - eg[0]).abs().max().item(),
                    "head_grad": (grads[1] - eg[1]).abs().max().item(),
                }
                assert grads[2] is None and grads[3] is None
                assert max(errors.values()) < 1e-10, errors
                cases.append(
                    dict(seed=seed, all_masked=masked, token_chunk=token_chunk, vocab_chunk=vocab_chunk, errors=errors)
                )
    self_kl = RecomputedKL.apply(hs, ws, hs.detach(), ws.detach(), 3, 4)
    self_grads = torch.autograd.grad(self_kl.sum(), (hs, ws))
    self_errors = {
        "value": self_kl.abs().max().item(),
        "hidden_grad": self_grads[0].abs().max().item(),
        "head_grad": self_grads[1].abs().max().item(),
    }
    assert max(self_errors.values()) < 1e-10
    with torch.no_grad():
        kl, _, mu = row_statistics(hs, ws, ht, wt, 3, 4)
        zp, zt = project(hs, ws), project(ht, wt)
        actions = torch.arange(len(hs))
        advantage = (zt.log_softmax(-1) - zp.log_softmax(-1))[actions, actions] + kl
        centered_gap = mu - (zp - zt)[actions, actions]
        baseline_error = (advantage - centered_gap).abs().max().item()
    assert baseline_error < 1e-10
    return {"cases": cases, "self_teacher_errors": self_errors, "centered_baseline_max_error": baseline_error}


def relative_error(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)).item()


def mean_result(fn, *args):
    return fn(*args).mean()


def value_mean(*args):
    return row_statistics(*args)[0].mean()


def measure(fn, hs, ws, backward, warmup, repeats):
    def once():
        hs.grad = ws.grad = None
        out = fn()
        if backward:
            out.backward()
        return out.detach()

    for _ in range(warmup):
        once()
    torch.cuda.synchronize()
    hs.grad = ws.grad = None
    gc.collect()
    torch.cuda.empty_cache()
    resident = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        begin = time.perf_counter()
        value = once()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - begin) * 1000)
    peak = torch.cuda.max_memory_allocated()
    result = dict(
        value=value.item(),
        wall_ms_median=statistics.median(times),
        wall_ms_samples=times,
        resident_gib=resident / 2**30,
        peak_allocated_gib=peak / 2**30,
        incremental_peak_gib=(peak - resident) / 2**30,
        peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
    )
    hs.grad = ws.grad = None
    return result


def cuda_probe(args):
    from liger_kernel.transformers.fused_linear_jsd import LigerFusedLinearJSD

    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    liger = LigerFusedLinearJSD(jsd_beta=1.0, temperature=1.0)
    liger_fp32 = LigerFusedLinearJSD(jsd_beta=1.0, temperature=1.0, accum_dtype=torch.float32)
    metadata = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "liger": importlib.metadata.version("liger-kernel"),
        "liger_source": str(__import__("inspect").getfile(LigerFusedLinearJSD)),
        "gpu": torch.cuda.get_device_name(0),
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "projection": "bf16 inputs, fp32 GEMM outputs and reductions",
        "seed": 42,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "scope": "synthetic hidden/head; no backbone, optimizer, FSDP, TQ or model weights",
    }
    rows = []
    for n in args.tokens:
        hs = torch.randn(n, args.hidden, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        ht = torch.randn_like(hs)
        ws = (
            torch.randn(args.vocab, args.hidden, device="cuda", dtype=torch.bfloat16) / args.hidden**0.5
        ).requires_grad_()
        wt = torch.randn_like(ws) / args.hidden**0.5
        labels = torch.zeros(n, device="cuda", dtype=torch.long)
        methods = {
            "dense": partial(mean_result, dense_rows, hs, ws, ht, wt),
            "token_recompute": partial(mean_result, RecomputedKL.apply, hs, ws, ht, wt, args.token_chunk, 0),
            "vocab_stream_recompute": partial(
                mean_result, RecomputedKL.apply, hs, ws, ht, wt, args.token_chunk, args.vocab_chunk
            ),
            "liger": partial(liger, hs, ws, ht, wt, labels),
            "liger_fp32_accum": partial(liger_fp32, hs, ws, ht, wt, labels),
        }
        expected = methods["dense"]()
        eg = torch.autograd.grad(expected, (hs, ws))
        expected_value = expected.item()
        eg_cpu = [x.detach().cpu() for x in eg]
        del expected, eg
        for name, fn in methods.items():
            actual = fn()
            grads = torch.autograd.grad(actual, (hs, ws))
            errors = dict(
                value_abs=abs(actual.item() - expected_value),
                hidden_grad_relative_l2=relative_error(grads[0].cpu(), eg_cpu[0]),
                head_grad_relative_l2=relative_error(grads[1].cpu(), eg_cpu[1]),
            )
            assert errors["value_abs"] < 1e-4, (name, errors)
            # A smoke bound, not the acceptance tolerance of the production adapter.
            assert max(errors["hidden_grad_relative_l2"], errors["head_grad_relative_l2"]) < 0.02, (name, errors)
            del actual, grads
            row = dict(
                tokens=n,
                vocab=args.vocab,
                hidden=args.hidden,
                method=name,
                mode="forward_backward",
                errors=errors,
                **measure(fn, hs, ws, True, args.warmup, args.repeats),
            )
            rows.append(row)
            print(json.dumps(row), flush=True)
        del eg_cpu
        for name, vocab_chunk in (("token_value", 0), ("vocab_stream_value", args.vocab_chunk)):
            fn = partial(value_mean, hs, ws, ht, wt, args.token_chunk, vocab_chunk)
            row = dict(
                tokens=n,
                vocab=args.vocab,
                hidden=args.hidden,
                method=name,
                mode="value_only",
                **measure(fn, hs, ws, False, args.warmup, args.repeats),
            )
            rows.append(row)
            print(json.dumps(row), flush=True)
        del methods, fn, hs, ht, ws, wt, labels
        gc.collect()
        torch.cuda.empty_cache()
    return dict(metadata=metadata, token_chunk=args.token_chunk, vocab_chunk=args.vocab_chunk, rows=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--tokens", type=int, nargs="+", default=[512, 4096])
    parser.add_argument("--hidden", type=int, default=1536)
    parser.add_argument("--vocab", type=int, default=151936)
    parser.add_argument("--token-chunk", type=int, default=512)
    parser.add_argument("--vocab-chunk", type=int, default=8192)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(*args.tokens, args.hidden, args.vocab, args.token_chunk, args.vocab_chunk, args.repeats) <= 0:
        parser.error("dimensions, chunk sizes and repeats must be positive")
    if args.warmup < 0:
        parser.error("warmup must be nonnegative")
    torch.set_num_threads(4)
    result = {"cpu_audit": cpu_audit()}
    if args.device == "cuda":
        result["cuda"] = cuda_probe(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
