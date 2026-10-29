# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Gate optimized training on analytic, full-model gradient, and timing evidence."""

import gc
import time

import torch

from . import fast_engine as engine
from .native import load, setup
from .round2_common import OUT, dump, initial_path, model_path
from .train import backward as reference_backward


def analytic_test():
    torch.manual_seed(42)
    h = torch.randn(19, 13, dtype=torch.float32, requires_grad=True)
    w = torch.randn(31, 13, dtype=torch.float32, requires_grad=True)
    q = torch.randn(19, 37).log_softmax(-1)  # Teacher extra vocabulary stays in normalizer.
    lp = (h @ w.T).log_softmax(-1)
    reference = (lp.exp() * (lp - q[:, :31])).sum() / 16384
    reference.backward()
    loss, dz = engine.head_vjp(h.detach(), w.detach(), q, 16384)
    errors = dict(
        loss=abs(float(loss) - float(reference.detach())),
        hidden=float((dz @ w.detach() - h.grad).abs().max()),
        head=float((dz.T @ h.detach() - w.grad).abs().max()),
    )
    assert max(errors.values()) < 1e-7, errors
    return errors


def gradients(model):
    return {n: p.grad.detach().cpu().clone() for n, p in model.named_parameters() if p.grad is not None}


def compare(model, reference):
    sums = torch.zeros(4, device="cuda", dtype=torch.float64)
    for name, p in model.named_parameters():
        assert p.grad is not None and name in reference
        old = reference[name].reshape(-1)
        new = p.grad.reshape(-1)
        for start in range(0, len(old), 2**20):
            a = old[start : start + 2**20].to("cuda")
            b = new[start : start + 2**20]
            sums += torch.stack(
                [
                    a.square().sum(dtype=torch.float64),
                    b.square().sum(dtype=torch.float64),
                    (a * b).sum(dtype=torch.float64),
                    (a - b).square().sum(dtype=torch.float64),
                ]
            )
    aa, bb, ab, delta = sums.tolist()
    return dict(
        relative_l2=(delta / aa) ** 0.5, cosine=ab / (aa * bb) ** 0.5, reference_norm=aa**0.5, candidate_norm=bb**0.5
    )


def measured(model, call):
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    loss = call()
    torch.cuda.synchronize()
    return dict(loss=loss, seconds=time.perf_counter() - started, peak_gib=torch.cuda.max_memory_allocated() / 2**30)


def main():
    setup()
    dest = OUT / "optimization"
    if (dest / "complete.json").exists():
        return
    dest.mkdir(parents=True, exist_ok=True)
    evidence = dict(
        analytic=analytic_test(),
        thresholds=dict(loss_absolute=1e-5, gradient_relative_l2=0.01, cosine=0.9999),
        probes=[],
    )
    dump(dest / "thresholds.json", evidence)
    rows = engine.panel()
    head, states = engine.teacher_cache("justrl", rows)
    probe = rows[:2]
    cache_start = time.perf_counter()
    qcache = engine.probabilities(head, states, probe)
    torch.cuda.synchronize()
    evidence["cache_seconds_two_rows"] = time.perf_counter() - cache_start
    best = None
    for label, path in [("S0", model_path("r1")), ("JustRL-step064", initial_path("CONT-J"))]:
        model = load("r1", student=True, path=path)
        engine.training_mode(model)
        # Warm kernels without changing parameters or optimizer state.
        measured(model, lambda m=model: reference_backward(m, head, states, probe[:1]))
        baseline = measured(model, lambda m=model: reference_backward(m, head, states, probe))
        reference = gradients(model)
        candidates = []
        for chunk in [128, 256, 512] if best is None else [best]:
            result = measured(model, lambda c=chunk, m=model: engine.backward(m, qcache, probe, c))
            result.update(chunk=chunk, **compare(model, reference))
            result["loss_error"] = abs(result["loss"] - baseline["loss"])
            result["accepted"] = (
                result["loss_error"] < 1e-5
                and result["relative_l2"] < 0.01
                and result["cosine"] > 0.9999
                and result["peak_gib"] < 65
            )
            candidates.append(result)
            print("PROFILE", label, result, flush=True)
        valid = [r for r in candidates if r["accepted"]]
        assert valid, candidates
        if best is None:
            best = min(valid, key=lambda r: r["seconds"])["chunk"]
        evidence["probes"].append(dict(model=label, reference=baseline, candidates=candidates))
        dump(dest / "evidence.json", evidence)
        del model, reference
        gc.collect()
        torch.cuda.empty_cache()
    evidence.update(
        chunk=best,
        microbatch=1,
        head="analytic FP32 full-vocabulary VJP",
        teacher_cache="FP32 log probabilities resident for current batch16",
        snapshots="disabled",
        optimizer="unchanged AdamW foreach=False",
    )
    dump(dest / "complete.json", evidence)


if __name__ == "__main__":
    main()
