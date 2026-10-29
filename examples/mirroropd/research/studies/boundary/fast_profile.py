# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Gate optimized training on analytic, full-model gradient, and timing evidence."""

import gc
import time

import torch

from examples.mirroropd.research.backends import hf_rkl
from examples.mirroropd.research.core.gradient_audit import compare
from examples.mirroropd.research.core.gradient_audit import measure as measured
from examples.mirroropd.research.core.gradient_audit import snapshot as gradients
from examples.mirroropd.research.core.io import dump
from examples.mirroropd.research.studies.boundary import fast_engine as engine
from examples.mirroropd.research.studies.boundary.native import load, setup
from examples.mirroropd.research.studies.boundary.round2_common import OUT, initial_path, model_path
from examples.mirroropd.research.studies.boundary.train import backward as reference_backward


def analytic_test():
    torch.manual_seed(42)
    h = torch.randn(19, 13, dtype=torch.float32, requires_grad=True)
    w = torch.randn(31, 13, dtype=torch.float32, requires_grad=True)
    q = torch.randn(19, 37).log_softmax(-1)  # Teacher extra vocabulary stays in normalizer.
    lp = (h @ w.T).log_softmax(-1)
    reference = (lp.exp() * (lp - q[:, :31])).sum() / 16384
    reference.backward()
    loss, dz = hf_rkl.head_vjp(h.detach(), w.detach(), q, 16384)
    errors = dict(
        loss=abs(float(loss) - float(reference.detach())),
        hidden=float((dz @ w.detach() - h.grad).abs().max()),
        head=float((dz.T @ h.detach() - w.grad).abs().max()),
    )
    assert max(errors.values()) < 1e-7, errors
    return errors


def main():
    import argparse

    argparse.ArgumentParser(description=__doc__).parse_args()
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
    qcache = hf_rkl.log_probabilities(head, states, probe)
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
