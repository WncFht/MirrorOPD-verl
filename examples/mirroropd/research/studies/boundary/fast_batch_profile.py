# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Read-only follow-up performance probes after freezing the six scientific arms."""

import gc
import time

import torch

from examples.mirroropd.research.backends import hf_rkl
from examples.mirroropd.research.core.gradient_audit import compare
from examples.mirroropd.research.core.gradient_audit import measure as measured
from examples.mirroropd.research.core.gradient_audit import snapshot as gradients
from examples.mirroropd.research.core.io import dump
from examples.mirroropd.research.studies.boundary import fast_engine as engine
from examples.mirroropd.research.studies.boundary.fast_train import restore
from examples.mirroropd.research.studies.boundary.native import load, setup
from examples.mirroropd.research.studies.boundary.round2_common import OUT, initial_path, model_path
from examples.mirroropd.research.studies.boundary.train import backward as reference_backward


def main():
    import argparse

    argparse.ArgumentParser(description=__doc__).parse_args()
    setup()
    dest = OUT / "optimization-batched"
    dest.mkdir(exist_ok=True)
    rows = engine.panel()
    head, states = engine.teacher_cache("justrl", rows)
    batch = rows[:16]
    qcache = hf_rkl.log_probabilities(head, states, batch, 512)
    evidence = dict(
        used_by_scientific_runs=False,
        thresholds=dict(loss_absolute=1e-5, gradient_relative_l2=0.01, cosine=0.9999, peak_including_adam_gib=74),
        probes=[],
    )
    dump(dest / "thresholds.json", evidence)
    for label, path in [("S0", model_path("r1")), ("JustRL-step064", initial_path("CONT-J"))]:
        model = load("r1", student=True, path=path)
        engine.training_mode(model)
        ref = measured(model, lambda m=model: reference_backward(m, head, states, batch[:8]))
        reference = gradients(model)
        current = measured(model, lambda m=model: engine.backward(m, qcache, batch, 512))
        probes = []
        for mb in [1, 2, 4, 8]:
            model.gradient_checkpointing_disable()
            try:
                numerical = measured(model, lambda m=model, b=mb: engine.backward(m, qcache, batch[:8], 512, b))
                numerical.update(**compare(model, reference))
                numerical["loss_error"] = abs(numerical["loss"] - ref["loss"])
                timing = measured(model, lambda m=model, b=mb: engine.backward(m, qcache, batch, 512, b))
                adam_gib = sum(p.numel() * 8 for p in model.parameters()) / 2**30
                accepted = (
                    numerical["loss_error"] < 1e-5
                    and numerical["relative_l2"] < 0.01
                    and numerical["cosine"] > 0.9999
                    and timing["peak_gib"] + adam_gib < 74
                )
            except torch.OutOfMemoryError:
                model.zero_grad(set_to_none=True)
                gc.collect()
                torch.cuda.empty_cache()
                probes.append(dict(microbatch=mb, accepted=False, reason="out_of_memory"))
                continue
            probes.append(
                dict(
                    microbatch=mb,
                    activation_checkpointing=False,
                    numerical=numerical,
                    timing=timing,
                    estimated_peak_with_adam_gib=timing["peak_gib"] + adam_gib,
                    speedup=current["seconds"] / timing["seconds"],
                    accepted=accepted,
                )
            )
            print("BATCH_PROFILE", label, probes[-1], flush=True)
        evidence["probes"].append(dict(model=label, reference=ref, current_batch16=current, candidates=probes))
        dump(dest / "evidence.json", evidence)
        del model, reference
        gc.collect()
        torch.cuda.empty_cache()
    accepted = [
        b
        for b in [1, 2, 4, 8]
        if all(any(c["microbatch"] == b and c["accepted"] for c in p["candidates"]) for p in evidence["probes"])
    ]
    assert accepted
    best = min(
        accepted,
        key=lambda b: sum(
            next(c["timing"]["seconds"] for c in p["candidates"] if c["microbatch"] == b) for p in evidence["probes"]
        ),
    )
    # A disposable step checks real Adam memory; no new checkpoint or science arm is produced.
    model = load("r1", student=True, path=initial_path("CONT-J"))
    engine.training_mode(model)
    model.gradient_checkpointing_disable()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-6, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, foreach=False
    )
    restore(optimizer, "CONT-J", rows)
    smoke = measured(model, lambda: engine.backward(model, qcache, batch, 512, best))
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, foreach=False)
    assert bool(torch.isfinite(norm))
    t0 = time.perf_counter()
    optimizer.step()
    torch.cuda.synchronize()
    smoke.update(
        optimizer_seconds=time.perf_counter() - t0,
        grad_norm=float(norm),
        actual_peak_with_adam_gib=torch.cuda.max_memory_allocated() / 2**30,
        scientific_update=False,
        exported_checkpoint=False,
    )
    assert {int(v["step"]) for v in optimizer.state.values()} == {65}
    assert smoke["actual_peak_with_adam_gib"] < 74
    evidence["disposable_adam_step"] = smoke
    evidence.update(
        microbatch=best,
        activation_checkpointing=False,
        chunk=512,
        time=time.time(),
        warning="Timing probes plus one disposable Adam step; not used for the six scientific trajectories",
    )
    dump(dest / "complete.json", evidence)


if __name__ == "__main__":
    main()
