# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Read-only repeated-backward probe at a legacy R1 continuation checkpoint."""

import argparse
import gc
import importlib.metadata
import os
from pathlib import Path

import torch

from examples.mirroropd.research.backends import hf_rkl
from examples.mirroropd.research.core.artifacts import file_hash, fingerprint
from examples.mirroropd.research.core.io import dump, read
from examples.mirroropd.research.studies.boundary import fast_engine as engine
from examples.mirroropd.research.studies.boundary.fast_train import extension_plan
from examples.mirroropd.research.studies.boundary.native import bank, load, setup
from examples.mirroropd.research.studies.boundary.round2_common import MODELS, OUT
from examples.mirroropd.research.studies.boundary.small_to_large import grad_compare


def probe(plan, arm, output):
    spec = extension_plan(plan, arm)
    config = spec["arms"][arm]
    rows = [r for r in bank("r1") if r["group"] == chr(65 + config["start"] // 16)]
    chunk = read(OUT / "optimization/complete.json")["chunk"]
    head, states = engine.teacher_cache(config["teacher"], rows, cache_root=spec["cache"])
    q = hf_rkl.log_probabilities(head, states, rows, chunk)
    model = load("r1", student=True, path=config["source_weights"])
    engine.training_mode(model)
    results = []
    for deterministic in [False, True]:
        os.environ["FLASH_ATTENTION_DETERMINISTIC"] = "1" if deterministic else "0"
        model.zero_grad(set_to_none=True)
        first = engine.backward(model, q, rows, chunk)
        saved = {name: p.grad.detach().cpu().clone() for name, p in model.named_parameters()}
        model.zero_grad(set_to_none=True)
        second = engine.backward(model, q, rows, chunk)
        comparison = grad_compare(model, saved)
        unequal = sum(not torch.equal(saved[name], p.grad.detach().cpu()) for name, p in model.named_parameters())
        results.append(
            dict(
                deterministic=deterministic,
                first_loss=first,
                second_loss=second,
                unequal_gradient_tensors=unequal,
                **comparison,
            )
        )
        del saved
        gc.collect()
        print("REPEATABILITY", results[-1], flush=True)
    historical = read(OUT / "arms" / arm / "training.json")[0]["loss"]
    dump(
        output,
        dict(
            arm=arm,
            student=list(MODELS["r1"]),
            teacher=list(MODELS[config["teacher"]]),
            source_weights=config["source_weights"],
            source_step=config["start"],
            bank=fingerprint(rows),
            scientific_updates=0,
            historical_next_loss=historical,
            results=results,
            source=file_hash(__file__),
            engine=file_hash(engine.__file__),
            backend=file_hash(hf_rkl.__file__),
            versions={name: importlib.metadata.version(name) for name in ["torch", "transformers", "flash-attn"]},
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    visible = [int(g) for g in os.environ["CUDA_VISIBLE_DEVICES"].split(",")]
    if len(visible) != 1 or not set(visible) <= set(range(6)):
        raise ValueError("This diagnostic uses one allocated GPU within 0-5")
    setup()
    probe(args.plan, args.arm, args.output)


if __name__ == "__main__":
    main()
