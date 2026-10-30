# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Verify a stage checkpoint after loading in a new GPU process, then discard one actual update."""

import argparse
import math
from pathlib import Path

import torch

from examples.mirroropd.assets.paths import resolve_location
from examples.mirroropd.research.backends import hf_models
from examples.mirroropd.research.backends import hf_rkl as engine
from examples.mirroropd.research.core import training_state
from examples.mirroropd.research.core.artifacts import fingerprint
from examples.mirroropd.research.core.io import dump, read
from examples.mirroropd.research.core.restore_audit import verify_loaded
from examples.mirroropd.research.runtime.cuda import setup
from examples.mirroropd.research.studies.fixed_transfer.data import required_rows
from examples.mirroropd.research.studies.fixed_transfer.spec import Execution, Study
from examples.mirroropd.research.studies.fixed_transfer.training import (
    arm_by_name,
    arm_directory,
    resume_signature,
    teacher_state,
    validate_checkpoint,
)


def verify(study, name):
    arm = arm_by_name(study, name)
    root = arm_directory(study, arm)
    marker = read(root / "resume/complete.json")
    weights = resolve_location(marker["weights"])
    validate_checkpoint(weights, verify_hash=True)
    execution = Execution(**read(root / "protocol.json")["execution"])
    model = hf_models.load(study.models[study.student], trainable=True, path=weights)
    hf_models.training_mode(model, execution)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=study.training.learning_rate,
        betas=study.training.betas,
        eps=study.training.epsilon,
        weight_decay=study.training.weight_decay,
        foreach=False,
    )
    step = training_state.restore(
        root / "resume",
        model,
        optimizer,
        signature=resume_signature(study, arm, execution),
        weight_identity=marker["weight_identity"],
    )
    checked_parameters, checked_slots = verify_loaded(root / "resume", weights, model, optimizer)
    # A real update checks that restored moments are usable and Adam advances from the saved step.
    rows = required_rows(study, arm)
    head, states = teacher_state(study, arm.teacher, rows)
    batch = [r for r in rows if r["group"] == study.training_group(step + 1)]
    q = engine.log_probabilities(head, states, batch, execution.token_chunk)
    tracked = model.get_decoder().norm.weight
    before = tracked.detach().clone()
    optimizer.zero_grad(set_to_none=True)
    loss = engine.backward(model, q, batch, study.training, execution, study.evaluation.stop_ids[0])
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), study.training.clip, foreach=False)
    assert math.isfinite(loss) and bool(torch.isfinite(norm))
    optimizer.step()
    assert {int(v["step"]) for v in optimizer.state.values()} == {step + 1}
    displacement = float((tracked.detach() - before).abs().max())
    assert displacement > 0
    dump(
        root / "resume/gpu-verified.json",
        dict(
            fingerprint=fingerprint(
                dict(step=step, signature=marker["signature"], weight_identity=marker["weight_identity"])
            ),
            rng_exact=True,
            step=step,
            signature=marker["signature"],
            weight_identity=marker["weight_identity"],
            parameters_checked=checked_parameters,
            optimizer_tensors_checked=checked_slots,
            exact_loaded_values=True,
            disposable_next_step=step + 1,
            loss=loss,
            norm_parameter_max_change=displacement,
            scientific_updates=0,
        ),
    )
    print("RESUME_VERIFIED", name, step, checked_parameters, checked_slots, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    args = parser.parse_args()
    setup()
    verify(Study.load(args.plan), args.arm)


if __name__ == "__main__":
    main()
