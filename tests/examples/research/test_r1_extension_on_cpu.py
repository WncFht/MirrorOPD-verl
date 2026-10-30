# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Legacy Adam restores against its original bank subset even when scoring coverage expands."""

import random

import torch

from examples.mirroropd.research.studies.boundary import fast_train


def test_restore_keeps_adam_and_checks_historical_subset(tmp_path, monkeypatch):
    model = torch.nn.Linear(2, 1)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-6, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    model(torch.ones(1, 2)).sum().backward()
    opt.step()
    for state in opt.state.values():
        state["step"].fill_(64)
    old = dict(id=0, prompt_ids=[1], response_ids=[2], source="bank", seed=42, group="A")
    extra = dict(old, id=1, group="I")
    payload = dict(
        step=64,
        arm="TB1-J",
        models=("r1", "justrl"),
        model_revisions={k: fast_train.MODELS[k] for k in ["r1", "justrl"]},
        trajectories=[{k: old[k] for k in ["id", "prompt_ids", "response_ids", "source", "seed"]}],
        optimizer=opt.state_dict(),
        torch_rng=torch.get_rng_state(),
        cuda_rng=[torch.tensor([1])],
        python_rng=random.getstate(),
    )
    torch.save(payload, tmp_path / "optimizer.pt")
    monkeypatch.setattr(fast_train, "initial_path", lambda _: tmp_path)
    restored_cuda = []
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", lambda value: restored_cuda.extend(value))
    fresh = torch.optim.AdamW(model.parameters(), lr=0.1)
    result = fast_train.restore(fresh, "CONT-J", [old, extra])
    assert result["trajectories_checked"] == 1 and result["optimizer_reset"] is False
    assert fresh.param_groups[0]["lr"] == 1e-6
    assert {int(s["step"]) for s in fresh.state.values()} == {64}
    assert len(restored_cuda) == 1
    for key, value in opt.state_dict()["state"].items():
        for slot, tensor in value.items():
            assert torch.equal(tensor, fresh.state_dict()["state"][key][slot])
