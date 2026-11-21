# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Compare the production hybrid update and resume against native optimizers."""

import copy

import pytest
import torch

from verl.utils.fsdp_optimizers import MuonAdamW, muon_parameter_groups
from verl.workers.config.optimizer import FSDPOptimizerConfig, build_optimizer


@pytest.mark.parametrize("shape", [(13, 7), (7, 13)])
@pytest.mark.parametrize("nesterov", [True, False])
def test_native_update_and_resume(shape, nesterov):
    torch.manual_seed(42)
    params = [torch.nn.Parameter(torch.randn(shape)), torch.nn.Parameter(torch.randn(5))]
    refs = [torch.nn.Parameter(p.detach().clone()) for p in params]
    model = torch.nn.Module()
    model.register_parameter("matrix", params[0])
    model.register_parameter("bias", params[1])
    config = FSDPOptimizerConfig(
        optimizer="MuonAdamW",
        optimizer_impl="verl.utils.fsdp_optimizers",
        lr=0.003,
        weight_decay=0.01,
        override_optimizer_config={"aux_lr": 0.001, "nesterov": nesterov},
    )
    opt = build_optimizer(params, config, model=model)
    muon = torch.optim.Muon([refs[0]], lr=0.003, weight_decay=0.01, nesterov=nesterov, adjust_lr_fn="match_rms_adamw")
    adam = torch.optim.AdamW([refs[1]], lr=0.001, weight_decay=0.01, foreach=False)
    for i in range(5):
        for p, r in zip(params, refs, strict=True):
            grad = torch.randn_like(p)
            p.grad, r.grad = grad, grad.clone()
        opt.step()
        muon.step()
        adam.step()
        for p, r in zip(params, refs, strict=True):
            torch.testing.assert_close(p, r, rtol=0, atol=0)
        if i == 2:
            state = copy.deepcopy(opt.state_dict())
            opt = MuonAdamW(muon_parameter_groups(model), lr=0.5, aux_lr=0.4)
            opt.load_state_dict(state)
            assert opt.auxiliary.state is opt.state
            assert [g["lr"] for g in opt.param_groups] == [0.003, 0.001]


def test_sgd_factory_does_not_pass_adam_betas():
    p = torch.nn.Parameter(torch.ones(2))
    opt = build_optimizer([p], FSDPOptimizerConfig(optimizer="SGD", lr=0.1, weight_decay=0))
    p.grad = torch.ones_like(p)
    opt.step()
    torch.testing.assert_close(p, torch.full_like(p, 0.9))
    assert not opt.state


def test_muon_factory_rejects_missing_model():
    with pytest.raises(ValueError, match="requires model"):
        build_optimizer([], FSDPOptimizerConfig(optimizer="MuonAdamW", optimizer_impl="verl.utils.fsdp_optimizers"))


def test_structural_selection_excludes_tied_head_and_embedding():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(7, 3)
            self.projection = torch.nn.Linear(3, 3)
            self.head = torch.nn.Linear(3, 7, bias=False)
            self.head.weight = self.embed.weight

        def get_output_embeddings(self):
            return self.head

    model = Model()
    groups = muon_parameter_groups(model)
    assert [id(p) for p in groups[0]["params"]] == [id(model.projection.weight)]
    assert {id(p) for p in groups[1]["params"]} == {id(model.embed.weight), id(model.projection.bias)}


def test_multiple_groups_scheduler_addition_closure_and_resume():
    torch.manual_seed(13)
    params = [torch.nn.Parameter(torch.randn(3, 3)) for _ in range(4)]
    refs = [torch.nn.Parameter(p.detach().clone()) for p in params]
    groups = [dict(params=[p], use_muon=i % 2 == 0, lr=0.001 * (i + 1)) for i, p in enumerate(params)]
    opt = MuonAdamW(groups[:3], lr=0.01)
    opt.add_param_group(groups[3])
    native = [
        (
            torch.optim.Muon([p], lr=g["lr"], weight_decay=0, adjust_lr_fn="match_rms_adamw")
            if g["use_muon"]
            else torch.optim.AdamW([p], lr=g["lr"], weight_decay=0, foreach=False)
        )
        for p, g in zip(refs, groups, strict=True)
    ]
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.5)
    for step in range(4):
        for p, ref in zip(params, refs, strict=True):
            p.grad = torch.randn_like(p)
            ref.grad = p.grad.clone()
        assert opt.step(lambda: torch.tensor(7.0)).item() == 7
        for ref_opt in native:
            ref_opt.step()
        for p, ref in zip(params, refs, strict=True):
            torch.testing.assert_close(p, ref, rtol=0, atol=0)
        scheduler.step()
        for ref_opt in native:
            ref_opt.param_groups[0]["lr"] *= 0.5
        if step == 1:
            state = copy.deepcopy(opt.state_dict())
            opt.load_state_dict(state)
    assert opt.partition["muon_tensors"] == 2
    assert opt.partition["aux_tensors"] == 2


def test_pure_muon_and_invalid_group():
    p = torch.nn.Parameter(torch.ones(3, 2))
    opt = MuonAdamW([dict(params=[p], use_muon=True)], lr=0.01)
    p.grad = torch.ones_like(p)
    opt.step()
    assert opt.auxiliary is None
    with pytest.raises(ValueError, match="explicit use_muon"):
        MuonAdamW([p], lr=0.01)
    with pytest.raises(ValueError, match="2D"):
        MuonAdamW([dict(params=[torch.nn.Parameter(torch.ones(2))], use_muon=True)], lr=0.01)
