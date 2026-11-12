# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
import itertools

import numpy as np
import pytest
import torch

from examples.mirroropd.research.studies.teacher_difference.response import (
    GradientBank,
    cotangents,
    split_request,
    token_signs,
)
from verl.trainer.distillation.numerics import kl_rows


@pytest.mark.parametrize("reduction,denominator", [("fixed_cap", 4), ("valid_tokens", 3)])
def test_bundled_vjp_matches_dense_tied_head_extra_teacher_support(reduction, denominator):
    torch.manual_seed(71)
    w = torch.randn(7, 3, dtype=torch.float64, requires_grad=True)
    x = torch.randn(3, 7, dtype=torch.float64)
    h = x @ w
    h0, w0 = torch.randn_like(h), torch.randn_like(w)
    ht, wt = torch.randn_like(h), torch.randn(9, 3, dtype=torch.float64)
    _, b = kl_rows(h @ w.T, ht @ wt.T)
    _, b0 = kl_rows(h0 @ w0.T, ht @ wt.T)
    parallel, perpendicular, _ = split_request(b, b0)
    torch.testing.assert_close((parallel * perpendicular).sum(-1), torch.zeros(3, dtype=w.dtype), atol=1e-12, rtol=0)
    signs = torch.tensor(list(itertools.product((-1.0, 1.0), repeat=3)), dtype=w.dtype)
    fields = ["live", "parallel", "frozen", *[f"sign-{i}" for i in range(len(signs))]]
    requests = [b, parallel, b0, *[b * sign[:, None] for sign in signs]]
    results = []
    for (field, dh, dw, moments), request in zip(
        cotangents(
            h, w, ht, wt, (h0, w0), fields=fields, cap=4, token_chunk=2, bundle=3, signs=signs, reduction=reduction
        ),
        requests,
        strict=True,
    ):
        actual = torch.autograd.grad((h, w), w, (dh, dw), retain_graph=True)[0]
        expected = torch.autograd.grad(((h @ w.T) * request.detach()).sum() / denominator, w, retain_graph=True)[0]
        torch.testing.assert_close(actual, expected)
        results.append(actual)
        if field == "live":
            torch.testing.assert_close(moments[1], b.square().sum() / denominator)
            torch.testing.assert_close(moments[3] + moments[4], moments[1])
    token_gradients = [
        torch.autograd.grad(((h @ w.T)[t] * b[t].detach()).sum(), w, retain_graph=True)[0] for t in range(3)
    ]
    exact_u = sum(v.square().sum() for v in token_gradients) / denominator
    estimated_u = denominator * torch.stack([v.square().sum() for v in results[3:]]).mean()
    torch.testing.assert_close(exact_u, estimated_u)
    frozen_signed = next(
        cotangents(h, w, ht, wt, (h0, w0), fields=["frozen-sign-0"], cap=4, signs=signs, reduction=reduction)
    )
    _, dh, dw, _ = frozen_signed
    actual = torch.autograd.grad((h, w), w, (dh, dw), retain_graph=True)[0]
    expected = torch.autograd.grad(((h @ w.T) * b0.detach() * signs[0, :, None]).sum() / denominator, w)[0]
    torch.testing.assert_close(actual, expected)


def test_zero_anchor_and_negative_scales_preserve_orthogonal_split():
    live = torch.tensor([[1.0, 2.0], [-2.0, 0.0]], dtype=torch.float64)
    initial = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float64)
    parallel, perpendicular, alpha = split_request(live, initial)
    torch.testing.assert_close(alpha, torch.tensor([0.0, -2.0], dtype=alpha.dtype))
    torch.testing.assert_close(parallel + perpendicular, live)
    torch.testing.assert_close((parallel * perpendicular).sum(-1), torch.zeros(2, dtype=live.dtype))


def test_group_bank_matches_mean_and_keeps_unused_parameters():
    model = torch.nn.Linear(3, 2)
    bank = GradientBank(model, ("live",))
    vectors = []
    for index in range(2):
        model.zero_grad(set_to_none=True)
        model.weight.sum().mul(index + 1).backward()
        vectors.append(torch.cat((model.weight.grad.flatten(), torch.zeros_like(model.bias))))
        bank.add("live")
    torch.testing.assert_close(bank.export("live", 2), torch.stack(vectors).mean(0))
    assert model.bias.grad is None


def test_random_sign_identity_is_paired_across_conditions():
    a = token_signs("question-1", 127, 16, 42, device="cpu")
    b = token_signs("question-1", 127, 16, 42, device="cpu")
    c = token_signs("question-2", 127, 16, 42, device="cpu")
    assert torch.equal(a, b) and not torch.equal(a, c)
    assert set(np.unique(a.numpy())) == {-1, 1}


def test_response_factors_and_four_cross_terms_reconstruct_native(tmp_path):
    import json

    from examples.mirroropd.research.studies.teacher_difference.report_response import component_transfer, factors

    rows = [dict(group=str(i), split="train" if i < 2 else "heldout", batch_index=i) for i in range(4)]
    bank = tmp_path / "bank.json"
    bank.write_text(json.dumps(rows))
    vectors = np.array([[2.0, 1.0], [1.0, 0.0], [1.0, 2.0], [0.0, 1.0], [3.0, 1.0], [1.0, 0.0], [1.0, 3.0], [0.0, 1.0]])
    data = dict(signature=dict(task=dict(cap=4, records=str(bank), stage=0), probes=2), groups=[], records=[])
    for i, row in enumerate(rows):
        for field in ("live", "parallel"):
            data["groups"].append(dict(**row, field=field, rows=[str(i)]))
        g, p = vectors[2 * i : 2 * i + 2]
        data["records"].append(
            dict(
                row_id=str(i),
                moments=[2.0, 3.0, 4.0, 1.0, 2.0, 0.0, 1.0],
                norm_sq={"live": g @ g, "parallel": p @ p, "sign-0": 2.0, "sign-1": 4.0},
                native_parallel_inner=g @ p,
            )
        )
    data["gram"] = (vectors @ vectors.T).tolist()
    out = factors(data, "heldout", np.ones((1, 2)) / 2)
    np.testing.assert_allclose(out["A"], out["A_token"] * out["R_agg"])
    np.testing.assert_allclose(out["second"], out["diagonal"] + out["offdiagonal"])
    np.testing.assert_allclose(
        out["second"], out["parallel_second"] + out["perpendicular_second"] + out["parameter_mixed"]
    )
    cross = component_transfer(data, "future-train")
    np.testing.assert_allclose(
        cross["native"],
        sum(
            cross[key]
            for key in (
                "parallel_parallel",
                "parallel_perpendicular",
                "perpendicular_parallel",
                "perpendicular_perpendicular",
            )
        ),
    )
    np.testing.assert_allclose(cross["native"], cross["parallel_source"] + cross["perpendicular_source"])
    native = vectors[[4, 6]].mean(0) @ vectors[[0, 2]].mean(0) / 2
    np.testing.assert_allclose(cross["native"], native)


def test_update_trials_restore_exact_master_and_equal_displacement():
    from examples.mirroropd.research.studies.teacher_difference.runtime import flatten_tensors
    from examples.mirroropd.research.studies.teacher_difference.update_probe import restore_step

    model = torch.nn.Linear(3, 2)
    parameters = tuple(model.parameters())
    baseline = flatten_tensors(parameters, [p.detach() for p in parameters])
    direction = torch.arange(1, baseline.numel() + 1, dtype=baseline.dtype)
    before = baseline.clone()
    for delta in (0.003, 0.01, 0.03):
        restore_step(model, baseline, direction, delta / direction.norm())
        current = flatten_tensors(parameters, [p.detach() for p in parameters])
        torch.testing.assert_close((current - baseline).norm(), torch.tensor(delta), atol=1e-7, rtol=1e-5)
        restore_step(model, baseline, None, 0)
        current = flatten_tensors(parameters, [p.detach() for p in parameters])
        assert torch.equal(current, baseline) and torch.equal(before, baseline)


def test_step_selection_uses_calibration_without_test_leakage():
    from examples.mirroropd.research.studies.teacher_difference.report_updates import analyze_updates

    vectors = np.array([[2.0, 1.0], [1.0, 0.0], [1.0, 2.0], [0.0, 1.0], [3.0, 1.0], [1.0, 0.0]])
    geometry = dict(groups=[], gram=(vectors @ vectors.T).tolist())
    for group, split in (("source", "train"), ("cal", "heldout"), ("test", "heldout")):
        for field in ("live", "parallel"):
            geometry["groups"].append(dict(group=group, split=split, field=field))
    baseline = [
        dict(row_id=str(i), group=g, split=s, loss=2.0)
        for i, (g, s) in enumerate((("source", "train"), ("cal", "heldout"), ("test", "heldout")))
    ]
    data = dict(signature=dict(task=dict(target="polaris"), calibration=["cal"]), baseline=baseline, trials=[])
    for field, direction in (
        ("native", vectors[0]),
        ("parallel", vectors[1]),
        ("perpendicular", vectors[0] - vectors[1]),
    ):
        for delta, cal_gain, test_gain in ((0.003, 0.2, 0.001), (0.01, 0.1, 0.3)):
            records = [
                dict(r, loss=r["loss"] - gain) for r, gain in zip(baseline, (0.1, cal_gain, test_gain), strict=True)
            ]
            data["trials"].append(
                dict(
                    source="source",
                    field=field,
                    displacement=delta,
                    gradient_norm=np.linalg.norm(direction),
                    records=records,
                )
            )
    _, _, selected = analyze_updates(data, geometry)
    assert all(r["displacement"] == 0.003 for r in selected)
    np.testing.assert_allclose([r["gain"] for r in selected], 0.001)
    # If every calibrated step harms, zero displacement keeps the test baseline.
    data["baseline"][2]["loss"] = 3.0
    for trial in data["trials"]:
        trial["records"][1]["loss"] = 2.1
    _, _, selected = analyze_updates(data, geometry)
    assert all(r["displacement"] == 0 and r["baseline"] == 3 and r["groups"] == 1 for r in selected)


def test_question_sensitivity_preserves_full_panel_and_energy_identity():
    from examples.mirroropd.research.studies.teacher_difference.report_response import question_sensitivity

    data = dict(signature=dict(task=dict(cap=4, target="justrl", stage=240), probes=2), records=[])
    for i, (p, u, e) in enumerate(((1.0, 20.0, 4.0), (2.0, 6.0, 2.0), (3.0, 4.0, 1.0))):
        data["records"].append(
            dict(row_id=i, split="heldout", moments=[0.0, p], norm_sq={"live": e, "sign-0": u / 4, "sign-1": u / 4})
        )
    rows = question_sensitivity(data)
    np.testing.assert_allclose(sum(r["token_energy_share"] for r in rows), 1)
    np.testing.assert_allclose(rows[0]["A_token_without"], 2)
    np.testing.assert_allclose(rows[0]["A_without"], 0.6)
    for row in rows:
        np.testing.assert_allclose(row["A_without"], row["A_token_without"] * row["R_agg_without"])
    assert len(data["records"]) == 3
