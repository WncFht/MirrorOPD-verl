# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
import numpy as np

from examples.mirroropd.research.studies.teacher_difference.analyze_matrix import describe, panel_statistics


def test_cross_panel_prediction_preserves_negative_transfer():
    vectors = np.array([[1.0, 0.0], [3.0, 0.0], [-1.0, 1.0], [-3.0, 1.0]])
    stats = panel_statistics(vectors @ vectors.T, [0, 1], [2, 3], [2.0, 2.0], [4.0, 4.0])
    assert stats["D"] == -4 and stats["D_over_L"] == -1
    assert stats["low"] < stats["D_over_L"] < stats["high"]
    np.testing.assert_allclose(stats["cosine"], -2 / np.sqrt(5))
    np.testing.assert_allclose(stats["q"], 5 / 4)


def test_frozen_response_uses_native_loss_and_future_selection():
    groups = []
    for field in ("live", "frozen"):
        for index, (split, label) in enumerate((("train", "A"), ("train", "P"), ("heldout", "H"))):
            groups.append(
                dict(
                    field=field,
                    split=split,
                    group=label,
                    rows=[str(index)],
                    losses=[2.0 if field == "live" else 200.0],
                    pressure=[3.0],
                    second=[4.0],
                )
            )
    vectors = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [2.0, 0.0], [4.0, 0.0], [6.0, 0.0]])
    stats = describe(groups, vectors @ vectors.T, field="frozen", future={"P"})
    assert stats["D"] == 24 and stats["D_over_L"] == 12 and stats["L"] == 2
    assert stats["source_groups"] == 1


def test_restored_adam_matches_pytorch_history_and_clipping():
    import copy

    import torch

    from examples.mirroropd.research.studies.teacher_difference.adam_probe import restored_step

    torch.manual_seed(87)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, betas=(0.8, 0.9), weight_decay=0.2, foreach=True)
    for _ in range(3):
        for p in model.parameters():
            p.grad = torch.randn_like(p)
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    state = copy.deepcopy(optimizer.state_dict())
    expected = copy.deepcopy(model)
    expected_optimizer = torch.optim.AdamW(expected.parameters())
    expected_optimizer.load_state_dict(copy.deepcopy(state))
    gradient = torch.arange(1, 9, dtype=torch.float32)
    offset = 0
    for p in expected.parameters():
        p.grad = gradient[offset : offset + p.numel()].reshape_as(p).clone()
        offset += p.numel()
    torch.nn.utils.clip_grad_norm_(expected.parameters(), 1.0)
    before = torch.cat([p.detach().flatten() for p in model.parameters()])
    expected_optimizer.step()
    delta, meta = restored_step(model, gradient, state)
    for p, q in zip(model.parameters(), expected.parameters(), strict=True):
        torch.testing.assert_close(p, q)
    torch.testing.assert_close(delta, before - torch.cat([p.detach().flatten() for p in expected.parameters()]))
    assert meta["step_before"] == 3 and meta["step_after"] == 4 and meta["gradient_norm"] > 1


def test_calibration_selection_does_not_use_test_outcomes():
    from examples.mirroropd.research.studies.teacher_difference.report_crossed import calibrated_test

    rows = [
        dict(
            branch="x",
            source="S",
            field="native",
            displacement=delta,
            role=role,
            group=role,
            baseline=2.0,
            gain=gain,
            predicted_gain=0.1,
        )
        for delta, role, gain in (
            (1.0, "calibration", 0.2),
            (2.0, "calibration", 0.1),
            (1.0, "test", -20.0),
            (2.0, "test", 200.0),
        )
    ]
    selected, _ = calibrated_test(rows)
    assert len(selected) == 1 and selected[0]["displacement"] == 1.0 and selected[0]["gain"] == -20
    for r in rows:
        if r["role"] == "calibration":
            r["gain"] = -1
    selected, _ = calibrated_test(rows)
    assert selected[0]["gain"] == selected[0]["displacement"] == 0


def test_adam_prediction_uses_native_test_groups_and_saved_scale():
    from examples.mirroropd.research.studies.teacher_difference.report_crossed import adam_groups

    rows = [
        dict(row_id=i, group=g, split=s, loss=2.0)
        for i, g, s in ((0, "P", "train"), (1, "H0", "heldout"), (2, "H1", "heldout"))
    ]
    data = dict(
        signature=dict(calibration=["H0"]),
        baseline=rows,
        trials=[
            dict(
                source="P",
                scale=0.5,
                records=[dict(r, loss=1.0) for r in rows],
                predictions=dict(P=0.3, H0=0.2, H1=-0.1),
            )
        ],
    )
    groups = adam_groups(data, "crossed")
    test = next(r for r in groups if r["role"] == "test")
    assert test["group"] == "H1" and test["gain"] == 1.0 and test["predicted_gain"] == -0.1
    assert test["displacement"] == 0.5


def test_factorial_ratios_separate_request_scale_from_student_response(tmp_path):
    import json

    from examples.mirroropd.research.studies.teacher_difference.analyze_matrix import factorial_contrasts

    cells = []
    for student, response in (("justrl", 3.0), ("polaris", 1.0)):
        for teacher, request in (("justrl", 2.0), ("polaris", 1.0)):
            groups = [
                dict(
                    field=f,
                    split="heldout",
                    group=g,
                    rows=[g],
                    losses=[2.0],
                    pressure=[request**2],
                    second=[(response * request) ** 2],
                )
                for f in ("live", "frozen")
                for g in ("H0", "H1")
            ]
            gram = np.full((4, 4), (response * request) ** 2)
            path = tmp_path / f"{student}-{teacher}.json"
            path.write_text(json.dumps(dict(groups=groups, gram=gram.tolist())))
            cells.append(dict(student=student, teacher=teacher, trajectory=str(path)))
    manifest = tmp_path / "cells.json"
    manifest.write_text(json.dumps(cells))
    rows = factorial_contrasts(manifest, replicates=20)
    lookup = {(r["field"], r["contrast"], r["metric"]): r["ratio"] for r in rows}
    assert lookup["live", "teacher-J-over-P-on-justrl", "P"] == 4
    assert lookup["live", "teacher-J-over-P-on-justrl", "A"] == 1
    assert lookup["frozen", "student-J-over-P-for-polaris", "A"] == 9
    assert lookup["live", "teacher-by-student-interaction", "q"] == 1


def test_paired_cell_contrast_uses_ratio_of_risks_without_pooling_teachers():
    from examples.mirroropd.research.studies.teacher_difference.report_crossed import paired_cell_contrasts

    rows = [
        dict(
            branch=branch,
            field="native",
            displacement=0.003,
            source=source,
            group=group,
            role="test",
            baseline=loss * (2 if branch == "polaris-on-justrl" else 1),
            gain=gain if branch == "justrl-on-justrl" else 0.0,
            predicted_gain=0.0,
        )
        for branch in ("justrl-on-justrl", "polaris-on-justrl", "justrl-on-polaris", "polaris-on-polaris")
        for source in ("P", "R")
        for group, loss, gain in (("H0", 1.0, 0.0), ("H1", 3.0, 2.0))
    ]
    values = paired_cell_contrasts(rows, replicates=40)
    contrast = next(r for r in values if r["contrast"] == "teacher-J-minus-P-on-J")
    assert contrast["difference"] == 0.5
