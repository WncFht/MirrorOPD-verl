# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Recover known log-preference directions, including native teacher tail mass."""

import pytest
import torch

from examples.mirroropd.research.core.preference_geometry import geometry, moments


def test_known_projection_and_teacher_support_shift():
    p0 = torch.tensor([[0.2, 0.3, 0.5], [0.4, 0.1, 0.5]], dtype=torch.float64).log()
    qa = torch.tensor([[1.0, 2.0, 0.0], [0.0, 1.0, 2.0]], dtype=torch.float64).log_softmax(-1)
    qb = torch.tensor([[2.0, 1.0, 0.0], [2.0, 0.0, 1.0]], dtype=torch.float64).log_softmax(-1)
    for scale in [-0.5, 0, 1, 2]:
        pk = (p0 + scale * (qb - qa)).log_softmax(-1)
        stats = moments(p0, qa, qb, pk)
        result = geometry(**{k: float(stats[k].sum()) for k in ["d2", "u2", "du"]})
        assert result["projection"] == pytest.approx(scale, abs=1e-12)
        assert result["residual"] == pytest.approx(0, abs=1e-12)
        shifted = moments(p0, qa + torch.tensor(0.8).double().log(), qb + torch.tensor(0.6).double().log(), pk)
        for key in ["d2", "u2", "du"]:
            torch.testing.assert_close(stats[key], shifted[key])
        torch.testing.assert_close(
            shifted["teacher_b_extra_mass"], torch.full((2,), 0.4, dtype=torch.float64), atol=1e-7, rtol=0
        )


def test_zero_stage_change_and_incomplete_student_support():
    p = torch.tensor([[0.4, 0.6]], dtype=torch.float64).log()
    stats = moments(p, p, p, p)
    assert not geometry(**{k: float(stats[k].sum()) for k in ["d2", "u2", "du"]})["identifiable"]
    with pytest.raises(ValueError, match="full normalized"):
        moments(p[:, :1], p[:, :1], p[:, :1], p[:, :1])


def test_paired_projection_uses_common_states_and_aggregated_moments():
    from examples.mirroropd.research.studies.fixed_transfer.stage_preferences import summarize

    records = []
    for a, b in [("L1", "L2"), ("L2", "L3")]:
        for student, alpha in [("L1", 0), ("L2", 1), ("L3", 2)]:
            for index, strength in enumerate([1.0, 4.0]):
                records.append(
                    dict(
                        id=index,
                        group="M01",
                        length=8,
                        a=a,
                        b=b,
                        student=student,
                        d2=strength,
                        du=alpha * strength,
                        u2=alpha**2 * strength,
                        teacher_a_extra_mass=0.0,
                        teacher_b_extra_mass=0.0,
                    )
                )
    summaries, paired = summarize(records, 8)
    assert len(paired) == 4  # math and pooled held-out, each with two stage contrasts
    assert all(r["later_minus_earlier_projection"] == pytest.approx(1) for r in paired)
    assert all(r["ci_low"] == pytest.approx(1) and r["ci_high"] == pytest.approx(1) for r in paired)
    assert all(r["projection"] == pytest.approx(int(r["student"][-1]) - 1) for r in summaries)
