# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
import numpy as np

from examples.mirroropd.research.studies.teacher_difference.report_trajectory import statistics


def test_remaining_risk_normalization_removes_pure_amplitude_scaling():
    vectors = np.array([[1.0, 2.0], [2.0, 1.0], [1.0, 1.0], [2.0, 2.0]])
    data = dict(
        gram=(vectors @ vectors.T).tolist(),
        groups=[dict(losses=[v], pressure=[v / 2]) for v in (2.0, 3.0, 2.0, 3.0)],
    )
    source, held = [0], [1]
    point = statistics(data, source, held, np.ones((1, 1)), np.ones((1, 1)), "live")
    alpha = 0.1
    scaled = dict(
        gram=(alpha**2 * vectors @ vectors.T).tolist(),
        groups=[dict(losses=[alpha**2 * v], pressure=[alpha**2 * v / 2]) for v in (2.0, 3.0, 2.0, 3.0)],
    )
    other = statistics(scaled, source, held, np.ones((1, 1)), np.ones((1, 1)), "live")
    for key in ("D_over_L", "M", "pressure_transfer", "gamma", "source_q", "held_q"):
        np.testing.assert_allclose(point[key], other[key])
    np.testing.assert_allclose(other["D"], point["D"] * alpha**2)
    np.testing.assert_allclose(point["M"], point["gamma"] * np.sqrt(point["source_q"] * point["held_q"]))


def test_token_alignment_separates_rescaling_rotation_and_reversal():
    import torch

    from examples.mirroropd.research.studies.teacher_difference.alignment_probe import alignment_moments

    initial = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    live = torch.tensor([[0.1, 0.0], [0.0, 2.0], [0.0, 0.0]])
    m = alignment_moments(live, initial).sum(0)
    assert m[2] / m[3] == 1
    assert m[2] / (m[0] * m[1]).sqrt() < 1
    assert m[4] == m[5] == 0
    changed = alignment_moments(torch.tensor([[0.0, 1.0], [0.0, -1.0], [1.0, 0.0]]), initial)
    torch.testing.assert_close(changed[:, 4], torch.tensor([1.0, 0.0, 1.0], dtype=torch.float64))
    torch.testing.assert_close(changed[:, 5], torch.ones(3, dtype=torch.float64))


def test_cross_field_components_close_for_arbitrary_requests():
    from examples.mirroropd.research.studies.teacher_difference.report_trajectory import decomposition

    rng = np.random.default_rng(11)
    live, frozen = rng.normal(size=(4, 7)), rng.normal(size=(4, 7))
    vectors = np.concatenate((live, frozen))
    groups = [dict(request=[[1.0]]) for _ in range(4)]
    for i in range(4):
        groups.append(dict(request=[[live[i] @ live[i], frozen[i] @ frozen[i], live[i] @ frozen[i], 1.0]]))
    result = decomposition(dict(groups=groups, gram=vectors @ vectors.T), [0, 1], [2, 3])
    a, b = result["alpha_source"], result["alpha_held"]
    expected = (live[:2] - a * frozen[:2]).mean(0) @ (live[2:] - b * frozen[2:]).mean(0)
    np.testing.assert_allclose(result["residual"], expected)
    np.testing.assert_allclose(
        sum(result[k] for k in ("parallel", "source_parallel", "held_parallel", "residual")), result["D"]
    )
