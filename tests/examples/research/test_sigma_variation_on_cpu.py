# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Spectral shape invariance, exact normalization and unequal-interval semantics."""

import pytest

from examples.mirroropd.research.studies.absorption_dynamics.sigma_variation import (
    distance,
    distribution,
    measure,
    trajectory,
)


def test_shape_distance_separates_scale_from_shape():
    a, b = distribution([4, 2, 1]), distribution([8, 4, 2])
    assert distance(a, b) == dict(sd_variation=0, hellinger=0)
    assert b["nuclear_norm"] == 2 * a["nuclear_norm"]
    assert b["frobenius_norm"] == 2 * a["frobenius_norm"]
    assert distance(a, distribution([7, 0, 0]))["sd_variation"] > 0
    assert distance(a, distribution([0, 0, 0])) is None
    with pytest.raises(ValueError, match="matching singular dimensions"):
        distance(a, distribution([1, 1]))
    with pytest.raises(ValueError, match="descending"):
        distribution([1, 2])


def test_unequal_intervals_are_net_speeds():
    points = [dict(step=t, **distribution(s)) for t, s in [(0, [3, 1]), (16, [2, 2]), (64, [3, 1])]]
    d = trajectory(points)
    assert d["intervals"][0]["sd_variation"] == pytest.approx(d["intervals"][1]["sd_variation"])
    assert d["intervals"][0]["net_speed"] == pytest.approx(3 * d["intervals"][1]["net_speed"])
    assert next(p for p in d["pairs"] if p["step_a"] == 0 and p["step_b"] == 64)["sd_variation"] == 0
    with pytest.raises(ValueError, match="unique and increasing"):
        trajectory(points[::-1])


def test_full_and_update_spectra_are_independent_and_teacher_baseline_guarded():
    raw = dict(
        initial_singular_values=[4, 2],
        teacher=dict(weight=dict(singular_values=[8, 4])),
        stages=[
            dict(step=16, weight=dict(singular_values=[4.1, 2]), singular_values=[0.2, 0.1]),
            dict(step=64, weight=dict(singular_values=[4.2, 2]), singular_values=[0.2, 0.2]),
        ],
    )
    d = measure(raw)
    assert d["weight"]["points"][0]["step"] == 0
    assert d["update"]["points"][0]["step"] == 16
    assert d["teacher_distances"][0]["remaining_fraction"] is None
    assert "teacher_update_distances" not in d
    assert d["update"]["intervals"][0]["sd_variation"] > d["weight"]["intervals"][-1]["sd_variation"]
    del raw["teacher"]
    assert "teacher_distances" not in measure(raw)


def test_uniform_spectral_scaling_and_energy_entropy():
    from examples.mirroropd.research.studies.absorption_dynamics.geometry_probes import spectrum_metrics

    initial = dict(step=0, **distribution([3, 3]))
    point = dict(step=16, **distribution([6, 6]))
    metrics = spectrum_metrics(point, initial)
    assert metrics["scale"] == pytest.approx(2)
    assert metrics["shape_residual"] == pytest.approx(0)
    assert metrics["entropy_rank"] == pytest.approx(2)
    assert metrics["maximum_perturbation"] == pytest.approx(1)
