# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Compare reductions with analytic dots, including cancellation and structural null."""

import pytest
import torch

from examples.mirroropd.research.backends.direction_projection import DirectionProjector, reduce_projection
from examples.mirroropd.research.studies.spectral.absorption.protocol import RunContext


def test_signed_projection_and_zero_direction():
    gradient = torch.tensor([[1.0, -1.0], [2.0, -2.0]])
    directions = {"aligned": gradient, "cancelled": torch.ones_like(gradient), "zero": torch.zeros_like(gradient)}
    norm, values = reduce_projection(gradient, directions, device="cpu")
    assert norm == pytest.approx(10**0.5)
    assert values["aligned"]["dot"] == 10
    assert values["aligned"]["unit_derivative"] == pytest.approx(norm)
    assert values["cancelled"]["dot"] == 0
    assert values["zero"] == dict(dot=0, norm=0, unit_derivative=None)
    with pytest.raises(ValueError, match="shapes"):
        reduce_projection(gradient, {"bad": torch.zeros(1)}, device="cpu")


def test_cache_and_reference_match(tmp_path):
    data = dict(learned=torch.randn(7, 11), off=torch.randn(7, 11), null=torch.zeros(7, 11))
    torch.save(data, tmp_path / "0000.pt")
    projector = DirectionProjector(tmp_path)
    gradient = torch.randn(7, 11)
    assert projector.project(0, gradient) == reduce_projection(gradient, data, device="cpu")
    (tmp_path / "0000.pt").unlink()
    assert projector.project(0, gradient) == reduce_projection(gradient, data, device="cpu")
    with pytest.raises(ValueError, match="requires a CUDA"):
        DirectionProjector(tmp_path, backend="cuda-stream").project(0, gradient)


def test_explicit_stop_roots(tmp_path):
    other = tmp_path / "other-stop"
    context = RunContext(tmp_path, stop_paths=(other,))
    context.check_stop()
    other.touch()
    with pytest.raises(InterruptedError, match="other-stop"):
        context.check_stop()


def test_group_journal_requires_identity_and_complete_tensor(tmp_path):
    from examples.mirroropd.research.studies.spectral.absorption.journal import GroupJournal
    from examples.mirroropd.research.studies.spectral.absorption.protocol import save_tensor

    journal = GroupJournal(tmp_path, {"checkpoint": "frozen-a"})
    save_tensor(tmp_path / "domain-gradient-sums/train_math.pt", torch.ones(2))
    journal.save("train:math", [{"position": 0}], [{"norm": 1}])
    assert GroupJournal(tmp_path, {"checkpoint": "frozen-a"}).completed["train:math"]["projections"] == [
        {"position": 0}
    ]
    with pytest.raises(ValueError, match="identity changed"):
        GroupJournal(tmp_path, {"checkpoint": "frozen-b"})
    (tmp_path / "domain-gradient-sums/train_math.pt").write_bytes(b"truncated")
    with pytest.raises(ValueError, match="incomplete group"):
        GroupJournal(tmp_path, {"checkpoint": "frozen-a"})


def test_claim_unique_and_waits_prerequisite(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    from examples.mirroropd.research.core.io import dump, read
    from examples.mirroropd.research.studies.spectral.absorption.queue import claim

    dump(
        tmp_path / "queue.json",
        {
            "jobs": [dict(task_id=i, status="pending") for i in range(20)]
            + [dict(task_id=20, status="pending", ready_file=str(tmp_path / "not-ready"))]
        },
    )
    with ThreadPoolExecutor(max_workers=8) as pool:
        jobs = list(pool.map(lambda gpu: claim(tmp_path, gpu), range(20)))
    assert sorted(job["task_id"] for job in jobs) == list(range(20))
    assert claim(tmp_path, 0) is None
    (tmp_path / "not-ready").touch()
    assert claim(tmp_path, 0)["task_id"] == 20
    (tmp_path / "STOP").touch()
    assert claim(tmp_path, 0) is None
    assert all(task["status"] == "running" for task in read(tmp_path / "queue.json")["jobs"])


def test_analysis_separates_cross_layer_from_state_cancellation():
    from examples.mirroropd.research.studies.spectral.absorption.analyze_gradients import aggregate

    # Each layer has a consistent sign across sites; concatenating layers cancels.
    rows = []
    vectors = []
    for key, sign in [("model.layers.0.self_attn.q_proj.weight", 1), ("model.layers.1.self_attn.q_proj.weight", -1)]:
        for site in range(2):
            rows.append(
                dict(
                    row_id=site,
                    position=0,
                    key=key,
                    gradient_norm=1,
                    directions={band: dict(dot=sign, norm=1) for band in ["learned", "off", "null"]},
                )
            )
        vectors.append(dict(key=key, sum_gradient_norm=2))
    keys = {r["key"] for r in rows}
    combined = aggregate(rows, vectors, keys)
    assert combined["directions"]["off"]["state_scalar_coherence"] is None
    assert combined["vector_coherence"] == pytest.approx(1)
    for key in keys:
        layer = aggregate(rows, vectors, {key})
        assert layer["directions"]["off"]["state_scalar_coherence"] == 1
