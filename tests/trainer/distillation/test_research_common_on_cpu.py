# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Research invariants: native support, paired statistics, identity and restoration."""

import copy
from types import SimpleNamespace

import pytest
import torch

from examples.mirroropd.research.core.cache import HiddenCache
from examples.mirroropd.research.core.io import dump, read
from examples.mirroropd.research.core.metrics import distances, vectors
from examples.mirroropd.research.core.statistics import mean_ci, ratio_ci
from examples.mirroropd.research.studies.cross_batch.protocol import LONG_PASS, validate_long_pass, validate_resume


def test_native_teacher_extra_mass_is_not_conditioned_away():
    p = torch.tensor([[0.6, 0.4]], dtype=torch.float64)
    q = torch.tensor([[0.3, 0.2, 0.5]], dtype=torch.float64)
    got = distances(p.log(), q.log())
    assert got["rkl"].item() == pytest.approx(torch.log(torch.tensor(2.0)).item(), abs=1e-7)
    assert got["fkl_shared"].item() == pytest.approx(0.0, abs=1e-14)
    assert got["tv"].item() == pytest.approx(0.5)
    padded = torch.cat((p, torch.zeros(1, 1, dtype=p.dtype)), dim=-1)
    mixture = 0.5 * (padded + q)
    expected_js = 0.5 * ((p * (p / mixture[:, :2]).log()).sum() + (q * (q / mixture).log()).sum())
    torch.testing.assert_close(got["js"].sum(), expected_js)


def test_rkl_gradient_matches_softmax_derivative_with_extra_teacher_rows():
    sp = torch.tensor([[0.2, -0.4, 0.8]], dtype=torch.float64, requires_grad=True)
    tq = torch.tensor([[0.6, -0.2, 0.3, 1.2]], dtype=torch.float64)
    loss = distances(sp, tq)["rkl"].sum()
    (gradient,) = torch.autograd.grad(loss, sp)
    p = sp.detach().softmax(-1)
    gap = sp.detach().log_softmax(-1) - tq.log_softmax(-1)[:, :3]
    expected = p * (gap - (p * gap).sum(-1, keepdim=True))
    torch.testing.assert_close(gradient, expected, atol=1e-14, rtol=1e-14)


def test_overlap_membership_and_probability_are_distinct():
    sp = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
    tq = torch.tensor([[4.0, 1.0, 3.0, 2.0, 0.0]])
    got = vectors(sp, tq, torch.tensor([0]), ks=(2,))
    assert got["overlap2"].item() == 0.5
    assert got["category2_3"].item() == 1.0
    assert got["common_p2"].item() == pytest.approx(sp.softmax(-1)[0, 0].item())
    for prefix in ["mass_p2_", "mass_q2_", "category2_"]:
        total = sum(got[prefix + str(i)] for i in range(4))
        torch.testing.assert_close(total, torch.ones_like(total), atol=2e-7, rtol=0)


def row(seed=1):
    return dict(source="student", id=7, seed=seed, prompt_ids=[1, 2], response_ids=[3, 4])


def test_cache_separates_same_prompt_rollout_seeds(tmp_path):
    cache = HiddenCache(tmp_path, "teacher", {"checkpoint": "q0", "precision": "bf16"})
    cache.get(row(1), lambda: torch.ones(2, 3))
    cache.get(row(2), lambda: torch.zeros(2, 3))
    assert cache.path(row(1)) != cache.path(row(2))
    assert cache.load(row(1)).sum() == 6
    assert cache.load(row(2)).sum() == 0


def test_cache_rejects_same_length_changed_prefix(tmp_path):
    cache = HiddenCache(tmp_path, "teacher", {"checkpoint": "q0"})
    cache.get(row(), lambda: torch.ones(2, 3))
    changed = row() | {"response_ids": [3, 5]}
    with pytest.raises(ValueError, match="identity mismatch"):
        cache.get(changed, lambda: pytest.fail("must not silently replace mismatched evidence"))


def test_cache_rejects_changed_model_and_malformed_tensor(tmp_path):
    cache = HiddenCache(tmp_path, "teacher", {"checkpoint": "q0"})
    cache.get(row(), lambda: torch.ones(2, 3))
    with pytest.raises(ValueError, match="identity mismatch"):
        HiddenCache(tmp_path, "teacher", {"checkpoint": "q1"}).load(row())
    payload = torch.load(cache.path(row()), weights_only=True)
    payload["hidden"] = torch.ones(1, 3)
    torch.save(payload, cache.path(row()))
    with pytest.raises(ValueError, match="shape"):
        cache.load(row())


def test_cache_does_not_trust_legacy_tensor_only_files(tmp_path):
    (tmp_path / "teacher").mkdir()
    torch.save(torch.zeros(2, 3), tmp_path / "teacher/student-7.pt")
    cache = HiddenCache(tmp_path, "teacher", {"checkpoint": "q0"})
    assert cache.get(row(), lambda: torch.ones(2, 3)).sum() == 6


def test_json_nonfinite_failure_preserves_existing_artifact(tmp_path):
    target = tmp_path / "result.json"
    dump(target, {"risk": 1.0})
    with pytest.raises(ValueError):
        dump(target, {"risk": float("nan")})
    assert read(target) == {"risk": 1.0}
    assert list(tmp_path.iterdir()) == [target]


def test_paired_ratio_and_mean_bootstrap():
    assert mean_ci([3.0] * 5) == [3.0, 3.0]
    assert ratio_ci([1.0, 4.0, 9.0], [2.0, 8.0, 18.0]) == [2.0, 2.0]
    assert mean_ci([1.0, 2.0, 9.0]) == mean_ci([1.0, 2.0, 9.0])
    with pytest.raises(ValueError, match="paired"):
        ratio_ci([1.0, 2.0], [2.0])
    with pytest.raises(ValueError, match="denominator"):
        ratio_ci([0.0], [1.0])


def test_long_protocol_rejects_stale_normalizer():
    record = dict(
        cap=1024,
        prompts_per_batch=16,
        consecutive_updates_per_batch=16,
        train_batches=20,
        total_batches=40,
        training_seed=42,
        lr=1e-6,
        betas=[0.9, 0.999],
        weight_decay=0.01,
        clip=1.0,
    )
    assert validate_long_pass(record).normalizer == 16384
    with pytest.raises(ValueError, match="cap"):
        validate_long_pass(record | {"cap": 2048})


def test_resume_requires_same_teacher_protocol_and_trajectories():
    identity = dict(teacher="justrl", settings=LONG_PASS.as_record(), bank=[row()])
    payload = dict(experiment=copy.deepcopy(identity), completed_batches=1, step=16)
    validate_resume(payload, identity, 20)
    with pytest.raises(ValueError, match="identity"):
        validate_resume(payload, identity | {"teacher": "polaris"}, 20)
    with pytest.raises(ValueError, match="identity"):
        validate_resume(payload, identity | {"bank": [row(2)]}, 20)
    with pytest.raises(ValueError, match="counter"):
        validate_resume(payload | {"step": 17}, identity, 20)
    with pytest.raises(ValueError, match="identity"):
        validate_resume({"completed_batches": 1, "step": 16}, identity, 20)


def test_fsdp_evaluation_restores_kernel_even_if_forward_fails(monkeypatch):
    from examples.mirroropd.research.backends import fsdp as engine

    original = engine.kernels.full_vocab_objective_result
    custom = lambda *args, **kwargs: None
    monkeypatch.setattr(engine.tu, "assign_non_tensor", lambda *args, **kwargs: None)

    def fail(data):
        assert engine.kernels.full_vocab_objective_result is custom
        raise RuntimeError("forward failed")

    with pytest.raises(RuntimeError, match="forward failed"):
        engine.evaluate(SimpleNamespace(infer_batch=fail), SimpleNamespace(clone=lambda: object()), reference_fn=custom)
    assert engine.kernels.full_vocab_objective_result is original
