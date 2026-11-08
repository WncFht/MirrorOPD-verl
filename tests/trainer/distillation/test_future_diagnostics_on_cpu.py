# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Future diagnostics preserve literal suffixes, gradients and fixed-bank evidence."""

import math

import pytest
import torch

from verl.trainer.distillation.evaluation import sample_position_sums, sample_sums
from verl.trainer.distillation.full_vocab import full_vocab_objective_result
from verl.trainer.distillation.output import FullVocabOutput
from verl.trainer.distillation.reduction import PackedReduction
from verl.trainer.evaluation.reporting import evaluation_metrics
from verl.utils.metric.observations import Observation
from verl.utils.metric.position import relative_position_bins
from verl.workers.config.distillation_spec import FutureCreditConfig, resolve_distillation_spec


@pytest.mark.parametrize("mode", ["sum", "mean", "discounted_sum", "discounted_mean", "discounted_sequence"])
@pytest.mark.parametrize("chunk", [1, 3, 20])
def test_future_readouts_match_literal_suffix_and_preserve_gradient(mode, chunk):
    torch.manual_seed(931)
    hs, ws, ht, wt, hq, wq = [
        torch.randn(*shape, dtype=torch.float64, requires_grad=True) for shape in [(5, 2), (6, 2)] * 3
    ]
    gamma = 0.8 if mode.startswith("discounted") else 1.0
    times = torch.tensor([0, 2, 5, 0, 0])
    layout = PackedReduction(
        (3, 0, 1, 1), times, torch.tensor([0, 1, 2, 3, 4]), future=FutureCreditConfig(mode=mode, gamma=gamma)
    )
    spec = resolve_distillation_spec({"loss_mode": "mirror", "mirror": {"eta": 0.2, "kappa": 0.5, "diagnostics": True}})
    result = full_vocab_objective_result(hs, ws, ht, wt, spec=spec, hq=hq, wq=wq, reduction=layout, token_chunk=chunk)
    masses, means = torch.zeros(5, dtype=torch.float64), torch.zeros(5, dtype=torch.float64)
    for t in range(3):
        weights = gamma ** (times[t + 1 : 3] - times[t]).double()
        masses[t] = weights.sum()
        if len(weights):
            means[t] = (weights * result.objective_rows[t + 1 : 3]).sum() / weights.sum()
    torch.testing.assert_close(result.future_mass, masses)
    torch.testing.assert_close(result.future_mean_cost, means)
    expected_f = means if mode in ("mean", "discounted_mean") else means * masses
    torch.testing.assert_close(result.future_coefficients, expected_f)
    bins = relative_position_bins(times, torch.tensor([6, 6, 6, 1, 1]))
    before = torch.autograd.grad(result.loss_sum, (hs, ws), retain_graph=True)
    output = FullVocabOutput.from_result(result, spec, 5, bins=bins, num_sequences=3)
    after = torch.autograd.grad(output.objective_sum, (hs, ws))
    for a, b in zip(before, after, strict=True):
        torch.testing.assert_close(a, b)
    metrics = {k: v.aggregate() for k, v in output.observations.items()}
    assert metrics["distillation/future/suffix_valid_tokens"] == 2
    assert metrics["distillation/future/suffix_mean_cost"] == pytest.approx(means[:2].mean().item())
    assert metrics["distillation/future/coefficient_p95"] == pytest.approx(expected_f.max().item())
    assert metrics["distillation/future/effective_mass"] == pytest.approx(masses.mean().item())
    for name, values in [("coefficient", expected_f), ("effective_mass", masses), ("suffix_mean_cost", means[:2])]:
        sums = sum(
            v
            for k, v in metrics.items()
            if k.startswith(f"distillation/future/{name}_position/") and k.endswith("/sum")
        )
        assert sums == pytest.approx(values.sum().item())
    assert all(not o.state.requires_grad for o in output.observations.values())


@pytest.mark.parametrize("eta", [0.2, 1.0])
def test_teacher_positions_do_not_merge_eos_or_scale_by_eta(eta):
    h = torch.ones(2, 1, dtype=torch.float64, requires_grad=True)
    p = torch.tensor([0.05, 0.45, 0.5], dtype=torch.float64).log().reshape(-1, 1).requires_grad_()
    t = torch.tensor([0.45, 0.05, 0.5], dtype=torch.float64).log().reshape(-1, 1)
    spec = resolve_distillation_spec({"loss_mode": "mirror", "mirror": {"eta": eta, "diagnostics": True}})
    result = full_vocab_objective_result(
        h, p, h.detach(), t, hq=h.detach(), wq=p.detach(), spec=spec, termination_token_ids=(0, 1)
    )
    output = FullVocabOutput.from_result(result, spec, 2, bins=torch.tensor([2, 7]), termination=True, num_sequences=1)
    expected = 0.4 * math.log(9)
    assert output.observations["distillation/teacher_rkl"].aggregate() == pytest.approx(expected)
    assert output.observations["distillation/teacher_rkl_per_sequence_sum"].aggregate() == pytest.approx(2 * expected)
    assert output.observations["distillation/teacher_rkl_position/p20_30/mean"].aggregate() == pytest.approx(expected)
    if eta < 1:
        assert output.observations["distillation/teacher_kl"].aggregate() == pytest.approx(0, abs=1e-14)


def test_bank_position_statistics_roundtrip_and_legacy_coverage():
    values = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    counts = [3, 1]
    bins = relative_position_bins(torch.tensor([0, 2, 4, 0]), torch.tensor([5, 5, 5, 1]))
    stats, totals = sample_position_sums(values, counts, bins), sample_sums(values, counts)
    rows = [
        dict(
            id=str(i),
            bank="public_fixed",
            domain="math",
            tokens=counts[i],
            rkl_sum=totals["rkl_sum"][i].item(),
            rkl_position_sum=stats["rkl_position_sum"][i].tolist(),
            rkl_position_count=stats["rkl_position_count"][i].tolist(),
        )
        for i in range(2)
    ]
    metrics = evaluation_metrics(rows, baselines={})
    prefix = "eval/rkl/public_fixed/math/position/"
    assert sum(v for k, v in metrics.items() if k.startswith(prefix) and k.endswith("/sum")) == 10
    assert sum(v for k, v in metrics.items() if k.startswith(prefix) and k.endswith("/token_count")) == 4
    assert metrics[prefix + "p50_60/mean"] == 3
    assert prefix + "p20_30/mean" not in metrics
    legacy = [dict(r) for r in rows]
    legacy[0].pop("rkl_position_sum")
    partial = evaluation_metrics(legacy, baselines={})
    assert partial[prefix + "rows"] == 1
    assert prefix + "p50_60/mean" not in partial
    rows[0]["rkl_position_count"][0] += 1
    with pytest.raises(ValueError, match="counts"):
        evaluation_metrics(rows, baselines={})


def test_coefficient_quantile_merges_tokens_instead_of_rank_percentiles():
    parts = [Observation.quantile(torch.tensor(x), 0.95) for x in [[0.0] * 19, [100.0], []]]
    assert Observation.merge_partitions(parts) == 0


@pytest.mark.parametrize("schema_name", ["RKL", "EVAL_RKL"])
def test_position_arrays_survive_parquet_and_legacy_nulls(tmp_path, schema_name):
    from verl.trainer.evaluation import artifacts

    schema = getattr(artifacts, schema_name)
    store = artifacts.ArtifactStore.create(tmp_path / schema_name, kind="diagnostic", metadata={})
    row = dict(
        id="a",
        bank="source_heldout",
        domain="math",
        rkl_sum=2.0,
        tokens=1,
        rkl_position_sum=[0.0] * 5 + [2.0] + [0.0] * 4,
        rkl_position_count=[0] * 5 + [1] + [0] * 4,
    )
    store.write_rows("scores.parquet", [{k: v for k, v in row.items() if k in schema.names}], schema=schema)
    restored = store.read_rows("scores.parquet")
    assert restored[0]["rkl_position_sum"] == row["rkl_position_sum"]
    assert restored[0]["rkl_position_count"] == row["rkl_position_count"]
    assert evaluation_metrics(restored, baselines={})["eval/rkl/source_heldout/position/p50_60/mean"] == 2
    legacy = dict(row, rkl_position_sum=None, rkl_position_count=None)
    metrics = evaluation_metrics([legacy], baselines={})
    assert metrics["eval/rkl/source_heldout/position/rows"] == 0
    assert "eval/rkl/source_heldout/position/p50_60/mean" not in metrics


@pytest.mark.parametrize("coarsened", [False, True])
def test_scoring_transport_preserves_position_arrays_and_measure(monkeypatch, coarsened):
    from types import SimpleNamespace

    from verl.trainer.evaluation import execution, scoring

    space = SimpleNamespace(
        spec=SimpleNamespace(mode="native"),
        termination_ids=(1, 2) if coarsened else (),
        metadata={"support": "fixture"},
    )
    fake_execution = SimpleNamespace(
        distribution=lambda **kwargs: space, balance=lambda meta: meta, score_teacher=lambda *args: None
    )
    monkeypatch.setattr(execution, "NativeEvaluationExecution", lambda trainer: fake_execution)
    monkeypatch.setattr(scoring, "hidden_dtype_bytes", lambda config: 4)
    monkeypatch.setattr(scoring, "numerical_profile", lambda config: {})
    monkeypatch.setattr(scoring, "payload_windows", lambda rows, **kwargs: [rows])
    row = dict(id="a", bank="source_heldout", source_id="a", response_mask=[1, 0, 1])
    meta = SimpleNamespace(keys=["a"], tags=[{}], partition_id="fixture")
    monkeypatch.setattr(scoring, "put_fixed_rows", lambda *args: (meta, {"a": row}))
    sums, counts = [1.0] + [0.0] * 8 + [2.0], [1] + [0] * 8 + [1]
    payload = dict(
        rkl_sum=torch.tensor([[3.0]]),
        rkl_count=torch.tensor([[2]]),
        rkl_position_sum=torch.tensor([sums]),
        rkl_position_count=torch.tensor([counts]),
    )
    if coarsened:
        payload.update(
            uncoarsened_rkl_sum=torch.tensor([[6.0]]),
            uncoarsened_rkl_position_sum=torch.tensor([[v * 2 for v in sums]]),
            student_excluded_mass_sum=torch.tensor([[0.0]]),
            teacher_excluded_mass_sum=torch.tensor([[0.0]]),
        )

    def get(**kwargs):
        assert set(kwargs["select_fields"]) == set(payload)
        return payload

    monkeypatch.setattr(scoring.tq, "kv_batch_get", get)
    monkeypatch.setattr(scoring.tq, "kv_clear", lambda **kwargs: None)
    worker = SimpleNamespace(
        begin_rkl_readout=lambda **kwargs: None, score_full_vocab=lambda meta: None, end_rkl_readout=lambda: None
    )
    trainer = SimpleNamespace(config={}, actor_optimizer_step=0, global_steps=0, actor_rollout_wg=worker)
    budget = SimpleNamespace(
        available_payload=lambda live: 1024, objective_space=coarsened, token_chunk=2, audit_state=False
    )
    scored, timing = scoring.score_native_rows(trainer, [row], hidden_size=2, budget=budget)
    assert scored[0]["rkl_position_sum"] == sums
    assert scored[0]["rkl_position_count"] == counts
    assert scored[0]["measure_id"] == timing["measure_id"]
    metrics = evaluation_metrics(scored, baselines={})
    assert metrics["eval/rkl/source_heldout/position/p90_100/mean"] == 2
    if coarsened:
        assert metrics["eval/rkl/source_heldout/uncoarsened_position/p90_100/mean"] == 4
