# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Partition-invariant policy diagnostics, including real Gloo reduction."""

import math

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from verl.trainer.ppo.policy_metrics import pair_observations, update_observations
from verl.utils.metric import Metric
from verl.utils.metric.observations import Observation, observations_to_cpu
from verl.utils.py_functional import append_to_dict


def merge(parts):
    names = set().union(*(part.keys() for part in parts))
    return {name: Observation.merge_partitions([p[name] for p in parts if name in p]) for name in names}


def observe(z, mask=None):
    mask = torch.ones_like(z, dtype=torch.bool) if mask is None else mask
    # These partition tests treat each token as a complete one-token response.
    # Sequence statistics are invariant to splitting rows, not splitting a response.
    if z.ndim == 1:
        z, mask = z.reshape(-1, 1), mask.reshape(-1, 1)
    return pair_observations(z, torch.zeros_like(z), mask, "current_behavior", quantile=True)


def assert_equal(left, right):
    assert left.keys() == right.keys()
    for key in left:
        if math.isnan(left[key]):
            assert math.isnan(right[key]), key
        else:
            assert left[key] == pytest.approx(right[key], rel=1e-10, abs=1e-12), key


def test_unequal_partitions_and_empty_rank_are_not_means_of_means():
    z = torch.tensor([0.01, -0.2, 0.5, -2.0, 1.4, 0.0, 0.2], dtype=torch.float64)
    expected = merge([observe(z)])
    got = merge([observe(z[:1]), observe(z[1:5]), observe(z[5:]), observe(z[:0])])
    assert_equal(expected, got)
    weights = z.exp()
    assert got["policy/current_behavior/ess_fraction"] == pytest.approx(
        (weights.sum().square() / (len(z) * weights.square().sum())).item()
    )
    assert got["policy/current_behavior/ratio_p99"] == pytest.approx(weights.max().item())


def test_engine_append_protocol_merges_state_and_does_not_keep_autograd():
    x = torch.tensor([1.0, 3.0, 5.0], requires_grad=True)
    acc = {}
    append_to_dict(acc, {"m": Observation.mean(x[:1])})
    append_to_dict(acc, {"m": Observation.mean(x[1:])})
    assert acc["m"].aggregate() == 3
    assert acc["m"].state.grad_fn is None
    observations_to_cpu(acc)
    assert acc["m"].state.device.type == "cpu"
    assert Metric.aggregate_dp([acc["m"], Observation.mean(torch.tensor([9.0]))]) == 4.5


def test_padding_nan_is_excluded_but_valid_nan_is_observable():
    x = torch.tensor([0.0, math.nan, math.inf])
    got = merge([observe(x, torch.tensor([True, False, False]))])
    assert got["policy/current_behavior/nonfinite_fraction"] == 0
    assert got["policy/current_behavior/log_ratio_m2"] == 0
    bad = merge([observe(x, torch.tensor([True, True, False]))])
    assert bad["policy/current_behavior/nonfinite_fraction"] == 0.5
    for key, value in bad.items():
        if not key.endswith(("nonfinite_fraction", "valid_token_count")):
            assert math.isnan(value), key
    assert bad["policy/current_behavior/valid_token_count"] == 1


def test_empty_and_zero_denominators_are_undefined():
    result = merge([observe(torch.empty(0))])
    assert all(math.isnan(value) for key, value in result.items() if not key.endswith("valid_token_count"))
    assert result["policy/current_behavior/valid_token_count"] == 0
    assert math.isnan(Observation.ratio(torch.tensor(0.0), 0).aggregate())


def test_extreme_finite_log_weights_have_stable_ess():
    got = merge([observe(torch.tensor([1000.0])), observe(torch.tensor([999.0, -1000.0]))])
    w = torch.tensor([1.0, math.exp(-1), 0.0], dtype=torch.float64)
    assert got["policy/current_behavior/ess_fraction"] == pytest.approx((w.sum() ** 2 / (3 * w.square().sum())).item())
    assert math.isnan(got["policy/current_behavior/ratio_p99"])
    assert got["policy/current_behavior/log_ratio_abs_max"] == 1000


def test_global_p99_is_not_average_of_rank_p99():
    a, b = torch.zeros(200), torch.tensor([math.log(100.0)])
    got = merge([observe(a), observe(b)])
    assert got["policy/current_behavior/ratio_p99"] == 1
    assert (
        merge([observe(a)])["policy/current_behavior/ratio_p99"]
        + merge([observe(b)])["policy/current_behavior/ratio_p99"]
    ) / 2 > 40


def test_coefficient_observations_use_actual_guard_and_clip_branches():
    current = torch.tensor([[-1.0, -30.0, -1.0, -1.0]])
    behavior = torch.tensor([[-30.0, -1.0, -2.0, -0.5]])
    teacher = torch.tensor([[-2.0, -20.0, -2.0, -2.0]])
    advantage = teacher - current
    mask = torch.ones_like(current, dtype=torch.bool)
    active = torch.tensor([[False, True, False, True]])
    records = update_observations(
        current,
        behavior,
        behavior,
        advantage,
        mask,
        component="distillation",
        teacher=teacher,
        mode="token_ppo",
        clip_active=active,
    )
    got = merge([records])
    z = current - behavior
    w = torch.where(z.abs() > 20, 0, z.clamp(-20, 20).exp())
    coefficients = (w * advantage).abs()
    assert got["correction/log_ratio_guard/fraction"] == 0.5
    assert got["correction/behavior_ppo/active_fraction"] == 0.5
    assert got["correction/behavior_ppo/active_mass"] == pytest.approx(
        (coefficients[active].sum() / coefficients.sum()).item()
    )


def _gloo_worker(rank, path):
    dist.init_process_group("gloo", init_method="file://" + path, rank=rank, world_size=2)
    try:
        # Unequal token counts and unequal local micro-batch counts.
        partitions = [[torch.tensor([0.1])], [torch.tensor([-0.3, 0.8]), torch.tensor([0.5, 0.0])]]
        local = {}
        for z in partitions[rank]:
            append_to_dict(local, observe(z))
        observations_to_cpu(local)
        gathered = [None, None]
        dist.all_gather_object(gathered, local)
        result = {k: Metric.aggregate_dp([p[k] for p in gathered]) for k in local}
        all_z = torch.cat([z for part in partitions for z in part])
        assert_equal(result, merge([observe(all_z)]))
        # Exercise the actual engine worker output boundary, including loss
        # contribution DP averaging and replicated learning-rate handling.
        from types import SimpleNamespace

        from verl.utils import tensordict_utils as tu
        from verl.workers import engine_workers

        engine_workers.get_torch_device = lambda: SimpleNamespace(
            max_memory_allocated=lambda: 0, max_memory_reserved=lambda: 0
        )
        worker = SimpleNamespace(
            device_name="cpu",
            flops_counter=None,
            engine=SimpleNamespace(get_data_parallel_group=lambda: dist.group.WORLD),
        )
        output = engine_workers.TrainingWorker._postprocess_output(
            worker,
            {
                "metrics": {**local, "lr_used": 0.01, "_optimizer_step_applied": True},
                "loss": [float(rank + 1)],
                "model_output": {},
            },
            global_token_num=None,
            delta_time=1.0,
            forward_only=False,
            images_seqlens=None,
        )
        metrics = tu.get(output, "metrics")
        assert metrics["lr_used"] == 0.01
        assert metrics["loss"] == 1.5
        assert metrics["_optimizer_step_applied"] == [True, True]
        assert_equal({k: Metric.aggregate_dp(metrics[k]) for k in local}, merge([observe(all_z)]))
    finally:
        dist.destroy_process_group()


def test_real_gloo_merge(tmp_path):
    mp.spawn(_gloo_worker, args=(str(tmp_path / "rendezvous"),), nprocs=2, join=True)


def test_clip_mass_uses_the_loss_sequence_weights():
    current = torch.full((2, 3), -5.0)
    behavior = current - math.log(2)
    teacher = torch.tensor([[-7.0, -7.0, -7.0], [-1.0, -1.0, -1.0]])
    mask = torch.tensor([[True, True, True], [True, False, False]])
    active = torch.tensor([[False, False, False], [True, False, False]])
    scale = torch.ones_like(current) / mask.sum(-1, keepdim=True)
    observed = update_observations(
        current,
        behavior,
        behavior,
        teacher - current,
        mask,
        component="distillation",
        teacher=teacher,
        mode="token_ppo",
        clip_active=active,
        coefficient_scale=scale,
    )
    assert observed["correction/behavior_ppo/active_mass"].aggregate() == pytest.approx(4 / 6)


def test_pair_uses_pre_rejection_tokens_while_advantage_uses_loss_mask():
    current = torch.tensor([[-1.0, -10.0]])
    behavior = torch.tensor([[-1.0, -1.0]])
    used_mask = torch.tensor([[True, False]])
    original_mask = torch.tensor([[True, True]])
    result = update_observations(
        current,
        behavior,
        None,
        torch.tensor([[2.0, 99.0]]),
        used_mask,
        component="task",
        observation_mask=original_mask,
    )
    assert result["policy/current_behavior/log_ratio_m2"].aggregate() == 81 / 2
    assert result["advantage/task/mean"].aggregate() == 2
    assert not any(k.startswith("policy/current_anchor/") for k in result)


def test_loss_contributions_and_raw_observations_cannot_share_a_reducer():
    with pytest.raises(ValueError, match="Raw observations"):
        Metric("sum", 1).append(Observation.mean(torch.tensor([2.0])))
    with pytest.raises(ValueError, match="reduction types"):
        Metric.aggregate_dp([Observation.mean(torch.tensor([2.0])), Metric("sum", 1)])


def test_sequence_m2_uses_complete_responses_not_flat_tokens():
    z = torch.tensor([[1.0, -1.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float64)
    mask = torch.tensor([[True, True, False], [True, True, True]])
    result = merge([observe(z[:1], mask[:1]), observe(z[1:], mask[1:])])
    assert_equal(result, merge([observe(z, mask)]))
    assert result["policy/current_behavior/log_ratio_m2"] == 1
    assert result["policy/current_behavior/sequence_log_ratio_m2"] == 4.5
    assert result["policy/current_behavior/sequence_mean_log_ratio_m2"] == 0.5


def test_std_and_log_moments_merge_with_empty_partitions():
    z = torch.tensor([-2.0, 0.0, 3.0], dtype=torch.float64)
    for kind, reference in [("log_mean", z.exp().mean()), ("log_std", z.exp().std(unbiased=False))]:
        assert Observation.merge_partitions(
            [
                Observation.log_moment(z[:1], kind),
                Observation.log_moment(z[1:], kind),
                Observation.log_moment(z[:0], kind),
            ]
        ) == pytest.approx(reference.item())
    assert Observation.merge_partitions([Observation.std(z[:1]), Observation.std(z[1:])]) == pytest.approx(
        z.std(unbiased=False).item()
    )
    zero = torch.full((3,), -math.inf)
    assert Observation.log_moment(zero, "log_mean").aggregate() == 0
    assert Observation.ess(zero).is_empty()


def test_applied_weights_and_rejection_use_original_denominator():
    from verl.trainer.ppo.policy_metrics import applied_weight_observations, rejection_observations

    w = torch.tensor([[2.0, 3.0], [4.0, 5.0]])
    raw = torch.ones_like(w, dtype=torch.bool)
    used = torch.tensor([[True, False], [False, False]])
    result = merge([applied_weight_observations(w, used, raw), rejection_observations(used, raw)])
    assert result["correction/is/applied_mean"] == 0.5
    assert result["correction/is/applied_ess_fraction"] == pytest.approx(0.25)
    assert result["correction/is/applied_ess"] == pytest.approx(1)
    assert result["correction/rejection/token_fraction"] == 0.75
    assert result["correction/rejection/sequence_any_fraction"] == 1
    assert result["correction/rejection/sequence_all_fraction"] == 0.5
