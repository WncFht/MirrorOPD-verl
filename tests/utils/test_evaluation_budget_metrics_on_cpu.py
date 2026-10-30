# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
import itertools
import math

import numpy as np
import pytest

from verl.utils.metric.evaluation import evaluation_metrics, evaluation_records, pass_at_k


def test_repetition_is_scored_and_counted_separately_from_eos():
    result = evaluation_metrics(
        ["math"] * 3,
        ["q"] * 3,
        {"acc": [1, 0, 1], "__finish_reason__": ["stop", "repetition", "repetition"]},
    )
    assert result["eval/math/acc/observed_mean"] == pytest.approx(2 / 3)
    assert result["eval/math/generation/finish_reason/stop_count"] == 1
    assert result["eval/math/generation/finish_reason/repetition_count"] == 2
    assert result["eval/math/generation/finish_reason/repetition_fraction"] == pytest.approx(2 / 3)


def test_pass_estimator_matches_all_subsets():
    for n in range(1, 8):
        for c in range(n + 1):
            labels = [1] * c + [0] * (n - c)
            for k in range(1, n + 1):
                reference = np.mean([max(s) for s in itertools.combinations(labels, k)])
                assert pass_at_k(n, c, k) == pytest.approx(reference)
    assert pass_at_k(16, 1, 16) == 1


def test_all_budgets_and_ties_are_not_bootstrap_best():
    result = evaluation_metrics(["math"] * 4, ["q"] * 4, {"acc": [1, 1, 0, 0], "pred": ["a", "a", "b", "b"]})
    for k in range(1, 5):
        assert f"eval/math/acc/pass@{k}" in result
        assert f"eval/math/acc/majority@{k}" in result
        assert result[f"eval/math/acc/majority@{k}"] == pytest.approx(0.5)
    assert result["eval/math/acc/pass@4"] == 1
    assert result["eval/math/acc/majority@4_mc_se"] == 0


def test_mixed_capabilities_failures_and_nonfinite():
    result = evaluation_metrics(
        ["math", "reward_only", "math"],
        ["a", "b", "a"],
        {"acc": [1, None, float("nan")], "reward": [1, 2.5, 0]},
        {("math", "a"): 4, ("reward_only", "b"): 1},
    )
    assert result["eval/reward_only/reward/mean"] == 2.5
    assert "eval/reward_only/acc/mean" not in result
    assert result["eval/math/acc/mean"] == 0.25
    assert result["eval/math/failed_count"] == 3
    assert result["eval/math/nonfinite_score_count"] == 1
    assert result["eval/math/acc/pass@4"] == 1


def test_local_rng_and_stable_sample_order():
    np.random.seed(123)
    state = np.random.get_state()
    sources = ["math"] * 16
    uids = ["q"] * 16
    infos = {"acc": [1] + [0] * 15, "pred": ["yes"] + [str(i) for i in range(15)]}
    first = evaluation_metrics(sources, uids, infos, sample_indices=list(range(16)))
    second = evaluation_metrics(
        sources, uids, {k: v[::-1] for k, v in infos.items()}, sample_indices=list(range(15, -1, -1))
    )
    assert first == second
    now = np.random.get_state()
    assert all(np.array_equal(a, b) for a, b in zip(state, now, strict=True))
    assert first["eval/math/acc/prefix_mean@1"] == 1
    assert first["eval/math/acc/pass@1"] == 1 / 16


def test_common_population_for_unequal_budgets_and_exact_order_stats():
    result = evaluation_metrics(["r"] * 5, ["a"] * 2 + ["b"] * 3, {"reward": [1, 3, 0, 2, 4]})
    assert result["eval/r/reward/mean"] == 2
    assert result["eval/r/reward/best@2"] == pytest.approx((3 + 10 / 3) / 2)
    assert "eval/r/reward/best@3" not in result
    assert result["eval/r/prompt_count"] == 2
    assert result["eval/r/samples_per_prompt_max"] == 3


def test_all_failed_reward_source_is_not_fabricated_accuracy():
    result = evaluation_metrics([], [], {}, {("custom_reward", "a"): 4}, source_fields={"custom_reward": ["reward"]})
    assert result["eval/custom_reward/reward/mean"] == 0
    assert not any("/acc/" in k for k in result)


def test_prompt_ci_is_distinct_from_response_variability():
    result = evaluation_metrics(["m"] * 8, ["a"] * 4 + ["b"] * 4, {"acc": [1, 0, 1, 0] * 2})
    assert result["eval/m/acc/within_prompt_std"] == 0.5
    assert result["eval/m/acc/mean_ci_low"] == result["eval/m/acc/mean_ci_high"] == 0.5
    assert all(math.isfinite(v) for v in result.values())


def test_complete_records_include_missing_generations():
    rows = evaluation_records(
        ["MATH-500"],
        ["q"],
        {"acc": [1], "reward": [1]},
        {("MATH-500", "q"): 2},
        sample_indices=[1],
        prompt_metadata={("MATH-500", "q"): {"prompt_id": "MATH-500:0", "prompt": "question"}},
        inputs=["question"],
        outputs=["answer"],
        scores=[1],
        ground_truths=["answer"],
        response_lengths=[2],
        turns=[1],
    )
    assert [r["sample_index"] for r in rows] == [0, 1]
    assert rows[0]["status"] == "generation_missing"
    assert rows[0]["acc"] is None
    assert rows[1]["status"] == "ok"
    assert rows[0]["prompt_id"] == rows[1]["prompt_id"]


def test_invalid_indices_and_score_collisions_fail():
    with pytest.raises(ValueError, match="duplicate"):
        evaluation_metrics(["m"] * 2, ["q"] * 2, {"acc": [1, 0]}, sample_indices=[0, 0])
    with pytest.raises(ValueError, match="collision"):
        evaluation_metrics(["m"], ["q"], {"format_score": [1], "format": [0]})


def test_all_failed_unknown_source_does_not_invent_score_semantics():
    result = evaluation_metrics([], [], {}, {("unknown_task", "q"): 2})
    assert result["eval/unknown_task/failed_count"] == 2
    assert result["eval/unknown_task/quality_defined"] == 0
    assert not any("/acc/" in k or "/reward/" in k for k in result)


def test_default_null_seed_and_numeric_predictions():
    kwargs = dict(data_sources=["m"] * 3, uids=["q"] * 3, infos={"acc": [1, 1, 0], "pred": [1, 1, 2]})
    result = evaluation_metrics(**kwargs, seed=None)
    assert result == evaluation_metrics(**kwargs, seed=42)
    assert result["eval/m/acc/majority@3"] == 1
    assert not any("/pred/" in k for k in result)
    with pytest.raises(ValueError, match="metric_fields"):
        evaluation_metrics([], [], {}, {("m", "q"): 2}, source_fields={"m": "acc"})


def test_tool_counts_and_finish_reasons_are_generation_metrics_not_scores():
    result = evaluation_metrics(
        ["m"] * 2,
        ["q"] * 2,
        {"acc": [1, 0], "tool_call_counts": [0, 2], "__tool_calls__": [0, 2], "__finish_reason__": ["stop", "length"]},
    )
    assert result["eval/m/generation/tool_calls/mean"] == 1
    assert result["eval/m/generation/finish_reason/length_count"] == 1
    assert not any("/tool_call_counts/" in k for k in result)


def test_observed_budget_curves_have_explicit_failure_conditioning():
    result = evaluation_metrics(["m"], ["q"], {"acc": [1], "pred": ["yes"]}, {("m", "q"): 2})
    assert result["eval/m/acc/pass@1"] == 0.5
    assert result["eval/m/acc_observed/pass@1"] == 1
    assert result["eval/m/acc_observed/response_count"] == 1
    assert result["eval/m/acc_observed/prompt_count"] == 1
    assert "eval/m/acc_observed/pass@2" not in result
    complete = evaluation_metrics(["m"], ["q"], {"acc": [1]})
    assert not any("/acc_observed/" in k for k in complete)


def test_unknown_tool_counts_are_not_zero_calls():
    from verl.trainer.ppo.metric_utils import compute_tool_call_metrics

    result = compute_tool_call_metrics([None, 0, 2, float("nan")])
    assert result["tool_call_counts/known_fraction"] == 0.5
    assert result["tool_call_counts/mean"] == 1
    assert compute_tool_call_metrics([None]) == {"tool_call_counts/known_fraction": 0}


def test_abstentions_keep_known_voting_curve_but_not_other_sources():
    result = evaluation_metrics(
        ["MATH-500"] * 2 + ["code"], ["a"] * 2 + ["b"], {"acc": [0, 0, 1], "pred": [None, None, None]}
    )
    assert result["eval/math500/acc/majority@2"] == 0
    assert result["eval/math500/acc/voting_defined"] == 1
    assert result["eval/code/acc/voting_defined"] == 0
    assert "eval/code/acc/majority@1" not in result


@pytest.mark.parametrize("source,slug", [("AIME24", "aime24"), ("AIME26", "aime26")])
def test_new_aime_benchmarks_report_accuracy_and_failures(source, slug):
    metrics = evaluation_metrics([source] * 2, ["question"] * 2, {"acc": [1, 0], "pred": ["42", "43"]})
    assert metrics[f"eval/{slug}/acc/mean"] == 0.5
    assert metrics[f"eval/{slug}/acc/pass@2"] == 1
    assert metrics[f"eval/{slug}/response_count"] == 2
    assert metrics[f"eval/{slug}/failed_count"] == 0
