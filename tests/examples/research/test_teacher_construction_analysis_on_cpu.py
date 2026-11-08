# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Check paired uncertainty, denominator stability and actual rollout costs."""

import json

import numpy as np
import pytest

from examples.mirroropd.research.studies.fixed_transfer import teacher_construction_analysis as analysis


def test_recovery_is_suppressed_without_teacher_gain():
    values = analysis.capability_statistics([0, 1] * 10, [0, 1] * 10, [1] * 20)
    assert values["teacher_gain"] == 0
    assert values["gain_recovered"] is None
    assert not values["recovery_denominator_stable"]


def test_metric_tail_audit_keeps_untrimmed_mean_and_reports_question_leverage():
    value = analysis.metric_tail_statistics([0, 1, 99], ["a", "b", "c"])
    assert value["mean"] == pytest.approx(100 / 3)
    assert value["largest_question_id"] == "c"
    assert value["largest_question_fraction"] == 0.99
    assert value["effective_question_count"] == pytest.approx(10000 / 9802)
    assert value["leave_one_out_mean_min"] == 0.5
    assert value["leave_one_out_mean_max"] == 50
    zero = analysis.metric_tail_statistics([0, 0], ["a", "b"])
    assert zero["mean_ci95"] == [0, 0]
    assert zero["largest_question_fraction"] is None
    negative = analysis.metric_tail_statistics([-1e-8, 1], ["a", "b"])
    assert not negative["nonnegative"]
    assert negative["minimum"] == -1e-8
    assert negative["effective_question_count"] is None
    with pytest.raises(ValueError, match="finite"):
        analysis.metric_tail_statistics([np.inf], ["a"])


def test_token_weighted_diagnostic_does_not_replace_equal_question_mean():
    points = [
        dict(
            arm="RAW",
            bank="development",
            domain="math",
            id="a",
            tokens=1,
            rkl=1,
            fkl=1,
            chi2=1,
            centered_logratio_variance=1,
        ),
        dict(
            arm="RAW",
            bank="development",
            domain="math",
            id="b",
            tokens=99,
            rkl=0,
            fkl=0,
            chi2=0,
            centered_logratio_variance=0,
        ),
    ]
    rows = analysis.feature_tail_table(points)
    assert len(rows) == 4
    assert rows[0]["mean"] == 0.5
    assert rows[0]["token_weighted_mean_diagnostic"] == 0.01
    assert rows[0]["largest_question_response_tokens"] == 1
    assert rows[0]["short_responses_count_le16"] == 1
    assert rows[0]["short_responses_metric_fraction"] == 1


def test_recovery_is_suppressed_when_gain_uncertainty_crosses_zero():
    values = analysis.capability_statistics([0] * 20, [1] + [0] * 19, [1] * 20)
    assert values["teacher_gain"] > 0
    assert values["teacher_gain_ci95"][0] == 0
    assert values["gain_recovered"] is None


def test_paired_recovery_and_difference_with_stable_gain():
    values = analysis.capability_statistics([0] * 20, [1] * 20, [1] * 10 + [0] * 10)
    assert values["recovery_denominator_stable"]
    assert values["gain_recovered"] == pytest.approx(0.5)
    assert values["student_minus_teacher"] == pytest.approx(-0.5)
    assert values["gain_recovered_ci95"][0] < 0.5 < values["gain_recovered_ci95"][1]
    assert values["base_ci95"] == [0, 0]
    assert values["teacher_ci95"] == [1, 1]
    assert values["student_ci95"][0] < 0.5 < values["student_ci95"][1]


def test_reuse_counts_training_presentations_but_not_fresh_generation_twice():
    rows = [dict(step=i + 1, outer=i // 2, sampled_tokens=10 * (i // 2 + 1)) for i in range(4)]
    value = analysis.token_costs([dict(step=1, presented_tokens=45)], rows)
    assert value["teacher_response_token_presentations"] == 45
    assert value["opd_fresh_response_tokens"] == 30
    assert value["opd_response_token_presentations"] == 60
    rows[-1]["sampled_tokens"] += 1
    with pytest.raises(ValueError, match="reuse"):
        analysis.token_costs([], rows)


def test_risk_ratio_does_not_clip_negative_absorption_or_near_zero_denominators():
    before = np.ones(2000)
    value = analysis.risk_statistics(1, 2, before, before * 2)
    assert value["residual_fraction"] == 2
    assert value["absolute_drop"] == -1
    assert value["residual_ci95"] == [2, 2]
    near_zero = analysis.risk_statistics(0, 0, before * 0, before * 0)
    assert near_zero["residual_fraction"] is None


def test_four_domain_macro_requires_all_domains_and_full_mode_rejects_partial(tmp_path):
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    prompts = [dict(id=d, domain=d) for d in analysis.DOMAINS]
    for name in ["evaluation", "dev"]:
        (prepared / f"{name}.jsonl").write_text("".join(json.dumps(v) + "\n" for v in prompts))
    (tmp_path / "protocol.json").write_text(json.dumps(dict(arms=["RAW"], evaluation=dict(checkpoints=[0, 16]))))
    out = tmp_path / "opd/RAW"
    out.mkdir(parents=True)
    values = [
        dict(
            id=d, domain=d, step=step, rkl=(index + 1) / (2 if step else 1), fkl=1, chi2=1, centered_logratio_variance=1
        )
        for index, d in enumerate(analysis.DOMAINS)
        for step in [0, 16]
    ]
    path = out / "heldout.jsonl"
    path.write_text("".join(json.dumps(v) + "\n" for v in values))
    rows, _ = analysis.risk_table(tmp_path, partial=True)
    macro = next(v for v in rows if v["domain"] == "macro4" and v["step"] == 16)
    assert macro["initial_rkl"] == pytest.approx(2.5)
    assert macro["rkl"] == pytest.approx(1.25)
    assert macro["residual_fraction"] == pytest.approx(0.5)
    points = analysis.feature_question_table(tmp_path, partial=True)
    assert len(points) == 4
    tails = analysis.feature_tail_table(points)
    assert len(tails) == 16
    assert next(v for v in tails if v["domain"] == "code" and v["metric"] == "rkl")["mean"] == 2
    with pytest.raises(ValueError, match="Incomplete initial features"):
        analysis.feature_question_table(tmp_path)
    path.write_text("".join(json.dumps(v) + "\n" for v in values if v["domain"] != "if"))
    rows, _ = analysis.risk_table(tmp_path, partial=True)
    assert not any(v["domain"] == "macro4" for v in rows)
    with pytest.raises(ValueError, match="Incomplete"):
        analysis.risk_table(tmp_path)


def test_live_snapshot_hashes_analyzed_bytes_and_records_omitted_fragment(tmp_path):
    root, output = tmp_path / "run", tmp_path / "analysis"
    root.mkdir()
    output.mkdir()
    source = root / "live.jsonl"
    source.write_bytes(b'{"id":1}\n{"id":')
    with pytest.raises(ValueError, match="Unterminated"):
        analysis.snapshot_inputs(root, output, [source])
    frozen, receipt = analysis.snapshot_inputs(root, output, [source], partial=True)
    source.write_bytes(b'{"id":1}\n{"id":2}\n')
    assert analysis.records(frozen / source.name) == [{"id": 1}]
    assert receipt[source.name]["omitted_trailing_bytes"] == 6
    assert receipt[source.name]["sha256"] == analysis.digest(frozen / source.name)
    assert (output / "source.py").is_file()


def test_capability_tables_require_frozen_ids_and_a_complete_paired_benchmark(tmp_path):
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    expected = [dict(id=str(i), benchmark="AIME24") for i in range(20)]
    (prepared / "benchmark.jsonl").write_text("".join(json.dumps(v) + "\n" for v in expected))
    (tmp_path / "protocol.json").write_text(json.dumps(dict(arms=["RAW"], populations=dict(benchmark=dict(count=20)))))
    for label, score in [("base", 0), ("RAW-teacher", 1), ("RAW-student", 0.5)]:
        path = tmp_path / "benchmarks" / label / "responses.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text(
            "".join(
                json.dumps(dict(**v, reward=score, dapo_verifier_reward=0, tokens=42, finish_reason="length")) + "\n"
                for v in expected
            )
        )
    capabilities, behavior = analysis.capability_tables(tmp_path)
    assert capabilities[0]["gain_recovered"] == 0.5
    teacher = next(v for v in behavior if v["checkpoint"] == "teacher")
    assert teacher["grader_disagreement"] == 1
    assert teacher["length_stop_fraction"] == 1
    path.write_text(path.read_text().splitlines()[0] + "\n")
    assert analysis.capability_tables(tmp_path, partial=True) == ([], [])
    with pytest.raises(ValueError, match="Incomplete"):
        analysis.capability_tables(tmp_path)
    with pytest.raises(ValueError, match="Duplicate"):
        analysis.identified([expected[0], expected[0]], {v["id"]: v for v in expected}, "test")


def test_prediction_features_only_use_independent_development_readouts():
    def risk(bank, step, drop):
        return dict(
            arm="RAW",
            domain="code",
            bank=bank,
            step=step,
            initial_rkl=2,
            rkl=2 - drop,
            absolute_drop=drop,
            residual_fraction=(2 - drop) / 2,
        )

    features = [dict(arm="RAW", domain="code", bank="development", rkl=2, fkl=3, chi2=4, centered_logratio_variance=5)]
    risks = [risk("development", 4, 0.2), risk("development", 16, 0.5), risk("Base-bank", 256, 1)]
    rows = analysis.prediction_table(risks, features)
    assert rows[0]["dev_step4_absolute_drop"] == 0.2
    assert rows[0]["test_residual_fraction"] == 0.5
    assert rows[0]["dev_initial_chi2"] == 4
    assert analysis.prediction_table(risks[1:], features) == []


def test_method_contrast_pairs_the_same_questions_and_keeps_direction(tmp_path):
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "evaluation.jsonl").write_text(
        "".join(json.dumps(dict(id=d, domain=d)) + "\n" for d in analysis.DOMAINS)
    )
    (tmp_path / "protocol.json").write_text(
        json.dumps(dict(arms=["RAW", "CCS-rho0"], evaluation=dict(checkpoints=[0, 16])))
    )
    for arm, endpoint in [("RAW", 0.5), ("CCS-rho0", 0.75)]:
        path = tmp_path / "opd" / arm / "heldout.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text(
            "".join(
                json.dumps(dict(id=d, domain=d, step=step, rkl=endpoint if step else 1)) + "\n"
                for d in analysis.DOMAINS
                for step in [0, 16]
            )
        )
    values = analysis.method_contrasts(tmp_path)
    assert len(values) == 5
    macro = next(v for v in values if v["domain"] == "macro4")
    assert macro["prefixes_matched"]
    assert macro["residual_rkl_left_minus_right"] == -0.25
    assert macro["absolute_drop_left_minus_right"] == 0.25
    assert macro["absolute_drop_difference_ci95"] == [0.25, 0.25]


def test_shared_generation_pool_is_counted_once_and_requires_complete_identity(tmp_path):
    prepared, generated = tmp_path / "prepared", tmp_path / "generated"
    prepared.mkdir()
    generated.mkdir()
    populations = {name: dict(count=2) for name in ["teacher", "dev", "evaluation"]}
    (tmp_path / "protocol.json").write_text(json.dumps(dict(populations=populations, generation=dict(candidates=2))))
    for name in populations:
        (prepared / f"{name}.jsonl").write_text('{"id":"a"}\n{"id":"b"}\n')
    for population, kind in [("teacher", "base"), ("teacher", "expert"), ("dev", "base"), ("evaluation", "base")]:
        candidates = [dict(reward=1, tokens=3, generated_ids=[1, 2, 3], finish_reason="length")]
        if population == "dev":
            candidates[0]["reward"] = None
        if population == "teacher" and kind == "base":
            candidates.append(dict(reward=0, tokens=1, generated_ids=[1], finish_reason="stop"))
        (generated / f"{population}-{kind}-0.jsonl").write_text(
            "".join(json.dumps(dict(id=key, candidates=candidates)) + "\n" for key in ["a", "b"])
        )
    values = analysis.generation_costs(tmp_path)
    base = next(v for v in values if v["population"] == "teacher" and v["generator"] == "base")
    assert base["generated_response_tokens"] == 8
    assert base["responses"] == 4
    assert base["candidate_mean_reward"] == 0.5
    assert base["length_stop_fraction"] == 0.5
    assert base["shared_by"] == "CCS-rho0,CCS-rho1"
    dev = next(v for v in values if v["population"] == "dev")
    assert dev["responses"] == 2
    assert dev["graded_responses"] == 0
    assert dev["candidate_mean_reward"] is None
    assert dev["length_stop_fraction"] == 1
    (generated / "teacher-expert-0.jsonl").unlink()
    with pytest.raises(ValueError, match="Incomplete"):
        analysis.generation_costs(tmp_path)
    expert = next(v for v in analysis.generation_costs(tmp_path, partial=True) if v["generator"] == "expert")
    assert not expert["complete"]
    assert expert["candidate_mean_reward"] is None
