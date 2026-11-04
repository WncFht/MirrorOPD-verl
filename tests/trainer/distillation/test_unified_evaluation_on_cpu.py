# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Scientific schedule, sample coverage, aggregation and lifecycle contracts."""

from types import SimpleNamespace

import pytest

from verl.trainer.evaluation.accuracy import final_answer, gpqa_letter, summarize_accuracy, verify_sample
from verl.trainer.evaluation.artifacts import GENERATION, ArtifactStore
from verl.trainer.evaluation.config import EvaluationConfig, GenerationProtocol, ScoreBudget
from verl.trainer.evaluation.plan import compile_plan, payload_windows
from verl.trainer.evaluation.reporting import evaluation_metrics
from verl.trainer.evaluation.trajectory import load_trajectories, select_panel, trajectory_index


def config(**changes):
    return EvaluationConfig.parse(
        {
            "output": "out",
            "source_file": "source",
            "source_enabled": True,
            "history_enabled": True,
            "bootstrap": "bootstrap",
            "datasets": [{"name": "gpqa", "domain": "science", "path": "gpqa.parquet", "verifier": "gpqa", "n": 4}],
            "protocols": {"source": {"max_new_tokens": 7168}, "public": {}},
            **changes,
        }
    )


def test_schedule_union_and_no_initial_regeneration():
    options = config()
    initial = compile_plan(options, outer=0, total=50, optimizer_steps=0)
    assert not initial.generate_current
    assert set(initial.tasks) == {
        "source_heldout",
        "public_initial",
        "public_teacher",
        "accuracy",
        "public_current",
        "teacher_accuracy",
    }
    for outer in range(1, 51):
        plan = compile_plan(options, outer=outer, total=50, optimizer_steps=outer * 32)
        assert ("public_teacher" in plan.tasks) == (outer in {1, 2} or outer % 5 == 0)
        assert plan.generate_current == (outer in {25, 50})
        assert {"source_heldout", "history"} <= set(plan.tasks)
        assert len(plan.tasks) == len(set(plan.tasks))


@pytest.mark.parametrize(
    "value",
    [
        {"temperature": float("nan")},
        {"top_p": 0},
        {"top_k": 0},
        {"repetition_penalty": float("nan")},
        {"repetition_detection": {"min_pattern_size": 16, "max_pattern_size": 128, "min_count": 1}},
        {"repetition_detection": True},
        {"max_new_tokens": True},
        {"unknown": 5},
    ],
)
def test_protocol_rejects_silent_configuration_errors(value):
    with pytest.raises(ValueError):
        GenerationProtocol.parse(value)


def test_delayed_periodic_schedules_still_establish_initial_baselines():
    options = config(schedules={k: {"at": [10], "final": True} for k in ("source", "history", "banks", "bench")})
    initial = compile_plan(options, outer=0, total=50, optimizer_steps=0)
    assert {"source_heldout", "public_initial", "public_teacher", "public_current"} <= set(initial.tasks)
    assert not initial.generate_current
    assert compile_plan(options, outer=1, total=50, optimizer_steps=32).tasks == ()


def test_per_request_seed_and_sampling_are_isolated():
    protocol = GenerationProtocol.parse({"template_kwargs": {"enable_thinking": False}})
    sample = protocol.sampling("gpqa", "q1", 0)
    assert sample["seed"] == protocol.sampling("gpqa", "q1", 0)["seed"]
    assert sample["seed"] != protocol.sampling("gpqa", "q1", 1)["seed"]
    sample["max_tokens"] = 2
    assert protocol.sampling("gpqa", "q1", 0)["max_tokens"] == 32768
    assert protocol.protocol_id != GenerationProtocol().protocol_id


def test_repetition_stop_is_default_and_part_of_generation_identity():
    enabled = GenerationProtocol()
    disabled = GenerationProtocol.parse({"repetition_detection": None})
    tuned = GenerationProtocol.parse(
        {"repetition_detection": {"min_pattern_size": 32, "max_pattern_size": 256, "min_count": 8}}
    )
    params = enabled.sampling("math", "q", 0)
    assert params["repetition_detection"] == {"min_pattern_size": 16, "max_pattern_size": 128, "min_count": 8}
    assert disabled.sampling("math", "q", 0)["repetition_detection"] is None
    assert len({enabled.protocol_id, disabled.protocol_id, tuned.protocol_id}) == 3
    params["repetition_detection"]["min_count"] = 99
    assert enabled.sampling("math", "q", 0)["repetition_detection"]["min_count"] == 8


def test_accuracy_reports_cap_and_repetition_separately():
    reasons = ["stop", "length", "repetition", None]
    rows = [dict(accuracy_row("q", i, 1.0), finish_reason=reason) for i, reason in enumerate(reasons)]
    expected = {(r["dataset"], r["source_id"], r["sample_index"]) for r in rows}
    metrics = summarize_accuracy(rows, expected)
    prefix = "eval/accuracy/gpqa/finish_reason/"
    assert metrics[prefix + "response_count"] == 4
    assert metrics[prefix + "length_fraction"] == metrics["eval/accuracy/gpqa/length_finish_fraction"] == 0.25
    assert metrics[prefix + "repetition_fraction"] == 0.25
    assert metrics[prefix + "truncated_fraction"] == 0.5
    assert metrics[prefix + "unknown_fraction"] == 0.25


def test_payload_windows_account_for_live_training_and_complete_rows():
    budget = ScoreBudget(payload_bytes=1000, host_budget_bytes=3000, copy_reserve_factor=2)
    assert budget.available_payload(700) == 800
    with pytest.raises(ValueError):
        budget.available_payload(1500)
    rows = [{"id": str(i), "valid_tokens": n} for i, n in enumerate([15, 20, 10, 5])]
    windows = payload_windows(rows, hidden_size=4, payload_bytes=400)
    assert sorted(r["id"] for w in windows for r in w) == ["0", "1", "2", "3"]
    assert all(sum(r["valid_tokens"] * 16 for r in w) <= 400 for w in windows)
    with pytest.raises(ValueError):
        payload_windows(rows + [rows[0]], hidden_size=4, payload_bytes=400)
    with pytest.raises(ValueError):
        payload_windows(rows, hidden_size=4, payload_bytes=100)


def test_answer_channel_does_not_grade_unfinished_reasoning():
    assert final_answer("Reasoning: answer is A", thinking_open=True, channel="after_thinking") == ""
    assert final_answer("A</think>\nB", thinking_open=True, channel="after_thinking") == "B"
    assert gpqa_letter("A could be correct, but this reasoning is unfinished") is None
    assert gpqa_letter("reasoning\nThe answer is (C).") == "C"
    assert gpqa_letter(r"therefore \boxed{D}") == "D"


def accuracy_row(q, i, score):
    return dict(
        id=f"{q}/{i}",
        source_id=q,
        dataset="gpqa",
        domain="science",
        sample_index=i,
        score=score,
        status="ok",
        finish_reason="stop",
        response_tokens=10,
        seed=42,
    )


def test_accuracy_is_avg_at_n_with_complete_coverage_not_best_of_n():
    rows = [accuracy_row(q, i, float(i == 0)) for q in ["q1", "q2"] for i in range(4)]
    expected = {(r["dataset"], r["source_id"], r["sample_index"]) for r in rows}
    metrics = summarize_accuracy(rows, expected)
    assert metrics["eval/accuracy/gpqa/avg_at_n"] == 0.25
    assert metrics["eval/accuracy/gpqa/questions"] == 2
    with pytest.raises(ValueError, match="coverage"):
        summarize_accuracy(rows[:-1], expected)
    with pytest.raises(ValueError, match="coverage"):
        summarize_accuracy(rows + [rows[0]], expected)
    rows[0]["status"] = "infrastructure_error"
    with pytest.raises(ValueError, match="verifier failure"):
        summarize_accuracy(rows, expected)


def test_official_ifeval_strict_and_gpqa_verifiers():
    row = accuracy_row("q", 0, 0)
    question = {
        "ground_truth": {
            "key": 1,
            "instruction_id_list": ["change_case:english_capital"],
            "prompt": "Write in capital letters.",
            "kwargs": [{}],
        }
    }
    assert verify_sample(row, question, "ifeval", "THIS IS A SHORT RESPONSE WRITTEN IN ENGLISH.")["score"] == 1
    assert verify_sample(row, question, "ifeval", "Hello world")["score"] == 0
    assert verify_sample(row, {"ground_truth": "C"}, "gpqa", "The answer is C.")["score"] == 1
    assert verify_sample(row, {"ground_truth": "C"}, "gpqa", "")["score"] == 0


def test_public_domain_equal_mean_and_initial_alias():
    rows = [
        dict(id=f"q{i}", source_id=f"q{i}", bank="public_initial", domain=domain, rkl_sum=risk * 10, tokens=10)
        for i, (domain, risk) in enumerate([("math", 1), ("math", 3), ("code", 8)])
    ]
    metrics = evaluation_metrics(rows, baselines={"public_initial": rows}, aliases={"public_current": "public_initial"})
    assert metrics["eval/rkl/public_initial/all/question_mean"] == 5
    assert metrics["eval/rkl/public_current/all/question_mean"] == 5
    assert metrics["eval/rkl/public_initial/all/reduction_percent"] == 0
    assert "eval/rkl/public_current/all/reduction_percent" not in metrics


def test_streaming_indices_preserve_exact_sample_zero_and_empty_generation(tmp_path):
    store = ArtifactStore.create(tmp_path / "bank", kind="generation", metadata={})
    rows = [
        dict(
            id=f"q/{i}",
            source_id="q",
            outer=0,
            prompt_ids=[1, 2],
            response_ids=[3] if i == 0 else [],
            response_mask=[1] if i == 0 else [],
            dataset="gpqa",
            domain="science",
            sample_index=i,
            valid_tokens=int(i == 0),
            producer="initial",
        )
        for i in range(2)
    ]
    store.write_rows("generations/part.parquet", rows, schema=GENERATION)
    store.finish()
    refs = trajectory_index(store, bank="public_initial")
    chosen = select_panel(refs, [SimpleNamespace(name="gpqa", panel_ids=("q",))])
    assert len(chosen) == 1 and chosen[0]["sample_index"] == 0
    restored = load_trajectories(chosen)
    assert restored[0]["id"] == "public_initial:q/0"
    assert restored[0]["response_ids"] == [3]
    assert restored[0]["bank"] == "public_initial"


def test_rollout_restore_tracks_residency_not_only_version():
    from omegaconf import OmegaConf

    from verl.experimental.rollout_policy.service import DefaultRolloutService

    calls = []
    trainer = SimpleNamespace(
        readout_callback=None,
        config=OmegaConf.create({"trainer": {}}),
        common_rollout_sampling=False,
        global_steps=0,
        timing_raw={},
        checkpoint_manager=SimpleNamespace(
            update_weights=lambda version: calls.append(("load", version)),
            sleep_replicas=lambda: calls.append(("sleep",)),
        ),
    )
    service = DefaultRolloutService(trainer, None)
    service.initialize()
    service.ensure_current(0)
    service.park()
    service.ensure_current(0)
    trainer.global_steps = 1
    service.after_update()
    assert calls == [("load", 0), ("sleep",), ("load", 0), ("load", 1)]


def test_rolling_admission_refills_while_an_early_request_is_slow():
    import asyncio

    from verl.trainer.evaluation.generation import bounded_results

    async def run():
        release = asyncio.Event()
        active, peak = 0, 0

        async def worker(i):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            try:
                if i == 0:
                    await release.wait()
                if i == 5:
                    release.set()
                return i
            finally:
                active -= 1

        async def abort():
            release.set()

        result = [
            r async for r in bounded_results([(i,) for i in range(6)], worker, concurrency=2, timeout=2, abort=abort)
        ]
        assert sorted(result) == list(range(6)) and result[0] != 0
        assert peak <= 2 and active == 0

    asyncio.run(run())


def test_generation_failure_waits_for_abort_and_settlement():
    import asyncio

    from verl.trainer.evaluation.generation import bounded_results

    async def run():
        release, settled, aborted = asyncio.Event(), False, False

        async def worker(i):
            nonlocal settled
            if i:
                raise ValueError("request failed")
            await release.wait()
            settled = True
            return i

        async def abort():
            nonlocal aborted
            aborted = True
            release.set()

        with pytest.raises(ValueError, match="request failed"):
            async for _ in bounded_results([(0,), (1,)], worker, concurrency=2, timeout=2, abort=abort):
                pass
        assert aborted and settled

    asyncio.run(run())


def test_code_test_reference_restores_function_metadata(tmp_path):
    import json

    import pyarrow as pa
    import pyarrow.parquet as pq

    from verl.trainer.evaluation.accuracy import referenced_tests

    tests = [{"input": "2\n3", "output": "5", "testtype": "functional"}]
    path = tmp_path / "raw.parquet"
    pq.write_table(pa.Table.from_pylist([{"tests": json.dumps(tests), "metadata": {"func_name": "add"}}]), path)
    restored = referenced_tests({"path": str(path), "row": 0})
    assert restored[0]["metadata"]["func_name"] == "add"
    assert restored[0]["input"] == tests[0]["input"]
    row = accuracy_row("q", 0, 0)
    question = {"ground_truth": {"tests_ref": {"path": str(path), "row": 0}}}
    code = "```python\nclass Solution:\n    def add(self, a, b):\n        return a + b\n```"
    assert verify_sample(row, question, "lcb", code)["score"] == 1
    assert verify_sample(row, question, "lcb", code.replace("a + b", "a - b"))["score"] == 0


def test_gap_recovery_preserves_ratio_and_zero_gap_availability():
    from verl.trainer.evaluation.accuracy import gap_recovery

    key = "eval/accuracy/math/avg_at_n"
    result = gap_recovery({key: 0.6}, {key: 0.5}, {key: 0.7})
    assert result["eval/accuracy/math/gap_recovery_percent"] == pytest.approx(50)
    result = gap_recovery({key: 0.6}, {key: 0.5}, {key: 0.5})
    assert result["eval/accuracy/math/gap_recovery_defined"] == 0
    assert "eval/accuracy/math/gap_recovery_percent" not in result


def test_code_reference_selects_nonzero_rows_across_row_groups(tmp_path):
    import json

    import pyarrow as pa
    import pyarrow.parquet as pq

    from verl.trainer.evaluation.accuracy import referenced_tests

    rows = [
        {
            "tests": json.dumps([{"input": str(i), "output": str(i + 1), "testtype": "functional"}]),
            "metadata": {"func_name": f"f{i}"},
        }
        for i in range(5)
    ]
    path = tmp_path / "grouped.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=2)
    for i in [1, 4, 2, 1]:
        tests = referenced_tests({"path": str(path), "row": i})
        assert tests[0]["input"] == str(i)
        assert tests[0]["metadata"]["func_name"] == f"f{i}"


def test_source_selection_protects_question_families_and_is_outcome_independent():
    from verl.trainer.evaluation.selection import fingerprints, source_split

    rows = [dict(id=str(i), family=f"f{i}", stratum="common" if i < 80 else "rare") for i in range(100)]
    train, heldout, report = source_split(rows, [{"family": "f0"}], train_count=32, heldout_count=8)
    assert len(train) == 32 and len(heldout) == 8
    assert not {r["family"] for r in train} & {r["family"] for r in heldout}
    assert "f0" not in {r["family"] for r in train + heldout}
    other = source_split(list(reversed(rows)), [{"family": "f0"}], train_count=32, heldout_count=8)
    assert (train, heldout, report) == other
    assert fingerprints("Compute 123 + 5")[1] == fingerprints("Compute 456 + 7")[1]


def test_frozen_generation_recovery_preserves_even_empty_answers(tmp_path):
    from verl.trainer.evaluation.artifacts import identity
    from verl.trainer.evaluation.config import DatasetSpec
    from verl.trainer.evaluation.generation_contract import sample_identity
    from verl.trainer.evaluation.generation_resume import replay_published

    protocol = GenerationProtocol()
    spec = DatasetSpec("gpqa", "science", "questions", "gpqa", n=2)
    questions = [{"id": "q", "prompt": [{"role": "user", "content": "question"}]}]
    meta = dict(
        producer="initial",
        model_identity="weights",
        outer=0,
        protocols={"public": protocol.metadata()},
        expected_samples=2,
        expected_version=None,
        question_set_sha256=identity([(spec.name, 2, "public", questions)]),
        engine_contract={"schema": 1, "support": "tokenizer_ids"},
        engine_contract_id="engine-v1",
    )
    old = ArtifactStore.create(tmp_path / "old", kind="generation", metadata=meta)
    row = dict(
        id=sample_identity("weights", 0, "gpqa", "q", 0, protocol.protocol_id, "engine-v1"),
        source_id="q",
        outer=0,
        prompt_ids=[1],
        response_ids=[],
        response_mask=[],
        dataset="gpqa",
        sample_index=0,
        protocol_id=protocol.protocol_id,
        seed=protocol.sample_seed("gpqa", "q", 0),
    )
    old.write_rows("generations/part-00000.parquet", [row], schema=GENERATION)
    new = ArtifactStore.create(tmp_path / "new", kind="generation", metadata=meta)
    replayed = []
    seen, shard, provenance = replay_published(
        new, old.root, [(spec, questions)], {"public": protocol}, on_batch=replayed.extend
    )
    assert seen == {("gpqa", "q", 0)} and shard == 1 and provenance["reused_samples"] == 1
    assert replayed[0][0]["response_ids"] == []
    assert new.read_rows("generations/part-00000.parquet")[0]["id"] == row["id"]
    changed = ArtifactStore.create(tmp_path / "wrong", kind="generation", metadata={**meta, "model_identity": "other"})
    with pytest.raises(ValueError, match="changed"):
        replay_published(changed, old.root, [(spec, questions)], {"public": protocol})
    for name, legacy_contract in (
        ("unrecorded", {}),
        ("raw", {"engine_contract": {"support": "head_rows"}, "engine_contract_id": "raw-head"}),
    ):
        legacy = {k: v for k, v in meta.items() if not k.startswith("engine_contract")}
        previous = ArtifactStore.create(tmp_path / name, kind="generation", metadata={**legacy, **legacy_contract})
        fresh = ArtifactStore.create(tmp_path / (name + "-destination"), kind="generation", metadata=meta)
        with pytest.raises(ValueError, match="changed"):
            replay_published(fresh, previous.root, [(spec, questions)], {"public": protocol})
        assert fresh.manifest()["files"] == {}


def test_completed_generation_is_not_timed_out_by_slow_cpu_consumption():
    import asyncio

    from verl.trainer.evaluation.generation import bounded_results

    async def run():
        async def worker(i):
            return i

        async def abort():
            raise AssertionError("Completed requests must not be aborted")

        result = []
        async for item in bounded_results([(i,) for i in range(4)], worker, concurrency=2, timeout=0.1, abort=abort):
            result.append(item)
            if len(result) == 1:
                await asyncio.sleep(0.2)
        assert sorted(result) == list(range(4))

    asyncio.run(run())
