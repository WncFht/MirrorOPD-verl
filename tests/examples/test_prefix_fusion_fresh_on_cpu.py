# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Fresh-state identities and paired inference must not confuse responses with questions."""

from copy import deepcopy
from pathlib import Path

import pytest

from examples.mirroropd.research.core.artifacts import file_hash, fingerprint, receipt
from examples.mirroropd.research.core.io import dump
from examples.mirroropd.research.studies.prefix_fusion.analysis import (
    Evidence,
    accuracy_values,
    build,
    macro5_pair,
    paired,
    prompt_values,
)
from examples.mirroropd.research.studies.prefix_fusion.fresh import (
    ENDPOINTS,
    load_fresh_bank,
    request_rows,
    study_manifest,
    validate_bank_rows,
)
from examples.mirroropd.research.studies.prefix_fusion.protocol import ARMS, PROTOCOL, protocol_identity


def prepared(root):
    questions = [
        dict(
            id=320 + i,
            group=f"M0{i // 16 + 1}",
            domain="math",
            split="heldout",
            prompt_ids=[1, 2],
            ground_truth="2",
            primary=True,
            long=i < 32,
        )
        for i in range(64)
    ]
    directory = root / "prepared-03"
    dump(directory / "heldout-questions.json", questions)
    dump(directory / "task-prompts.json", [])
    dump(
        directory / "complete.json",
        dict(
            inputs=dict(protocol=protocol_identity()),
            artifacts={name: file_hash(directory / name) for name in ["heldout-questions.json", "task-prompts.json"]},
        ),
    )
    dump(root / "assets/legal-token-ids.json", [1, 2, 11, 151643])
    dump(root / "assets/complete.json", dict(models={}))
    return questions


def manifest(path, *, expanded=False):
    pairs = {
        p: {
            a: dict(run=str(path.parent / "outputs" / f"{p}-{a}"), audit=str(path.parent / "audits" / f"{p}-{a}.json"))
            for a in ARMS
        }
        for p in ["J", "P"]
    }
    if expanded:
        value = dict(
            protocol=protocol_identity(),
            conditions=[
                dict(pair=p, arm=a, key=f"{p}-{a}", attempt=1, run=pairs[p][a]["run"], target_updates=640)
                for p in pairs
                for a in ARMS
            ],
            old_failures={"J-S0": "/diagnostic-only/v3"},
        )
    else:
        value = dict(schema=1, protocol=protocol_identity(), pairs=pairs)
    dump(path, value)
    return value


def bank_rows(questions, source="initial", *, smoke=False):
    requested = request_rows(questions, source, smoke=smoke)
    rows = [
        dict(r, response_ids=[11, 151643], behavior_log_probs=[-0.5, -0.2], finish_reason="stop", output="2")
        for r in requested
    ]
    return requested, rows


def test_initial_and_trained_s0_are_distinct_with_shared_per_question_seeds(tmp_path):
    questions = prepared(tmp_path)
    initial = request_rows(questions, "initial")
    trained = request_rows(questions, "S0")
    assert len(ENDPOINTS) == 7 and ENDPOINTS[:2] == ("initial", "S0")
    assert len(initial) == 128 and len({r["id"] for r in initial}) == 128
    for first, second in zip(initial, trained, strict=True):
        assert first["source"] != second["source"]
        assert first["seed"] == second["seed"] == 62000000 + 100 * first["question_id"] + first["sample_index"]
        assert first["cap"] == 7168


def test_duplicate_generation_questions_rejected():
    question = dict(id=4)
    with pytest.raises(ValueError, match="Duplicate"):
        request_rows([question, question], "PM")


@pytest.mark.parametrize(
    "mutation", ["seed", "duplicate", "missing", "token", "probability", "stop", "premature-length"]
)
def test_invalid_generation_never_becomes_a_frozen_bank(tmp_path, mutation):
    requested, rows = bank_rows(prepared(tmp_path))
    if mutation == "seed":
        rows[0]["seed"] += 1
    if mutation == "duplicate":
        rows[-1] = deepcopy(rows[0])
    if mutation == "missing":
        rows.pop()
    if mutation == "token":
        rows[0]["response_ids"][0] = 88
    if mutation == "probability":
        rows[0]["behavior_log_probs"][0] = float("nan")
    if mutation == "stop":
        rows[0]["response_ids"][-1] = 11
    if mutation == "premature-length":
        rows[0]["finish_reason"] = "length"
    with pytest.raises(ValueError):
        validate_bank_rows(rows, requested, [1, 2, 11, 151643])


def test_fresh_bank_checksum_and_protocol_are_verified(tmp_path):
    questions = prepared(tmp_path)
    _, rows = bank_rows(questions)
    out = tmp_path / "bank"
    dump(out / "bank.json", rows)
    inputs = dict(
        protocol=protocol_identity(),
        smoke=False,
        model=dict(endpoint="initial", pair="shared"),
        questions=fingerprint(questions),
        sampling=PROTOCOL["fresh"],
        support=file_hash(tmp_path / "assets/legal-token-ids.json"),
    )
    receipt(out, inputs, rows=128, questions=64, samples=2, bank_sha256=file_hash(out / "bank.json"))
    assert len(load_fresh_bank(tmp_path, out)[0]) == 128
    changed = deepcopy(inputs)
    changed["sampling"]["temperature"] = 0.7
    receipt(out, changed, rows=128, questions=64, samples=2, bank_sha256=file_hash(out / "bank.json"))
    with pytest.raises(ValueError, match="sampling protocol"):
        load_fresh_bank(tmp_path, out)
    receipt(out, inputs, rows=128, questions=64, samples=2, bank_sha256="changed")
    with pytest.raises(ValueError, match="data changed"):
        load_fresh_bank(tmp_path, out)


def test_smoke_bank_not_accepted_as_formal(tmp_path):
    questions = prepared(tmp_path)[:2]
    _, rows = bank_rows(questions, smoke=True)
    out = tmp_path / "bank"
    dump(out / "bank.json", rows)
    inputs = dict(
        protocol=protocol_identity(),
        smoke=True,
        model=dict(endpoint="initial"),
        questions=fingerprint(questions),
        sampling=PROTOCOL["fresh"],
        support=file_hash(tmp_path / "assets/legal-token-ids.json"),
    )
    receipt(out, inputs, rows=4, questions=2, samples=2, bank_sha256=file_hash(out / "bank.json"))
    with pytest.raises(ValueError, match="Short fresh"):
        load_fresh_bank(tmp_path, out)
    assert len(load_fresh_bank(tmp_path, out, allow_smoke=True)[0]) == 4


def test_expanded_manifest_excludes_diagnostic_old_attempts(tmp_path):
    path = tmp_path / "manifest.json"
    manifest(path, expanded=True)
    normalized = study_manifest(path)
    assert normalized["pairs"]["J"]["S0"]["run"] == str(tmp_path / "outputs/J-S0")
    assert normalized["pairs"]["P"]["PM"]["audit"] == str(tmp_path / "attempts/P-PM/a01/audit.json")
    assert "old_failures" not in normalized


def test_manifest_rejects_reused_run_directory(tmp_path):
    path = tmp_path / "manifest.json"
    value = manifest(path)
    value["pairs"]["J"]["S"] = value["pairs"]["J"]["S0"]
    dump(path, value)
    with pytest.raises(ValueError, match="share a run"):
        study_manifest(path)


def test_risk_averages_two_responses_before_question_bootstrap():
    rows = [
        dict(id=2 * q + i, question_id=q, sample_index=i, stats={"all": {"rkl": value, "tokens": 1}})
        for q, i, value in [(1, 0, 10), (1, 1, 20), (2, 0, 0), (2, 1, 10)]
    ]
    values = prompt_values(rows, "rkl", 10, samples=2)
    assert values == {1: 1.5, 2: 0.5}  # Fixed H=10, not the single observed token count.
    result = paired(values, {1: 2.5, 2: 1.5})
    assert result["delta"] == result["ci_low"] == result["ci_high"] == 1
    with pytest.raises(ValueError, match="Missing"):
        prompt_values(rows[:-1], "rkl", 10, samples=2)
    with pytest.raises(ValueError, match="Duplicate"):
        prompt_values(rows + [rows[0]], "rkl", 10, samples=2)


def test_task_samples_are_one_question_unit():
    rows = [dict(data_source="MATH", prompt_id="q", sample_index=i, acc=i < 2, status="ok") for i in range(4)]
    assert accuracy_values(rows) == {"MATH": {"q": 0.5}}
    with pytest.raises(ValueError, match="Missing"):
        accuracy_values(rows[:-1])


def test_macro5_weights_datasets_equally_and_resamples_within_dataset():
    before = {"large": {str(i): 0.0 for i in range(100)}, **{f"small{i}": {"q": 0.0} for i in range(4)}}
    after = deepcopy(before)
    after["large"] = {str(i): 1.0 for i in range(100)}
    result = macro5_pair(before, after)
    assert result["questions"] == 104
    assert result["delta"] == pytest.approx(0.2)
    assert result["ci_low"] == result["ci_high"] == pytest.approx(0.2)
    assert result == macro5_pair(before, after)
    del after["small0"]["q"]
    with pytest.raises(ValueError, match="paired questions"):
        macro5_pair(before, after)


def test_partial_report_never_fabricates_missing_conditions(tmp_path):
    root = tmp_path / "study"
    prepared(root)
    path = tmp_path / "manifest.json"
    manifest(path)
    result = build(root, path, root / "fresh", tmp_path / "partial")
    assert not result["complete"] and result["counts"] == dict(risk_points=0, accuracy_rows=0, fresh_cells=0)
    assert len([x for x in result["missing"] if x.startswith("fresh/")]) == 98
    assert not list((tmp_path / "partial").glob("*.csv"))
    with pytest.raises(ValueError, match="incomplete"):
        build(root, path, root / "fresh", tmp_path / "rejected", require_complete=True)
    assert not (tmp_path / "rejected").exists()


def test_receipt_reader_rejects_tampered_identity(tmp_path):
    inputs = dict(protocol=protocol_identity(), value=1)
    receipt(tmp_path, inputs)
    assert Evidence().receipt(tmp_path)["inputs"] == inputs
    dump(tmp_path / "complete.json", dict(inputs=inputs, fingerprint="wrong"))
    with pytest.raises(ValueError, match="identity/protocol"):
        Evidence().receipt(tmp_path)


@pytest.mark.parametrize("dtype,accepted", [("float32", True), ("float16", False)])
def test_trained_endpoint_requires_accepted_fp32_weights(tmp_path, dtype, accepted):
    import numpy as np
    from safetensors.numpy import save_file

    from examples.mirroropd.research.studies.prefix_fusion.fresh import model_identity

    root = tmp_path / "study"
    path = tmp_path / "manifest.json"
    bindings = manifest(path)
    config = dict(
        model_type="qwen2",
        hidden_size=2,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        vocab_size=151936,
        eos_token_id=151643,
    )
    original = root / "assets/S0"
    dump(original / "config.json", config)
    dump(root / "assets/complete.json", dict(models={"S0": dict(directory=str(original))}))
    dump(root / "teacher-cache/J/complete.json", dict(fingerprint="target-J"))
    entry = bindings["pairs"]["J"]["PM"]
    dest = Path(entry["run"]) / "science/outer-040"
    weights = dest / "weights/huggingface"
    dump(weights / "config.json", config)
    save_file({"weight": np.array([0.5], dtype=dtype)}, str(weights / "model.safetensors"))
    identity = fingerprint({name: file_hash(weights / name) for name in ["config.json", "model.safetensors"]})
    dump(dest / "fixed/complete.json", dict(inputs=dict(target="target-J", weights=identity)))
    dump(
        dest / "complete.json",
        dict(
            outer=40,
            optimizer_step=640,
            protocol=protocol_identity(),
            smoke=False,
            retained_weights=True,
            fixed_receipt=file_hash(dest / "fixed/complete.json"),
        ),
    )
    dump(entry["audit"], dict(passed=True, formal_complete=True, applied_updates=640, pair="J", arm="PM"))
    if accepted:
        point = model_identity(root, "J", "PM", path)
        assert point["identity"] == identity and point["endpoint"] == "PM"
    else:
        with pytest.raises(ValueError, match="FP32"):
            model_identity(root, "J", "PM", path)
    dump(entry["audit"], dict(passed=False, formal_complete=False, applied_updates=16, pair="J", arm="PM"))
    with pytest.raises(ValueError, match="full run acceptance"):
        model_identity(root, "J", "PM", path)


def test_combined_risk_bootstrap_keeps_domain_weights_fixed():
    from examples.mirroropd.research.studies.prefix_fusion.analysis import fixed_panel_pair

    domains = {i: "math" if i < 64 else "code" if i < 96 else "if" for i in range(128)}
    before = {i: 0.0 for i in domains}
    after = {i: float(domains[i] == "math") for i in domains}
    result = fixed_panel_pair(before, after, domains)
    assert result["delta"] == result["ci_low"] == result["ci_high"] == 0.5
    with pytest.raises(ValueError, match="domain allocation"):
        fixed_panel_pair(
            {i: v for i, v in before.items() if i != 0}, {i: v for i, v in after.items() if i != 0}, domains
        )


def test_expanded_manifest_relative_cli_path_resolves_audit_bindings(tmp_path, monkeypatch):
    path = tmp_path / "manifest.json"
    manifest(path, expanded=True)
    monkeypatch.chdir(tmp_path)
    value = study_manifest(Path("manifest.json"))
    assert value["pairs"]["J"]["PM"]["audit"] == str(tmp_path / "attempts/J-PM/a01/audit.json")
