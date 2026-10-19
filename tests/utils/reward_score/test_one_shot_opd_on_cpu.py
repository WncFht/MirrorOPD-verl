# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Numerical and execution checks for the migrated four-domain verifiers."""

import copy
import importlib.util
import json
import multiprocessing
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from verl.utils.reward_score.one_shot_opd import CODE_SOURCES, FC_SOURCES, IF_SOURCES, MATH_SOURCES, reward_func


@pytest.mark.parametrize("source", ["DAPO", "MATH-500", "AMC23", "AIME25"])
@pytest.mark.parametrize("answer,expected", [(r"\boxed{42}", 1.0), (r"\boxed{43}", 0.0)])
def test_boxed_math(source, answer, expected):
    result = reward_func(source, answer, "42")
    assert result["score"] == expected
    assert result["format_score"] == 1.0


def test_math_equivalence_and_last_box():
    assert reward_func("DAPO", r"\boxed{\frac{1}{2}}", "0.5")["acc"]
    assert reward_func("DAPO", r"\boxed{0}\quad\boxed{42}", "42")["acc"]


def test_unboxed_math_fallback():
    result = reward_func("DAPO", "The answer is 42.", "42")
    assert result["score"] == 1.0
    assert result["format_score"] == 0.0


def test_math_from_reward_worker_thread():
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda s: reward_func("DAPO", s, "42"), [r"\boxed{42}", r"\boxed{43}"]))
    assert [r["score"] for r in results] == [1.0, 0.0]


@pytest.mark.parametrize("source", ["taco", "livecodebench"])
@pytest.mark.parametrize("operator,expected", [("+", 1.0), ("-", 0.0)])
def test_code_stdin(source, operator, expected):
    code = f"```python\na,b=map(int,input().split())\nprint(a {operator} b)\n```"
    tests = [{"input": "2 3\n", "output": "5\n", "testtype": "stdin", "metadata": {}}]
    assert reward_func(source, code, json.dumps(tests))["score"] == expected


def test_code_functional_dict_schema():
    code = "```python\nclass Solution:\n    def add(self,a,b): return a+b\n```"
    tests = {"inputs": ["2\n3"], "outputs": ["5"], "fn_name": "add"}
    assert reward_func("taco", code, tests)["acc"]


def test_code_from_reward_thread_cleans_up_children():
    before = {child.pid for child in multiprocessing.active_children()}
    code = "```python\nprint(int(input())+1)\n```"
    truth = [{"input": "4\n", "output": "5\n", "testtype": "stdin", "metadata": {}}]
    with ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(reward_func, "taco", code, truth).result(timeout=30)
    assert result["acc"]
    assert {child.pid for child in multiprocessing.active_children()} <= before


@pytest.mark.parametrize("operator,expected", [("+", 1.0), ("-", 0.0)])
def test_humaneval_supplied_tests(operator, expected):
    code = f"```python\ndef add(a,b): return a {operator} b\n```"
    tests = {"test": "assert add(2,3)==5", "entry_point": "add"}
    assert reward_func("humanevalplus", code, json.dumps(tests))["score"] == expected


def test_code_requires_fence_and_has_timeout():
    from verl.utils.reward_score.one_shot_opd.code_eval_v2._heplus_runner import run_test

    assert reward_func("taco", "print(5)", []) == {"score": 0.0, "acc": False, "format_score": 0.0}
    passed, output = run_test("while True: pass", "assert True", timeout=1)
    assert not passed
    assert "timeout" in output.lower() or "timed out" in output.lower()


@pytest.mark.parametrize("as_json", [False, True])
def test_xlam_order_and_argument_normalization(as_json):
    truth = [{"name": "f", "arguments": {"x": 1}}, {"name": "g", "arguments": {"y": True}}]
    pred = [{"name": "g", "arguments": {"y": "true"}}, {"name": "f", "arguments": {"x": "1"}}]
    assert reward_func("xlam", json.dumps(pred), json.dumps(truth) if as_json else truth)["acc"]
    pred[1]["arguments"]["extra"] = 2
    assert not reward_func("xlam", json.dumps(pred), truth)["acc"]


@pytest.mark.parametrize("source", sorted(FC_SOURCES - {"xlam"}))
def test_bfcl_optional_and_allowed_values(source):
    truth = [{"weather": {"city": ["Paris", "paris"], "unit": ["", "C"]}}]
    assert reward_func(source, '[{"name":"weather","arguments":{"city":"Paris"}}]', truth)["acc"]
    assert not reward_func(source, '[{"name":"weather","arguments":{"city":"London"}}]', truth)["acc"]


def test_fc_empty_and_malformed_output():
    assert reward_func("xlam", "[]", "[]")["acc"]
    assert not reward_func("xlam", "not a function call", "[]")["acc"]
    truth = [{"f": {"x": [1]}}, {"g": {"x": [1]}}]
    assert not reward_func("bfcl_parallel", '[{"name":"f","arguments":{"x":1}}]', truth)["acc"]


def test_unknown_sources_do_not_fall_back_to_math():
    with pytest.raises(ValueError, match="Unknown One-Shot-OPD"):
        reward_func("livecodebenhc", r"\boxed{42}", "42")
    sources = [MATH_SOURCES, CODE_SOURCES, IF_SOURCES, FC_SOURCES]
    assert len(set.union(*(set(s) for s in sources))) == sum(map(len, sources))


def test_dispatch_does_not_import_other_domains(monkeypatch):
    import verl.utils.reward_score.one_shot_opd as router

    original = router.import_module
    loaded = []

    def import_one(name):
        loaded.append(name)
        return original(name)

    monkeypatch.setattr(router, "import_module", import_one)
    assert reward_func("xlam", "[]", [])["acc"]
    assert loaded == ["verl.utils.reward_score.one_shot_opd.fc_eval"]


def test_if_missing_dependency_is_not_scored_as_wrong(monkeypatch):
    from verl.utils.reward_score.one_shot_opd import if_eval

    monkeypatch.delenv("OPENCOMPASS_ROOT", raising=False)
    with pytest.raises(RuntimeError, match="OPENCOMPASS_ROOT unset"):
        if_eval._load_opencompass_modules()


@pytest.mark.skipif(not os.environ.get("OPENCOMPASS_ROOT"), reason="requires OpenCompass MultiIF source")
def test_real_multiif_checkers_and_threaded_loading():
    from verl.utils.reward_score.one_shot_opd.if_eval import score_single_turn

    reference = {
        "key": 1,
        "prompt": "Return JSON.",
        "instruction_id_list": ["detectable_format:json_format"],
        "kwargs": [{"unused": None}],
    }
    original = copy.deepcopy(reference)
    responses = ['<think>not json</think>{"x":1}', "not json"]
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda text: reward_func("MultiIF", text, reference), responses))
    assert [r["score"] for r in results] == [1.0, 0.0]
    assert reference == original
    offline = score_single_turn("MultiIF", "Return JSON.", reference["instruction_id_list"], [{}], responses[0])
    assert offline["if_inst_total"] == 1
    assert offline["prompt_strict_acc"] == 1.0


def test_hydra_path_entry_point():
    root = Path(__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location("custom_reward_function", root / "examples/mirroropd/reward.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.reward_func("xlam", "[]", [])["score"] == 1.0
