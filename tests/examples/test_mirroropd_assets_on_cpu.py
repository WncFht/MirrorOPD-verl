# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Check data conversion and local asset validation using tiny CPU fixtures."""

import importlib
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[2]


def load_asset_scripts():
    """Import the same sibling modules used by the direct Python command-line scripts."""
    sys.path.insert(0, str(ROOT / "examples/mirroropd"))
    try:
        return tuple(importlib.import_module(name) for name in ("check_assets", "prepare_math_data", "prepare_models"))
    finally:
        sys.path.pop(0)


assets, prepare_data, prepare_models = load_asset_scripts()


def write_rows(path, rows):
    """Write a fixture using the actual prepared-data schema."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=assets.SCHEMA), path)


@pytest.fixture
def raw_math(tmp_path, monkeypatch):
    """Supply duplicate source IDs and both 15-question AIME parts without downloading."""
    for name, count in (("DAPO", 2), ("MATH-500", 1), ("AMC23", 1)):
        monkeypatch.setitem(assets.DATASETS[name], "rows", count)
    raw = tmp_path / "raw"
    raw.mkdir()
    rows = [
        {
            "prompt": [{"role": "user", "content": prepare_data.DAPO_PREFIX + "first" + prepare_data.DAPO_SUFFIX}],
            "extra_info": {"index": 10},
            "reward_model": {"ground_truth": "37"},
        },
        {
            "prompt": [{"role": "user", "content": "discard repeated source ID"}],
            "extra_info": {"index": 10},
            "reward_model": {"ground_truth": "wrong"},
        },
        {
            "prompt": [{"role": "user", "content": "second"}],
            "extra_info": {"index": 11},
            "reward_model": {"ground_truth": "2"},
        },
    ]
    pq.write_table(pa.Table.from_pylist(rows), raw / "dapo-math-17k.parquet")
    (raw / "math500_test.jsonl").write_text(json.dumps({"problem": "math", "answer": "3"}) + "\n")
    pq.write_table(pa.Table.from_pylist([{"problem": "amc", "answer": 4}]), raw / "amc23.parquet")
    for part in ("I", "II"):
        (raw / f"aime2025-{part}.jsonl").write_text(
            "".join(json.dumps({"question": f"{part}-{index}", "answer": index}) + "\n" for index in range(15))
        )
    return raw


def test_raw_conversion_preserves_first_occurrence_and_aime_order(raw_math):
    result = dict(prepare_data.build_rows(raw_math))
    assert [row["prompt"][0]["content"] for row in result["DAPO"]] == ["first", "second"]
    assert result["DAPO"][0]["reward_model"]["ground_truth"] == "37"
    assert result["AIME25"][14]["prompt"][0]["content"] == "I-14"
    assert result["AIME25"][15]["prompt"][0]["content"] == "II-0"


@pytest.mark.parametrize("answer", [None, float("nan")])
def test_missing_raw_answer_is_not_converted_to_a_plausible_string(answer):
    with pytest.raises(ValueError, match="missing or nonfinite answer"):
        prepare_data.make_row("question", answer, "DAPO", 0)


def test_prepare_and_manifest_detects_changed_answers(raw_math, tmp_path, monkeypatch):
    root = tmp_path / "data"
    monkeypatch.setattr(sys, "argv", ["prepare_math_data.py", "--raw-dir", str(raw_math), "--data-root", str(root)])
    prepare_data.main()
    manifest = json.loads((root / "math_manifest.json").read_text())
    path = root / assets.DATASETS["DAPO"]["path"]
    assert manifest["data"]["DAPO"]["sha256"] == assets.file_sha256(path)
    rows = pq.read_table(path).to_pylist()
    rows[0]["reward_model"]["ground_truth"] = "different but well-formed"
    write_rows(path, rows)
    monkeypatch.setattr(sys, "argv", ["check_assets.py", "--data-only", "--data-root", str(root)])
    with pytest.raises(ValueError, match="SHA256 does not match"):
        assets.main()


def test_bad_rebuild_does_not_replace_existing_outputs(raw_math, tmp_path, monkeypatch):
    root = tmp_path / "data"
    path = root / assets.DATASETS["DAPO"]["path"]
    path.parent.mkdir(parents=True)
    path.write_bytes(b"previous successful output")
    monkeypatch.setitem(assets.DATASETS["DAPO"], "rows", 3)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_math_data.py",
            "--raw-dir",
            str(raw_math),
            "--data-root",
            str(root),
            "--overwrite",
        ],
    )
    with pytest.raises(ValueError, match="expected 3"):
        prepare_data.main()
    assert path.read_bytes() == b"previous successful output"


@pytest.mark.parametrize("invalid", ["source", "answer", "role", "index", "columns"])
def test_invalid_prepared_rows_fail_early(tmp_path, monkeypatch, invalid):
    monkeypatch.setitem(assets.DATASETS["DAPO"], "rows", 2)
    rows = [prepare_data.make_row("question", "1", "DAPO", index) for index in range(2)]
    if invalid == "source":
        rows[0]["data_source"] = "dapo-typo"
    elif invalid == "answer":
        rows[0]["reward_model"]["ground_truth"] = None
    elif invalid == "role":
        rows[0]["prompt"][0]["role"] = "assistant"
    elif invalid == "index":
        rows[1]["extra_info"]["index"] = rows[0]["extra_info"]["index"]
    path = tmp_path / "invalid.parquet"
    write_rows(path, rows)
    if invalid == "columns":
        pq.write_table(pq.read_table(path).drop(["ability"]), path)
    with pytest.raises(ValueError):
        assets.check_data(path, "DAPO")


def test_duplicate_questions_are_reported_without_changing_the_recipe(tmp_path, monkeypatch):
    monkeypatch.setitem(assets.DATASETS["DAPO"], "rows", 2)
    path = tmp_path / "duplicate.parquet"
    write_rows(path, [prepare_data.make_row("same", "1", "DAPO", index) for index in range(2)])
    report, counts = assets.check_data(path, "DAPO")
    assert report["rows"] == 2 and report["duplicate_prompt_rows"] == 1
    assert counts["same"] == 2


def make_tiny_model(folder, swapped=False):
    """Save a real one-layer Qwen2 checkpoint and a four-token fast tokenizer."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    folder.mkdir(parents=True)
    Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=4,
            hidden_size=4,
            intermediate_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=131072,
        )
    ).save_pretrained(folder)
    vocabulary = {"[UNK]": 0, "<eos>": 1, "1": 3 if swapped else 2, "+": 2 if swapped else 3}
    backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", eos_token="<eos>")
    tokenizer.chat_template = "{{ messages[0]['content'] }}"
    tokenizer.save_pretrained(folder)


def test_local_model_prepare_materializes_symlinks_and_checks_hashes(tmp_path, monkeypatch):
    source = tmp_path / "source"
    destination = tmp_path / "model"
    for index, name in enumerate(assets.MODELS):
        folder = source / name
        make_tiny_model(folder)
        weights = folder / "model.safetensors"
        shared = tmp_path / f"weights-{index}.safetensors"
        weights.rename(shared)
        weights.symlink_to(shared)
        monkeypatch.setitem(assets.MODELS[name], "weight_sha256", assets.file_sha256(shared))
    monkeypatch.setattr(
        sys, "argv", ["prepare_models.py", "--source-root", str(source), "--model-root", str(destination)]
    )
    prepare_models.main()
    for name in assets.MODELS:
        assert (destination / name / "model.safetensors").is_file()
        assert not (destination / name / "model.safetensors").is_symlink()
    assert (destination / "model_manifest.json").is_file()


def test_missing_layer_tensor_is_rejected(tmp_path):
    from safetensors.numpy import load_file, save_file

    name = next(iter(assets.MODELS))
    folder = tmp_path / name
    make_tiny_model(folder)
    weights = folder / "model.safetensors"
    tensors = load_file(weights)
    tensors.pop("model.layers.0.mlp.gate_proj.weight")
    save_file(tensors, weights)
    with pytest.raises(ValueError, match="tensor names"):
        assets.check_model(folder, name)


def test_incompatible_token_id_mappings_are_rejected(tmp_path, monkeypatch):
    for index, name in enumerate(assets.MODELS):
        make_tiny_model(tmp_path / name, swapped=bool(index))
    monkeypatch.setattr(sys, "argv", ["check_assets.py", "--model-root", str(tmp_path)])
    with pytest.raises(ValueError, match="token-to-ID mappings differ"):
        assets.main()
