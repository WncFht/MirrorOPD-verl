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


def test_prepared_data_is_checked_from_actual_rows(raw_math, tmp_path, monkeypatch):
    root = tmp_path / "data"
    monkeypatch.setattr(sys, "argv", ["prepare_math_data.py", "--raw-dir", str(raw_math), "--data-root", str(root)])
    prepare_data.main()
    manifest = json.loads((root / "math_manifest.json").read_text())
    path = root / assets.DATASETS["DAPO"]["path"]
    assert manifest["data"]["DAPO"]["rows"] == 2
    monkeypatch.setattr(sys, "argv", ["check_assets.py", "--data-only", "--data-root", str(root)])
    assets.main()
    report = json.loads((root / "asset_check.json").read_text())
    assert report["models"] == {} and report["data"]["DAPO"]["rows"] == 2
    rows = pq.read_table(path).to_pylist()
    rows[0]["reward_model"]["ground_truth"] = " "
    write_rows(path, rows)
    with pytest.raises(ValueError, match="ground_truth must be a nonempty string"):
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


def make_tiny_model(folder):
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
    vocabulary = {"[UNK]": 0, "<eos>": 1, "1": 2, "+": 3}
    backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", eos_token="<eos>")
    tokenizer.chat_template = "{{ messages[0]['content'] }}"
    tokenizer.save_pretrained(folder)


def test_local_model_prepare_materializes_symlinks_without_generation_config(tmp_path, monkeypatch):
    source = tmp_path / "source"
    destination = tmp_path / "model"
    for index, name in enumerate(assets.MODELS):
        folder = source / name
        make_tiny_model(folder)
        (folder / "generation_config.json").unlink()
        weights = folder / "model.safetensors"
        shared = tmp_path / f"weights-{index}.safetensors"
        weights.rename(shared)
        weights.symlink_to(shared)
    monkeypatch.setattr(
        sys,
        "argv",
        ["prepare_models.py", "--source-root", str(source), "--model-root", str(destination), "--models", "all"],
    )
    prepare_models.main()
    for name in assets.MODELS:
        assert (destination / name / "model.safetensors").is_file()
        assert not (destination / name / "model.safetensors").is_symlink()
        assert not (destination / name / "generation_config.json").exists()
        # The separate chat template must survive materialization for prompt length checks.
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(destination / name, local_files_only=True)
        assert tokenizer.apply_chat_template([{"role": "user", "content": "1 + 1"}], tokenize=False) == "1 + 1"
    assert not (destination / "model_manifest.json").exists()
    # Reusing an already prepared model must not recopy or overwrite it.
    prepare_models.main()


def test_missing_weight_file_is_rejected(tmp_path):
    name = next(iter(assets.MODELS))
    folder = tmp_path / name
    make_tiny_model(folder)
    weights = folder / "model.safetensors"
    weights.unlink()
    with pytest.raises(FileNotFoundError, match="model.safetensors"):
        assets.check_model(folder, name)


def test_sharded_model_check_requires_every_indexed_shard(tmp_path):
    folder = tmp_path / "Qwen3-8B"
    make_tiny_model(folder)
    (folder / "model.safetensors").unlink()
    (folder / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "a": "model-00001-of-00002.safetensors",
                    "b": "model-00002-of-00002.safetensors",
                }
            }
        )
    )
    (folder / "model-00001-of-00002.safetensors").write_bytes(b"first")
    with pytest.raises(FileNotFoundError, match="model-00002"):
        assets.check_model(folder, folder.name)
    (folder / "model-00002-of-00002.safetensors").write_bytes(b"second")
    assert assets.check_model(folder, folder.name)["weight_bytes"] == 11


def test_selected_sharded_model_import_from_hf_cache_path(tmp_path, monkeypatch):
    source = tmp_path / ".cache" / "downloads"
    folder = source / "Qwen3-8B"
    make_tiny_model(folder)
    (folder / "model.safetensors").rename(folder / "model-00001-of-00001.safetensors")
    (folder / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "weight": "model-00001-of-00001.safetensors",
                }
            }
        )
    )
    target = tmp_path / "model"
    monkeypatch.setattr(
        sys,
        "argv",
        ["prepare_models.py", "--source-root", str(source), "--model-root", str(target), "--models", "Qwen3-8B"],
    )
    prepare_models.main()
    assert sorted(p.name for p in target.iterdir()) == ["Qwen3-8B"]
    assert (target / "Qwen3-8B/model.safetensors.index.json").is_file()
    assert (target / "Qwen3-8B/model-00001-of-00001.safetensors").is_file()


def test_prompt_length_check_uses_token_rows_not_batch_encoding_keys(tmp_path, monkeypatch):
    from transformers import AutoTokenizer

    model = tmp_path / "tiny"
    make_tiny_model(model)
    tokenizer = AutoTokenizer.from_pretrained(model)
    monkeypatch.setitem(assets.DATASETS["DAPO"], "rows", 1)
    path = tmp_path / "train.parquet"
    write_rows(path, [prepare_data.make_row("1 + 1", "2", "DAPO", 0)])
    report, _ = assets.check_data(path, "DAPO", tokenizer, max_prompt_length=2)
    assert report["max_prompt_tokens"] == 3
    assert report["overlong_prompts"] == 1
