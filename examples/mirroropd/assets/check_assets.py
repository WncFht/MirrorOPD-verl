# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Validate math data on CPU, with minimal local model file checks."""

import argparse
import json
import os
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pyarrow as pa
import pyarrow.parquet as pq

WORKSPACE = Path(__file__).resolve().parents[4]
MODEL_ROOT = Path(os.environ.get("MODEL_ROOT", WORKSPACE / "model"))
DATA_ROOT = Path(os.environ.get("DATA_ROOT", WORKSPACE / "data"))
MODELS = {
    "DeepSeek-R1-Distill-Qwen-1.5B": {
        "repo": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "revision": "ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562",
    },
    "JustRL-DeepSeek-1.5B": {
        "repo": "hbx/JustRL-DeepSeek-1.5B",
        "revision": "0637e4096c789c67f9eecbe8355e0bdeddede1c2",
    },
    "DeepSeek-R1-Distill-Qwen-7B": {"repo": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"},
    "Polaris-7B-Preview": {"repo": "POLARIS-Project/Polaris-7B-Preview"},
    "Qwen3-1.7B-Base": {"repo": "Qwen/Qwen3-1.7B-Base"},
    "Qwen3-8B": {"repo": "Qwen/Qwen3-8B"},
    "Qwen3-4B": {"repo": "Qwen/Qwen3-4B"},
    "Qwen3-1.7B": {"repo": "Qwen/Qwen3-1.7B"},
    "Qwen3-4B-Instruct-2507": {"repo": "Qwen/Qwen3-4B-Instruct-2507"},
}
DEFAULT_MODELS = ("DeepSeek-R1-Distill-Qwen-1.5B", "JustRL-DeepSeek-1.5B")
MODEL_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json")
OPTIONAL_MODEL_FILES = ("generation_config.json", "chat_template.jinja", "README.md")
DATASETS = {
    "DAPO": {"path": "math/DAPO/train.parquet", "rows": 17917, "split": "train"},
    "MATH-500": {"path": "math/MATH-500/test.parquet", "rows": 500, "split": "test"},
    "AMC23": {"path": "math/AMC23/test.parquet", "rows": 40, "split": "test"},
    "AIME25": {"path": "math/AIME25/test.parquet", "rows": 30, "split": "test"},
}
SCHEMA = pa.schema(
    [
        ("prompt", pa.list_(pa.struct([("content", pa.string()), ("role", pa.string())]))),
        ("data_source", pa.large_string()),
        ("ability", pa.large_string()),
        ("reward_model", pa.struct([("ground_truth", pa.string()), ("style", pa.string())])),
        (
            "extra_info",
            pa.struct(
                [
                    ("difficulty", pa.string()),
                    ("index", pa.string()),
                    ("n_tests", pa.int64()),
                    ("source", pa.string()),
                    ("split", pa.string()),
                ]
            ),
        ),
    ]
)


def write_json(path, value):
    """Publish manifests only after a complete JSON document is on disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    partial.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    partial.replace(path)


def canonical_type(kind):
    """Compare Arrow field types independently of struct order and string/list offset width."""
    if pa.types.is_struct(kind):
        return pa.struct(sorted((field.name, canonical_type(field.type)) for field in kind))
    if pa.types.is_list(kind) or pa.types.is_large_list(kind):
        return pa.list_(canonical_type(kind.value_type))
    return pa.string() if pa.types.is_large_string(kind) else kind


def check_data(path, name, tokenizer=None, max_prompt_length=4096):
    """Validate every row without changing duplicate questions or benchmark membership."""
    path = Path(path)
    table = pq.read_table(path)
    missing = set(SCHEMA.names) - set(table.column_names)
    extra_columns = set(table.column_names) - set(SCHEMA.names)
    if missing or extra_columns:
        raise ValueError(f"{path}: schema columns mismatch; missing={sorted(missing)}, extra={sorted(extra_columns)}")
    for field in SCHEMA:
        if canonical_type(table.schema.field(field.name).type) != canonical_type(field.type):
            raise ValueError(f"{path}: invalid Arrow type for {field.name}")
    rows = table.to_pylist()
    spec = DATASETS[name]
    if len(rows) != spec["rows"]:
        raise ValueError(f"{path}: {len(rows)} rows, expected {spec['rows']}")
    ids, prompts = set(), []
    for index, row in enumerate(rows):
        location = f"{path}: row {index}"
        prompt, reward, extra = row["prompt"], row["reward_model"], row["extra_info"]
        if not isinstance(prompt, list) or len(prompt) != 1 or not isinstance(prompt[0], dict):
            raise ValueError(f"{location}: expected one user message")
        if prompt[0].get("role") != "user" or not isinstance(prompt[0].get("content"), str):
            raise ValueError(f"{location}: invalid prompt role/content")
        content = prompt[0]["content"]
        if not content.strip():
            raise ValueError(f"{location}: empty prompt")
        if row["data_source"] != name or row["ability"] != "math":
            raise ValueError(f"{location}: expected data_source={name}, ability=math")
        if not isinstance(reward, dict) or reward.get("style") != "rule":
            raise ValueError(f"{location}: reward_model must use rule scoring")
        answer = reward.get("ground_truth")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(f"{location}: ground_truth must be a nonempty string")
        if not isinstance(extra, dict) or extra.get("source") != name or extra.get("split") != spec["split"]:
            raise ValueError(f"{location}: extra_info source/split mismatch")
        identifier = extra.get("index")
        if not isinstance(identifier, str) or not identifier or identifier in ids:
            raise ValueError(f"{location}: missing or duplicate extra_info.index: {identifier!r}")
        ids.add(identifier)
        prompts.append(content.strip())
    counts = Counter(prompts)
    report = {
        "path": str(path.resolve()),
        "rows": len(rows),
        "duplicate_prompt_rows": len(prompts) - len(counts),
    }
    if tokenizer is not None:
        lengths = []
        for start in range(0, len(rows), 128):
            tokens = tokenizer.apply_chat_template(
                [row["prompt"] for row in rows[start : start + 128]],
                add_generation_prompt=True,
                tokenize=True,
                return_dict=False,
            )
            lengths.extend(len(item) for item in tokens)
        report.update(
            max_prompt_tokens=max(lengths),
            overlong_prompts=sum(length > max_prompt_length for length in lengths),
            max_prompt_length=max_prompt_length,
        )
    print(f"OK {name}: {len(rows)} rows, {report['duplicate_prompt_rows']} duplicate prompt rows", flush=True)
    return report, counts


def selected_models(names):
    if "all" in names:
        if names != ["all"]:
            raise ValueError("Use --models all without additional model names")
        return list(MODELS)
    return list(dict.fromkeys(names))


def model_weight_files(folder):
    """Resolve standard safetensors layouts from metadata, without reading weights."""
    folder = Path(folder)
    index = folder / "model.safetensors.index.json"
    if index.is_file():
        files = sorted(set(json.loads(index.read_text())["weight_map"].values()))
        if not files:
            raise ValueError(f"Empty weight map: {index}")
        for name in files:
            if Path(name).is_absolute() or ".." in Path(name).parts:
                raise ValueError(f"Weight shard must be inside the model directory: {name}")
        return files
    return ["model.safetensors"]


def check_model(folder, name):
    """Check required files without reading weights; generation configuration is optional."""
    folder = Path(folder)
    for filename in MODEL_FILES:
        path = folder / filename
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing or empty model file: {path}")
        if filename.endswith(".json"):
            json.loads(path.read_text())
    config = json.loads((folder / "config.json").read_text())
    weights = [folder / filename for filename in model_weight_files(folder)]
    for path in weights:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing or empty model file: {path}")
    report = {
        "path": str(folder.resolve()),
        "weight_bytes": sum(path.stat().st_size for path in weights),
        "model_type": config.get("model_type"),
    }
    print(f"OK model {name}: required files present", flush=True)
    return report


def main():
    """Check full math assets without loading model tensors or initializing CUDA."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, default=MODEL_ROOT)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--data-only", action="store_true", help="skip models and prompt tokenization")
    parser.add_argument("--models", nargs="+", choices=[*MODELS, "all"], default=list(DEFAULT_MODELS))
    parser.add_argument("--max-prompt-length", type=int, default=4096)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = {"checked_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "models": {}, "data": {}}
    tokenizer = None
    if not args.data_only:
        from transformers import AutoTokenizer

        for name in selected_models(args.models):
            report["models"][name] = check_model(args.model_root / name, name)
        student = args.model_root / selected_models(args.models)[0]
        tokenizer = AutoTokenizer.from_pretrained(student, local_files_only=True, trust_remote_code=False)
    prompt_sets = {}
    for name, spec in DATASETS.items():
        result, prompts = check_data(args.data_root / spec["path"], name, tokenizer, args.max_prompt_length)
        report["data"][name] = result
        prompt_sets[name] = set(prompts)
    report["exact_train_validation_overlap"] = {
        name: len(prompt_sets["DAPO"] & prompts) for name, prompts in prompt_sets.items() if name != "DAPO"
    }
    print("Exact DAPO/validation question overlap:", report["exact_train_validation_overlap"], flush=True)
    for name, result in report["data"].items():
        if "overlong_prompts" in result:
            print(f"{name}: max prompt {result['max_prompt_tokens']} tokens; {result['overlong_prompts']} above limit")
    destination = args.report or args.data_root / "asset_check.json"
    write_json(destination, report)
    print(f"Asset checks passed; report: {destination.resolve()}")


if __name__ == "__main__":
    main()
