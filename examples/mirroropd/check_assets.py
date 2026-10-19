# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Validate the fixed math recipe's local models and parquet files on CPU."""

import argparse
import hashlib
import json
import os
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pyarrow as pa
import pyarrow.parquet as pq

WORKSPACE = Path(__file__).resolve().parents[3]
MODEL_ROOT = Path(os.environ.get("MODEL_ROOT", WORKSPACE / "model"))
DATA_ROOT = Path(os.environ.get("DATA_ROOT", WORKSPACE / "data"))
MODELS = {
    "DeepSeek-R1-Distill-Qwen-1.5B": {
        "repo": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "revision": "ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562",
        "weight_sha256": "58858233513d76b8703e72eed6ce16807b523328188e13329257fb9594462945",
    },
    "JustRL-DeepSeek-1.5B": {
        "repo": "hbx/JustRL-DeepSeek-1.5B",
        "revision": "0637e4096c789c67f9eecbe8355e0bdeddede1c2",
        "weight_sha256": "1bedcdf243c4e2e633fa04c05617809cf6a1bbc1a07221609035f6347efecffb",
    },
}
MODEL_FILES = ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors")
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


def file_sha256(path):
    """Hash files incrementally, including multi-gigabyte model weights when requested."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


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
        "sha256": file_sha256(path),
        "duplicate_prompt_rows": len(prompts) - len(counts),
    }
    if tokenizer is not None:
        lengths = []
        for start in range(0, len(rows), 128):
            tokens = tokenizer.apply_chat_template(
                [row["prompt"] for row in rows[start : start + 128]], add_generation_prompt=True, tokenize=True
            )
            lengths.extend(len(item) for item in tokens)
        report.update(
            max_prompt_tokens=max(lengths),
            overlong_prompts=sum(length > max_prompt_length for length in lengths),
            max_prompt_length=max_prompt_length,
        )
    print(f"OK {name}: {len(rows)} rows, {report['duplicate_prompt_rows']} duplicate prompt rows", flush=True)
    return report, counts


def check_model(folder, name, verify_sha256=False):
    """Check complete fixed snapshots, safetensors payload boundaries and a loadable tokenizer."""
    from safetensors import safe_open
    from transformers import AutoTokenizer

    folder = Path(folder)
    for filename in MODEL_FILES:
        path = folder / filename
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing or empty model file: {path}")
        if filename.endswith(".json"):
            json.loads(path.read_text())
    config = json.loads((folder / "config.json").read_text())
    if config["model_type"] != "qwen2":
        raise ValueError(f"{folder}: expected qwen2 model, got {config['model_type']}")
    weights = folder / "model.safetensors"
    hidden = config["hidden_size"]
    intermediate = config["intermediate_size"]
    key_value = config["num_key_value_heads"] * (hidden // config["num_attention_heads"])
    expected_shapes = {
        "model.embed_tokens.weight": [config["vocab_size"], hidden],
        "lm_head.weight": [config["vocab_size"], hidden],
        "model.norm.weight": [hidden],
    }
    layer_shapes = {
        "input_layernorm.weight": [hidden],
        "post_attention_layernorm.weight": [hidden],
        "self_attn.q_proj.weight": [hidden, hidden],
        "self_attn.q_proj.bias": [hidden],
        "self_attn.k_proj.weight": [key_value, hidden],
        "self_attn.k_proj.bias": [key_value],
        "self_attn.v_proj.weight": [key_value, hidden],
        "self_attn.v_proj.bias": [key_value],
        "self_attn.o_proj.weight": [hidden, hidden],
        "mlp.gate_proj.weight": [intermediate, hidden],
        "mlp.up_proj.weight": [intermediate, hidden],
        "mlp.down_proj.weight": [hidden, intermediate],
    }
    for layer in range(config["num_hidden_layers"]):
        expected_shapes.update({f"model.layers.{layer}.{key}": shape for key, shape in layer_shapes.items()})
    with safe_open(str(weights), framework="pt", device="cpu") as tensors:
        keys = set(tensors.keys())
        if keys != set(expected_shapes):
            raise ValueError(f"{folder}: checkpoint tensor names differ from the complete Qwen2 configuration")
        for key, shape in expected_shapes.items():
            if tensors.get_slice(key).get_shape() != shape:
                raise ValueError(f"{folder}: incompatible tensor shape for {key}")
        tensor_count = len(keys)
    tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True, trust_remote_code=False)
    if max(tokenizer.get_vocab().values()) >= config["vocab_size"]:
        raise ValueError(f"{folder}: tokenizer IDs exceed model vocabulary size")
    tokenizer.apply_chat_template([{"role": "user", "content": "1 + 1 = ?"}], add_generation_prompt=True, tokenize=True)
    report = {
        "path": str(folder.resolve()),
        "expected_repo": MODELS[name]["repo"],
        "expected_revision": MODELS[name]["revision"],
        "weight_bytes": weights.stat().st_size,
        "tensor_count": tensor_count,
        "vocab_size": config["vocab_size"],
        "tokenizer_size": len(tokenizer),
        "max_position_embeddings": config["max_position_embeddings"],
        "weight_sha256_verified": verify_sha256,
    }
    if verify_sha256:
        report["weight_sha256"] = file_sha256(weights)
        if report["weight_sha256"] != MODELS[name]["weight_sha256"]:
            raise ValueError(f"{weights}: SHA256 differs from the pinned model snapshot")
    print(f"OK model {name}: {tensor_count} tensors, {len(tokenizer)} tokenizer entries", flush=True)
    return report, tokenizer


def main():
    """Check full math assets without loading model tensors or initializing CUDA."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, default=MODEL_ROOT)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--data-only", action="store_true", help="skip models and prompt tokenization")
    parser.add_argument("--verify-model-sha256", action="store_true", help="read and hash both complete weight files")
    parser.add_argument("--max-prompt-length", type=int, default=4096)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = {"checked_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "models": {}, "data": {}}
    tokenizer = None
    if not args.data_only:
        tokenizers = []
        for name in MODELS:
            result, tok = check_model(args.model_root / name, name, args.verify_model_sha256)
            report["models"][name] = result
            tokenizers.append(tok)
        if tokenizers[0].get_vocab() != tokenizers[1].get_vocab():
            raise ValueError(
                "Student and teacher token-to-ID mappings differ; token-level distillation is incompatible"
            )
        if tokenizers[0].all_special_ids != tokenizers[1].all_special_ids:
            raise ValueError("Student and teacher special token IDs differ")
        tokenizer = tokenizers[0]
        report["tokenizer_ids_match"] = True
    prompt_sets = {}
    manifest_path = args.data_root / "math_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    report["data_manifest_verified"] = manifest is not None
    for name, spec in DATASETS.items():
        result, prompts = check_data(args.data_root / spec["path"], name, tokenizer, args.max_prompt_length)
        if manifest is not None and manifest["data"][name]["sha256"] != result["sha256"]:
            raise ValueError(f"{name}: file SHA256 does not match math_manifest.json")
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
