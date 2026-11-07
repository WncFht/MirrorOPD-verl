#!/usr/bin/env python3
# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Extract teacher states and reproduce PCA/K-means coverage from One-Shot OPD."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from .core import (
    assign_clusters,
    coverage,
    deterministic_rollout_split,
    fit_kmeans,
    fit_pca,
    relative_positions,
    transform_pca,
)

TOKEN_FIELDS = {"prompt_ids", "response_ids", "response_mask"}


def _json_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


def _trajectory_files(inputs: list[Path]) -> list[Path]:
    files = []
    for path in inputs:
        if not path.exists():
            raise FileNotFoundError(path)
        if path.is_dir():
            files.extend(
                candidate
                for candidate in path.rglob("*")
                if candidate.is_file()
                and (
                    candidate.suffix == ".parquet"
                    or candidate.suffix == ".jsonl"
                    or candidate.name.endswith(".jsonl.gz")
                    or candidate.suffix == ".ndjson"
                )
            )
        else:
            files.append(path)
    result = sorted({path.resolve() for path in files})
    unsupported = [
        path
        for path in result
        if not (path.suffix == ".parquet" or path.suffix in {".jsonl", ".ndjson"} or path.name.endswith(".jsonl.gz"))
    ]
    if unsupported:
        raise ValueError(f"Unsupported trajectory input(s): {unsupported}")
    if not result:
        raise ValueError("No Parquet or JSONL trajectory files found")
    return result


def _read_trajectories(inputs: list[Path]) -> list[dict]:
    rows = []
    for path in _trajectory_files(inputs):
        if path.suffix == ".parquet":
            try:
                import pyarrow.parquet as pq
            except ImportError as error:
                raise RuntimeError("Reading Parquet requires pyarrow") from error
            rows.extend(pq.read_table(path).to_pylist())
        else:
            opener = gzip.open if path.name.endswith(".gz") else open
            with opener(path, "rt", encoding="utf-8") as stream:
                rows.extend(json.loads(line) for line in stream if line.strip())
    if not rows:
        raise ValueError("Trajectory inputs contain no rows")
    return rows


def _normalize_trajectory(row: dict, source: str) -> dict:
    prompt = [int(token) for token in row.get("prompt_ids", [])]
    response = [int(token) for token in row.get("response_ids", [])]
    if not prompt or not response:
        raise ValueError(f"Trajectory {row.get('id')!r} needs non-empty prompt_ids and response_ids")
    mask = row.get("response_mask")
    if mask is None:
        mask = [True] * len(response)
    if len(mask) != len(response):
        raise ValueError(f"Trajectory {row.get('id')!r} has mismatched response_ids/response_mask lengths")
    valid_positions = [index for index, valid in enumerate(mask) if bool(valid)]
    if not valid_positions:
        raise ValueError(f"Trajectory {row.get('id')!r} has no response positions selected by response_mask")
    trajectory_id = str(row.get("id") or row.get("rollout_key") or "")
    if not trajectory_id:
        raise ValueError("Each trajectory needs a stable id or rollout_key")
    metadata = {
        key: _json_scalar(value)
        for key, value in row.items()
        if key not in TOKEN_FIELDS and isinstance(value, str | int | float | bool | np.generic)
    }
    return {
        "trajectory_id": f"{source}::{trajectory_id}",
        "original_trajectory_id": trajectory_id,
        "source_id": str(row.get("source_id", "")),
        "source": source,
        "prompt_ids": prompt,
        "response_ids": response,
        "valid_positions": valid_positions,
        "metadata": metadata,
    }


def _empty_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Output directory must be absent or empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _load_hf_model(args):
    try:
        import torch
        from transformers import AutoModelForCausalLM
    except ImportError as error:
        raise RuntimeError("State extraction requires torch and transformers") from error

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if args.dtype == "auto":
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    else:
        dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        local_files_only=not args.allow_download,
        trust_remote_code=args.trust_remote_code,
    ).to(device)
    model.eval()
    model.requires_grad_(False)
    return torch, model, device


def _last_hidden(model, *, input_ids, attention_mask, position_ids):
    """Run only the decoder when the HF causal-LM wrapper exposes it."""
    backbone = getattr(model, "model", None) or getattr(model, "transformer", None)
    if backbone is not None:
        output = backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            return_dict=True,
        )
        return output.last_hidden_state
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
        output_hidden_states=True,
        return_dict=True,
    )
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    return output.hidden_states[-1]


def extract(args) -> None:
    if args.positions < 1 or args.batch_size < 1:
        raise ValueError("positions and batch-size must be positive")
    trajectories = [_normalize_trajectory(row, args.source) for row in _read_trajectories(args.inputs)]
    seen = set()
    for row in trajectories:
        if row["trajectory_id"] in seen:
            raise ValueError(f"Duplicate trajectory id in source {args.source!r}: {row['original_trajectory_id']}")
        seen.add(row["trajectory_id"])
        row["positions"] = relative_positions(row["valid_positions"], args.positions)
        row["rank_by_position"] = {position: rank for rank, position in enumerate(row["valid_positions"])}
        row["sequence"] = row["prompt_ids"] + row["response_ids"][: max(row["positions"])]

    torch, model, device = _load_hf_model(args)
    pad_id = model.config.pad_token_id
    if pad_id is None:
        pad_id = model.config.eos_token_id
    if pad_id is None:
        pad_id = 0

    vector_batches = []
    state_records = []
    with torch.inference_mode():
        for start in range(0, len(trajectories), args.batch_size):
            batch = trajectories[start : start + args.batch_size]
            width = max(len(row["sequence"]) for row in batch)
            input_ids = torch.full((len(batch), width), int(pad_id), dtype=torch.long, device=device)
            attention_mask = torch.zeros((len(batch), width), dtype=torch.long, device=device)
            for batch_index, row in enumerate(batch):
                sequence = torch.tensor(row["sequence"], dtype=torch.long, device=device)
                input_ids[batch_index, width - len(sequence) :] = sequence
                attention_mask[batch_index, width - len(sequence) :] = 1
            position_ids = (attention_mask.cumsum(-1) - 1).masked_fill(attention_mask == 0, 0)
            hidden = _last_hidden(
                model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
            batch_vectors = []
            for batch_index, row in enumerate(batch):
                pad_width = width - len(row["sequence"])
                prompt_length = len(row["prompt_ids"])
                for position in row["positions"]:
                    token_state_position = prompt_length + position - 1
                    batch_vectors.append(hidden[batch_index, pad_width + token_state_position].float().cpu().numpy())
                    record = dict(row["metadata"])
                    record.update(
                        {
                            "trajectory_id": row["trajectory_id"],
                            "original_trajectory_id": row["original_trajectory_id"],
                            "source_id": row["source_id"],
                            "source": row["source"],
                            "state_index": len(state_records),
                            "response_position": int(position),
                            "valid_position_rank": int(row["rank_by_position"][position]),
                            "response_length": len(row["valid_positions"]),
                            "relative_position": float(
                                row["rank_by_position"][position] / max(1, len(row["valid_positions"]) - 1)
                            ),
                            "finish_reason": row["metadata"].get("finish_reason"),
                        }
                    )
                    state_records.append(record)
            vector_batches.append(np.stack(batch_vectors).astype(np.float32))

    vectors = np.concatenate(vector_batches, axis=0) if vector_batches else np.empty((0, 0), dtype=np.float32)
    if len(vectors) != len(state_records):
        raise RuntimeError("Internal state/vector count mismatch")
    output_dir = args.output
    _empty_output(output_dir)
    np.save(output_dir / "vectors.npy", vectors, allow_pickle=False)
    with (output_dir / "states.jsonl").open("w", encoding="utf-8") as stream:
        for record in state_records:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    config = {key: value for key, value in model.config.to_dict().items() if key != "_name_or_path"}
    config_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    ).hexdigest()
    model_path = Path(args.model).expanduser()
    teacher_model = str(model_path.resolve()) if model_path.exists() else str(args.model)
    summary = {
        "format_version": 1,
        "source": args.source,
        "model": str(args.model),
        "teacher_model": teacher_model,
        "model_config_sha256": config_hash,
        "model_config_name": config.get("_name_or_path"),
        "hidden_size": int(vectors.shape[1]),
        "state_count": int(len(vectors)),
        "trajectory_count": int(len(trajectories)),
        "positions_per_rollout": args.positions,
        "position_mode": "relative response-mask quantiles",
        "state_representation": "teacher final-layer hidden state at the last token of the prefix",
        "state_definition": "prompt_ids + response_ids[:response_position]",
        "dtype": str(next(model.parameters()).dtype),
        "device": device,
        "inputs": [str(path.resolve()) for path in _trajectory_files(args.inputs)],
        "trust_remote_code": bool(args.trust_remote_code),
    }
    (output_dir / "metadata.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _load_extractions(paths: list[Path]) -> tuple[np.ndarray, list[dict], list[dict]]:
    matrices = []
    records = []
    metadata = []
    for directory in paths:
        vector_path = directory / "vectors.npy"
        states_path = directory / "states.jsonl"
        metadata_path = directory / "metadata.json"
        if not vector_path.is_file() or not states_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(f"Expected vectors.npy, states.jsonl, and metadata.json in {directory}")
        vectors = np.load(vector_path, mmap_mode="r", allow_pickle=False)
        current_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        with states_path.open(encoding="utf-8") as stream:
            current_records = [json.loads(line) for line in stream if line.strip()]
        if vectors.ndim != 2 or len(vectors) != len(current_records):
            raise ValueError(f"Invalid vector/state row alignment in {directory}")
        if matrices and vectors.shape[1] != matrices[0].shape[1]:
            raise ValueError("All extraction directories must use the same hidden size")
        matrices.append(vectors)
        records.extend(current_records)
        metadata.append(current_metadata)
    if not matrices:
        raise ValueError("At least one extraction directory is required")
    return np.concatenate(matrices, axis=0), records, metadata


def _check_teacher_identity(metadata: list[dict]) -> dict:
    required = ("teacher_model", "model_config_sha256")
    if any(not all(key in current for key in required) for current in metadata):
        raise ValueError("Every extraction must have teacher_model and model_config_sha256 in metadata.json")
    identity = {key: metadata[0][key] for key in required}
    if any({key: current[key] for key in required} != identity for current in metadata[1:]):
        raise ValueError("All extractions must use the same teacher model and model config")
    return identity


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def fit(args) -> None:
    vectors, records, extraction_metadata = _load_extractions(args.extractions)
    teacher_identity = _check_teacher_identity(extraction_metadata)
    reference_ids, heldout_ids = deterministic_rollout_split(
        records,
        source=args.reference_source,
        group_field=args.split_field,
        seed=args.seed,
        reference_fraction=args.reference_fraction,
    )
    reference_mask = np.fromiter((row["trajectory_id"] in reference_ids for row in records), dtype=bool)
    reference_vectors = vectors[reference_mask]
    if args.pca_dim > min(reference_vectors.shape):
        raise ValueError(f"pca-dim {args.pca_dim} exceeds reference matrix shape {reference_vectors.shape}")
    pca = fit_pca(reference_vectors, args.pca_dim)
    projected = transform_pca(reference_vectors, pca)
    centers, kmeans_info = fit_kmeans(
        projected,
        args.clusters,
        seed=args.seed,
        max_iter=args.max_iter,
        tol=args.tol,
    )

    output_dir = args.output
    _empty_output(output_dir)
    np.savez_compressed(
        output_dir / "pca.npz",
        mean=pca["mean"],
        components=pca["components"],
        explained_variance=pca["explained_variance"],
        explained_variance_ratio=pca["explained_variance_ratio"],
        total_explained_variance_ratio=np.asarray(pca["total_explained_variance_ratio"]),
    )
    np.save(output_dir / "centroids.npy", centers, allow_pickle=False)
    split_metadata = {
        "reference_source": args.reference_source,
        **teacher_identity,
        "split_field": args.split_field,
        "reference_fraction": args.reference_fraction,
        "seed": args.seed,
        "reference_rollouts": sorted(reference_ids),
        "heldout_rollouts": sorted(heldout_ids),
    }
    _write_json(output_dir / "rollout_split.json", split_metadata)
    fit_metadata = {
        "format_version": 1,
        "reference_source": args.reference_source,
        **teacher_identity,
        "split_field": args.split_field,
        "seed": args.seed,
        "reference_fraction": args.reference_fraction,
        "reference_rollout_count": len(reference_ids),
        "heldout_rollout_count": len(heldout_ids),
        "reference_state_count": int(reference_mask.sum()),
        "input_state_count": len(records),
        "input_hidden_size": int(vectors.shape[1]),
        "pca_dim": args.pca_dim,
        "pca_total_explained_variance_ratio": float(pca["total_explained_variance_ratio"]),
        "n_clusters": args.clusters,
        "kmeans": kmeans_info,
        "algorithm": "centered full-SVD PCA; Lloyd K-means with k-means++ initialization",
        "extractions": [str(path.resolve()) for path in args.extractions],
    }
    _write_json(output_dir / "fit.json", fit_metadata)
    print(json.dumps(fit_metadata, ensure_ascii=False, indent=2))


def _numeric_step(value, field: str, trajectory_id: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Step field {field!r} must be numeric; {trajectory_id} has {value!r}") from error
    if not math.isfinite(number):
        raise ValueError(f"Step field {field!r} must be finite; {trajectory_id} has {value!r}")
    return number


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty result table: {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def score(args) -> None:
    vectors, records, extraction_metadata = _load_extractions(args.extractions)
    teacher_identity = _check_teacher_identity(extraction_metadata)
    fit_dir = args.fit
    fit_metadata = json.loads((fit_dir / "fit.json").read_text(encoding="utf-8"))
    if any(fit_metadata.get(key) != teacher_identity[key] for key in teacher_identity):
        raise ValueError("Scoring extractions use a different teacher identity than the fitted codebook")
    with np.load(fit_dir / "pca.npz", allow_pickle=False) as data:
        pca = {
            "mean": data["mean"],
            "components": data["components"],
        }
    centers = np.load(fit_dir / "centroids.npy", allow_pickle=False)
    if vectors.shape[1] != len(pca["mean"]):
        raise ValueError("State hidden size does not match the fitted PCA input size")
    if args.rollouts_per_step < 1 or args.repeats < 1:
        raise ValueError("rollouts-per-step and repeats must be positive")
    projected = transform_pca(vectors, pca)
    labels = assign_clusters(projected, centers, chunk_size=args.distance_chunk_size)

    by_source_step: dict[str, dict[float, dict[str, list[int]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    step_values: dict[tuple[str, float], object] = {}
    for index, row in enumerate(records):
        source = str(row["source"])
        if args.step_field not in row or row[args.step_field] is None:
            raise ValueError(f"State record lacks requested scoring field {args.step_field!r}")
        step_number = _numeric_step(row[args.step_field], args.step_field, row["trajectory_id"])
        step_values[(source, step_number)] = row[args.step_field]
        by_source_step[source][step_number][str(row["trajectory_id"])].append(int(labels[index]))

    split = json.loads((fit_dir / "rollout_split.json").read_text(encoding="utf-8"))
    heldout_ids = set(split["heldout_rollouts"])
    reference_source = str(fit_metadata["reference_source"])
    n_clusters = int(fit_metadata["n_clusters"])
    rng = np.random.default_rng(args.seed)
    replicate_rows = []
    for source in sorted(by_source_step):
        groups = by_source_step[source]
        if source == reference_source:
            groups = defaultdict(
                dict,
                {
                    step: {
                        rollout_id: cluster_ids
                        for rollout_id, cluster_ids in rollouts.items()
                        if rollout_id in heldout_ids
                    }
                    for step, rollouts in groups.items()
                },
            )
        available_steps = [step for step in sorted(groups) if args.max_step is None or step <= args.max_step]
        if not available_steps:
            continue
        for repeat in range(args.repeats):
            repeat_seed = int(rng.integers(0, 2**31 - 1))
            repeat_rng = np.random.default_rng(repeat_seed)
            seen_clusters: set[int] = set()
            cumulative_rollouts = 0
            cumulative_states = 0
            for step in available_steps:
                rollouts = groups[step]
                rollout_ids = sorted(rollouts)
                if len(rollout_ids) < args.rollouts_per_step and not args.allow_fewer:
                    raise ValueError(
                        f"Source {source!r}, step {step:g} has {len(rollout_ids)} rollouts; "
                        f"need {args.rollouts_per_step}. Use --allow-fewer only for exploratory smoke runs."
                    )
                sample_count = min(args.rollouts_per_step, len(rollout_ids))
                chosen = repeat_rng.choice(rollout_ids, size=sample_count, replace=False).tolist()
                step_labels = [label for rollout_id in chosen for label in rollouts[rollout_id]]
                seen_clusters.update(step_labels)
                cumulative_rollouts += len(chosen)
                cumulative_states += len(step_labels)
                visited, value = coverage(np.fromiter(seen_clusters, dtype=np.int32), n_clusters)
                replicate_rows.append(
                    {
                        "source": source,
                        "step": step_values[(source, step)],
                        "step_numeric": step,
                        "repeat": repeat,
                        "seed": repeat_seed,
                        "coverage": value,
                        "visited_clusters": visited,
                        "n_clusters": n_clusters,
                        "cumulative_rollouts": cumulative_rollouts,
                        "cumulative_states": cumulative_states,
                        "rollouts_sampled_this_step": len(chosen),
                    }
                )

    if not replicate_rows:
        raise ValueError("No states remained after source/step filtering")
    summary_groups: dict[tuple[str, float, object], list[float]] = defaultdict(list)
    for row in replicate_rows:
        summary_groups[(row["source"], row["step_numeric"], row["step"])].append(row["coverage"])
    summary_rows = []
    for (source, step_numeric, original_step), values in sorted(summary_groups.items()):
        summary_rows.append(
            {
                "source": source,
                "step": original_step,
                "step_numeric": step_numeric,
                "repeats": len(values),
                "coverage_mean": float(np.mean(values)),
                "coverage_std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            }
        )
    output_dir = args.output
    _empty_output(output_dir)
    _write_csv(output_dir / "coverage-repeats.csv", replicate_rows)
    _write_csv(output_dir / "coverage-summary.csv", summary_rows)
    _write_json(
        output_dir / "coverage.json",
        {
            "format_version": 1,
            "fit": str(fit_dir.resolve()),
            "step_field": args.step_field,
            "rollouts_per_step": args.rollouts_per_step,
            "repeats": args.repeats,
            "seed": args.seed,
            "max_step": args.max_step,
            "heldout_reference_rollouts_only": reference_source,
            "allow_fewer": args.allow_fewer,
            "extractions": [str(path.resolve()) for path in args.extractions],
            "summary": summary_rows,
        },
    )
    print(
        json.dumps(
            {"sources": sorted({row["source"] for row in replicate_rows}), "summary": summary_rows},
            ensure_ascii=False,
            indent=2,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    extract_parser = commands.add_parser("extract", help="extract teacher final-layer response-prefix states")
    extract_parser.add_argument(
        "--input", dest="inputs", type=Path, nargs="+", required=True, help="Parquet/JSONL file(s) or directories"
    )
    extract_parser.add_argument("--output", type=Path, required=True)
    extract_parser.add_argument("--model", required=True, help="Frozen teacher Hugging Face model path or ID")
    extract_parser.add_argument("--source", required=True, help="Stable policy/source label for these trajectories")
    extract_parser.add_argument(
        "--positions", type=int, default=8, help="Unique relative response positions per rollout"
    )
    extract_parser.add_argument("--batch-size", type=int, default=1)
    extract_parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    extract_parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    extract_parser.add_argument("--allow-download", action="store_true", help="allow Transformers to fetch model files")
    extract_parser.add_argument("--trust-remote-code", action="store_true")
    extract_parser.set_defaults(func=extract)

    fit_parser = commands.add_parser("fit", help="fit one shared reference PCA and K-means codebook")
    fit_parser.add_argument("--extractions", type=Path, nargs="+", required=True)
    fit_parser.add_argument("--output", type=Path, required=True)
    fit_parser.add_argument("--reference-source", required=True)
    fit_parser.add_argument(
        "--split-field", required=True, help="metadata field that identifies independent rollout groups, usually step"
    )
    fit_parser.add_argument("--pca-dim", type=int, default=50)
    fit_parser.add_argument("--clusters", type=int, default=200)
    fit_parser.add_argument("--reference-fraction", type=float, default=0.6)
    fit_parser.add_argument("--seed", type=int, default=17)
    fit_parser.add_argument("--max-iter", type=int, default=100)
    fit_parser.add_argument("--tol", type=float, default=1e-4)
    fit_parser.set_defaults(func=fit)

    score_parser = commands.add_parser("score", help="score cumulative shared-cluster state coverage")
    score_parser.add_argument("--extractions", type=Path, nargs="+", required=True)
    score_parser.add_argument("--fit", type=Path, required=True)
    score_parser.add_argument("--output", type=Path, required=True)
    score_parser.add_argument("--step-field", required=True, help="numeric training-step metadata field")
    score_parser.add_argument("--rollouts-per-step", type=int, default=2)
    score_parser.add_argument("--repeats", type=int, default=30)
    score_parser.add_argument("--seed", type=int, default=29)
    score_parser.add_argument("--max-step", type=float)
    score_parser.add_argument("--distance-chunk-size", type=int, default=4096)
    score_parser.add_argument(
        "--allow-fewer", action="store_true", help="permit underfilled steps; for smoke runs only"
    )
    score_parser.set_defaults(func=score)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
