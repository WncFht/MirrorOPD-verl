# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Shared fixed-state readouts; frozen histories never enter an optimizer."""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from examples.mirroropd.research.core.artifacts import complete, file_hash, fingerprint, receipt
from examples.mirroropd.research.core.io import dump, finite, read
from examples.mirroropd.research.studies.prefix_fusion.assets import verify_model_assets
from examples.mirroropd.research.studies.prefix_fusion.protocol import LONG_GROUPS, protocol_identity


def load_bank(path):
    path = Path(path)
    manifest = read(path / "complete.json")
    if (
        manifest["inputs"]["protocol"] != protocol_identity()
        or file_hash(path / "bank.json") != manifest["bank_sha256"]
    ):
        raise ValueError("Frozen bank identity changed")
    rows = read(path / "bank.json")
    if len(rows) != 128 or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Incomplete or duplicate bank rows")
    return rows, manifest


def fixed_views(rows, horizon, groups=None):
    if horizon < 1:
        raise ValueError("Fixed risk needs a positive horizon")
    selected = [r for r in rows if groups is None or r["group"] in groups]
    views = []
    for row in selected:
        view = dict(
            row,
            response_ids=row["response_ids"][:horizon],
            view_horizon=horizon,
            original_response_length=len(row["response_ids"]),
        )
        if "behavior_log_probs" in row:
            view["behavior_log_probs"] = row["behavior_log_probs"][:horizon]
        views.append(view)
    return views


def require_allocated_gpu():
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {str(gpu) for gpu in range(8)}:
        raise ValueError("Use one explicitly allocated physical GPU 0 through 7")


def cache_teacher(assets, banks, pair, out):
    require_allocated_gpu()
    import torch

    from examples.mirroropd.research.backends import hf_models
    from examples.mirroropd.research.backends.hf_rkl import save_tensor

    assets, banks, out = Path(assets), Path(banks), Path(out)
    asset = read(assets / "complete.json")["models"][pair]
    verify_model_assets(asset)
    loaded = {source: load_bank(banks / source) for source in ("S0", pair)}
    inputs = dict(
        protocol=protocol_identity(),
        pair=pair,
        teacher=asset["identity"],
        contract=1,
        banks={s: m["bank_sha256"] for s, (_, m) in loaded.items()},
        model_code=file_hash(hf_models.__file__),
        cache_code=file_hash(__file__),
        precision="BF16 teacher backbone; FP32 head during scoring; native full head",
    )
    if complete(out, inputs):
        return
    out.mkdir(parents=True, exist_ok=True)
    if (out / "inputs.json").exists() and read(out / "inputs.json") != inputs:
        raise ValueError("Partial teacher cache belongs to different inputs")
    dump(out / "inputs.json", inputs)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = hf_models.load(SimpleNamespace(directory=Path(asset["directory"])))
    save_tensor(out / "head.pt", model.get_output_embeddings().weight.detach().cpu())
    probes = []
    with torch.no_grad():
        for source, (rows, _) in loaded.items():
            (out / source).mkdir(exist_ok=True)
            for i, row in enumerate(rows):
                target = out / source / f"{row['id']:04d}.pt"
                if target.exists():
                    value = torch.load(target, map_location="cpu", weights_only=True)
                    if value["row_identity"] != fingerprint(row):
                        raise ValueError("Partial teacher row identity changed")
                    if i != 0:
                        continue
                hidden = hf_models.hidden(model, [row], pad_id=151643)
                if len(hidden) != len(row["response_ids"]) or not torch.isfinite(hidden).all():
                    raise ValueError("Invalid cached teacher hidden states")
                if i == 0:
                    positions = sorted({0, len(row["response_ids"]) // 2, len(row["response_ids"]) - 1})
                    absolute = torch.tensor([len(row["prompt_ids"]) - 1 + j for j in positions], device="cuda")
                    head = model.get_output_embeddings()

                    def project(module, args, output):
                        with torch.autocast("cuda", enabled=False):
                            return torch.nn.functional.linear(args[0].float(), module.weight.float())

                    hook = head.register_forward_hook(project)
                    try:
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            reference = (
                                model(
                                    input_ids=torch.tensor(
                                        [row["prompt_ids"] + row["response_ids"][:-1]], device="cuda"
                                    ),
                                    use_cache=False,
                                    logits_to_keep=absolute,
                                )
                                .logits[0]
                                .float()
                            )
                    finally:
                        hook.remove()
                    actual = torch.nn.functional.linear(hidden[positions].float(), head.weight.float())
                    difference = (actual - reference).abs().max().item()
                    if difference >= 1e-4:
                        raise ValueError(f"Teacher hidden/head reconstruction failed: {difference}")
                    probes.append(dict(source=source, id=row["id"], max_logit_difference=difference))
                if not target.exists():
                    save_tensor(
                        target, dict(row_identity=fingerprint(row), hidden=hidden.cpu(), teacher=asset["identity"])
                    )
                if i % 16 == 0:
                    print("TEACHER_CACHE", pair, source, i, flush=True)
    row_files = {
        str(p.relative_to(out)): file_hash(p) for source in ("S0", pair) for p in sorted((out / source).glob("*.pt"))
    }
    if len(row_files) != 256:
        raise ValueError("Incomplete teacher cache")
    receipt(
        out, inputs, rows=256, head_sha256=file_hash(out / "head.pt"), row_files=row_files, reconstruction_checks=probes
    )


def score(assets, banks, cache, weights, out, outer, long_panel=False, limit_per_source=None, horizon=None):
    require_allocated_gpu()
    import torch

    from examples.mirroropd.research.backends import hf_models, hf_rkl

    assets, banks, cache, weights, out = map(Path, (assets, banks, cache, weights, out))
    target = read(cache / "complete.json")
    if target["inputs"].get("contract") != 1 or target["inputs"]["model_code"] != file_hash(hf_models.__file__):
        raise ValueError("Teacher hidden cache uses a different scoring model contract")
    pair = target["inputs"]["pair"]
    if pair not in {"J", "P"} or set(target["inputs"]["banks"]) != {"S0", pair}:
        raise ValueError("Unexpected teacher cache bank set")
    asset = read(assets / "complete.json")["models"]["S0"]
    config = read(weights / "config.json")
    original = read(Path(asset["directory"]) / "config.json")
    for key in (
        "model_type",
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "vocab_size",
    ):
        if config[key] != original[key]:
            raise ValueError(f"Readout is not the declared learner: {key}")
    weight_files = sorted(weights.glob("*.safetensors"))
    if not weight_files:
        raise ValueError("No learner safetensors readout")
    if weights.resolve() != Path(asset["directory"]).resolve():
        from safetensors import safe_open

        for path in weight_files:
            with safe_open(path, framework="pt", device="cpu") as handle:
                for name in handle.keys():
                    value = handle.get_tensor(name)
                    if value.is_floating_point() and value.dtype != torch.float32:
                        raise ValueError("Fixed-state readout must preserve FP32 master weights")
    weight_identity = fingerprint({p.name: file_hash(p) for p in [weights / "config.json", *weight_files]})
    if read(assets / "complete.json")["models"][pair]["identity"] != target["inputs"]["teacher"]:
        raise ValueError("Teacher cache belongs to another target model")
    inputs = dict(
        protocol=protocol_identity(),
        weights=weight_identity,
        target=target["fingerprint"],
        outer=outer,
        horizon=7168 if long_panel else (horizon or 1024),
        limit_per_source=limit_per_source,
        scorer=file_hash(hf_rkl.__file__),
        adapter=file_hash(__file__),
        precision="FP32 learner master/head; BF16 backbone autocast; native full-head RKL",
    )
    if complete(out, inputs):
        return
    if out.exists():
        raise ValueError("Partial scoring output exists; retain it and use a distinct attempt")
    out.mkdir(parents=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = hf_models.load(SimpleNamespace(directory=Path(asset["directory"])), path=weights, trainable=True)
    model.requires_grad_(False)
    model.eval()
    if file_hash(cache / "head.pt") != target["head_sha256"]:
        raise ValueError("Teacher head cache changed")
    head = torch.load(cache / "head.pt", map_location="cuda", weights_only=True).float()
    all_results = []
    for source in ("S0", pair):
        raw, manifest = load_bank(banks / source)
        if manifest["bank_sha256"] != target["inputs"]["banks"][source]:
            raise ValueError("Teacher cache and scoring bank differ")
        rows = fixed_views(raw, inputs["horizon"], LONG_GROUPS if long_panel else None)
        if limit_per_source is not None:
            if limit_per_source < 1:
                raise ValueError("Positive readout subset size required")
            rows = rows[:limit_per_source]
        original_rows = {r["id"]: r for r in raw}
        states = {}
        for row in rows:
            name = f"{source}/{row['id']:04d}.pt"
            if file_hash(cache / name) != target["row_files"][name]:
                raise ValueError("Teacher hidden tensor changed")
            value = torch.load(cache / name, map_location="cpu", weights_only=True)
            if (
                value["row_identity"] != fingerprint(original_rows[row["id"]])
                or value["teacher"] != target["inputs"]["teacher"]
            ):
                raise ValueError("Teacher cached row identity changed")
            states[row["id"]] = value["hidden"][: len(row["response_ids"])]
        bands = (
            [("all", 0, 7168), ("0_1024", 0, 1024), ("1024_4096", 1024, 4096), ("4096_7168", 4096, 7168)]
            if long_panel
            else None
        )
        results = hf_rkl.score(
            model,
            head,
            states,
            rows,
            outer * 16,
            SimpleNamespace(horizon=inputs["horizon"]),
            SimpleNamespace(microbatch=1, token_chunk=512),
            151643,
            bands=bands,
        )
        finite(results)
        all_results.extend(results)
    dump(out / "per-question.json", all_results)
    receipt(
        out,
        inputs,
        questions=len(all_results),
        metrics_sha256=file_hash(out / "per-question.json"),
        optimizer_updates=0,
        source_file_sha256=file_hash(__file__),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("cache-teacher", "score"):
        p = commands.add_parser(name)
        p.add_argument("--assets", type=Path, required=True)
        p.add_argument("--banks", type=Path, required=True)
        p.add_argument("--out", type=Path, required=True)
        if name == "cache-teacher":
            p.add_argument("--pair", choices=["J", "P"], required=True)
        else:
            p.add_argument("--cache", type=Path, required=True)
            p.add_argument("--weights", type=Path, required=True)
            p.add_argument("--outer", type=int, required=True)
            p.add_argument("--long-panel", action="store_true")
            p.add_argument("--limit-per-source", type=int)
            p.add_argument("--horizon", type=int)
    args = parser.parse_args()
    require_allocated_gpu()
    from examples.mirroropd.benchmarks.fusion.fingerprint import source_fingerprint

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tag = args.out.parent / f"{args.out.name}-run-{time.time_ns()}"
    dump(
        str(tag) + "-source.json",
        dict(
            source=source_fingerprint(),
            command=vars(args) | {key: str(value) for key, value in vars(args).items() if isinstance(value, Path)},
        ),
    )
    monitor = Path(__file__).resolve().parents[3] / "benchmarks/fusion/monitor.py"
    observer = subprocess.Popen(
        [sys.executable, str(monitor), "--parent", str(os.getpid()), "--out", str(tag) + "-resources.jsonl"]
    )
    assert observer.pid > 0
    if args.command == "cache-teacher":
        cache_teacher(args.assets, args.banks, args.pair, args.out)
    else:
        score(
            args.assets,
            args.banks,
            args.cache,
            args.weights,
            args.out,
            args.outer,
            args.long_panel,
            args.limit_per_source,
            args.horizon,
        )


if __name__ == "__main__":
    main()
