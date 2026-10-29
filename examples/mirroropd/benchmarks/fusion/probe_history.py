# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Probe full distributions after replaying real, long decode histories into both KV caches."""

import argparse
import json
import os
from pathlib import Path

import torch
from verify_numerics import TOLERANCES

from verl.experimental.rollout_policy.hf_decoder import DecodeRequest, HFFusionDecoder
from verl.experimental.rollout_policy.sampling import SamplingConfig, fusion_log_probs, validate_tokenizers


class HistoryProbe(HFFusionDecoder):
    def set_replay(self, requests, histories, points, config):
        self.replay_requests = requests
        self.histories = histories
        self.points = points
        self.sampling = config
        self.checks = []

    def _select_tokens(self, log_probs, generators, step):
        ids = [
            self.histories[r.request_id][step] if step < len(self.histories[r.request_id]) else self.pad_id
            for r in self.replay_requests
        ]
        tokens = torch.tensor(ids, device=log_probs.device)
        return tokens, log_probs.gather(-1, tokens[:, None]).squeeze(-1)

    def _forward_models(self, models, ids, mask, positions, caches, extras, step):
        outputs = super()._forward_models(models, ids, mask, positions, caches, extras, step)
        if step not in self.points:
            return outputs
        cached = {name: out.logits[:, -1] for name, out, _ in outputs}
        device = ids.device
        legal = list(self.sampling.legal_ids)
        for row, request in enumerate(self.replay_requests):
            history = self.histories[request.request_id]
            if step >= len(history):
                continue
            prefix = list(request.prompt_ids) + history[:step]
            inputs = torch.tensor([prefix], device=device)
            reference = {}
            check = {"request_id": request.request_id, "step": step, "prefix_length": len(prefix)}
            # One row at a time bounds reference-prefill activation memory for
            # the 30B MoE, and also checks independence from batch padding.
            for name, model in models:
                reference[name] = model(
                    input_ids=inputs,
                    attention_mask=torch.ones_like(inputs),
                    position_ids=torch.arange(inputs.shape[1], device=device)[None],
                    use_cache=False,
                    logits_to_keep=1,
                ).logits[:, -1]
                check[name + "_logits_abs"] = (
                    (cached[name][row : row + 1, legal] - reference[name][:, legal]).abs().max().item()
                )
            actual = fusion_log_probs(
                cached["student"][row : row + 1], cached["teacher"][row : row + 1], 0.5, self.sampling
            )
            expected = fusion_log_probs(reference["student"], reference["teacher"], 0.5, self.sampling)
            check["probability_abs"] = (actual.exp() - expected.exp()).abs().max().item()
            selected = history[step]
            check["selected_logprob_abs"] = abs(actual[0, selected].item() - expected[0, selected].item())
            self.checks.append(check)
        print(json.dumps({"device": device.index, "probe_step": step, "checks": len(self.checks)}), flush=True)
        return outputs


def worker(rank, source, args):
    import pyarrow.parquet as pq
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from verl.experimental.rollout_policy.precision import configure_precision
    from verl.utils.tokenizer.continuous_token_wiring import create_continuous_token_builder

    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    cfg = source["args"]
    models = [
        AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", local_files_only=True
        )
        .eval()
        .to(rank)
        for path in (cfg["student"], cfg["teacher"])
    ]
    for model in models:
        configure_precision(model, args.precision)
    tokenizers = [
        AutoTokenizer.from_pretrained(path, local_files_only=True) for path in (cfg["student"], cfg["teacher"])
    ]
    legal = validate_tokenizers(*tokenizers, models[0].config.vocab_size, models[1].config.vocab_size)
    tokenizer = AutoTokenizer.from_pretrained(cfg.get("prompt_model") or cfg["student"], local_files_only=True)
    builder = create_continuous_token_builder(
        tokenizer,
        model_family="qwen3",
        hf_model_type=models[0].config.model_type,
        chat_template_kwargs=source["chat_template_kwargs"],
    )
    rows = pq.read_table(cfg["data"], columns=["prompt"]).slice(0, cfg["batch"]).to_pylist()
    requests = [
        DecodeRequest(str(i), tuple(builder.build_initial_tokens(row["prompt"])), cfg["seed"] + i)
        for i, row in enumerate(rows)
    ]
    histories = {}
    for path in sorted(args.trajectory.glob("rank-*.jsonl")):
        for line in path.open():
            value = json.loads(line)
            if value["kind"] == "measurement" and value["repeat"] == 0 and value["compact"] == args.source_compact:
                histories.update({item["request_id"]: item["token_ids"] for item in value["results"]})
                break
    if len(histories) != cfg["batch"]:
        raise ValueError("Incomplete source trajectory batch")
    config = SamplingConfig(max_tokens=max(args.points) + 1, legal_ids=legal)
    decoder = HistoryProbe(
        *models,
        pad_id=tokenizer.pad_token_id,
        preallocate_kv=args.executor == "optimized",
        flash_kv=True,
        flash_kv_splits=1,
        cuda_graph=args.executor == "optimized",
    )
    ordered = sorted(requests, key=lambda item: len(item.prompt_ids)) if args.source_compact else requests
    microbatch = cfg["microbatch"][0]
    chunks = [ordered[i : i + microbatch] for i in range(0, len(ordered), microbatch)]
    records = []
    for chunk in chunks[rank :: args.replay_replicas]:
        decoder.set_replay(chunk, histories, set(args.points), config)
        decoder.generate(chunk, 0.5, config, deadline_seconds=7200)
        records.extend(decoder.checks)
    failures = [
        r
        for r in records
        if any(
            r[key] > threshold
            for key, threshold in (
                ("student_logits_abs", TOLERANCES["logits_abs"]),
                ("teacher_logits_abs", TOLERANCES["logits_abs"]),
                ("probability_abs", TOLERANCES["probability_abs"]),
                ("selected_logprob_abs", TOLERANCES["selected_logprob_abs"]),
            )
        )
    ]
    (args.out / f"rank-{rank}.json").write_text(json.dumps({"checks": records, "failures": failures}, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--executor", choices=["optimized", "reference"], default="optimized")
    parser.add_argument("--source-compact", type=int, choices=[0, 1, 2], default=2)
    parser.add_argument("--precision", choices=["stable", "stable_moe", "stable_moe_attention"], default="stable")
    parser.add_argument("--points", type=int, nargs="+", default=[0, 1, 2, 128, 512, 1024, 2048, 4096, 6144, 7167])
    args = parser.parse_args()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible not in {"6", "7", "6,7"}:
        parser.error("Allocate CUDA_VISIBLE_DEVICES=6, 7, or 6,7 through run_fusion_local_fsdp2.sh")
    args.replay_replicas = len(visible.split(","))
    if args.out.exists():
        parser.error("Probe output must be a new directory")
    source = json.loads((args.trajectory / "rank-0.jsonl").open().readline())
    if source["args"]["layout"] != "colocated" or source["args"]["alpha"] != 0.5:
        parser.error("Probe requires the colocated alpha=0.5 acceptance workload")
    if len(source["args"]["microbatch"]) != 1:
        parser.error("Probe requires one frozen microbatch setting; replay never changes its groups")
    if args.source_compact not in source["args"]["compact"]:
        parser.error("Requested source compaction is absent from the trajectory run")
    if any(step < 0 or step >= source["args"]["cap"] for step in args.points):
        parser.error("Probe points must lie inside the response cap")
    args.out.mkdir(parents=True)
    import torch.multiprocessing as mp
    from fingerprint import source_fingerprint

    executor_source = source_fingerprint()
    mp.spawn(worker, args=(source, args), nprocs=args.replay_replicas, join=True)
    reports = [json.loads(path.read_text()) for path in sorted(args.out.glob("rank-*.json"))]
    result = {
        "trajectory": str(args.trajectory),
        "source_replicas": source["args"]["gpus"],
        "replay_replicas": args.replay_replicas,
        "physical_gpus": [int(device) for device in visible.split(",")],
        "executor": args.executor,
        "source_compact": args.source_compact,
        "source": executor_source,
        "precision": args.precision,
        "tolerances": {key: TOLERANCES[key] for key in ("logits_abs", "probability_abs", "selected_logprob_abs")},
        "checks": sum(len(row["checks"]) for row in reports),
        "failures": [x for row in reports for x in row["failures"]],
    }
    result["passed"] = len(reports) == args.replay_replicas and not result["failures"]
    (args.out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
