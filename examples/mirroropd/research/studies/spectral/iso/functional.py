# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""In-memory full-weight spectrum swaps, matched native risk and task readouts."""

import argparse
import concurrent.futures
import multiprocessing
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from examples.mirroropd.research.core.io import dump, read
from examples.mirroropd.research.core.math_grading import grade, init_grader

from ...teacher_difference.runtime import collect_memory, load_model
from ..absorption.banks import score
from ..absorption.protocol import RunContext
from ..scan import thin_svd
from ..weights import DEFAULT_SCOPE, TensorCheckpoint, tensor_digest
from .math import spectrum_edit


def generate(context, model, tokenizer, prompts, output, cap=4096):
    records = []
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=2, mp_context=multiprocessing.get_context("spawn"), initializer=init_grader
    ) as pool:
        for start in range(0, len(prompts), 4):
            context.check_stop()
            batch = prompts[start : start + 4]
            texts = [
                tokenizer.apply_chat_template(
                    p["messages"], tokenize=False, add_generation_prompt=True, **p.get("template_options", {})
                )
                for p in batch
            ]
            encoded = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                answers = model.generate(
                    **encoded, do_sample=False, max_new_tokens=cap, pad_token_id=tokenizer.pad_token_id, use_cache=True
                )[:, encoded.input_ids.shape[1] :]
            pending = []
            for prompt, answer in zip(batch, answers.cpu().tolist(), strict=True):
                eos = model.generation_config.eos_token_id or tokenizer.eos_token_id
                stops = set(eos if isinstance(eos, list) else [eos])
                for end, token in enumerate(answer):
                    if token in stops:
                        answer = answer[: end + 1]
                        break
                item = dict(
                    prompt_id=prompt["prompt_id"],
                    data_source=prompt["data_source"],
                    gts=prompt["gts"],
                    response_ids=answer,
                    output=tokenizer.decode(answer, skip_special_tokens=True),
                    response_length=len(answer),
                    truncated=len(answer) == cap,
                )
                pending.append(pool.submit(grade, item))
            records.extend(f.result() for f in pending)
            dump(output.with_suffix(".progress.json"), dict(completed=len(records), expected=len(prompts)))
    dump(
        output, dict(records=records, cap=cap, protocol="HF FP32-master BF16-autocast greedy; paired prompt identities")
    )


def run(manifest, job_id, output, stop, smoke=False, scale_controls_only=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    context = RunContext(output, stop_paths=(Path(stop),))
    job = next(j for j in read(manifest)["jobs"] if j["job_id"] == job_id)
    evidence = Path(job["evidence_attempt"])
    banks = read(evidence / "banks.json")
    if smoke:
        banks = {origin: rows[:1] for origin, rows in banks.items()}
    reference, head = torch.load(evidence / "reference.pt", weights_only=True, mmap=True)
    head = head.cuda()
    base = TensorCheckpoint(job["base"])
    tokenizer = AutoTokenizer.from_pretrained(job["base"], local_files_only=True, trust_remote_code=True)
    prompts = read(job["readout_prompts"])
    if job["family"] == "Qwen":
        prompts = [dict(p, template_options=dict(enable_thinking="nonthinking" not in job["run_id"])) for p in prompts]
    started = time.time()
    for role, path in [("teacher", job["teacher_path"]), ("student", job["endpoint"])]:
        model = load_model(path, trainable=True, device="cuda").eval().requires_grad_(False)
        target = TensorCheckpoint(path)
        keys = base.selected(DEFAULT_SCOPE)
        if smoke:
            keys = [k for k in keys if ".layers.0." in k and ".k_proj." in k]
        # Two CPU FP32 edit tensors per selected matrix. No frames/models written.
        edits = {}
        receipts = []
        for index, key in enumerate(keys):
            context.check_stop()
            original = target.get(key)
            wb = base.get(key).cuda().double()
            wt = original.cuda().double()
            u0, s0, vh0 = thin_svd(wb, driver="gesvd", transpose_wide=True)
            u1, s1, vh1 = thin_svd(wt, driver="gesvd", transpose_wide=True)
            error = float(((u1 * s1) @ vh1 - wt).norm() / wt.norm())
            if error > 1e-10:
                raise ValueError(f"SVD identity gate failed: {key}: {error}")
            ratio = float(s1.norm() / s0.norm())
            edits[key] = (
                ((u1 * (s0 - s1)) @ vh1).float().cpu(),
                ((u0 * s1) @ vh0 - wt).float().cpu() if not scale_controls_only else None,
                ratio,
            )
            receipts.append(
                dict(
                    tensor=key,
                    shape=list(wt.shape),
                    target_sha256=tensor_digest(original),
                    base_sha256=tensor_digest(base.get(key)),
                    svd_identity_error=error,
                    restoration_norm=float(edits[key][0].double().norm()),
                    reverse_norm=float(edits[key][1].double().norm()) if edits[key][1] is not None else None,
                    target_to_base_frobenius_ratio=ratio,
                )
            )
            dump(output / f"{role}-prepare-progress.json", dict(completed=index + 1, total=len(keys)))
            del wb, wt, u0, s0, vh0, u1, s1, vh1
        dump(output / f"{role}-edits.json", receipts)
        conditions = [("restore", a) for a in (1.0, 0.0, 0.25, 0.5, 0.75)] + [("reverse", 1.0)]
        if scale_controls_only:
            conditions = [("restore", 1.0), ("norm-only", 1.0), ("shape-only", 1.0)]
        for kind, alpha in conditions:
            context.check_stop()
            with torch.no_grad():
                for key, (restore, reverse, ratio) in edits.items():
                    original = target.get(key).cuda().float()
                    model.get_parameter(key).copy_(spectrum_edit(original, restore, reverse, ratio, kind, alpha))
            label = f"{role}-{kind}-alpha-{alpha}"
            result = score(
                context,
                model,
                banks,
                reference,
                head,
                output / "scores" / f"{label}.json",
                dict(role=role, kind=kind, alpha=alpha),
            )
            if role == "student" and kind == "restore" and alpha == 1:
                baseline = read(evidence / "scores/endpoint.json")
                expected = {(r["origin"], r["row_id"]): r["rkl"] for r in baseline["rows"]}
                error = max(abs(r["rkl"] - expected[r["origin"], r["row_id"]]) for r in result["rows"])
                dump(output / "no-edit-gate.json", dict(max_rkl_error=error, passed=error <= 1e-6))
                if error > 1e-6:
                    raise ValueError(f"Endpoint no-edit gate failed: {error}")
            if role == "teacher" and kind == "restore" and alpha == 1:
                error = max(abs(row["rkl"]) for row in result["rows"])
                if error > 1e-6:
                    raise ValueError(f"Teacher-self matched execution gate failed: {error}")
            if not smoke and not scale_controls_only and (kind == "reverse" or alpha in (0.0, 1.0)):
                generate(context, model, tokenizer, prompts, output / "generation" / f"{label}.json")
            dump(output / "progress.json", dict(condition=label, wall_seconds=time.time() - started))
        del edits, model
        collect_memory()
    dump(
        output / "complete.json",
        dict(job_id=job_id, smoke=smoke, scale_controls_only=scale_controls_only, wall_seconds=time.time() - started),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--job-id", type=int, required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--stop", required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--scale-controls-only", action="store_true")
    args = vars(p.parse_args())
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    run(**args)


if __name__ == "__main__":
    main()
