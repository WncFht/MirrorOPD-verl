# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Shared-base expert merging: ISO, paper ablations, task arithmetic and averaging."""

import argparse
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from examples.mirroropd.research.core.io import dump, read

from ...teacher_difference.runtime import collect_memory, load_model
from ..absorption.banks import score
from ..absorption.protocol import RunContext
from ..scan import thin_svd
from ..weights import TensorCheckpoint, tensor_digest
from .functional import generate
from .math import compose


def run(manifest, job_ids, output, stop, smoke=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    context = RunContext(output, stop_paths=(Path(stop),))
    jobs = [next(j for j in read(manifest)["jobs"] if j["job_id"] == i) for i in job_ids]
    if len({j["base"] for j in jobs}) != 1:
        raise ValueError("ISO merging requires exactly the same base checkpoint")
    stores = [TensorCheckpoint(jobs[0]["base"]), *[TensorCheckpoint(j["teacher_path"]) for j in jobs]]
    model = load_model(jobs[0]["base"], trainable=True, device="cuda").eval().requires_grad_(False)
    parameters = dict(model.named_parameters())
    keys = list(parameters)
    if smoke:
        keys = [key for key in keys if ".layers.13.self_attn.k_proj." in key]
    edits = {}
    started = time.time()
    for index, key in enumerate(keys):
        context.check_stop()
        values = [store.get(key).cuda().double() for store in stores]
        if len(values[0].shape) == 2:
            frames = [thin_svd(w, driver="gesvd", transpose_wide=True) for w in values]
            merged = {}
            facts = {}
            for label, keep, uniform in [("iso", 0.9, False), ("iso-uniform", 0.9, True), ("iso-keep-all", 1.0, False)]:
                merged[label], facts[label] = compose(frames[0], frames[1:], keep=keep, uniform=uniform)
            initial_s = frames[0][1]
            err = float((torch.linalg.svdvals(merged["iso"], driver="gesvd") - initial_s).norm() / initial_s.norm())
            if err > 1e-8:
                raise ValueError(f"Merged base-spectrum gate failed: {key}: {err}")
            facts["spectrum_relative_error"] = err
        else:
            merged = {label: sum(values[1:]) / len(jobs) for label in ["iso", "iso-uniform", "iso-keep-all"]}
            facts = dict(vector_policy="mean expert task vectors")
        merged["task-arithmetic"] = values[0] + sum(w - values[0] for w in values[1:])
        merged["weight-average"] = sum(values[1:]) / len(jobs)
        edits[key] = {label: value.float().cpu() for label, value in merged.items()}
        dump(
            output / "matrices" / f"{index:05d}.json",
            dict(
                tensor=key, shape=list(values[0].shape), inputs=[tensor_digest(s.get(key)) for s in stores], facts=facts
            ),
        )
        dump(output / "prepare-progress.json", dict(completed=index + 1, total=len(keys)))
        del values, merged
    tokenizer = AutoTokenizer.from_pretrained(jobs[0]["base"], local_files_only=True, trust_remote_code=True)
    prompts = read(jobs[0]["readout_prompts"])
    if jobs[0]["family"] == "Qwen":
        prompts = [
            dict(p, template_options=dict(enable_thinking="nonthinking" not in jobs[0]["run_id"])) for p in prompts
        ]
    for label in ["iso", "iso-uniform", "iso-keep-all", "task-arithmetic", "weight-average"]:
        with torch.no_grad():
            for key, variants in edits.items():
                parameters[key].copy_(variants[label].cuda())
        for job in jobs:
            evidence = Path(job["evidence_attempt"])
            banks = read(evidence / "banks.json")
            if smoke:
                banks = {k: v[:1] for k, v in banks.items()}
            reference, head = torch.load(evidence / "reference.pt", weights_only=True, mmap=True)
            head = head.cuda()
            score(
                context,
                model,
                banks,
                reference,
                head,
                output / "scores" / f"{label}-expert-{job['job_id']}.json",
                dict(merge=label, reference_expert=job["job_id"]),
            )
            del reference, head
        if not smoke:
            generate(context, model, tokenizer, prompts, output / "generation" / f"{label}.json")
        dump(output / "progress.json", dict(condition=label, wall_seconds=time.time() - started))
    del model, edits
    collect_memory()
    dump(output / "complete.json", dict(job_ids=job_ids, smoke=smoke, wall_seconds=time.time() - started))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["manifest", "output", "stop"]:
        p.add_argument("--" + name, required=True)
    p.add_argument("--job-ids", nargs="+", type=int, required=True)
    p.add_argument("--smoke", action="store_true")
    torch.set_num_threads(2)
    run(**vars(p.parse_args()))


if __name__ == "__main__":
    main()
