# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""GPU acceptance benchmark: exact all-matrix edits and unchanged full-response probes."""

import argparse
import json
import time
from pathlib import Path

import torch

from verl.trainer.distillation.distribution import DistributionSpec

from ..spectral.scan import singular_values
from ..spectral.update_dynamics import atomic_json
from ..spectral.weights import TensorCheckpoint
from .spectral import SCOPE
from .spectrum_bank_acceleration import ReferenceCache, gram_restore
from .spectrum_bank_worker import load_model, read_job, reference_cache, restore_sigma0, score_row


def timed(call):
    torch.cuda.synchronize()
    started = time.perf_counter()
    result = call()
    torch.cuda.synchronize()
    return result, time.perf_counter() - started


@torch.inference_mode()
def benchmark(job, output, device):
    output.mkdir(parents=True, exist_ok=False)
    torch.cuda.set_device(device)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    # Shortest and longest full responses in every bank/domain; no prefix truncation.
    rows = []
    for bank in ("public_initial", "public_teacher"):
        for domain in ("math", "code", "if"):
            group = sorted(
                (r for r in job["rows"] if r["bank"] == bank and r["domain"] == domain),
                key=lambda r: sum(r["response_mask"]),
            )
            rows.extend((group[0], group[-1]))
    cache = reference_cache(job, output, device, rows)
    references = ReferenceCache(cache, device, max_bytes=24 * 1024**3)
    point = job["points"][-1]
    model = load_model(point["checkpoint"], device)
    space = DistributionSpec.parse(job["distribution"]).compile(
        len(model.get_output_embeddings().weight), len(references.head)
    )

    def scores(chunk, resident):
        values = []
        for row in rows:
            if resident:
                ht = references.get(row)
            else:
                from verl.trainer.evaluation.artifacts import identity

                ht = torch.load(cache / f"{identity(row['id'])}.pt", weights_only=True, map_location="cpu")["hidden"]
            value = score_row(model, row, ht, references.head, space, chunk)
            values.append(value["rkl_sum"] / value["tokens"])
        return torch.tensor(values, dtype=torch.float64)

    scores(256, True)  # Warm up CUDA kernels and resident cache before timing.
    baseline, _ = timed(lambda: scores(256, True))
    chunks = []
    for chunk in (256, 512, 1024, 2048):
        for resident in (False, True):
            durations, errors = [], []
            for _ in range(2):
                values, seconds = timed(lambda chunk=chunk, resident=resident: scores(chunk, resident))
                durations.append(seconds)
                errors.append(float((values - baseline).abs().max()))
            chunks.append(
                dict(
                    chunk=chunk,
                    resident=resident,
                    seconds=durations,
                    max_rkl_error=max(errors),
                    passed=max(errors) <= 1e-6,
                )
            )
    atomic_json(output / "scoring.json", dict(pair=job["pair"], rows=len(rows), chunks=chunks))
    base, target = TensorCheckpoint(job["base"]), TensorCheckpoint(point["checkpoint"])
    edits, candidates = [], {}
    for key in base.selected(SCOPE):
        w = target.get(key).to(device=device, dtype=torch.float64)
        s0 = singular_values(base.get(key).to(device=device, dtype=torch.float64), driver="gesvd", transpose_wide=True)
        reference, svd_seconds = timed(lambda w=w, s0=s0: restore_sigma0(w, s0, driver="gesvd"))
        candidate, gram_seconds = timed(
            lambda w=w, s0=s0: gram_restore(w, s0, fallback=lambda a, b: restore_sigma0(a, b, driver="gesvd"))
        )
        ref, receipt = reference
        new, gram_receipt = candidate
        param = model.get_parameter(key)
        ref_native, new_native = ref.to(param.dtype), new.to(param.dtype)
        edits.append(
            dict(
                tensor=key,
                shape=list(w.shape),
                svd_seconds=svd_seconds,
                gram_seconds=gram_seconds,
                fp32_max_abs_error=float((ref - new).abs().max()),
                native_different_elements=int((ref_native != new_native).sum()),
                native_max_abs_error=float((ref_native.float() - new_native.float()).abs().max()),
                reference=receipt,
                candidate=gram_receipt,
            )
        )
        param.copy_(ref_native)
        candidates[key] = new_native.cpu()
        atomic_json(
            output / "progress.json",
            dict(phase="edit-comparison", complete=len(edits), total=len(base.selected(SCOPE))),
        )
    edited_svd, score_svd_seconds = timed(lambda: scores(256, True))
    for key, value in candidates.items():
        model.get_parameter(key).copy_(value)
    edited_gram, score_gram_seconds = timed(lambda: scores(256, True))
    error = float((edited_svd - edited_gram).abs().max())
    # Require both tensor agreement and end-to-end score agreement.
    passed = error <= 1e-6 and all(e["fp32_max_abs_error"] <= 1e-6 for e in edits)
    result = dict(
        pair=job["pair"],
        step=point["step"],
        rows=len(rows),
        matrix_count=len(edits),
        edits=edits,
        chunks=chunks,
        svd_seconds=sum(e["svd_seconds"] for e in edits),
        gram_seconds=sum(e["gram_seconds"] for e in edits),
        native_different_elements=sum(e["native_different_elements"] for e in edits),
        edited_rkl_max_abs_error=error,
        svd_rkl=edited_svd.tolist(),
        gram_rkl=edited_gram.tolist(),
        score_svd_seconds=score_svd_seconds,
        score_gram_seconds=score_gram_seconds,
        passed=passed,
        reference_cache=references.stats(),
    )
    atomic_json(output / "complete.json", result)
    if not passed:
        raise ValueError("Candidate failed exact-edit acceptance; preserve benchmark and keep SVD")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--pair", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    item = json.loads(args.plan.read_text())["pairs"][args.pair]
    benchmark(read_job(item["input"]), args.output, args.device)


if __name__ == "__main__":
    main()
