# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Exact full-weight sigma0 edits and paired full-response bank RKL, without training."""

import gzip
import json
import math
import shutil
import time
from pathlib import Path

import torch

from examples.mirroropd.research.backends.hf_models import load
from examples.mirroropd.research.studies.spectral.scan import singular_values, thin_svd
from examples.mirroropd.research.studies.spectral.update_dynamics import atomic_json
from examples.mirroropd.research.studies.spectral.weights import TensorCheckpoint, tensor_digest
from verl.trainer.distillation.distribution import DistributionSpec
from verl.trainer.distillation.evaluation import score_distribution_rows
from verl.trainer.evaluation.artifacts import identity
from verl.trainer.evaluation.reporting import evaluation_metrics
from verl.utils.metric.position import position_statistics, relative_position_bins

from ..teacher_difference.runtime import checkpoint_identity
from .spectral import SCOPE
from .spectrum_bank_acceleration import ReferenceCache, gram_restore


def check_stop(output):
    if any((p / "STOP").exists() for p in (output, output.parent)):
        raise InterruptedError("Spectrum bank STOP: preserve completed row/condition artifacts")


def atomic_tensor(path, payload):
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


@torch.inference_mode()
def hidden(model, row):
    device = model.get_input_embeddings().weight.device
    prompt, response = row["prompt_ids"], row["response_ids"]
    ids = torch.tensor([prompt + response[:-1]], device=device)
    output = model.get_decoder()(input_ids=ids, use_cache=False).last_hidden_state[0, len(prompt) - 1 :]
    mask = torch.tensor(row["response_mask"], device=device, dtype=torch.bool)
    if len(output) != len(mask):
        raise ValueError("Model response prediction positions do not match fixed bank")
    return output[mask].contiguous()


def load_model(path, device):
    model = load(path=path, device=device, attention="flash_attention_2", trainable=False, trust_remote_code=True)
    model.requires_grad_(False)
    return model


@torch.inference_mode()
def restore_sigma0(original, initial_singular, *, driver=None):
    """Full W intervention; delta form preserves the exact target at alpha=1."""
    target = original.double()
    u, s, vh = thin_svd(target, driver=driver, transpose_wide=True)
    if s.shape != initial_singular.shape:
        raise ValueError("Base/target singular value dimensions differ")
    error = float(((u * s) @ vh - target).norm() / target.norm())
    if not math.isfinite(error) or error > 1e-10:
        raise ValueError("Full-weight SVD reconstruction gate failed")
    correction = (u * (initial_singular - s)) @ vh
    edited = (target + correction).float()
    return edited, dict(
        reconstruction_relative_error=error,
        correction_norm=float(correction.norm()),
        original_norm=float(target.norm()),
        edited_norm=float(edited.double().norm()),
    )


def reference_cache(job, output, device, rows):
    """One teacher forward per fixed trajectory; no token-by-vocabulary disk cache."""
    cache = output / "teacher-cache"
    cache.mkdir(exist_ok=True)
    binding = identity(dict(teacher=job["teacher"], banks=job["banks"], rows=rows, execution=job["execution"]))
    marker = cache / "identity.json"
    if marker.exists() and json.loads(marker.read_text())["identity"] != binding:
        raise ValueError("Teacher hidden cache identity mismatch")
    atomic_json(marker, dict(identity=binding, execution=job["execution"]))
    missing = [r for r in rows if not (cache / f"{identity(r['id'])}.pt").exists()]
    if not missing and (cache / "head.pt").exists():
        return cache
    model = load_model(job["teacher"], device)
    started = time.monotonic()
    atomic_tensor(cache / "head.pt", model.get_output_embeddings().weight.detach().cpu())
    for index, row in enumerate(missing):
        check_stop(output)
        h = hidden(model, row).cpu()
        atomic_tensor(cache / f"{identity(row['id'])}.pt", dict(row_identity=identity(row), hidden=h))
        atomic_json(
            output / "progress.json",
            dict(
                phase="teacher-cache",
                cached=len(rows) - len(missing) + index + 1,
                rows=len(rows),
                seconds=time.monotonic() - started,
            ),
        )
    del model
    torch.cuda.empty_cache()
    return cache


def score_row(model, row, teacher, head, space, token_chunk):
    hs = hidden(model, row)
    ht = teacher.to(device=hs.device)
    if len(ht) != len(hs):
        raise ValueError("Teacher cache response mask differs")
    scores = score_distribution_rows(
        hs, model.get_output_embeddings().weight, ht, head, token_chunk=token_chunk, distribution=space
    )
    values = scores["rkl"]
    if not torch.isfinite(values).all() or float(values.min()) < -1e-6:
        raise ValueError("Nonfinite or negative bank RKL")
    positions = torch.tensor([i for i, keep in enumerate(row["response_mask"]) if keep], device=hs.device)
    bins = relative_position_bins(positions, torch.full_like(positions, len(row["response_ids"])))
    sums, counts = position_statistics(values, bins)
    result = {k: row[k] for k in ("id", "bank", "source_id", "domain", "dataset", "sample_index", "trajectory_id")}
    result.update(
        rkl_sum=float(values.double().sum()),
        tokens=len(values),
        rkl_position_sum=sums.cpu().tolist(),
        rkl_position_count=counts.cpu().tolist(),
        measure_id=identity(dict(space=space.metadata, engine="hf-bf16-fa2-mainline-rkl-v1")),
    )
    for name in ("uncoarsened_rkl", "student_excluded_mass", "teacher_excluded_mass"):
        if name in scores:
            result[name + "_sum"] = float(scores[name].double().sum())
    return result


def score_condition(model, rows, cache, destination, output, distribution, *, token_chunk, references=None):
    if destination.exists():
        result = json.loads(destination.read_text())
        if result["row_identity"] != identity(rows):
            raise ValueError("Completed condition row identity changed")
        return result
    head = (
        references.head if references else torch.load(cache / "head.pt", weights_only=True, map_location=model.device)
    )
    space = DistributionSpec.parse(distribution).compile(len(model.get_output_embeddings().weight), len(head))
    journal = destination.with_suffix(".jsonl")
    scored = []
    if journal.exists():
        # Interrupted appends may leave one partial line; completed rows remain immutable.
        payload = journal.read_bytes()
        cut = payload.rfind(b"\n") + 1
        journal.write_bytes(payload[:cut])
        scored = [json.loads(line) for line in payload[:cut].splitlines()]
        if [r["id"] for r in scored] != [r["id"] for r in rows[: len(scored)]]:
            raise ValueError("Partial condition row order changed")
    started = time.monotonic()
    with journal.open("a") as stream:
        for row in rows[len(scored) :]:
            check_stop(output)
            if references:
                teacher = references.get(row)
            else:
                reference = torch.load(cache / f"{identity(row['id'])}.pt", weights_only=True, map_location="cpu")
                if reference["row_identity"] != identity(row):
                    raise ValueError("Teacher cache row changed")
                teacher = reference["hidden"]
            value = score_row(model, row, teacher, head, space, token_chunk)
            stream.write(json.dumps(value, allow_nan=False) + "\n")
            stream.flush()
            scored.append(value)
            atomic_json(
                output / "progress.json",
                dict(phase=destination.stem, completed=len(scored), rows=len(rows), seconds=time.monotonic() - started),
            )
    result = dict(
        rows=scored,
        row_identity=identity(rows),
        metrics=evaluation_metrics(scored, baselines={}),
        seconds=time.monotonic() - started,
    )
    atomic_json(destination, result)
    del head
    return result


def paired_summary(original, edited, training):
    before, after = original["rows"], edited["rows"]
    expected = {r["id"]: r for r in training}
    if [(r["id"], r["tokens"]) for r in before] != [(r["id"], r["tokens"]) for r in after]:
        raise ValueError("Paired spectrum intervention changed row masks")
    rows = []
    native_errors = []
    for a, b in zip(before, after, strict=True):
        ref = expected[a["id"]]
        if a["tokens"] != ref["tokens"]:
            raise ValueError("Original training response mask differs")
        native_errors.append(abs(a["rkl_sum"] - ref["rkl_sum"]) / a["tokens"])
        rows.append(
            dict(
                a,
                rkl_sum=b["rkl_sum"] - a["rkl_sum"],
                rkl_position_sum=[y - x for x, y in zip(a["rkl_position_sum"], b["rkl_position_sum"], strict=True)],
                **{k: None for k in ("uncoarsened_rkl_sum", "student_excluded_mass_sum", "teacher_excluded_mass_sum")},
            )
        )
    return dict(
        original=original["metrics"],
        restored=edited["metrics"],
        delta=evaluation_metrics(rows, baselines={}),
        training_comparison=dict(
            max_question_abs_error=max(native_errors),
            mean_question_abs_error=sum(native_errors) / len(native_errors),
            exact_native_gate=all(
                math.isclose(
                    a["rkl_sum"] / a["tokens"], expected[a["id"]]["rkl_sum"] / a["tokens"], rel_tol=1e-4, abs_tol=1e-6
                )
                for a in before
            ),
        ),
    )


@torch.inference_mode()
def run_pair(
    job,
    output,
    device,
    *,
    token_chunk=256,
    smoke=False,
    keep_cache=False,
    probe_only=False,
    edit_method="svd",
    reference_cache_mib=0,
):
    if edit_method not in ("svd", "gram") or reference_cache_mib < 0 or token_chunk < 1:
        raise ValueError("Invalid edit method, reference cache size or scoring chunk")
    if probe_only and not smoke:
        raise ValueError("Repeated-forward probe requires explicit smoke coverage")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    source_identities = {
        path: checkpoint_identity(path)
        for path in {job["base"], job["teacher"], *(p["checkpoint"] for p in job["points"])}
    }
    contract = dict(
        job_identity=identity(job),
        token_chunk=token_chunk,
        smoke=smoke,
        execution=job["execution"],
        engine="sigma0-bank-v2",
        probe_only=probe_only,
        checkpoint_files=source_identities,
        edit_method=edit_method,
        reference_cache_mib=reference_cache_mib,
    )
    marker = output / "protocol.json"
    if marker.exists() and json.loads(marker.read_text()) != contract:
        raise ValueError("Pair execution protocol changed")
    atomic_json(marker, contract)
    if (output / "complete.json").exists():
        return json.loads((output / "complete.json").read_text())
    torch.set_num_threads(2)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    rows = job["rows"]
    if smoke:
        rows = [
            next(r for r in rows if r["bank"] == bank and r["domain"] == domain)
            for bank in ("public_initial", "public_teacher")
            for domain in ("math", "code", "if")
        ]
    points = [job["points"][0]] if smoke else job["points"]
    cache = reference_cache(job, output, device, rows)
    references = ReferenceCache(cache, device, max_bytes=reference_cache_mib * 1024**2) if reference_cache_mib else None
    scoring = dict(token_chunk=token_chunk, references=references)
    base = TensorCheckpoint(job["base"])
    base_singular = {}
    summary = []
    for point in points:
        check_stop(output)
        stage = output / f"step-{point['step']:04d}"
        stage.mkdir(exist_ok=True)
        if (stage / "complete.json").exists():
            summary.append(json.loads((stage / "complete.json").read_text()))
            continue
        started = time.monotonic()
        model = load_model(point["checkpoint"], device)
        original = score_condition(model, rows, cache, stage / "original.json", output, job["distribution"], **scoring)
        gate_rows = [
            next(r for r in rows if r["bank"] == bank and r["domain"] == domain)
            for bank in ("public_initial", "public_teacher")
            for domain in ("math", "code", "if")
        ]
        repeated = score_condition(
            model, gate_rows, cache, stage / "no-edit-repeat.json", output, job["distribution"], **scoring
        )
        baseline = {r["id"]: r for r in original["rows"]}
        gate_error = max(abs(r["rkl_sum"] - baseline[r["id"]]["rkl_sum"]) / r["tokens"] for r in repeated["rows"])
        if gate_error > 1e-8:
            raise ValueError("Matched no-edit repeated-forward RKL gate failed")
        atomic_json(stage / "no-edit-gate.json", dict(passed=True, rows=len(gate_rows), max_rkl_error=gate_error))
        if probe_only:
            summary.append(
                dict(
                    step=point["step"],
                    no_edit_gate=dict(passed=True, max_rkl_error=gate_error),
                    training_comparison=paired_summary(original, repeated, point["training_rows"])[
                        "training_comparison"
                    ],
                )
            )
            del model
            torch.cuda.empty_cache()
            continue
        target = TensorCheckpoint(point["checkpoint"])
        keys = base.selected(SCOPE)
        if keys != target.selected(SCOPE):
            raise ValueError("Base/checkpoint projection identities differ")
        receipts = []
        editing_started = time.monotonic()
        for key in keys:
            check_stop(output)
            raw = target.get(key)
            current = raw.to(device=device, dtype=torch.float64)
            if key not in base_singular:
                raw_base = base.get(key)
                base_singular[key] = dict(
                    s=singular_values(
                        raw_base.to(device=device, dtype=torch.float64), driver="gesvd", transpose_wide=True
                    ).cpu(),
                    sha256=tensor_digest(raw_base),
                )
            initial = base_singular[key]["s"].to(device)
            if edit_method == "gram":
                edited, receipt = gram_restore(
                    current, initial, fallback=lambda w, s: restore_sigma0(w, s, driver="gesvd")
                )
            elif edit_method == "svd":
                edited, receipt = restore_sigma0(current, initial, driver="gesvd")
            else:
                raise ValueError("Unknown spectrum restoration method")
            parameter = model.get_parameter(key)
            if not torch.equal(parameter, raw.to(device=device, dtype=parameter.dtype)):
                raise ValueError("Loaded unedited parameter differs from immutable checkpoint")
            converted = edited.to(parameter.dtype)
            receipt.update(
                tensor=key,
                target_sha256=tensor_digest(raw),
                base_sha256=base_singular[key]["sha256"],
                native_changed_elements=int((converted != parameter).sum()),
                elements=parameter.numel(),
            )
            parameter.copy_(converted)
            receipts.append(receipt)
            atomic_json(
                output / "progress.json",
                dict(phase="edit", step=point["step"], completed=len(receipts), total=len(keys)),
            )
        atomic_json(stage / "edits.json", receipts)
        edit_seconds = time.monotonic() - editing_started
        edited = score_condition(model, rows, cache, stage / "sigma0.json", output, job["distribution"], **scoring)
        compared = paired_summary(original, edited, point["training_rows"])
        result = dict(
            pair=job["pair"],
            step=point["step"],
            rows=len(rows),
            tokens=sum(r["tokens"] for r in original["rows"]),
            edit_seconds=edit_seconds,
            reference_cache=references.stats() if references else None,
            execution_options=dict(
                edit_method=edit_method, token_chunk=token_chunk, reference_cache_mib=reference_cache_mib
            ),
            no_edit_gate=dict(passed=True, max_rkl_error=gate_error),
            seconds=time.monotonic() - started,
            **compared,
        )
        for path in (point["checkpoint"], job["base"], job["teacher"]):
            if checkpoint_identity(path) != source_identities[path]:
                raise ValueError("Source checkpoint changed during intervention scoring")
        atomic_json(stage / "complete.json", result)
        summary.append(result)
        del model, parameter, current, edited, converted
        torch.cuda.empty_cache()
    result = dict(
        pair=job["pair"], smoke=smoke, stages=summary, complete=len(summary) == len(points), execution=job["execution"]
    )
    atomic_json(output / "complete.json", result)
    if not keep_cache:
        shutil.rmtree(cache)
    return result


def read_job(path):
    return json.loads(gzip.decompress(Path(path).read_bytes()))
