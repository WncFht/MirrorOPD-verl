# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Execute one explicitly selected job and phase."""

import json
import shutil
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from examples.mirroropd.research.core.io import dump

from ...teacher_difference.runtime import collect_memory, load_model
from ..weights import TensorCheckpoint
from .banks import collect_reference, generated_bank, score
from .directions import apply, prepare_directions, variants
from .gradients import gradients
from .protocol import save_tensor


def run(context, job, out, smoke=False, phase="all"):
    if phase == "gradients":
        torch.set_num_threads(2)
        banks = json.loads((out / "banks.json").read_text())
        reference, head = torch.load(out / "reference.pt", weights_only=True, mmap=True)
        records = json.loads((out / "matrix-index.json").read_text())
        gradients(context, job, out, banks, reference, head, records)
        dump(out / "complete.json", dict(finished_at=time.time(), phase="all-complete"))
        return
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(job["base"], local_files_only=True, trust_remote_code=True)
    peer = AutoTokenizer.from_pretrained(job["teacher_path"], local_files_only=True, trust_remote_code=True)
    if tokenizer.get_vocab() != peer.get_vocab():
        raise ValueError("Native full-vocabulary scoring requires identical token-ID vocabulary")
    bank_path = out / "banks.json"
    if bank_path.exists():
        banks = json.loads(bank_path.read_text())
    else:
        prompts = []
        sources = json.loads((context.root / "prompts.json").read_text())
        if smoke:
            sources = sources[:2]
        for source in sources:
            options = dict(tokenize=False, add_generation_prompt=True)
            if job["family"] == "Qwen":
                options["enable_thinking"] = "nonthinking" not in job["run_id"]
            rendered = tokenizer.apply_chat_template(source["messages"], **options)
            ids = tokenizer.encode(rendered, add_special_tokens=False)
            if not ids or any(type(token) is not int for token in ids):
                raise ValueError("Chat template did not produce a flat integer token sequence")
            prompts.append(source | dict(prompt_ids=ids))
        banks = {}
        for origin, path in [("initial-student", job["base"]), ("teacher", job["teacher_path"])]:
            model = load_model(path, trainable=True, device="cuda")
            model.eval().requires_grad_(False)
            banks[origin] = generated_bank(context, model, tokenizer, prompts, origin)
            del model
            collect_memory()
        dump(bank_path, banks)
    reference_path = out / "reference.pt"
    if reference_path.exists():
        reference, head = torch.load(reference_path, weights_only=True, mmap=True)
    else:
        reference, head = collect_reference(context, job["teacher_path"], banks)
        save_tensor(reference_path, (reference, head))
    head = head.cuda()
    # Teacher-self uses identical model execution as students.
    model = load_model(job["teacher_path"], trainable=True, device="cuda")
    model.eval().requires_grad_(False)
    floor = score(context, model, banks, reference, head, out / "scores/teacher-self.json", dict(name="teacher-self"))
    if max(abs(r["rkl"]) for r in floor["rows"]) > 1e-6:
        raise ValueError("Teacher-self matched-execution floor failed")
    del model
    collect_memory()
    for label, path in [("base", job["base"]), ("endpoint", job["endpoint"])]:
        model = load_model(path, trainable=True, device="cuda")
        model.eval().requires_grad_(False)
        score(context, model, banks, reference, head, out / f"scores/{label}.json", dict(name=label, checkpoint=path))
        del model
        collect_memory()
    # Independent serialized checkpoint, followed by a matched-execution reload.
    exported = out / "no-edit-export"
    if not (exported / "complete.json").exists():
        exported.mkdir(exist_ok=True)
        checkpoint = TensorCheckpoint(job["endpoint"])
        for filename in sorted(set(checkpoint.files.values())):
            context.check_stop()
            source_file = Path(job["endpoint"]) / filename
            target_file = exported / filename
            target_file.parent.mkdir(parents=True, exist_ok=True)
            if not target_file.exists():
                shutil.copy2(source_file, target_file)
            if source_file.stat().st_size != target_file.stat().st_size:
                raise ValueError("Incomplete no-edit serialized copy")
        for source_file in Path(job["endpoint"]).iterdir():
            if source_file.is_file() and source_file.suffix in {".json", ".py", ".model", ".txt", ".jinja"}:
                shutil.copy2(source_file, exported / source_file.name)
        dump(
            exported / "complete.json",
            dict(source=job["endpoint"], policy="byte-copy immutable shards, reload and compare per-question risk"),
        )
    model = load_model(exported, trainable=True, device="cuda")
    model.eval().requires_grad_(False)
    reloaded = score(
        context, model, banks, reference, head, out / "scores/no-edit-reload.json", dict(name="no-edit-reload")
    )
    original = json.loads((out / "scores/endpoint.json").read_text())
    error = max(abs(a["rkl"] - b["rkl"]) for a, b in zip(original["rows"], reloaded["rows"], strict=True))
    dump(out / "no-edit-gate.json", dict(max_rkl_error=error, passed=error <= 1e-6))
    if error > 1e-6:
        raise ValueError("No-edit export/reload risk gate failed")
    del model
    collect_memory()
    records = prepare_directions(context, job, out, smoke=smoke)
    model = load_model(job["endpoint"], trainable=True, device="cuda")
    model.eval().requires_grad_(False)
    conditions = variants()
    if smoke:
        conditions = [conditions[0], next(v for v in conditions if v["name"] == "missing-off-beta-1.0")]
        job = job | dict(gradient_stages={"endpoint": job["endpoint"]})
    for variant in conditions:
        dest = out / "scores" / (variant["name"] + ".json")
        if dest.exists():
            continue
        apply(context, model, job, out, records, variant)
        score(context, model, banks, reference, head, dest, variant)
        dump(
            out / "progress.json",
            dict(phase="interventions", latest=variant["name"], scores=len(list((out / "scores").glob("*.json")))),
        )
    del model
    collect_memory()
    dump(out / "interventions-complete.json", dict(finished_at=time.time()))
    if phase == "interventions":
        return
    gradients(context, job, out, banks, reference, head, records)
    dump(
        out / "complete.json",
        dict(
            finished_at=time.time(),
            scores=len(list((out / "scores").glob("*.json"))),
            gradient_stages=job["gradient_stages"],
        ),
    )
