# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Native fixed-H full-vocabulary risk for reconstructed HF exports."""

import torch

from examples.mirroropd.research.core.io import dump
from examples.mirroropd.research.studies.teacher_difference.inputs import (
    load_records,
    normalize_records,
    source_fingerprint,
    validate_records,
)
from examples.mirroropd.research.studies.teacher_difference.runtime import (
    checkpoint_identity,
    collect_memory,
    load_model,
    response_hidden,
    teacher_cache,
)


@torch.no_grad()
def score(
    model_path,
    teacher_path,
    records,
    output,
    *,
    cache,
    device="cuda",
    cap=1024,
    token_chunk=16,
    vocab_chunk=8192,
    teacher_execution="native",
):
    """Score against the original teacher, including its extra native head rows."""
    from transformers import AutoTokenizer

    from verl.trainer.distillation.reverse_vocab import full_reverse_kl_result

    if teacher_execution not in ("native", "matched"):
        raise ValueError("teacher_execution must be native or matched")
    if min(cap, token_chunk, vocab_chunk) < 1:
        raise ValueError("cap/token_chunk/vocab_chunk must be positive")
    rows = normalize_records(load_records(records), cap=cap)
    validate_records(rows)
    if any(not r["prompt_ids"] or not r.get("split") for r in rows):
        raise ValueError("Every question needs a nonempty prompt and explicit split")
    vocabularies = [
        AutoTokenizer.from_pretrained(p, local_files_only=True, trust_remote_code=True).get_vocab()
        for p in (model_path, teacher_path)
    ]
    if any(vocabularies[1].get(token) != index for token, index in vocabularies[0].items()):
        raise ValueError("Native risk requires aligned student/teacher token IDs")
    teacher = teacher_cache(cache, teacher_path, rows, device=device, student=teacher_execution == "matched")
    model = load_model(model_path, trainable=True, device=device)
    model.requires_grad_(False)
    device = model.lm_head.weight.device
    teacher_head = teacher["head"].to(device)
    values = []
    try:
        for row in rows:
            hs = response_hidden(model, row).to(device)
            ht = teacher["hidden"][row["row_id"]].to(device)
            result = full_reverse_kl_result(
                hs,
                model.lm_head.weight,
                ht,
                teacher_head,
                backend="streaming",
                token_chunk=token_chunk,
                vocab_chunk=vocab_chunk,
            )
            values.append(
                dict(
                    row_id=row["row_id"], split=row["split"], tokens=len(hs), loss=float(result.loss_sum.double() / cap)
                )
            )
        dump(
            output,
            dict(
                model=checkpoint_identity(model_path),
                teacher=teacher["identity"],
                bank=source_fingerprint(rows),
                cap=cap,
                teacher_execution=teacher_execution,
                denominator="fixed H per question",
                token_chunk=token_chunk,
                vocab_chunk=vocab_chunk,
                rows=values,
            ),
        )
    finally:
        del model, teacher_head
        collect_memory()
