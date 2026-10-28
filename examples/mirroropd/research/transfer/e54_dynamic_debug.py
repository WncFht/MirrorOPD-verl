# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Read-only diagnostics injected into an isolated E54 runtime by prepare_e54_debug.

Records describe pre-update distributions on fixed sampled prefixes, not the
effect of a free logit displacement on shared network parameters. No RNG calls,
loss changes, or additional backbone forwards are made here.
"""

import json
import os
import time
from pathlib import Path

import torch

STOP_IDS = (151643, 151645)


def append(name, rows):
    root = Path(os.environ["E54_DEBUG_DIR"])
    root.mkdir(parents=True, exist_ok=True)
    with (root / name).open("a") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def rank():
    return torch.distributed.get_rank() if torch.distributed.is_initialized() else 0


def begin_update(engine):
    start = int(os.environ.get("E54_DEBUG_UPDATE_START", "0"))
    engine.e54_debug_update = getattr(engine, "e54_debug_update", start) + 1


def update_coordinates(update):
    start = int(os.environ.get("E54_DEBUG_UPDATE_START", "0"))
    outer_start = int(os.environ.get("E54_DEBUG_OUTER_START", "0"))
    reuse = int(os.environ.get("E54_DEBUG_REUSE", "16"))
    assert reuse > 0 and update > start
    local = update - start - 1
    return {"outer": outer_start + local // reuse + 1, "inner": local % reuse + 1, "reuse": reuse}


def rope_check(module):
    rows = []
    for name, child in module.named_modules():
        if hasattr(child, "inv_freq"):
            assert child.inv_freq.dtype == torch.float32, (name, child.inv_freq.dtype)
            rows.append({"module": name, "dtype": str(child.inv_freq.dtype)})
    assert rows, "Expected Qwen RoPE buffers"
    append(f"rope-rank-{rank()}.jsonl", [{"time": time.time(), "buffers": rows}])


def stop_statistics(student_logits, teacher_logits, ids=STOP_IDS):
    """FP64 coarsened KL and derivative when all stopping logits shift together."""
    lp = student_logits.double().log_softmax(-1)
    lq = teacher_logits.double().log_softmax(-1)
    p = lp.exp()
    columns = list(ids)
    log_p_stop = lp[:, columns].logsumexp(-1)
    log_q_stop = lq[:, columns].logsumexp(-1)
    p_stop, q_stop = log_p_stop.exp(), log_q_stop.exp()
    raw = (p * (lp - lq)).sum(-1)
    individual_stop = (p[:, columns] * (lp[:, columns] - lq[:, columns])).sum(-1)
    coarse = raw - individual_stop + p_stop * (log_p_stop - log_q_stop)
    return {
        "student_stop": p_stop,
        "teacher_stop": q_stop,
        "student_eot": lp[:, columns[0]].exp(),
        "student_im_end": lp[:, columns[1]].exp(),
        "raw_kl": raw,
        "coarse_kl": coarse,
        "stop_logit_gradient": p_stop * (log_p_stop - log_q_stop - coarse),
        "student_top1": lp.argmax(-1),
        "teacher_top1": lq.argmax(-1),
    }


@torch.no_grad()
def observe(engine, micro_batch, positions, hs, ws, ht, wt):
    from verl.trainer.distillation.full_vocab import project

    update = engine.e54_debug_update
    keys = list(micro_batch["sample_keys"])
    selected, metadata = [], []
    offset = 0
    for key, full, response, pred in zip(
        keys,
        micro_batch["input_ids"].unbind(),
        micro_batch["responses"].unbind(),
        positions,
        strict=True,
    ):
        prompt_length = len(full) - len(response)
        targets = (pred + 1 - prompt_length).cpu().tolist()
        wanted = {0, 128, 512, 1024, 2048, 4096, len(response) - 2, len(response) - 1}
        chosen = [(j, t) for j, t in enumerate(targets) if t in wanted]
        for j, target in chosen:
            selected.append(offset + j)
            metadata.append(
                {
                    "sample_key": key,
                    "update_attempt": update,
                    **update_coordinates(update),
                    "position": target,
                    "response_length": len(response),
                    "target_token": int(response[target]),
                    "terminal_target": int(response[target]) in STOP_IDS,
                    "stage": "pre_update",
                }
            )
        offset += len(pred)
    assert offset == len(hs)
    for start in range(0, len(selected), 16):
        idx = torch.tensor(selected[start : start + 16], device=hs.device)
        stats = stop_statistics(project(hs[idx], ws), project(ht[idx], wt))
        cpu = {k: v.cpu().tolist() for k, v in stats.items()}
        for j, row in enumerate(metadata[start : start + 16]):
            row.update({k: v[j] for k, v in cpu.items()})
    append(f"states-rank-{rank()}.jsonl", metadata)


def dump_rollout(trainer, batch):
    import transfer_queue as tq

    data = tq.kv_batch_get(
        keys=batch.keys,
        partition_id=batch.partition_id,
        select_fields=["prompts", "responses", "response_mask", "rm_scores"],
    )
    rows = []
    for i, (prompt, response, mask) in enumerate(
        zip(data["prompts"].unbind(), data["responses"].unbind(), data["response_mask"].unbind(), strict=True)
    ):
        prompt, response, mask = prompt.tolist(), response.tolist(), mask.tolist()
        rows.append(
            {
                "outer": trainer.global_steps,
                "sample_key": batch.keys[i],
                "prompt_ids": prompt,
                "response_ids": response,
                "response_mask": mask,
                "last_token": response[-1] if response else None,
                "stopped_on_eos": bool(response and response[-1] in STOP_IDS),
                "length": len(response),
                "at_response_cap": len(response) >= trainer.config.data.max_response_length,
                "score": float(data["rm_scores"][i].sum()),
                "text": trainer.tokenizer.decode(response, skip_special_tokens=False),
            }
        )
    append(f"rollout-{trainer.global_steps}.jsonl", rows)
    summary = {
        "outer": trainer.global_steps,
        "samples": len(rows),
        "mean_length": sum(r["length"] for r in rows) / len(rows),
        "cap_count": sum(r["at_response_cap"] for r in rows),
        "eos_count": sum(r["stopped_on_eos"] for r in rows),
        "mean_score": sum(r["score"] for r in rows) / len(rows),
        "time": time.time(),
    }
    append("rollout-summary.jsonl", [summary])
    print("E54_DYNAMIC_DEBUG " + json.dumps(summary), flush=True)
