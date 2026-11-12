# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Full-parameter P/A/C computation for frozen-prefix reverse KL."""

from __future__ import annotations

from contextlib import nullcontext

import torch

from .metrics import squared_norm
from .reductions import response_denominator
from .requests import trajectory_projection
from .runtime import flatten_grad as _flatten_grad
from .runtime import gradient_norms, hidden_states
from .signals import fisher_diagnostics, position_signal  # noqa: F401 -- compatibility exports


def compute_batch(
    student,
    teacher_head,
    teacher_hidden,
    rows,
    *,
    cap=1024,
    retain_samples=True,
    initial=None,
    token_chunk=128,
    reduction="fixed_cap",
):
    """Compute per-record gradients and P/A/C inputs.

    ``teacher_hidden`` maps each row's ``row_id`` (or ``id``) to hidden states
    aligned with response positions. The function intentionally computes one
    record at a time so exact per-record ``g(s)`` is available; callers can
    disable retention when only a batch mean is needed.
    """
    if not rows:
        raise ValueError("rows must not be empty")
    context = gradient_norms(student) if not retain_samples else nullcontext(None)
    student.zero_grad(set_to_none=True)
    with context as norm_buffer:
        return _compute_records(
            student,
            teacher_head,
            teacher_hidden,
            rows,
            cap=cap,
            reduction=reduction,
            retain_samples=retain_samples,
            norm_buffer=norm_buffer,
            initial=initial,
            token_chunk=token_chunk,
        )


def _compute_records(
    student, teacher_head, teacher_hidden, rows, *, cap, retain_samples, norm_buffer, initial, token_chunk, reduction
):
    head_q = teacher_head.float()
    gradients, pressures, losses, gradient_norms, diagnostics = [], [], [], [], []
    lengths, denominators = [], []
    for row in rows:
        if norm_buffer is not None:
            norm_buffer.zero_()
        if retain_samples:
            student.zero_grad(set_to_none=True)
        hp = hidden_states(student, row["prompt_ids"] + row["response_ids"][:-1])
        hp = hp[len(row["prompt_ids"]) - 1 :]
        hq = teacher_hidden[str(row.get("row_id", row.get("id")))]
        hq = hq[: len(hp)].to(hp.device)
        length = min(len(hp), cap)
        denominator = response_denominator(length, cap, reduction)
        lengths.append(length)
        denominators.append(denominator)
        hp = hp[:length]
        hq = hq[:length]
        anchor = None
        if initial is not None:
            anchor = (initial["hidden"][str(row["row_id"])][:length].to(hp.device), initial["head"])
        result = trajectory_projection(
            hp,
            student.lm_head.weight.float(),
            hq,
            head_q,
            initial=anchor,
            cap=cap,
            reduction=reduction,
            token_chunk=token_chunk,
        )
        loss = result[0]
        info = result[3].double().sum(0) / denominator
        pressure = info[0 if initial is None else 1]
        diagnostics.append(info.detach().cpu().tolist())
        current_loss = result[1].double().sum() / denominator
        loss.backward()
        if retain_samples:
            gradient = _flatten_grad(student)
            gradient_norms.append(squared_norm(gradient))
            gradients.append(gradient)
        else:
            gradient_norms.append(float(norm_buffer))
        pressures.append(float(pressure.detach()))
        losses.append(float(current_loss.detach()))
    if retain_samples:
        gradient_tensor = torch.stack(gradients)
        gradient_norm_sq = torch.tensor(gradient_norms, dtype=torch.float64)
    else:
        gradient_tensor = _flatten_grad(student, scale=1 / len(rows)).unsqueeze(0)
        gradient_norm_sq = torch.tensor(gradient_norms, dtype=torch.float64)
    return {
        "gradient_vectors": gradient_tensor,
        "gradient_norm_sq": gradient_norm_sq,
        "pressure": torch.tensor(pressures, dtype=torch.float64),
        "losses": losses,
        "token_counts": lengths,
        "denominators": denominators,
        "request_diagnostics": diagnostics,
        "parameter_count": int(gradient_tensor.shape[-1]),
        "records": len(rows),
    }
