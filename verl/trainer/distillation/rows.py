# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Objective block formulas and native-coordinate logit VJPs."""

import torch

from verl.trainer.distillation.profiling import full_kl_range
from verl.utils.linear import project

_MIRROR_DIAGNOSTICS = ("target_kl", "target_log_z", "anchor_kl", "teacher_kl")


def dense_rows(hs, ws, ht, wt):
    from verl.trainer.distillation.numerics import kl_rows

    return kl_rows(project(hs, ws), project(ht.detach(), wt.detach()))[0]


def precompute_kl_rows(hs, ws, ht, wt, *, student_logits=None):
    """One block's exact KL and logit VJP; release logits before gradient GEMMs."""
    with full_kl_range("student_projection"):
        zp = project(hs, ws) if student_logits is None else student_logits
    with full_kl_range("teacher_projection"):
        zt = project(ht, wt)
    from verl.trainer.distillation.numerics import kl_rows

    with full_kl_range("normalization_and_kl"):
        kl, dz = kl_rows(zp, zt)
    return kl, dz if hs.dtype == torch.float16 else dz.to(hs.dtype)


def termination_equivalent_kl_rows(hs, ws, ht, wt, *, actions, fused=False, student_logits=None):
    """Coarsened loss/VJP, with native-vocabulary KL retained as its metric."""
    from verl.trainer.distillation.numerics import kl_rows, kl_values

    zp, zt = project(hs, ws) if student_logits is None else student_logits, project(ht, wt)
    dtype = torch.float32 if hs.dtype == torch.float16 else hs.dtype
    student, student_error = actions.coarsen(zp, compensated=fused)
    teacher, teacher_error = actions.coarsen(zt, compensated=fused)
    if fused:
        from verl.trainer.distillation.full_vocab_triton import difference, fused_rows, weighted_rows

        raw_kl, _, _ = weighted_rows(zp, difference(zp, zt), dtype, gradient=False)
        coarse_kl, gradient = fused_rows(student, teacher, torch.float32, corrections=(student_error, teacher_error))
    else:
        raw_kl = kl_values(zp, zt)
        coarse_kl, gradient = kl_rows(student, teacher)
    gradient = actions.expand_gradient(gradient, zp)
    return coarse_kl.to(zp.dtype), gradient.to(dtype), raw_kl


def conditional_kl_rows(hs, ws, ht, wt, *, space, actions=None, fused=False):
    """Reuse projections for conditional loss, VJP and both excluded masses."""
    from verl.trainer.distillation.numerics import kl_rows, kl_values

    zp, zt = project(hs, ws), project(ht, wt)
    p = space.select(zp.T, "student").T
    q = space.select(zt.T, "teacher").T
    masses = torch.stack(
        [space.excluded_from_logits(z, role) for z, role in ((zp, "student"), (zt, "teacher"))], dim=-1
    )
    cp, cq = p, q
    corrections = None
    if actions is not None:
        cp, ep = actions.coarsen(p, compensated=fused)
        cq, eq = actions.coarsen(q, compensated=fused)
        corrections = (ep, eq) if fused else None
    if fused:
        # The Triton row kernels require packed rows; slicing a wider head leaves a row gap.
        cp, cq = cp.contiguous(), cq.contiguous()
        from verl.trainer.distillation.full_vocab_triton import fused_rows

        value, gradient = fused_rows(cp, cq, torch.float32, corrections=corrections)
    else:
        value, gradient = kl_rows(cp, cq)
    raw = kl_values(p, q) if actions is not None else value
    if actions is not None:
        gradient = actions.expand_gradient(gradient, p)
    dz = torch.zeros_like(zp, dtype=torch.float32 if hs.dtype == torch.float16 else hs.dtype)
    if space.spec.common_vocab_size is not None:
        dz[:, : space.spec.common_vocab_size] = gradient.to(dz.dtype)
    else:
        dz.index_copy_(1, torch.tensor(space.student_ids, device=dz.device), gradient.to(dz.dtype))
    return value, dz, raw, masses


def _mirror_rows(hs, ws, ht, wt, hq, wq, spec, *, fused=False, actions=None, student_logits=None):
    with full_kl_range("teacher_projection_and_normalization"):
        zt = project(ht, wt)
    with full_kl_range("anchor_projection_and_normalization"):
        zq = project(hq, wq)
    with full_kl_range("student_projection"):
        z = project(hs, ws) if student_logits is None else student_logits
    with full_kl_range("mirror_rows"):
        original_z = z
        corrections = None
        if actions is not None:
            z, ez = actions.coarsen(z, compensated=fused)
            zq, eq = actions.coarsen(zq, compensated=fused)
            zt, et = actions.coarsen(zt, compensated=fused)
            corrections = (ez, eq, et) if fused else None
        if fused:
            from verl.trainer.distillation.full_vocab_triton import fused_mirror_logits

            dtype = torch.float32 if actions is not None or hs.dtype == torch.float16 else hs.dtype
            rows, dz, diag = fused_mirror_logits(z, zq, zt, dtype, spec=spec, corrections=corrections)
        else:
            from verl.trainer.distillation.numerics import mirror_logits_rows

            rows, dz, info = mirror_logits_rows(
                z, zq, zt, objective=spec.objective, kappa=spec.kappa, eta=spec.eta, diagnostics=spec.diagnostics
            )
            diag = (
                torch.stack([info[k] for k in _MIRROR_DIAGNOSTICS], dim=-1)
                if spec.diagnostics
                else z.new_empty((len(hs), 0))
            )
        if actions is not None:
            dz = actions.expand_gradient(dz, original_z)
        dz = dz.float() if hs.dtype == torch.float16 else dz.to(hs.dtype)
    return rows, dz, diag


def prepare_objective(hs, ws, ht, wt, hq, wq, *, spec, fused, actions, future):
    from verl.trainer.distillation.numerics import kl_rows, support_correction

    z = project(hs, ws)
    if not spec.teacher_only_kl:
        values, dz, info = _mirror_rows(hs, ws, ht, wt, hq, wq, spec, fused=fused, actions=actions, student_logits=z)
        return values, dz, values, info, z if future else None
    if actions is not None:
        values, dz, raw = termination_equivalent_kl_rows(hs, ws, ht, wt, actions=actions, fused=fused, student_logits=z)
    else:
        teacher = project(ht, wt)
        if fused:
            from verl.trainer.distillation.full_vocab_triton import fused_rows

            values, dz = fused_rows(z, teacher[:, : len(ws)].contiguous(), torch.float32)
            values = values + support_correction(teacher, len(ws))
        else:
            values, dz = kl_rows(z, teacher)
        raw = values
    return values, dz, raw, z.new_empty((len(hs), 0)), z if future else None


def prepare_conditional(hs, ws, ht, wt, hq, wq, *, space, actions, fused):
    values, dz, raw, info = conditional_kl_rows(hs, ws, ht, wt, space=space, actions=actions, fused=fused)
    return values, dz, raw, info, None
