# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Native-support distribution metrics. Historical shared-support probes stay explicit.

Teacher extra rows remain in its normalizer. Only fkl_shared conditions the
teacher on the student's support. This module never chooses an averaging denominator.
"""

import math

import torch


def _measure(sp, tq, *, shared_fkl=False):
    if sp.ndim != 2 or tq.ndim != 2 or sp.shape[0] != tq.shape[0] or sp.shape[1] > tq.shape[1]:
        raise ValueError("Expected aligned [state, vocabulary] logits with teacher support covering the student")
    lp = sp.log_softmax(-1)
    lq = tq.log_softmax(-1)
    p = lp.exp()
    q = lq.exp()
    v = sp.shape[-1]
    m = torch.logaddexp(lp, lq[:, :v]) - math.log(2)
    out = {
        "rkl": (p * (lp - lq[:, :v])).sum(-1),
        "js": 0.5 * ((p * (lp - m)).sum(-1) + (q[:, :v] * (lq[:, :v] - m)).sum(-1) + q[:, v:].sum(-1) * math.log(2)),
        "tv": 0.5 * ((p - q[:, :v]).abs().sum(-1) + q[:, v:].sum(-1)),
        "hp": -(p * lp).sum(-1),
        "hq": -(q * lq).sum(-1),
    }

    if shared_fkl:
        lqc = tq[:, :v].log_softmax(-1)
        out["fkl_shared"] = (lqc.exp() * (lqc - lp)).sum(-1)
    return out, lp, lq, p, q


def native_metrics(sp, tq):
    """Native-support distances without top-k sorting or conditional FKL diagnostics."""
    out, _, _, _, q = _measure(sp, tq)
    out["extra_teacher_mass"] = q[:, sp.shape[-1] :].sum(-1)
    return out


def distances(sp, tq):
    return _measure(sp, tq, shared_fkl=True)[0]


def fixed_metrics(sp, tq):
    """Historical long-pass profile: native distances and logit-ranked top-16."""
    out, _, _, p, q = _measure(sp, tq, shared_fkl=True)
    si, ti = sp.topk(16, -1).indices, tq.topk(16, -1).indices
    common = (si[:, :, None] == ti[:, None, :]).any(-1)
    out.update(
        overlap16=common.float().mean(-1),
        common_p16=(p.gather(1, si) * common).sum(-1),
        common_q16=(q.gather(1, si) * common).sum(-1),
    )
    return out


def vectors(zp, zt, actions, *, ks=(8, 16, 64)):
    zp, zt = zp.float(), zt.float()
    out, lp, lq, p, q = _measure(zp, zt)
    v = p.shape[-1]
    out["sample_gap"] = lp.gather(1, actions[:, None]).squeeze(1) - lq.gather(1, actions[:, None]).squeeze(1)
    maxsp = p.topk(max(ks), -1).indices
    maxtq = q.topk(max(ks), -1).indices
    for k in ks:
        sp = maxsp[:, :k]
        tq = maxtq[:, :k]
        common = (sp[:, :, None] == tq[:, None, :]).any(-1)
        inter = common.sum(-1)
        pc = (p.gather(1, sp) * common).sum(-1)
        qt = (q.gather(1, sp) * common).sum(-1)
        ps = p.gather(1, sp).sum(-1)
        qs = q.gather(1, sp).sum(-1)
        pt = p.gather(1, tq.clamp(max=v - 1)).mul(tq < v).sum(-1)
        qq = q.gather(1, tq).sum(-1)
        ins = (sp == actions[:, None]).any(-1)
        int_ = (tq == actions[:, None]).any(-1)
        cat = 2 * ins.long() + int_.long()
        out[f"overlap{k}"] = inter.float() / k
        out[f"jaccard{k}"] = inter.float() / (2 * k - inter)
        out[f"common_p{k}"] = pc
        out[f"common_q{k}"] = qt
        for c, pm, qm in [
            (0, 1 - ps - pt + pc, 1 - qs - qq + qt),
            (1, pt - pc, qq - qt),
            (2, ps - pc, qs - qt),
            (3, pc, qt),
        ]:
            out[f"category{k}_{c}"] = (cat == c).float()
            out[f"mass_p{k}_{c}"] = pm
            out[f"mass_q{k}_{c}"] = qm
    return out
