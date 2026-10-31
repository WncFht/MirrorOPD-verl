# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Independent stopping-class semantics and compensated logit representation."""

import pytest
import torch

from verl.trainer.distillation.termination import TerminationActionMap


def coarse_oracle(z, q, t, ids, kappa, eta):
    """80-digit probability-space objective and original-vocabulary gradient."""
    mp = pytest.importorskip("mpmath")
    with mp.workdps(80):

        def probabilities(values):
            p = [mp.exp(mp.mpf(float(x))) for x in values]
            return [x / sum(p) for x in p]

        p, q, t = [probabilities(v) for v in (z, q, t)]
        keep = [i for i in range(len(p)) if i not in ids]
        pc, qc, tc = [[v[i] for i in keep] + [sum(v[i] for i in ids)] for v in (p, q, t)]
        e, k = mp.mpf(eta), mp.mpf(kappa)
        b = [
            (1 - e) * mp.log(a) + e * mp.log(c) if kappa == 0 else mp.log((1 - e) * a**k + e * c**k) / k
            for a, c in zip(qc, tc, strict=True)
        ]
        log_z = mp.log(sum(mp.exp(x) for x in b))
        residual = [mp.log(a) - x for a, x in zip(pc, b, strict=True)]
        value = sum(a * x for a, x in zip(pc, residual, strict=True))
        gradient = [mp.mpf(0)] * len(p)
        for i, j in enumerate(keep):
            gradient[j] = p[j] * (residual[i] - value)
        for j in ids:
            gradient[j] = p[j] * (residual[-1] - value)
        return float(value), torch.tensor([float(x) for x in gradient], dtype=torch.float64), float(value + log_z)


@pytest.mark.parametrize("ids", [[1, 4], [4, 1], [0, 2, 4], [0, 1, 2, 3, 4]])
def test_reference_coarsening_and_chain_rule(ids):
    torch.manual_seed(41)
    z = torch.randn(3, 5, dtype=torch.float64, requires_grad=True)
    mapping = TerminationActionMap.create(5, ids, z.device)
    coarse, _ = mapping.coarsen(z)
    original = z.softmax(-1)
    expected = torch.cat((original[:, mapping.keep], original[:, mapping.stop].sum(-1, keepdim=True)), -1)
    torch.testing.assert_close(coarse.softmax(-1), expected, atol=1e-14, rtol=1e-14)
    # A true distributional loss has a zero-sum coarse logit gradient.
    teacher = torch.randn_like(coarse).softmax(-1)
    loss = (coarse.softmax(-1) * (coarse.log_softmax(-1) - teacher.log())).sum()
    gc = torch.autograd.grad(loss, coarse, retain_graph=True)[0]
    reference = torch.autograd.grad(loss, z)[0]
    torch.testing.assert_close(mapping.expand_gradient(gc, z), reference, atol=1e-14, rtol=1e-14)


def test_small_class_increment_survives_fp32_storage():
    mapping = TerminationActionMap.create(4, [1, 3], "cpu")
    z = torch.tensor([[0.0, 128.0, -3.0, 104.0]])
    coarse, correction = mapping.coarsen(z, compensated=True)
    assert coarse.dtype == torch.float32 and correction.dtype == torch.float64
    assert coarse[0, -1] == 128 and correction[0] > 0
    expected = torch.log1p(torch.exp(torch.tensor(-24.0, dtype=torch.float64)))
    torch.testing.assert_close(correction[0], expected, atol=1e-25, rtol=1e-14)
