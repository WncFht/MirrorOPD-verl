# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""CPU/FP64 examples auditing probability-space contracts; no model loading or training.

The probability-space formulas below are independent references. Production
calls document current semantics and explicit unsupported cases, not new support.
"""

import argparse
import json
import math
from pathlib import Path

import torch

from verl.trainer.distillation.evaluation import score_rkl_rows
from verl.trainer.distillation.full_vocab import full_reverse_kl_result
from verl.trainer.evaluation.config import GenerationProtocol


def tensor(values):
    return torch.tensor(values, dtype=torch.float64)


def kl(p, q):
    return (p * (p.log() - q.log())).sum()


def audit():
    result = {}
    h = torch.ones(1, 1, dtype=torch.float64)
    p, q = tensor([0.6, 0.4]), tensor([0.3, 0.2, 0.5])
    native = score_rkl_rows(h, p.log()[:, None], h, q.log()[:, None], token_chunk=1).item()
    conditional = kl(p, q[:2] / q[:2].sum()).item()
    assert abs(native - (conditional - math.log(q[:2].sum()))) < 1e-12
    result["teacher_extra_support"] = {
        "native_rkl": native,
        "conditional_rkl": conditional,
        "teacher_excluded_mass": q[2].item(),
    }

    z = tensor([0.2, 0.3, 0.1, 0.4]).log().requires_grad_()
    teacher = tensor([0.1, 0.2, 0.7])
    pc = z[:3].softmax(-1)
    kc = kl(pc, teacher)
    grad = torch.autograd.grad(kc, z)[0]
    assert grad[-1].item() == 0
    rejected = {}
    for name, call in {
        "readout": lambda: score_rkl_rows(h, z[:, None], h, teacher.log()[:, None], token_chunk=1),
        "training": lambda: full_reverse_kl_result(h, z[:, None], h, teacher.log()[:, None], backend="dense"),
    }.items():
        try:
            call()
        except ValueError as exc:
            rejected[name] = str(exc)
    assert set(rejected) == {"readout", "training"}
    result["student_extra_support"] = {
        "conditional_reference": kc.item(),
        "excluded_mass": z.softmax(-1)[-1].item(),
        "excluded_logit_gradient": grad[-1].item(),
        "native": "infinite if teacher mass is zero",
        "current_rejections": rejected,
    }

    p, q = tensor([0.45, 0.05, 0.5]), tensor([0.05, 0.45, 0.5])
    loss = full_reverse_kl_result(
        h, p.log()[:, None], h, q.log()[:, None], backend="precompute", termination_token_ids=(0, 1), token_chunk=1
    )
    raw = score_rkl_rows(h, p.log()[:, None], h, q.log()[:, None], token_chunk=1).item()
    assert abs(loss.loss_sum.item()) < 1e-12 and raw > 0.8
    assert abs(loss.token_kl.item() - raw) < 1e-12
    result["termination_equivalence"] = {
        "training_coarse_rkl": loss.loss_sum.item(),
        "training_raw_diagnostic": loss.token_kl.item(),
        "readout_raw_rkl": raw,
    }

    anchor, teacher = tensor([0.8, 0.1, 0.1]), tensor([0.1, 0.8, 0.1])
    mix = (anchor * teacher).sqrt()
    mix /= mix.sum()
    mix_then_group = torch.stack([mix[:2].sum(), mix[2]])
    grouped_anchor = torch.stack([anchor[:2].sum(), anchor[2]])
    grouped_teacher = torch.stack([teacher[:2].sum(), teacher[2]])
    group_then_mix = (grouped_anchor * grouped_teacher).sqrt()
    group_then_mix /= group_then_mix.sum()
    assert not torch.allclose(group_then_mix, mix_then_group)
    result["grouping_and_geometric_target"] = {
        "group_then_mix": group_then_mix.tolist(),
        "mix_then_group": mix_then_group.tolist(),
    }

    # Whole-group conditioning commutes with grouping; selecting only part of a
    # stop group does not define the same quotient and must be rejected/declared.
    p = tensor([0.1, 0.2, 0.3, 0.4])
    conditioned_then_group = torch.stack([p[:2].sum(), p[2]]) / p[:3].sum()
    grouped = torch.stack([p[:2].sum(), p[2], p[3]])
    torch.testing.assert_close(conditioned_then_group, grouped[:2] / grouped[:2].sum())
    result["condition_complete_groups"] = {"equal": True, "partial_group_requires_explicit_policy": True}

    p, original = tensor([0.6, 0.4]), tensor([0.6, 0.4])
    stored_teacher = original.flip(0)
    wrong = kl(p, stored_teacher).item()
    mapped = kl(p, stored_teacher[tensor([1, 0]).long()]).item()
    assert wrong > 0 and abs(mapped) < 1e-12
    result["permuted_token_ids"] = {"row_order_rkl": wrong, "mapped_rkl": mapped}

    default = GenerationProtocol()
    alternate = GenerationProtocol(answer_channel="whole_response")
    assert default.sampling("bench", "q", 0) == alternate.sampling("bench", "q", 0)
    assert default.protocol_id != alternate.protocol_id
    result["generation_and_verifier_identity"] = {"same_sampling": True, "different_protocol_id": True}

    a, b, alpha, beta = tensor([0.6, 0.4]), tensor([0.2, 0.8]), 0.7, 0.9
    p, q = torch.cat([alpha * a, tensor([1 - alpha])]), torch.cat([beta * b, tensor([1 - beta])])
    decomposed = alpha * kl(a, b) + kl(tensor([alpha, 1 - alpha]), tensor([beta, 1 - beta]))
    assert abs(kl(p, q).item() - decomposed.item()) < 1e-12
    result["outside_bucket"] = {
        "coarse_kl": kl(p, q).item(),
        "conditional_kl": kl(a, b).item(),
        "decomposition_error": abs(kl(p, q).item() - decomposed.item()),
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
