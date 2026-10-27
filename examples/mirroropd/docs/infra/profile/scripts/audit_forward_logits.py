#!/usr/bin/env python3
"""Check the production fused rows on previously frozen real-model logits."""

import argparse
import json
from pathlib import Path

import torch

from verl.trainer.distillation.forward_vocab import forward_rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(0.10)
    records = []
    for source in ["student", "polaris"]:
        for index in [0, 8, 16, 20]:
            name = f"{source}-{index:02d}.pt"
            z = torch.load(a.cache / "student" / name, weights_only=True).cuda()
            all_t = torch.load(a.cache / "polaris" / name, weights_only=True).cuda()
            t = all_t[:, : z.shape[1]].contiguous()
            extra = all_t[:, z.shape[1] :].contiguous()
            ref = forward_rows(z.double(), t.double(), extra.double())
            fast = forward_rows(z, t, extra, fused=True)
            error = float((fast[0].double() - ref[0]).abs().max())
            grad = float((fast[1].double() - ref[1]).abs().max())
            assert error < 3e-6 and grad < 3e-6, (name, error, grad)
            assert fast[0].min() >= -1e-12
            records.append(
                dict(
                    case=name,
                    tokens=len(z),
                    loss_max_abs_error=error,
                    gradient_max_abs_error=grad,
                    forward_kl_min=float(fast[0].min()),
                    forward_kl_max=float(fast[0].max()),
                )
            )
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
