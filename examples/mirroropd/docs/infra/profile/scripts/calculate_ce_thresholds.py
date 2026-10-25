"""CPU checks of hard-label CE precision cliffs; no model/GPU state is touched.

The LSE-minus-target route models the arithmetic structure of fused CE, but
is executed with PyTorch CPU and is not a measurement of a CUDA/Triton kernel.
All references reuse the same representable logits and compute loss/gradient
in FP64 using log1p and the non-target mass (never p_target - 1).
"""

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F


def evaluate(margin, route="cross_entropy", offset=0.0, dtype=torch.float32):
    z = torch.tensor([[offset, offset-margin]], dtype=dtype, requires_grad=True)
    target = torch.tensor([0])
    if route == "cross_entropy":
        loss = F.cross_entropy(z, target)
    elif route == "lse_minus_target":
        loss = (z.logsumexp(-1)-z[:, 0]).mean()
    elif route == "autocast_ce":
        with torch.autocast("cpu", dtype=torch.bfloat16):
            loss = F.cross_entropy(z, target)
    else:
        raise ValueError(route)
    gradient = torch.autograd.grad(loss, z)[0].double()[0]
    actual_margin = (z.detach().double()[0, 0]-z.detach().double()[0, 1]).item()
    r = math.exp(-actual_margin)
    reference_loss = math.log1p(r)
    reference_gradient = -r/(1+r)
    return {
        "margin": actual_margin, "target_logit": z[0, 0].item(),
        "loss": loss.item(), "loss_dtype": str(loss.dtype),
        "reference_loss": reference_loss,
        "target_gradient": gradient[0].item(), "other_gradient": gradient[1].item(),
        "reference_target_gradient": reference_gradient,
        "target_gradient_relative_error": abs(gradient[0].item()/reference_gradient-1),
    }


def cliff(route, offset, dtype):
    low, high = 2., 30.
    for _ in range(30):
        mid = (low+high)/2
        if evaluate(mid, route, offset, dtype)["target_gradient"] == 0:
            high = mid
        else:
            low = mid
    return {"route": route, "input_dtype": str(dtype), "offset": offset,
            "last_nonzero": evaluate(low, route, offset, dtype),
            "first_zero": evaluate(high, route, offset, dtype)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = {"torch": torch.__version__, "device": "CPU", "ideal": [], "cliffs": [], "sweep": []}
    for name, precision in [("BF16", 8), ("FP16", 11), ("FP32", 24), ("FP64", 53)]:
        u = 2.**(-precision)
        result["ideal"].append({"arithmetic_dtype": name, "unit_roundoff": u,
            "ce_at_1_plus_r_absorption": math.log1p(u),
            "margin_at_absorption": -math.log(u), "one_minus_p_target": u/(1+u)})
    for route, dtype in [("cross_entropy", torch.float32), ("cross_entropy", torch.bfloat16),
                         ("autocast_ce", torch.bfloat16)]:
        result["cliffs"].append(cliff(route, 0., dtype))
    for offset in [0., 8., 16., 32., 64.]:
        row = cliff("lse_minus_target", offset, torch.float32)
        if offset:
            row["half_ulp_at_target"] = 2.**(math.floor(math.log2(offset))-24)
        result["cliffs"].append(row)
    for ce in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]:
        margin = -math.log(math.expm1(ce))
        result["sweep"].append({"requested_reference_ce": ce,
            "standard_fp32": evaluate(margin),
            "lse_minus_target_at_32": evaluate(margin, "lse_minus_target", 32.)})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    for row in result["ideal"]:
        print("ideal", row)
    for row in result["cliffs"]:
        print("cliff",row["route"],row["input_dtype"],row["offset"],
              row["first_zero"]["margin"],row["first_zero"]["reference_loss"])
    for row in result["sweep"]:
        print("sweep",row["requested_reference_ce"],
              row["standard_fp32"]["target_gradient_relative_error"],
              row["lse_minus_target_at_32"]["target_gradient_relative_error"])


if __name__ == "__main__":
    main()
