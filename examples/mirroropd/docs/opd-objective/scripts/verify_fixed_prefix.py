"""CPU-only checks for fixed-prefix OPD estimators; no model or GPU is loaded."""

import argparse
import importlib
import json
import math
import sys
from pathlib import Path

import torch


def gradient(logits, objective):
    logits = logits.detach().clone().requires_grad_()
    return torch.autograd.grad(objective(logits.log_softmax(-1)), logits)[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--async-opd-root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    upstream = None
    if args.async_opd_root is not None:
        sys.path.insert(0, str(args.async_opd_root.resolve()))
        upstream = importlib.import_module("opd.loss.kl").policy_gradient_kl

    checks = []
    for seed in range(20):
        generator = torch.Generator().manual_seed(seed)
        logits = torch.randn(7, generator=generator, dtype=torch.float64)
        mu = torch.randn(7, generator=generator, dtype=torch.float64).softmax(-1)
        teacher = torch.randn(7, generator=generator, dtype=torch.float64).softmax(-1)
        log_mu, log_teacher = mu.log(), teacher.log()
        exact = gradient(logits, lambda lp: (lp.exp() * (lp - log_teacher)).sum())
        current = gradient(
            logits, lambda lp: -(mu * (lp - log_mu).exp() * (log_teacher - lp).detach()).sum()
        )
        fixed = gradient(
            logits, lambda lp: -(mu * (lp - log_mu).exp() * (log_teacher - log_mu)).sum()
        )
        fixed_regularized = gradient(
            logits,
            lambda lp: -(mu * (lp - log_mu).exp() * (log_teacher - log_mu)).sum()
            + (lp.exp() * (lp - log_mu)).sum(),
        )
        corrected = gradient(
            logits,
            lambda lp: (
                mu * ((lp - log_mu).exp() * (lp - log_teacher - 1) + (log_teacher - log_mu).exp())
            ).sum(),
        )
        no_is = gradient(logits, lambda lp: -(mu * (log_teacher - lp).detach() * lp).sum())
        mse = gradient(logits, lambda lp: (mu * 0.5 * (lp - log_teacher).square()).sum())
        errors = {
            "current_vs_exact": (current - exact).abs().max().item(),
            "fixed_plus_kl_vs_exact": (fixed_regularized - exact).abs().max().item(),
            "corrected_scalar_vs_exact": (corrected - exact).abs().max().item(),
            "no_is_vs_logprob_mse": (no_is - mse).abs().max().item(),
        }
        if upstream is not None:
            def upstream_objective(lp, online):
                return sum(
                    mu[i]
                    * upstream(
                        student_token_logps=lp[i].reshape(1, 1),
                        teacher_token_logps=log_teacher[i].reshape(1, 1),
                        student_old_logprobs=log_mu[i].reshape(1, 1),
                        mask=torch.ones(1, 1, dtype=torch.bool),
                        clip_eps=1e6,
                        use_importance_sampling=True,
                        online_advantage=online,
                        use_decoupled_loss=False,
                    )
                    for i in range(len(mu))
                )

            current_upstream = gradient(logits, lambda lp: upstream_objective(lp, True))
            fixed_upstream = gradient(logits, lambda lp: upstream_objective(lp, False))
            errors["upstream_current_vs_exact"] = (current_upstream - exact).abs().max().item()
            errors["upstream_fixed_vs_fixed_formula"] = (fixed_upstream - fixed).abs().max().item()
        if max(errors.values()) >= 1e-6:
            raise AssertionError(f"Gradient mismatch, seed={seed}: {errors}")
        checks.append(errors)

    mu = torch.tensor([0.2, 0.8], dtype=torch.float64)
    teacher = torch.tensor([0.35, 0.65], dtype=torch.float64)
    example = []
    for first_probability in [0.2, 0.26, 0.35, 0.5]:
        row = {"first_probability": first_probability}
        for current_advantage in [False, True]:
            for clip in [False, True]:
                z = torch.tensor(
                    math.log(first_probability / (1 - first_probability)),
                    dtype=torch.float64,
                    requires_grad=True,
                )
                lp = torch.stack(
                    [torch.nn.functional.logsigmoid(z), torch.nn.functional.logsigmoid(-z)]
                )
                advantage = (teacher.log() - (lp if current_advantage else mu.log())).detach()
                ratio = (lp - mu.log()).exp()
                term = ratio * advantage
                if clip:
                    term = torch.minimum(term, ratio.clamp(0.8, 1.2) * advantage)
                loss = -(mu * term).sum()
                name = ("current" if current_advantage else "fixed") + ("_clip" if clip else "_is")
                row[name] = torch.autograd.grad(loss, z)[0].item()
        example.append(row)

    result = {
        "backend": "CPU",
        "torch": torch.__version__,
        "cases": len(checks),
        "upstream_checked": upstream is not None,
        "upstream_reference": (
            "furiosa-ai/async-opd@e3f79e916aad9dba886256d202bab96f8504a447"
            if upstream is not None else None
        ),
        "upstream_note": "Optional source path must be prepared at the cited revision; this script does not fetch it.",
        "max_absolute_gradient_errors": {key: max(row[key] for row in checks) for key in checks[0]},
        "two_token_example": example,
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
