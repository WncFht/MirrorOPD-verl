# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Measure trajectory overhead and traversal controls on identical synthetic heads.

Controls force the trajectory implementation with zero future credit. They are
measurement-only paths, not alternative production backends. Dense gates each
layout/mode before timing; all timings include hidden and head backward.
"""

import argparse
import json
import math
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from unittest.mock import patch

import torch

from examples.mirroropd.benchmarks.full_vocab.probe_full_kl import measure, relative_error
from verl.trainer.distillation import full_vocab, trajectory
from verl.trainer.distillation.full_vocab import full_vocab_objective_result
from verl.trainer.distillation.reduction import PackedReduction
from verl.trainer.distillation.trajectory import trajectory_objective_result
from verl.workers.config.distillation_spec import DistillationLossSettings, FutureCreditConfig


@dataclass(frozen=True)
class TraversalControl(PackedReduction):
    """Change only block visitation; zero credit makes cross-answer carry moot."""

    traversal: str = "answer-reverse"

    def blocks(self, chunk):
        if self.traversal == "answer-reverse":
            yield from super().blocks(chunk)
        else:
            blocks = [(start, min(start + chunk, len(self.times)), True) for start in range(0, len(self.times), chunk)]
            yield from reversed(blocks) if self.traversal == "contiguous-reverse" else blocks


def objective(inputs, settings, layout, backend, chunk, *, control=False):
    hs, ws, ht, wt = inputs
    kwargs = dict(spec=settings, reduction=layout, backend=backend, token_chunk=chunk)
    if control:
        result = trajectory_objective_result(
            hs, ws, ht, wt, hq=None, wq=None, vocab_chunk=8192, termination_token_ids=None, distribution=None, **kwargs
        )
    else:
        result = full_vocab_objective_result(hs, ws, ht, wt, **kwargs)
    return result.loss_sum / len(hs)


def projection_counts(fn):
    """Count untimed projection calls/rows without a second timed execution path."""
    calls = []
    original = full_vocab.project

    def record(hidden, weight):
        calls.append(len(hidden))
        return original(hidden, weight)

    with patch.object(full_vocab, "project", record), patch.object(trajectory, "project", record):
        value = fn()
    del value
    return dict(projection_calls=len(calls), projected_rows=sum(calls))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--answer-lengths", type=int, nargs="+", default=[1024, 64])
    parser.add_argument("--chunk", type=int, default=256)
    parser.add_argument("--vocab", type=int, default=151936)
    parser.add_argument("--hidden", type=int, default=1536)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if min(args.tokens, args.chunk, args.vocab, args.hidden, *args.answer_lengths) < 1:
        parser.error("Shapes and chunk must be positive")
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    hs = (torch.randn(args.tokens, args.hidden, device="cuda", dtype=torch.bfloat16) * 0.2).requires_grad_()
    ws = (
        torch.randn(args.vocab, args.hidden, device="cuda", dtype=torch.bfloat16) / math.sqrt(args.hidden)
    ).requires_grad_()
    ht, wt = torch.randn_like(hs), torch.randn_like(ws) / math.sqrt(args.hidden)
    inputs = hs, ws, ht, wt
    settings = DistillationLossSettings(names=("mirror",), payload_kind="hidden", objective="mdpo", eta=1.0)
    rows = []
    modes = ["none", "sum", "mean", "discounted_sum", "discounted_mean", "discounted_sequence"]
    for length in args.answer_lengths:
        counts = tuple(min(length, args.tokens - start) for start in range(0, args.tokens, length))
        times = torch.cat([torch.arange(n, device="cuda") for n in counts])
        actions = torch.randint(args.vocab, (args.tokens,), device="cuda")
        for aggregate in ["token-mean", "seq-mean-token-mean"]:
            for mode in modes:
                future = FutureCreditConfig(mode=mode, gamma=0.97 if mode.startswith("discounted") else 1.0)
                layout = PackedReduction(counts, times, actions, aggregate, future)
                expected = objective(inputs, settings, layout, "dense", args.chunk)
                gradients = torch.autograd.grad(expected, (hs, ws))
                expected_value, expected_grad = expected.item(), [g.cpu() for g in gradients]
                del expected, gradients
                variants = [("production", layout)]
                if mode == "none":
                    variants += [
                        (name, TraversalControl(counts, times, actions, aggregate, future, name))
                        for name in ["contiguous-forward", "contiguous-reverse", "answer-reverse"]
                    ]
                for backend in ["precompute", "precompute_fused"]:
                    for path, packed in variants:
                        fn = partial(
                            objective, inputs, settings, packed, backend, args.chunk, control=path != "production"
                        )
                        actual = fn()
                        gradients = torch.autograd.grad(actual, (hs, ws))
                        errors = dict(
                            value_abs=abs(actual.item() - expected_value),
                            hidden_relative_l2=relative_error(gradients[0].cpu(), expected_grad[0]),
                            head_relative_l2=relative_error(gradients[1].cpu(), expected_grad[1]),
                        )
                        assert all(math.isfinite(v) for v in errors.values()), errors
                        assert errors["value_abs"] < 1e-4 and max(errors.values()) < 0.03, errors
                        del actual, gradients
                        row = dict(
                            answer_length=length,
                            aggregate=aggregate,
                            future=mode,
                            backend=backend,
                            path=path,
                            errors=errors,
                            **projection_counts(fn),
                            **measure(fn, hs, ws, True, warmup=2, repeats=5),
                        )
                        rows.append(row)
                        print(json.dumps(row), flush=True)
                        args.output.parent.mkdir(parents=True, exist_ok=True)
                        args.output.write_text(
                            json.dumps(
                                dict(
                                    scope="synthetic head forward/backward; no backbone/FSDP/optimizer",
                                    torch=torch.__version__,
                                    gpu=torch.cuda.get_device_name(),
                                    tokens=args.tokens,
                                    chunk=args.chunk,
                                    vocab=args.vocab,
                                    hidden=args.hidden,
                                    warmup=2,
                                    repeats=5,
                                    rows=rows,
                                ),
                                indent=2,
                            )
                            + "\n"
                        )


if __name__ == "__main__":
    main()
