# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Spectral diagnostics and offline reconstruction, without launching training."""

import argparse

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    weights = commands.add_parser("weights", help="Stream compatible checkpoint matrices and null controls")
    weights.add_argument("--cache", help="Content-keyed reusable teacher scalar summaries")
    weights.add_argument("--resume", action="store_true", help="Resume matching versioned per-matrix results")
    weights.add_argument("--driver", choices=("gesvd", "gesvdj"), help="Exact CUDA SVD solver; CPU uses default")
    reconstruction = commands.add_parser("reconstruct", help="Publish a new HF checkpoint with selected matrix edits")
    residuals = commands.add_parser("residuals", help="Fit calibration output SVD and score fixed bands")
    hidden = commands.add_parser("representations", help="Fit same-coordinate hidden residual SVD")
    head = commands.add_parser("head", help="Exact centered-head bands in shared hidden coordinates")
    extraction = commands.add_parser("extract", help="Extract full-vocabulary log probabilities and sampled states")
    scoring = commands.add_parser("score", help="Native original-teacher fixed-H RKL for an HF export")
    for sub in (weights, reconstruction, residuals, hidden, head):
        sub.add_argument("--base", required=True)
    for sub in (weights, residuals, hidden, head):
        sub.add_argument("--teacher", required=True)
        sub.add_argument("--opd", required=True)
    for sub in (extraction, scoring, reconstruction):
        sub.add_argument("--model", required=True)
    scoring.add_argument("--teacher", required=True)
    scoring.add_argument("--cache", required=True)
    scoring.add_argument(
        "--teacher-execution",
        choices=("native", "matched"),
        default="native",
        help="Teacher native BF16 or actor-matched FP32 master/BF16 backbone",
    )
    for sub in (extraction, scoring):
        sub.add_argument("--records", required=True)
        sub.add_argument("--cap", type=int, default=1024)
    for sub in (weights, reconstruction, extraction, scoring, residuals, hidden, head):
        sub.add_argument("--output", required=True)
    for sub in (weights, reconstruction, extraction, scoring, head):
        sub.add_argument("--device", default="cpu" if sub in (weights, reconstruction) else "cuda")
    for sub in (weights, reconstruction):
        from .weights import DEFAULT_SCOPE

        sub.add_argument("--scope", default=DEFAULT_SCOPE)
        sub.add_argument("--arithmetic", choices=("float32", "float64"), default="float64")
    for sub in (weights, residuals, reconstruction):
        sub.add_argument("--seed", type=int, default=0)
    for sub in (residuals, hidden):
        sub.add_argument("--calibration", default="train")
        sub.add_argument("--evaluation", default="heldout")
        sub.add_argument("--rank", type=int, default=16)
    residuals.add_argument("--oversample", type=int, default=8)
    residuals.add_argument("--iterations", type=int, default=1)
    residuals.add_argument("--block", type=int, default=32)
    residuals.add_argument("--metric", choices=("fisher", "logit", "probability"), default="fisher")
    head.add_argument("--base-weights", required=True)
    head.add_argument("--teacher-weights", required=True)
    head.add_argument("--rank", type=int, default=16)
    head.add_argument("--shared-coordinates", action="store_true", required=True)
    hidden.add_argument("--layer", default="final")
    hidden.add_argument(
        "--shared-coordinates",
        action="store_true",
        required=True,
        help="Assert shared hidden coordinates/ancestry; equal width alone is insufficient",
    )
    extraction.add_argument("--positions", type=int, default=16)
    extraction.add_argument("--chunk", type=int, default=16)
    extraction.add_argument("--precision", choices=("master", "native"), default="master")
    extraction.add_argument("--layers", type=int, nargs="*", default=[])
    scoring.add_argument("--token-chunk", type=int, default=16)
    scoring.add_argument("--vocab-chunk", type=int, default=8192)
    reconstruction.add_argument(
        "--mode",
        required=True,
        choices=("top", "delete-diag", "delete-off", "delete-null", "patch-diag", "patch-off", "patch-null"),
    )
    reconstruction.add_argument("--control", choices=("none", "gaussian", "random_frames"), default="none")
    reconstruction.add_argument("--anchor")
    reconstruction.add_argument("--fraction", type=float, default=0.01)
    reconstruction.add_argument("--restore-norm", action="store_true")
    reconstruction.add_argument("--beta", type=float, default=1.0)
    reconstruction.add_argument("--untouched", choices=("base", "target"))
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command in ("weights", "reconstruct"):
        from . import weights as module

        args["pattern"] = args.pop("scope")
        args["dtype"] = getattr(torch, args.pop("arithmetic"))
        if command == "reconstruct":
            args["target"] = args.pop("model")
            args["untouched"] = args["untouched"] or ("base" if args["mode"] == "top" else "target")
        function = module.analyze if command == "weights" else module.reconstruct
    elif command == "residuals":
        from .residuals import analyze as function
    elif command == "representations":
        from .representations import analyze as function

        args.pop("shared_coordinates")
    elif command == "head":
        from .representations import analyze_head as function

        args.pop("shared_coordinates")
    elif command == "extract":
        from .extract import extract as function

        args["model_path"] = args.pop("model")
    else:
        from .scoring import score as function

        args["model_path"] = args.pop("model")
        args["teacher_path"] = args.pop("teacher")
    function(**args)


if __name__ == "__main__":
    main()
