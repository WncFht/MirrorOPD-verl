# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""TB2 common-input scoring: two student sizes on both frozen S0 banks."""

import argparse
import gc

import torch

from examples.mirroropd.research.core.io import dump
from examples.mirroropd.research.studies.boundary.common import ARMS, CACHE, OUT
from examples.mirroropd.research.studies.boundary.native import bank, load, setup
from examples.mirroropd.research.studies.boundary.train import load_teacher, score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("arm", choices=["TB1-P", "TB2-P7", "TB2-L1", "TB2-L7"])
    args = parser.parse_args()
    setup()
    student, teacher = ARMS[args.arm]
    for step in [0, 64]:
        path = None if step == 0 else CACHE / "checkpoints" / args.arm / "step064"
        model = load(student, student=True, path=path)
        for source in ["r1", "r7"]:
            dest = OUT / "cross" / args.arm / f"{source}-{step:03d}.json"
            if dest.exists():
                continue
            rows = [r for r in bank(source) if r["group"] in ["M01", "M02", "M03", "M04"]]
            assert len(rows) == 64
            head, states = load_teacher(teacher, source, rows, model.lm_head.weight.device)
            dump(dest, score(model, head, states, rows, step))
            del head, states
        del model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
