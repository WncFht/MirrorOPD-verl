# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Canonical math-answer grading shared by transfer studies and historical evaluators."""

import os


def init_grader():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = "1"
    # Spawned pool workers use multiprocessing's exit path, which otherwise
    # joins the math verifier's still-idle nested workers before their executor
    # receives shutdown. Close that executor before the child-join phase.
    from multiprocessing.util import Finalize

    Finalize(None, close_grader, exitpriority=20)


def close_grader():
    from verl.utils.reward_score import math_verify

    pool = math_verify._pool
    if pool is not None:
        pool.shutdown(wait=True, cancel_futures=True)
        math_verify._pool = None


def grade(record):
    from verl.utils.reward_score.one_shot_opd.math import reward_func

    result = reward_func(record["data_source"], record["output"], record["gts"])
    result = {key: value.item() if hasattr(value, "item") else value for key, value in result.items()}
    ids = record["response_ids"]
    windows = max(0, len(ids) - 7)
    repeat8 = 1 - len(set(zip(*(ids[i:] for i in range(8)), strict=False))) / windows if windows else 0.0
    return record | result | {"repeat8": repeat8, "status": "ok"}
