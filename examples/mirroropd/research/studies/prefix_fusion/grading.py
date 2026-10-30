# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Canonical math decisions inside an existing CPU worker, without nested cold-start timeouts."""

import os


def init_grader():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = "1"
    # Import before accepting answers. The native verifier already runs here on
    # the worker's main thread, where its signal-based expression limits work.
    from verl.utils.reward_score.math_verify import _verify_in_subprocess
    from verl.utils.reward_score.one_shot_opd.ttrl_math import compute_score

    assert callable(_verify_in_subprocess) and callable(compute_score)


def grade(record):
    from verl.utils.reward_score.math_verify import _verify_in_subprocess
    from verl.utils.reward_score.one_shot_opd.ttrl_math import compute_score

    result = compute_score(record["output"], str(record["gts"]))
    if result["format_score"] != 1.0:
        # Same parser/verification function as reward_func, directly in this
        # process. Infrastructure exceptions propagate; they never become 0s.
        correct = float(_verify_in_subprocess("\\boxed{" + str(record["gts"]) + "}", record["output"][-2000:]))
        result = dict(score=correct, format_score=0.0, acc=bool(correct), extracted_gt=record["gts"], pred="")
    result = {key: value.item() if hasattr(value, "item") else value for key, value in result.items()}
    ids = record["response_ids"]
    windows = max(0, len(ids) - 7)
    repeat8 = 1 - len(set(zip(*(ids[i:] for i in range(8)), strict=False))) / windows if windows else 0.0
    return record | result | dict(repeat8=repeat8, status="ok")
