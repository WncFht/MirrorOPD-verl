# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""One-Shot-OPD task verifiers with explicit, lazy domain routing.

The implementations come from Thinking-Space/One-Shot-OPD at
6b6b1782cc3f04749dd077de19353f839f6bc048. See examples/mirroropd/README.md for dependencies
and protocol boundaries. These scores do not define the distillation loss.
"""

from importlib import import_module

MATH_SOURCES = frozenset({"DAPO", "MATH-500", "AMC23", "AIME24", "AIME25", "AIME26", "math", "math_dapo"})
CODE_SOURCES = frozenset(
    {
        "livecodebench",
        "livecodebench/code_generation",
        "livecodebench/code_generation_lite",
        "taco",
        "BAAI/TACO",
        "apps",
        "codeparrot/apps",
        "code_contests",
        "deepmind/code_contests",
        "codeforces",
        "humanevalplus",
        "evalplus/humanevalplus",
        "openai/openai_humaneval",
        "openai_humaneval",
    }
)
IF_SOURCES = frozenset({"MultiIF"})
FC_SOURCES = frozenset(
    {"xlam", "bfcl_simple", "bfcl_multiple", "bfcl_parallel", "bfcl_parallel_multiple", "bfcl_live_simple"}
)
_ROUTES = {
    source: module
    for sources, module in (
        (MATH_SOURCES, "math"),
        (CODE_SOURCES, "code_eval_v2"),
        (IF_SOURCES, "if_eval"),
        (FC_SOURCES, "fc_eval"),
    )
    for source in sources
}


def reward_func(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """Score one completion; reject unknown datasets instead of silently grading them as math."""
    if data_source not in _ROUTES:
        raise ValueError(f"Unknown One-Shot-OPD data_source {data_source!r}; expected one of {sorted(_ROUTES)}")
    scorer = import_module(f"{__name__}.{_ROUTES[data_source]}").reward_func
    return scorer(data_source, solution_str, ground_truth, extra_info=extra_info, **kwargs)
