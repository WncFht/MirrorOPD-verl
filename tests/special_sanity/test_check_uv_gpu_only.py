# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Keep GPU gating independent from the launcher's preferred package manager."""

import pytest
from check_uv_gpu_only import check_script


@pytest.mark.parametrize("default", [0, 1])
def test_uv_gate_accepts_both_defaults(default):
    lines = [
        f'if [ "${{VERL_USE_UV:-{default}}}" != 0 ] && [ "${{DEVICE:-gpu}}" = gpu ]; then',
        "  uv run python3 train.py",
        "fi",
    ]
    assert not check_script(lines, "examples/demo.sh")
    assert check_script(lines, "examples/ascend_extras/demo.sh")


def test_uv_gate_still_requires_gpu_and_then_branch():
    assert check_script(['if [ "${VERL_USE_UV:-0}" != 0 ]; then', "uv run train.py", "fi"], "examples/demo.sh")
    assert check_script(
        ['if [ "${VERL_USE_UV:-0}" != 0 ] && [ "${DEVICE:-gpu}" = gpu ]; then', ":", "else", "uv run train.py", "fi"],
        "examples/demo.sh",
    )
