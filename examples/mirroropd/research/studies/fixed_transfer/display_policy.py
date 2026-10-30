# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""User-selected appendix placement; scientific records and execution plans remain intact."""

MAX_DISPLAY_UPDATES = 320

APPENDIX_TEACHERS = frozenset({"rlvr-gsm8k"})


def in_main_comparison(teacher):
    return teacher not in APPENDIX_TEACHERS
