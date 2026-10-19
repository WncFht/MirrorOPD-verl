# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Hydra custom_reward_function entry point for the four One-Shot-OPD domains."""

from verl.utils.reward_score.one_shot_opd import reward_func

__all__ = ["reward_func"]
