# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Unified CLI; execution and measurements share the same implementation."""

import importlib
import sys

COMMANDS = {
    "audit": "audit",
    "measure": "run_pac",
    "worker": "stages",
    "prefix": "point_probe",
    "logit": "logit_probe",
    "groups": "group_probe",
    "trajectories": "trajectory_probe",
    "alignment": "alignment_probe",
    "responses": "response_probe",
    "norms": "norm_probe",
    "updates": "update_probe",
    "adam": "adam_probe",
    "analyze-matrix": "analyze_matrix",
    "report-crossed": "report_crossed",
    "report-updates": "report_updates",
    "report-responses": "report_response",
    "report-trajectories": "report_trajectory",
    "report-groups": "report_groups",
    "batch": "batch",
    "sweep": "sweep",
    "report": "report",
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ["-h", "--help"]:
        print("teacher_difference {" + ", ".join(COMMANDS) + "} --help")
        return
    command = sys.argv.pop(1)
    if command not in COMMANDS:
        raise SystemExit(f"Unknown command: {command}")
    importlib.import_module(f"{__package__}.{COMMANDS[command]}").main()


if __name__ == "__main__":
    main()
