# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Select the scientific report cohort before aggregation or plotting."""


def report_eligible(record):
    """Exclude user-identified contaminated teacher conditions by identity."""
    identities = [
        str(record.get(k, "")).lower() for k in ("teacher", "run_id", "pair_id", "label", "teacher_path", "branch")
    ]
    return not any(
        value in {"raw", "ccs", "ccs-rho0", "ccs-rho1"}
        or any(token in value for token in ("controlled-", "controlled_", "ccs-rho", "ccs_rho"))
        for value in identities
    )
