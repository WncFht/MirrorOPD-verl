# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Substitute published rounded coefficients; do not refit or digitize any figure."""
import json
from pathlib import Path

PEAK = dict(A=.97, alpha=.304, beta=-.267, zeta=.952)
RATE = dict(B=.116, gamma=.189, delta=-.609, xi=1.896)


def main():
    records = []
    for label, nt, gt, observed, observed_m in [
        ("1.5B RL endpoint", 1.5, .636, .775, .558),
        ("3B RL intermediate", 3, .660, .738, None),
        ("3B RL endpoint", 3, .736, .803, .670),
    ]:
        ns = 7
        error = PEAK["A"] * ns ** -PEAK["alpha"] * nt ** -PEAK["beta"] * (1-gt) ** PEAK["zeta"]
        rate = RATE["B"] * ns ** -RATE["gamma"] * nt ** RATE["delta"] * (1-gt) ** -RATE["xi"]
        records.append(dict(teacher=label, student_b=ns, teacher_effective_b=nt, teacher_gold=gt,
                            calculated_peak_error=error, calculated_peak=1-error,
                            paper_observed_peak=observed, calculated_rate=rate,
                            paper_observed_rate=observed_m))
    first = records[0]
    assert abs(first["calculated_peak"]-.7714154813906473) < 1e-12
    assert abs(first["calculated_rate"]-.4262325963163943) < 1e-12
    result = dict(source="https://arxiv.org/pdf/2609.32722v1",
                  sections="Equation 6, PDF Tables 3/9 and Appendix H",
                  interpretation="Formula substitution using rounded published coefficients; not an author-data refit",
                  units="Gold scores in [0,1]; rate is gold probability per unit sqrt token-mean KL",
                  peak_coefficients=PEAK, rate_coefficients=RATE, cases=records,
                  elasticities=dict(student_doubling_error_multiplier=2**-PEAK["alpha"],
                                    teacher_doubling_fixed_score_error_multiplier=2**-PEAK["beta"],
                                    teacher_error_0364_to_0340_multiplier=(.34/.364)**PEAK["zeta"]))
    Path(__file__).with_name("prediction-example.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print("Published-coefficient substitution verified; no fitting performed.")


if __name__ == "__main__":
    main()
