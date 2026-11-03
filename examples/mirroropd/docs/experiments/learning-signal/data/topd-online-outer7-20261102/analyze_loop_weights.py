"""Describe length/repetition and loss coefficients of the fixed outer7 bank.

Labels are automatic verifier outcomes. Token coefficients and objective-value
shares are not parameter-gradient contributions or causal effects on accuracy.
"""

import hashlib
import json
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def records(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def repeated8(tokens):
    windows = len(tokens) - 7
    return 0.0 if windows <= 0 else 1 - len({tuple(tokens[i : i + 8]) for i in range(windows)}) / windows


def main():
    out = ROOT / "online-loop-weights-outer7-20261102.json"
    assert not out.exists()
    score_dir = ROOT / "online-outer7-scores-20261102"
    completion = json.loads((score_dir / "complete.json").read_text())
    assert completion["scores_sha256"] == digest(score_dir / "scores.jsonl")
    scores = {(s["arm"], s["request_id"]): s for s in records(score_dir / "scores.jsonl")}
    arms, selected_cases = {}, []
    for arm in ["reuse-k0", "reuse-k05", "reuse-k1"]:
        folder = ROOT / "online" / arm / "outer07"
        trained = json.loads((folder / "complete.json").read_text())
        assert trained["status"] == "complete" and trained["updates"] == 112 and trained["outer"] == 7
        assert trained["bank_sha256"] == completion["summary"][arm]["bank_sha256"]
        bank = records(folder / "bank.jsonl")
        assert len(bank) == 64 and digest(folder / "bank.jsonl") == completion["summary"][arm]["bank_sha256"]
        rows = []
        risks = {name: json.loads((folder / f"current-{name}.json").read_text()) for name in ["before", "after"]}
        risk_index = {name: {r["request_id"]: r for r in risk["responses"]} for name, risk in risks.items()}
        for row in bank:
            score = scores[arm, row["request_id"]]
            identity = hashlib.sha256(json.dumps(row, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            assert score["status"] == "scored" and score["row_sha256"] == identity
            for name in risks:
                assert risk_index[name][row["request_id"]]["tokens"] == len(row["response_ids"])
            item = {
                "request_id": row["request_id"],
                "tokens": len(row["response_ids"]),
                "capped": row["finish_reason"] == "length",
                "native_correct": bool(score["result"]["acc"]),
                "boxed_correct": bool(score["result"]["acc"]) and score["result"]["format_score"] == 1,
                "duplicate8_fraction": repeated8(row["response_ids"]),
                "before": risk_index["before"][row["request_id"]],
                "after": risk_index["after"][row["request_id"]],
            }
            rows.append(item)
        total = sum(r["tokens"] for r in rows)
        for name in risks:
            assert total == risks[name]["tokens"]
            for key in ["objective", "teacher_kl", "anchor_kl"]:
                value = sum(r["tokens"] * r[name][key] for r in rows) / total
                assert abs(value - risks[name][key]) < 1e-9
        groups = {}
        for label, subset in [
            ("all", rows),
            ("capped", [r for r in rows if r["capped"]]),
            ("stopped", [r for r in rows if not r["capped"]]),
            ("capped_native_incorrect", [r for r in rows if r["capped"] and not r["native_correct"]]),
        ]:
            n = sum(r["tokens"] for r in subset)
            values = {
                "responses": len(subset),
                "tokens": n,
                "token_loss_weight": n / total,
                "counterfactual_equal_response_weight": len(subset) / 64,
                "native_correct": sum(r["native_correct"] for r in subset),
                "boxed_correct": sum(r["boxed_correct"] for r in subset),
                "median_duplicate8_fraction": median(r["duplicate8_fraction"] for r in subset) if subset else None,
                "mean_duplicate8_fraction": mean(r["duplicate8_fraction"] for r in subset) if subset else None,
            }
            if subset:
                for name in risks:
                    values[name] = {
                        k: sum(r["tokens"] * r[name][k] for r in subset) / n
                        for k in ["objective", "teacher_kl", "anchor_kl"]
                    }
                    values[name]["objective_value_share"] = sum(
                        r["tokens"] * r[name]["objective"] for r in subset
                    ) / sum(r["tokens"] * r[name]["objective"] for r in rows)
            groups[label] = values
        capped_ids = sorted(r["request_id"] for r in rows if r["capped"])
        case_ids = set(capped_ids[:3]) | {r["request_id"] for r in rows if r["capped"] and r["native_correct"]}
        for row in bank:
            if row["request_id"] in case_ids:
                selected_cases.append({"arm": arm, **row})
        arms[arm] = {
            "groups": groups,
            "rows": rows,
            "risk_sha256": {name: digest(folder / f"current-{name}.json") for name in risks},
            "bank_sha256": digest(folder / "bank.jsonl"),
        }
        print(arm, json.dumps(groups["capped"]), flush=True)
    cases = ROOT / "online-loop-cases-outer7-20261102.jsonl"
    assert not cases.exists()
    with cases.open("w") as stream:
        for row in selected_cases:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "scope": __doc__,
        "outer": 7,
        "generation_parent_updates": 96,
        "after_bank_training_updates": 112,
        "arms": arms,
        "script_sha256": digest(Path(__file__)),
        "scores_sha256": completion["scores_sha256"],
        "selected_cases_sha256": digest(cases),
        "case_selection": "First3 capped request IDs plus any capped automatically-correct answer; illustrative.",
        "limits": [
            "Different kappas generate different states; cross-arm current KL is not fixed-state risk.",
            "Before/after within one arm uses its identical fixed bank.",
            "Duplicate8 measures exact token-window repetition, not semantic correctness.",
            "Equal-response weights are a calculated alternative; no such training intervention was run.",
        ],
    }
    out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
