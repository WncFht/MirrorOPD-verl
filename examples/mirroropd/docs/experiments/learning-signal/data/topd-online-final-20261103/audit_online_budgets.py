"""Reconstruct completed online budgets from banks and every optimizer log row.

Counts are per arm, including shared initial-bank exposure. They are not total
project generation calls, semantic diversity estimates or task-performance
evidence. Comparisons use the latest outer completed by all four arms.
"""

import argparse
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
ARMS = ["reuse-k0", "reuse-k05", "reuse-k1", "fresh-k1"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lines(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def prompt_identity(row):
    return hashlib.sha256(json.dumps(row["prompt_ids"], separators=(",", ":")).encode()).hexdigest()


def load_completed(seed, arm, outer, initial, common_bank):
    if seed == 42 and outer == 1:
        folder = ROOT / "training" / initial[arm]
        done = json.loads((folder / "complete.json").read_text())
        protocol = json.loads((folder / "protocol.json").read_text())
        schedule = protocol["schedule"]
        by_id = {r["request_id"]: r for r in common_bank}
        rows = [by_id[rid] for rid in dict.fromkeys(rid for batch in schedule for rid in batch)]
        bank_hash = digest(ROOT / "bank.jsonl")
    else:
        parent = ROOT / "online" if seed == 42 else ROOT / f"replications/seed{seed}"
        folder = parent / arm / f"outer{outer:02d}"
        if not (folder / "complete.json").exists():
            return None
        done = json.loads((folder / "complete.json").read_text())
        bank_done = json.loads((folder / "bank-complete.json").read_text())
        bank_hash = digest(folder / "bank.jsonl")
        assert bank_hash == bank_done["sha256"] == done["bank_sha256"]
        rows = lines(folder / "bank.jsonl")
        assert len(rows) == bank_done["responses"] == len({r["request_id"] for r in rows})
        ids = [r["request_id"] for r in rows]
        schedule = [ids] * 16 if arm != "fresh-k1" else [ids[i : i + 64] for i in range(0, 1024, 64)]
    assert done["status"] == "complete" and done["updates"] == 16 * outer
    assert len(rows) == (1024 if arm == "fresh-k1" else 64)
    assert all(len(r["response_ids"]) == r["response_length"] for r in rows)
    lengths = {r["request_id"]: len(r["response_ids"]) for r in rows}
    assert len(schedule) == 16 and all(len(batch) == 64 for batch in schedule)
    expected_tokens = [sum(lengths[rid] for rid in batch) for batch in schedule]
    updates = lines(folder / "updates.jsonl")
    assert len(updates) == 16
    assert [u["update"] for u in updates] == list(range((outer - 1) * 16 + 1, outer * 16 + 1))
    assert [u["tokens"] for u in updates] == expected_tokens
    assert all(math.isfinite(u["objective_before_update"]) and math.isfinite(u["grad_norm_preclip"]) for u in updates)
    if seed == 42 and outer == 1:
        assert protocol["token_visits"] == sum(expected_tokens)
    else:
        assert done["token_visits"] == sum(expected_tokens)
        assert done["response_visits"] == 1024 and done["new_trajectories"] == len(rows)
    risk = json.loads((folder / "heldout-after.json").read_text())
    n = sum(r["tokens"] for r in risk["responses"])
    value = sum(r["tokens"] * r["teacher_kl"] for r in risk["responses"]) / n
    assert n == risk["tokens"] and abs(value - risk["teacher_kl"]) < 1e-10
    assert abs(value - done["heldout_teacher_kl_after"]) < 1e-10
    assert Path(done["checkpoint"]).is_dir()
    return {
        "outer": outer,
        "optimizer_steps": 16,
        "new_trajectory_records": len(rows),
        "response_visits": 1024,
        "generated_tokens": sum(lengths.values()),
        "training_token_visits": sum(expected_tokens),
        "fixed_heldout_teacher_kl": value,
        "prompt_hashes": [prompt_identity(row) for row in rows],
        "heldout_identity": sorted((x["request_id"], x["tokens"]) for x in risk["responses"]),
        "bank_sha256": bank_hash,
        "completion_sha256": digest(folder / "complete.json"),
        "updates_sha256": digest(folder / "updates.jsonl"),
        "risk_sha256": digest(folder / "heldout-after.json"),
    }


def cumulative(rows):
    prompts = [p for row in rows for p in row["prompt_hashes"]]
    assert len(prompts) == len(set(prompts)), "A prompt was reused across online outers"
    result = {
        name: sum(row[name] for row in rows)
        for name in [
            "optimizer_steps",
            "new_trajectory_records",
            "response_visits",
            "generated_tokens",
            "training_token_visits",
        ]
    }
    return result | {
        "outer": rows[-1]["outer"],
        "unique_prompt_token_sequences": len(set(prompts)),
        "fixed_heldout_teacher_kl": rows[-1]["fixed_heldout_teacher_kl"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists()
    plan = json.loads((ROOT / "online/plan.json").read_text())
    initial = {a["name"]: a["initial_label"] for a in plan["arms"]}
    common = lines(ROOT / "bank.jsonl")
    assert digest(ROOT / "bank.jsonl") == json.loads((ROOT / "bank-complete.json").read_text())["sha256"]
    all_rows, summaries, pending = {}, {}, []
    heldout = None
    for seed in [42, 43, 44]:
        groups = {}
        for arm in ARMS:
            values = []
            for outer in range(1, 9):
                row = load_completed(seed, arm, outer, initial, common)
                if row is None:
                    pending.append(f"seed{seed}/{arm}/outer{outer:02d}")
                    break
                if heldout is None:
                    heldout = row["heldout_identity"]
                assert row["heldout_identity"] == heldout
                values.append(row)
            groups[arm] = values
            all_rows[f"seed{seed}/{arm}"] = values
        horizon = min(map(len, groups.values()))
        if horizon:
            equal = {arm: cumulative(values[:horizon]) for arm, values in groups.items()}
            summaries[str(seed)] = {
                "common_completed_outer": horizon,
                "arms": equal,
                "fresh_to_reuse_k1_training_token_ratio": equal["fresh-k1"]["training_token_visits"]
                / equal["reuse-k1"]["training_token_visits"],
            }
            print(seed, json.dumps(summaries[str(seed)]), flush=True)
    for values in all_rows.values():
        for row in values:
            row.pop("prompt_hashes")
            row.pop("heldout_identity")
    report = {
        "scope": __doc__,
        "checked_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "rows": all_rows,
        "equal_horizon": summaries,
        "next_incomplete": pending,
        "script_sha256": digest(Path(__file__)),
        "limits": [
            "Checkpoint directories were checked for existence; model shards were not rehashed here.",
            "Budget evidence does not substitute for held-out generated task outcomes.",
            "Prompt counts use exact token sequences, not semantic novelty.",
            "Different arms/seeds may share initial records or questions; counts are per arm.",
            "Training visits exclude generation, cache preparation, risk readouts and evaluation compute.",
        ],
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
