"""Validate saved four-GPU acceptance evidence; uses only Python stdlib."""

import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
results = json.loads((root / "results.json").read_text())
expected = [
    "full-qwen2",
    "full-qwen3-fused",
    "sampled-current-is",
    "sampled-frozen-ppo",
    "prepare-validation",
    "real-sampled",
    "real-full",
]
assert [r["name"] for r in results] == expected
assert all(r["exit_code"] == 0 for r in results)
summary = {"device_indices": [0, 1, 2, 3], "integration": {}, "real_runs": {}}
for name in expected[:2]:
    data = json.loads((root / (name + ".json")).read_text())
    assert data["world_size"] == 4
    for key in ["checkpoint_roundtrip", "scheduler_optimizer_roundtrip", "resumed_next_update_equivalent"]:
        assert data[key] is True
    assert len(data["records"]) == 3
    errors = []
    for row, count in zip(data["records"], [16, 12, 16], strict=True):
        assert sum(row["position_token_counts"]) == count
        error = abs(sum(row["position_kl_sums"]) / count - row["loss"])
        assert error < 1e-7
        errors.append(error)
    summary["integration"][name] = {
        "passed": True,
        "global_token_counts": [16, 12, 16],
        "position_reconstruction_max_abs_error": max(errors),
        "resumed_next_update_equivalent": True,
    }
for name, mode, steps in [
    ("sampled-current-is", "token_is/current", 14),
    ("sampled-frozen-ppo", "token_ppo/rollout", 8),
]:
    marker = f"test_shared_teacher_fsdp2 passed: {mode}, optimizer steps after resume={steps}"
    assert marker in (root / (name + ".log")).read_text()
    summary["integration"][name] = {"passed": True, "optimizer_steps_after_resume": steps}
for mode in ["sampled", "full"]:
    run = root / ("real-" + mode)
    records = [json.loads(line)["data"] for line in (run / "metrics.jsonl").read_text().splitlines()]
    updates = [r for r in records if "optim/actor/update_applied" in r]
    assert len(updates) == 4
    for key in ["step/update", "step/optimizer"]:
        assert [r[key] for r in updates] == [1, 2, 3, 4]
    assert [r["step/outer"] for r in updates] == [1, 1, 2, 2]
    assert all(r["optim/actor/update_applied"] == 1 for r in updates)
    assert all(r["optim/actor/update_skipped"] == 0 for r in updates)
    for r in updates:
        for key in ["optim/actor/loss", "optim/actor/grad_norm_preclip"]:
            assert math.isfinite(r[key]), (mode, key, r[key])
    lr_next = [r["step/update"] for r in updates if "optim/actor/lr_next" in r]
    assert lr_next == [2, 4]
    flat = {k: v for r in records for k, v in r.items()}
    validation = {name: flat[f"eval/{name}/acc/mean"] for name in ["math500", "amc23", "aime25"]}
    assert all(math.isfinite(value) for value in validation.values())
    result = {
        "outer_steps": 2,
        "optimizer_updates": 4,
        "lr_next_at_updates": lr_next,
        "losses": [r["optim/actor/loss"] for r in updates],
        "grad_norms": [r["optim/actor/grad_norm_preclip"] for r in updates],
        "validation": validation,
        "source_commit": (run / "source_commit.txt").read_text().strip(),
    }
    if mode == "full":
        errors = []
        counts = []
        for r in updates:
            prefixes = [f"distill/full_kl_position/p{i * 10:02d}_{(i + 1) * 10:02d}/" for i in range(10)]
            sums = [r[p + "sum"] for p in prefixes]
            tokens = [r[p + "token_count"] for p in prefixes]
            assert all(t > 0 for t in tokens)
            for p, s, n in zip(prefixes, sums, tokens, strict=True):
                assert abs(s / n - r[p + "mean"]) < 1e-7
            error = abs(sum(sums) / sum(tokens) - r["distill/full_kl"])
            assert error < 1e-7
            errors.append(error)
            counts.append(sum(tokens))
        result["position_reconstruction_max_abs_error"] = max(errors)
        result["position_token_counts"] = counts
    summary["real_runs"][mode] = result
(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
