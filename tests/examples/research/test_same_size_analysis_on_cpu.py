# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Replay accounting and pre-update loss checks reject inconsistent extension evidence."""

from pathlib import Path

import pytest

from examples.mirroropd.research.core.artifacts import file_hash, fingerprint
from examples.mirroropd.research.core.io import dump, read
from examples.mirroropd.research.studies.boundary import same_size_analysis as audit


@pytest.mark.parametrize("replica", [False, True])
def test_audit_counts_only_new_updates_and_rejects_corrupt_loss(tmp_path, monkeypatch, replica):
    groups = list("ABCDEFGHIJKLMNOPQRST") + ["M01", "M02", "M03", "M04", "Code01", "Code02", "IF01", "IF02"]
    bank = [
        dict(id=i, group=g, source="bank", response_ids=[1]) for i, g in enumerate(g for g in groups for _ in range(16))
    ]
    rows = [
        dict(
            id=r["id"], group=r["group"], source="bank", length=1, stats={"all": {"rkl": 102.4, "js": 10.24, "hq": 4.0}}
        )
        for r in bank
    ]
    monkeypatch.setattr(audit, "bank", lambda _: bank)
    monkeypatch.setattr(audit, "extension_plan", lambda *a: None)
    spec = dict(
        output=str(tmp_path / "new"),
        legacy_output=str(tmp_path / "old"),
        scoring={"groups": groups},
        models={"r1": ["repo", "revision"]},
        training={"horizon": 1024},
        replay_acceptance={"max_loss_absolute_tolerance": 1e-4, "per_prompt_rkl_absolute_tolerance": 1e-4},
        arms={
            "CONT-J": dict(start=64, prior_completed_stop=128, replay_through=128, new_updates=192, teacher="justrl")
        },
    )
    if replica:
        spec["execution_condition"] = "deterministic_from_initial"
        spec["arms"]["CONT-J"].update(start=0, new_updates=320)
    start = spec["arms"]["CONT-J"]["start"]
    plan = tmp_path / "plan.json"
    dump(plan, spec)
    root = Path(spec["output"]) / "arms/CONT-J"
    old = Path(spec["legacy_output"]) / "arms/CONT-J"
    history = [dict(step=s, group=chr(65 + (s - 1) // 16), loss=0.1, grad_norm=1.0) for s in range(start + 1, 321)]
    dump(
        root / "protocol.json",
        dict(
            extension_plan=spec,
            optimizer_exports=True,
            weights_dtype="FP32",
            flash_attention_deterministic=replica,
            resume={"fresh": replica},
        ),
    )
    dump(root / "training.json", history)
    dump(old / "training.json", [r for r in history if 65 <= r["step"] <= 128])
    for s in sorted({0, *range(start, 321, 16)}):
        dump(root / f"fixed-{s:03d}.json", rows)
    dump(old / "fixed-128.json", rows[:128] + rows[320:])
    dump(root / "replay-acceptance.json", dict(step=128, optimizer_reset=False, loss_max_error=0.0, risk_max_error=0.0))
    if replica:
        reference = [
            dict(r, stats={"all": dict(r["stats"]["all"], rkl=r["stats"]["all"]["rkl"] + 1.024)})
            for r in rows[:128] + rows[320:]
        ]
        dump(old / "fixed-128.json", reference)
        dump(
            root / "replay-acceptance.json",
            dict(
                step=128,
                optimizer_reset=False,
                loss_max_error=0.0,
                risk_max_error=0.001,
                reference_policy="diagnostic",
                exact_old_trajectory_claim=False,
            ),
        )
        dump(
            root / "deterministic-gate.json",
            dict(
                scientific_updates=0,
                unequal_gradient_tensors=0,
                reference_loss=0.1,
                candidate_loss=0.1,
                repeat_loss=0.1,
                relative_l2=0.0,
                cosine=1.0,
            ),
        )
    weights = tmp_path / "weights"
    dump(weights / "config.json", {})
    (weights / "weights.safetensors").write_bytes(b"fixture")
    files = [dict(name="weights.safetensors", bytes=7, sha256=file_hash(weights / "weights.safetensors"))]
    wid = fingerprint(dict(weights=files, config=file_hash(weights / "config.json")))
    dump(weights / "complete.json", dict(step=320, optimizer_saved=True, files=files, weight_identity=wid))
    dump(
        root / "resume/complete.json",
        dict(
            step=320,
            bytes=1,
            weight_identity=wid,
            signature=fingerprint(
                dict(
                    models=spec["models"],
                    training=spec["training"],
                    teacher="justrl",
                    bank=fingerprint(bank),
                    **({"execution_condition": spec["execution_condition"]} if replica else {}),
                )
            ),
        ),
    )
    (root / "resume/optimizer.pt").write_bytes(b"x")
    dump(
        root / "complete.json",
        dict(
            plan_identity=fingerprint(spec),
            stop=320,
            start=start,
            new_updates=320 if replica else 192,
            weights=str(weights),
        ),
    )
    audit.analyze(plan)
    proof = read(Path(spec["output"]) / "analysis/verification.json")
    assert proof["complete"] and proof["new_scientific_updates"] == (320 if replica else 192)
    assert proof["verified"][0]["replay_updates"] == (0 if replica else 64)
    assert proof["verified"][0]["payload_hashes_rechecked"] is False
    history[0]["loss"] = 0.2
    dump(root / "training.json", history)
    with pytest.raises(AssertionError):
        audit.analyze(plan)
