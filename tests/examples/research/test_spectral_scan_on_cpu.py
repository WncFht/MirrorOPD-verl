# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Numerical equivalence and resume integrity for grouped exact spectrum scans."""

import json

import pytest
import torch
from safetensors.torch import save_file

from examples.mirroropd.research.studies.spectral.decomposition import weight_summary
from examples.mirroropd.research.studies.spectral.scan import iso_scalars, scan, thin_svd
from tests.examples.research.test_spectral_on_cpu import make_checkpoint


@pytest.mark.parametrize("shape", [(7, 7), (5, 13), (13, 5)])
def test_iso_scalars_match_full_frame_diagnostics(shape):
    torch.manual_seed(19)
    base = torch.randn(*shape, dtype=torch.float64)
    target = base + torch.randn_like(base) * 1e-4
    frame = thin_svd(base, transpose_wide=True)
    actual = iso_scalars(base, target, frame, torch.linalg.svdvals(target))
    expected = weight_summary(base, target, frame=frame)
    for field in ("delta_sigma", "rho_sigma", "kappa_spec"):
        assert actual[field] == pytest.approx(expected[field], rel=1e-11, abs=1e-14)
    torch.testing.assert_close((frame[0] * frame[1]) @ frame[2], base)
    zero = iso_scalars(base, base, frame, frame[1])
    assert zero["rho_sigma"] is None and zero["kappa_spec"] is None


def test_grouped_stages_share_frame_cache_and_reject_changed_bytes(tmp_path, monkeypatch):
    key = "model.layers.0.mlp.up_proj.weight"
    torch.manual_seed(9)
    b = torch.randn(5, 3, dtype=torch.float64)
    paths = {
        name: str(make_checkpoint(tmp_path / name, {key: b + i * 0.1}))
        for i, name in enumerate(("base", "teacher", "endpoint", "early"))
    }
    manifest = tmp_path / "manifest.json"
    job = dict(
        job_id=0,
        teacher="clean-sft",
        base=paths["base"],
        teacher_path=paths["teacher"],
        endpoint=paths["endpoint"],
        gradient_stages={"initial": paths["base"], "step16": paths["early"]},
    )
    manifest.write_text(json.dumps({"jobs": [job]}))
    original = torch.linalg.svd
    calls = []

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(torch.linalg, "svd", counted)
    scan(manifest, tmp_path / "cold", stages=True, cache=tmp_path / "cache")
    assert len(calls) == 1  # All teacher and student stages share this base frame.
    record = next(
        json.loads(p.read_text()) for p in (tmp_path / "cold").glob("*.json") if "targets" in json.loads(p.read_text())
    )
    assert record["svd_calls"] == 4
    scan(manifest, tmp_path / "warm", stages=True, cache=tmp_path / "cache")
    warm = next(
        json.loads(p.read_text()) for p in (tmp_path / "warm").glob("*.json") if "targets" in json.loads(p.read_text())
    )
    assert warm["svd_calls"] == 1 and record["targets"] == warm["targets"]
    calls.clear()
    scan(manifest, tmp_path / "warm", stages=True, cache=tmp_path / "cache", resume=True)
    assert not calls
    save_file({key: b + 2}, str(tmp_path / "teacher/model.safetensors"))
    with pytest.raises(ValueError):
        scan(manifest, tmp_path / "warm", stages=True, cache=tmp_path / "cache", resume=True)
    stop = tmp_path / "STOP"
    stop.touch()
    with pytest.raises(InterruptedError):
        scan(manifest, tmp_path / "stopped", stop_paths=[stop])


def test_excluded_condition_cannot_be_requested(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"jobs": [dict(job_id=7, teacher="controlled-raw")]}))
    with pytest.raises(ValueError, match="excluded"):
        scan(manifest, tmp_path / "out", job_ids=[7])


def test_matrix_shards_partition_without_duplicate_work(tmp_path):
    tensors = {f"model.layers.{i}.mlp.up_proj.weight": torch.randn(4, 3) for i in range(5)}
    paths = [str(make_checkpoint(tmp_path / name, tensors)) for name in ("base", "teacher", "endpoint")]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            dict(jobs=[dict(job_id=0, teacher="clean", base=paths[0], teacher_path=paths[1], endpoint=paths[2])])
        )
    )
    partitions = []
    for index in range(2):
        output = tmp_path / f"shard-{index}"
        scan(manifest, output, shards=2, shard_index=index, updates=True)
        for path in output.glob("*.json"):
            record = json.loads(path.read_text())
            if "targets" in record:
                for facts in record["targets"].values():
                    assert facts["update_spectrum"]["singular_values"] == [0.0, 0.0, 0.0]
                assert record["pairs"][0]["missing_spectrum"]["singular_values"] == [0.0, 0.0, 0.0]
        partitions.append({d["tensor"] for p in output.glob("*.json") if "targets" in (d := json.loads(p.read_text()))})
    assert not partitions[0] & partitions[1]
    assert partitions[0] | partitions[1] == set(tensors)
    with pytest.raises(ValueError, match="shards"):
        scan(manifest, tmp_path / "invalid", shards=2, shard_index=2)
