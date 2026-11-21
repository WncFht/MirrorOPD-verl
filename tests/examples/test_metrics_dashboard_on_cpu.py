# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""History must load by explicit ID; partial appends must not corrupt live curves."""

import json

import pytest

from examples.mirroropd.tools.metrics_dashboard.__main__ import Dashboard, read_records


def test_partial_append_and_metric_namespaces(tmp_path):
    path = tmp_path / "metrics.jsonl"
    good = dict(
        step=1,
        data={
            "eval/rkl/public_initial/math/question_mean": 0.4,
            "eval/rkl/public_initial/dataset/math/question_mean": 0.3,
            "unused": 99,
        },
    )
    path.write_text(json.dumps(good) + '\n{"step":')
    rows, malformed = read_records(path)
    assert malformed == 0
    assert rows[0]["data"] == {k: v for k, v in good["data"].items() if k != "unused"}
    path.write_text(path.read_text() + "\n")
    assert read_records(path)[1] == 1


def test_catalog_does_not_read_unselected_history_and_cache_refreshes(tmp_path):
    study = tmp_path / "lane"
    (tmp_path / "plan.json").write_text(
        json.dumps(
            dict(outer=40, reuse=16, lanes=[dict(study=str(study), experiment="E124", lr=0.01, devices=[0, 1, 2, 3])])
        )
    )
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps(dict(runs=[dict(id="historical", metrics=str(tmp_path / "missing.jsonl"))])))
    dashboard = Dashboard(tmp_path, catalog)
    assert len(dashboard.catalog_data()["runs"]) == 2
    assert dashboard.metrics([])["runs"] == []
    assert dashboard.cache == {}
    assert dashboard.metrics(["E124"])["runs"][0]["records"] == []
    path = study / "training-sgd/E124/metrics.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(dict(step=0, data={"step/outer": 0})) + "\n")
    assert len(dashboard.metrics(["E124"])["runs"][0]["records"]) == 1
    with path.open("a") as file:
        file.write(json.dumps(dict(step=1, data={"step/outer": 1})) + "\n")
    assert len(dashboard.metrics(["E124"])["runs"][0]["records"]) == 2
    assert str(tmp_path / "missing.jsonl") not in dashboard.cache
    with pytest.raises(ValueError, match="Unknown"):
        dashboard.metrics(["not-registered"])


def test_pair_completion_overrides_later_shared_lane_failure(tmp_path):
    (tmp_path / "plan.json").write_text(json.dumps(dict(lanes=[])))
    status = tmp_path / "status.json"
    status.write_text(json.dumps(dict(state="failed", error="Later experiment failed")))
    receipt = tmp_path / "E102-complete.json"
    receipt.write_text("{}")
    catalog = tmp_path / "catalog.json"
    catalog.write_text(
        json.dumps(
            dict(
                runs=[
                    dict(
                        id="E102", metrics=str(tmp_path / "metrics.jsonl"), status=str(status), completion=str(receipt)
                    )
                ]
            )
        )
    )
    run = Dashboard(tmp_path, catalog).catalog_data()["runs"][0]
    assert run["state"] == "complete"
    assert run["error"] is None
