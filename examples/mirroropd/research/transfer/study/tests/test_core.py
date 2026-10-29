# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""CPU regressions for scientific identities, resume safety and real transport retries."""

import asyncio
import json
from dataclasses import replace
from pathlib import Path

import pytest

from ..artifacts import complete, read, receipt, records
from ..evaluation import collect
from ..spec import Evaluation, Model, Study, fingerprint

PLAN = Path(__file__).parents[1] / "plans/qwen3-lineage.json"


def test_pinned_protocol_and_budgets():
    study = Study.load(PLAN)
    assert study.evaluation.samples == 4
    assert sum(a.updates for a in study.arms) == 2112
    assert len(study.models) == 10 and len(study.arms) == 9
    assert study.training.denominator == 16384
    with pytest.raises(ValueError):
        Evaluation(samples=1)
    with pytest.raises(ValueError):
        Model("bad", "Qwen/Qwen3-1.7B", "main", "unused")
    with pytest.raises(ValueError):
        replace(study, gpus=(0, 1, 2, 3))
    assert fingerprint({"ids": (1, 2)}) == fingerprint({"ids": [1, 2]})


def test_receipt_rejects_changed_inputs(tmp_path):
    receipt(tmp_path, dict(model="a", bank="x"), rows=16)
    assert complete(tmp_path, dict(model="a", bank="x"))
    with pytest.raises(ValueError):
        complete(tmp_path, dict(model="b", bank="x"))


def test_only_incomplete_final_append_is_repaired(tmp_path):
    path = tmp_path / "records.jsonl"
    path.write_bytes(b'{"id": 1}\n{"id":')
    assert records(path, repair=True) == [dict(id=1)]
    assert records(path) == [dict(id=1)]
    assert read(tmp_path / "append-repair.json")["valid_records"] == 1
    path.write_text('{bad}\n{"id":2}\n')
    with pytest.raises(json.JSONDecodeError):
        records(path, repair=True)


def test_collection_retries_identical_requests_and_resumes(tmp_path):
    async def exercise():
        from aiohttp import web

        calls = []

        async def handle(request):
            body = await request.json()
            calls.append(body)
            if len(calls) == 1:
                request.transport.abort()
                return web.json_response({})
            ids = body.get("allowed_token_ids") or [42, 151645]
            return web.json_response(
                dict(choices=[dict(text=r"\boxed{2}", token_ids=ids, finish_reason="stop", stop_reason=ids[-1])])
            )

        app = web.Application()
        app.router.add_post("/v1/completions", handle)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        prompts = [
            dict(
                prompt_id="synthetic:0", seed_index=0, prompt_ids=[1, 2], input="1+1?", data_source="MATH-500", gts="2"
            )
        ]
        profile = Evaluation(response_cap=16, model_length=32, stop_ids=(151643, 151645))
        try:
            assert await collect(prompts, profile, port, "synthetic", tmp_path, 2) == 4
            assert calls[0] == calls[1]
            before = len(calls)
            assert await collect(prompts, profile, port, "synthetic", tmp_path, 2) == 4
            assert len(calls) == before + len(profile.stop_ids)
            assert len(records(tmp_path / "records.jsonl")) == 4
            assert read(tmp_path / "metrics.json")["eval/math500/acc/mean"] == 1.0
        finally:
            await runner.cleanup()

    asyncio.run(exercise())


def test_queue_rejects_changed_science_and_mismatched_readout(tmp_path):
    from ..artifacts import write
    from ..runner import finished, ready, validate_graph

    checkpoint = tmp_path / "checkpoint.json"
    result = tmp_path / "evaluation.json"
    write(checkpoint, dict(inputs=dict(teacher="teacher-a", step=64), weight_identity="weights-a"))
    write(result, dict(inputs=dict(weights="weights-b")))
    job = dict(
        name="readout",
        marker=str(result),
        ready_file=str(checkpoint),
        ready_inputs=dict(teacher="teacher-a", step=64),
        completion_weight_from_ready=True,
    )
    assert ready(job)
    with pytest.raises(ValueError, match="different checkpoint"):
        finished(job)
    job["ready_inputs"]["teacher"] = "teacher-b"
    with pytest.raises(ValueError, match="inputs changed"):
        ready(job)
    with pytest.raises(ValueError, match="cycle"):
        validate_graph([dict(name="a", dependencies=["b"]), dict(name="b", dependencies=["a"])])
    with pytest.raises(ValueError, match="Unknown dependency"):
        validate_graph([dict(name="a", dependencies=["missing"])])
    validate_graph([dict(name="train", dependencies=[]), dict(name="step64", dependencies=[], producer="train")])
