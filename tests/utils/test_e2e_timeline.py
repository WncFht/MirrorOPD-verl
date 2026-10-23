# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from verl.utils.profiler import timeline


def records(path):
    return [json.loads(s) for p in path.glob("events-*.jsonl") for s in p.read_text().splitlines()]


def test_disabled_is_noop(monkeypatch, tmp_path):
    monkeypatch.delenv("VERL_E2E_TIMELINE_DIR", raising=False)
    with timeline.span("unused"):
        pass
    assert not list(tmp_path.iterdir())


def test_threads_and_exception_are_recorded(monkeypatch, tmp_path):
    monkeypatch.setenv("VERL_E2E_TIMELINE_DIR", str(tmp_path))
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(lambda i: timeline.emit("worker", rank=i), range(20)))
    with pytest.raises(ValueError):
        with timeline.span("failure"):
            raise ValueError("expected")
    rows = records(tmp_path)
    assert len(rows) == 21 and rows[-1]["status"] == "error"
    assert sorted(r["rank"] for r in rows if r["name"] == "worker") == list(range(20))
    assert all(r["start_ns"] <= r["end_ns"] for r in rows)


def test_async_observer_preserves_request_and_output(monkeypatch, tmp_path):
    monkeypatch.setenv("VERL_E2E_TIMELINE_DIR", str(tmp_path))

    class Server:
        replica_rank = 2
        global_steps = 3

        @timeline.observe("rollout/request", "vllm")
        async def generate(self, prompt_ids, request_id):
            await asyncio.sleep(0)
            return SimpleNamespace(token_ids=[4, 5], extra_fields={"finish_reason": "length"}, stop_reason="completed")

    async def run():
        return await asyncio.gather(Server().generate([1, 2, 3], "a"), Server().generate([1], "b"))

    results = asyncio.run(run())
    assert [len(r.token_ids) for r in results] == [2, 2]
    rows = records(tmp_path)
    assert {r["request_id"]: r["input_tokens"] for r in rows} == {"a": 3, "b": 1}
    assert all(r["output_tokens"] == 2 and r["finish_reason"] == "length" and r["replica"] == 2 for r in rows)
    assert all("prompt_ids" not in r for r in rows)
