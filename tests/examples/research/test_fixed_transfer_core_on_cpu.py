# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""CPU regressions for scientific identities, resume safety and real transport retries."""

import asyncio
import json
from dataclasses import replace

import pytest

from examples.mirroropd.research.core.artifacts import complete, receipt, records
from examples.mirroropd.research.core.io import read
from examples.mirroropd.research.core.paths import MIRROROPD
from examples.mirroropd.research.studies.fixed_transfer.evaluation import collect
from examples.mirroropd.research.studies.fixed_transfer.spec import Evaluation, Model, Study, fingerprint

PLAN = MIRROROPD / "research/studies/fixed_transfer/plans/qwen3-lineage.json"


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
        replace(study, gpus=(4, 4))
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
    from examples.mirroropd.research.core.io import dump as write
    from examples.mirroropd.research.studies.fixed_transfer.runner import finished, ready, validate_graph

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


def test_external_queue_waits_without_launching_duplicate(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from examples.mirroropd.research.core.io import dump as write
    from examples.mirroropd.research.studies.fixed_transfer import runner

    marker = tmp_path / "baseline.json"
    job = dict(name="baseline", marker=str(marker), dependencies=[], external=True)
    monkeypatch.setattr(runner, "jobs", lambda *_: [job])
    monkeypatch.setattr(runner.time, "sleep", lambda _: write(marker, {}))
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **kw: pytest.fail("External job launched twice"))
    runner.queue(SimpleNamespace(gpus=(0, 1, 2, 3)), None, "training", tmp_path, "test")
    assert read(tmp_path / "jobs-complete.json")["done"] == ["baseline"]


def test_resource_recovery_checks_allocated_devices(tmp_path, monkeypatch):
    from examples.mirroropd.research.core.io import dump as write
    from examples.mirroropd.research.runtime.resources import Resource

    config = tmp_path / "front.json"
    write(config, dict(gpus=[0, 1, 2, 3], state_dir=str(tmp_path), lock=str(tmp_path / "lock")))
    monkeypatch.setenv("MIRROROPD_RESOURCE_CONFIG", str(config))
    resource = Resource()
    assert resource.environment()["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"
    state = dict(health="running", status=dict(cuda_visible_devices="4,5,6,7", workers={}))
    assert not resource.restored(state)
    state["status"] = dict(
        cuda_visible_devices="0,1,2,3",
        workers={str(i): dict(phase="running", allocated_mib=2000) for i in range(4)},
    )
    assert resource.restored(state)


def test_resource_lease_waits_for_owner_release(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor, TimeoutError
    from threading import Event

    from examples.mirroropd.research.core.io import dump as write
    from examples.mirroropd.research.runtime.resources import Resource

    config = tmp_path / "resource.json"
    write(config, dict(gpus=[4, 5, 6, 7], lock=str(tmp_path / "lease"), state_dir=str(tmp_path)))
    monkeypatch.setenv("MIRROROPD_RESOURCE_CONFIG", str(config))
    owner, waiter = Resource(), Resource()
    lease = owner.acquire()
    entered = Event()

    def acquire():
        entered.set()
        return waiter.acquire(wait=True)

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(acquire)
        try:
            assert entered.wait(2)
            with pytest.raises(TimeoutError):
                future.result(timeout=0.1)
            with pytest.raises(BlockingIOError):
                waiter.acquire()
        finally:
            lease.close()
        next_lease = future.result(timeout=2)
        next_lease.close()


def test_status_does_not_confuse_stale_queue_with_live_worker(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from examples.mirroropd.research.core.io import dump as write
    from examples.mirroropd.research.studies.fixed_transfer import runner

    control = tmp_path / "control" / "run"
    control.mkdir(parents=True)
    write(control / "owner.json", dict(identity=[123, "old"]))
    write(control / "queue.json", dict(running=["train-a"]))
    monkeypatch.setattr(runner, "identity", lambda pid: [123, "new"])
    study = SimpleNamespace(directory=tmp_path, name="test")
    item = runner.status(study)["controllers"][0]
    assert item["state"] == "stopped" and not item["alive"]
    (control / "owner.json").unlink()
    write(control / "resource-wait.json", dict(identity=[123, "new"]))
    assert runner.status(study)["controllers"][0]["state"] == "waiting_resource"
    write(control / "owner.json", dict(identity=[123, "new"]))
    assert runner.status(study)["controllers"][0]["state"] == "running"
    write(control / "jobs-complete.json", dict(done=["train-a"]))
    assert runner.status(study)["controllers"][0]["state"] == "recovering"
    monkeypatch.setattr(runner, "identity", lambda pid: None)
    assert runner.status(study)["controllers"][0]["state"] == "jobs_complete_recovery_unverified"
    write(control / "recovered.json", dict(health="running"))
    assert runner.status(study)["controllers"][0]["state"] == "completed"
    write(control / "superseded.json", dict(reason="replacement queue"))
    assert runner.status(study)["controllers"][0]["state"] == "superseded"


def test_status_finds_shared_baseline_without_local_alias(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from examples.mirroropd.research.core.io import dump as write
    from examples.mirroropd.research.studies.fixed_transfer import runner

    study = SimpleNamespace(
        directory=tmp_path / "study", cache_directory=tmp_path / "cache", name="reused", models={"teacher": None}
    )
    inputs = dict(weights="fixed", prompts="same")
    monkeypatch.setattr(runner, "evaluation_inputs", lambda *_: inputs)
    write(study.directory / "prepared/complete.json", {})
    cached = study.cache_directory / "evaluations" / fingerprint(inputs)
    receipt(cached, inputs)
    write(cached / "progress.json", dict(completed=2520, expected=2520))
    result = runner.status(study)
    assert len(result["evaluations"]) == 1
    assert result["evaluations"][0]["complete"]
    assert not (study.directory / "evaluations").exists()
    receipt(cached, dict(weights="different"))
    with pytest.raises(ValueError, match="Completion identity mismatch"):
        runner.status(study)


def test_banks_phase_has_no_accuracy_jobs(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from examples.mirroropd.research.core.io import dump as write
    from examples.mirroropd.research.studies.fixed_transfer import runner

    write(tmp_path / "prepared/complete.json", dict(bank_key="frozen"))
    monkeypatch.setattr(runner, "bank_directory", lambda study: tmp_path / "bank")
    jobs = runner.jobs(SimpleNamespace(directory=tmp_path), "banks")
    assert len(jobs) == 4
    assert all(j["args"][0] == "bank" for j in jobs)


def test_shared_system_prefix_keeps_question_overlap_check_valid():
    from examples.mirroropd.research.studies.fixed_transfer.data import apply_prompt_policy, user_content

    policy = dict(system_prompt="detailed thinking off")
    first = apply_prompt_policy([dict(role="user", content="Question A")], policy)
    second = apply_prompt_policy([dict(role="user", content="Question B")], policy)
    assert first[0] == second[0]
    assert user_content(first) != user_content(second)
    assert apply_prompt_policy(first, None) is first
    with pytest.raises(ValueError, match="existing system"):
        apply_prompt_policy(first, policy)


def test_cache_receipt_does_not_hide_missing_required_rows(tmp_path):
    from examples.mirroropd.research.core.io import dump as write
    from examples.mirroropd.research.studies.fixed_transfer.runner import finished

    write(tmp_path / "complete.json", {})
    job = dict(name="cache", marker=str(tmp_path / "complete.json"), required_files=[str(tmp_path / "new-row.pt")])
    assert not finished(job)
    (tmp_path / "new-row.pt").write_bytes(b"row")
    assert finished(job)


@pytest.mark.parametrize("phase", ["fit", "targets", "full"])
def test_content_bound_jobs_wait_for_missing_banks(tmp_path, monkeypatch, phase):
    from examples.mirroropd.research.studies.fixed_transfer import runner

    state = {"banks_ready": False}
    calls = []

    def jobs(study, requested):
        assert requested == "banks", "Do not construct teacher identities before banks exist"
        return [{"name": "bank-0"}]

    def queue(study, plan, requested, directory, tag):
        calls.append(requested)
        if requested in {"banks", "baselines"}:
            state["banks_ready"] = True
        else:
            assert state["banks_ready"]

    monkeypatch.setattr(runner, "jobs", jobs)
    monkeypatch.setattr(runner, "finished", lambda _: state["banks_ready"])
    monkeypatch.setattr(runner, "queue", queue)
    runner.queue_with_banks(None, None, phase, tmp_path, "test")
    assert calls == ["baselines" if phase == "full" else "banks", phase]
    calls.clear()
    runner.queue_with_banks(None, None, phase, tmp_path, "test")
    assert calls == [phase]


def test_mode_probes_do_not_require_a_bank_or_accuracy(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from examples.mirroropd.research.studies.fixed_transfer import mode_probe, runner

    study = SimpleNamespace(directory=tmp_path, student="student", models={"student": None, "teacher": None})
    monkeypatch.setattr(mode_probe, "probe_inputs", lambda study, name: {"model": name, "prompt": "fixed"})
    monkeypatch.setattr(runner, "bank_directory", lambda _: pytest.fail("Probe must not read a trajectory bank"))
    monkeypatch.setattr(runner, "evaluation_inputs", lambda *_: pytest.fail("Probe must not create accuracy jobs"))
    jobs = runner.jobs(study, "probes")
    assert len(jobs) == 1 and jobs[0]["name"] == "probe-teacher"
    assert jobs[0]["dependencies"] == []
    assert jobs[0]["module"].endswith("mode_probe")


def test_endpoint_only_queue_requires_existing_weight_receipts(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from examples.mirroropd.research.core.io import dump
    from examples.mirroropd.research.studies.fixed_transfer import runner, training

    checkpoint = tmp_path / "weights/complete.json"

    def endpoint_jobs(study):
        return [
            dict(
                name="eval-test",
                args=["endpoint"],
                producer="train-test",
                dependencies=[],
                ready_file=str(checkpoint),
                ready_inputs={"step": 64},
            )
        ]

    monkeypatch.setattr(training, "endpoint_jobs", endpoint_jobs)
    with pytest.raises(ValueError, match="requires saved weights"):
        runner.jobs(SimpleNamespace(), "endpoints")
    dump(checkpoint, {"inputs": {"step": 64}, "weight_identity": "fixed"})
    jobs = runner.jobs(SimpleNamespace(), "endpoints")
    assert len(jobs) == 1 and "producer" not in jobs[0]
    assert jobs[0]["ready_inputs"] == {"step": 64}


def test_native_template_controls_preserve_student_vocabulary(monkeypatch, tmp_path):
    from types import SimpleNamespace

    import transformers

    from examples.mirroropd.research.studies.fixed_transfer.data import prompt_tokenizers

    class Tokenizer:
        def __init__(self, template, vocab):
            self.chat_template = template
            self.vocab = vocab

        def get_vocab(self):
            return self.vocab

    student = Tokenizer("student-default", {"token": 1})
    source = Tokenizer("specialist-native", {"token": 1})
    study = SimpleNamespace(
        student="s",
        prompt_policy={"template_model": "t"},
        models={
            "s": SimpleNamespace(directory=tmp_path / "s"),
            "t": SimpleNamespace(directory=tmp_path / "t"),
        },
    )
    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", lambda path, **kw: student if path.name == "s" else source
    )
    adapted, native = prompt_tokenizers(study)
    assert adapted is student and native is source
    assert adapted.chat_template == "specialist-native" and adapted.get_vocab() == {"token": 1}
    source.vocab = {"token": 2}
    with pytest.raises(ValueError, match="token-to-ID"):
        prompt_tokenizers(study)


def test_resource_waiters_are_priority_fifo_and_drop_dead_identities(tmp_path, monkeypatch):
    from examples.mirroropd.research.core.io import dump
    from examples.mirroropd.research.runtime import resources

    monkeypatch.setattr(resources, "identity", lambda pid: [pid, "live"] if pid != 4 else None)
    for pid, priority, arrival in [(1, 100, 1), (2, 20, 3), (3, 20, 2), (4, 0, 0)]:
        dump(tmp_path / f"{pid}.json", dict(identity=[pid, "live"], priority=priority, enqueued_ns=arrival))
    assert [entry[2] for entry in resources.live_waiters(tmp_path)] == [3, 2, 1]
    assert not (tmp_path / "4.json").exists()


def test_resource_wait_acquisition_removes_its_ticket(tmp_path, monkeypatch):
    from examples.mirroropd.research.runtime import resources

    resource = resources.Resource.__new__(resources.Resource)
    resource.config = {"lock": str(tmp_path / "gpu.lock")}
    attempts = []

    def acquire(handle, flags):
        attempts.append(flags)
        if len(attempts) == 1:
            raise BlockingIOError()

    monkeypatch.setattr(resources.fcntl, "flock", acquire)
    monkeypatch.setattr(resources.time, "sleep", lambda _: None)
    with resource.acquire(wait=True):
        assert len(attempts) == 2
        assert not list((tmp_path / "gpu.waiters").glob("*.json"))


def test_waiter_can_disappear_between_listing_and_read(tmp_path, monkeypatch):
    from examples.mirroropd.research.runtime import resources

    ticket = tmp_path / "finished.json"
    ticket.write_text("{}")

    def disappeared(path):
        path.unlink()
        raise FileNotFoundError(path)

    monkeypatch.setattr(resources, "read", disappeared)
    assert resources.live_waiters(tmp_path) == []
