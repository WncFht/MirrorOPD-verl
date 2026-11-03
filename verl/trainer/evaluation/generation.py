# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Per-request evaluation through verl's native single-turn loop and vLLM transport."""

import asyncio
import copy
import itertools
import json
import time
import uuid
from collections import OrderedDict
from contextlib import aclosing

from omegaconf import OmegaConf, open_dict

from verl.experimental.agent_loop.agent_loop import DictConfigWrap
from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop
from verl.experimental.rollout_policy.collector import UnsafeDrainError
from verl.experimental.rollout_policy.sessions import LocalPhaseClient
from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.workers.rollout.router import GlobalRequestLoadBalancer

from .artifacts import GENERATION, ArtifactStore, identity
from .generation_contract import engine_contract, sample_identity


async def _rolling_results(requests, worker, *, concurrency, timeout, abort):
    """Refill as requests finish; on failure or consumer exit, drain owned requests."""
    iterator, pending = iter(requests), {}

    def admit(count):
        for args in itertools.islice(iterator, count):
            pending[asyncio.create_task(worker(*args))] = time.monotonic()

    try:
        admit(concurrency)
        while pending:
            now = time.monotonic()
            if any(not task.done() and now - started >= timeout for task, started in pending.items()):
                raise TimeoutError("Evaluation request exceeded its wall-time budget")
            done = {task for task in pending if task.done()}
            if not done:
                remaining = timeout - (time.monotonic() - min(pending.values()))
                if remaining <= 0:
                    raise TimeoutError("Evaluation request exceeded its wall-time budget")
                done, _ = await asyncio.wait(pending, timeout=remaining, return_when=asyncio.FIRST_COMPLETED)
                if not done:
                    raise TimeoutError("Evaluation request exceeded its wall-time budget")
            outputs = []
            for task in done:
                outputs.append(task.result())
                pending.pop(task)
            admit(len(done))
            for result in outputs:
                yield result
    finally:
        if pending:
            try:
                await asyncio.wait_for(abort(), 60)
            except BaseException as exc:
                raise UnsafeDrainError(
                    "Generation abort was not acknowledged; terminate before model switching"
                ) from exc
            _, unsettled = await asyncio.wait(pending, timeout=60)
            if unsettled:
                raise UnsafeDrainError("Generation requests did not drain; terminate before model switching")
            await asyncio.gather(*pending, return_exceptions=True)


async def bounded_results(requests, worker, *, concurrency, timeout, abort):
    """Own admission independently of publication, with at most one window buffered.

    Closing the consumer cancels admission and acknowledges/drains every owned RPC
    before returning. Backpressure bounds memory even if publication stops entirely.
    """
    if concurrency < 1 or timeout <= 0:
        raise ValueError("Positive concurrency and request timeout are required")
    queue = asyncio.Queue(maxsize=concurrency)

    async def produce():
        async with aclosing(
            _rolling_results(requests, worker, concurrency=concurrency, timeout=timeout, abort=abort)
        ) as stream:
            async for result in stream:
                await queue.put(result)

    producer = asyncio.create_task(produce())
    getter = None
    try:
        while True:
            if producer.done():
                producer.result()
                if queue.empty():
                    break
            getter = asyncio.create_task(queue.get())
            await asyncio.wait((getter, producer), return_when=asyncio.FIRST_COMPLETED)
            if producer.done():
                producer.result()
                if not getter.done() and queue.empty():
                    break
            yield await getter
            getter = None
    finally:
        if getter is not None:
            getter.cancel()
            await asyncio.gather(getter, return_exceptions=True)
        if not producer.done():
            producer.cancel()
        try:
            await producer
        except asyncio.CancelledError:
            pass


async def settled_thread_call(function, *args, **kwargs):
    """Cancellation must acknowledge an owned disk writer before releasing its caller."""
    task = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        try:
            await asyncio.wait_for(asyncio.shield(task), 60)
        except (TimeoutError, asyncio.CancelledError) as exc:
            raise UnsafeDrainError("Generation publication did not settle; terminate before model switching") from exc
        raise


class CheckedClient(LocalPhaseClient):
    def __init__(self, config, servers, *, max_model_len, prompt_limit, token_mapping=None, router=None):
        super().__init__(config, servers)
        self.max_model_len, self.prompt_limit = max_model_len, prompt_limit
        self.token_mapping = token_mapping
        self.router = router

    async def generate(self, request_id, *, prompt_ids, sampling_params, **kwargs):
        cap = sampling_params["max_tokens"]
        if len(prompt_ids) > self.prompt_limit or len(prompt_ids) + cap > self.max_model_len:
            raise ValueError("Evaluation input exceeds the declared context; refusing silent response truncation")
        mapping = self.token_mapping
        if mapping is not None and not mapping.is_identity:
            prompt_ids = mapping.tokens(prompt_ids)
            sampling_params = dict(sampling_params)
            if sampling_params.get("stop_token_ids"):
                sampling_params["stop_token_ids"] = mapping.tokens(sampling_params["stop_token_ids"])
        route = None
        try:
            if self.router is not None:
                route, _ = self.router.acquire_server(request_id)
                sampling_params = {**sampling_params, "_rollout_replica": int(route)}
            output = await super().generate(
                request_id, prompt_ids=prompt_ids, sampling_params=sampling_params, **kwargs
            )
        finally:
            if route is not None:
                self.router.release_server(route, request_id=request_id)
        if mapping is not None and not mapping.is_identity:
            output = output.model_copy(update={"token_ids": mapping.tokens(output.token_ids, reverse=True)})
        return output


class CachedSingleTurn(SingleTurnAgentLoop):
    """Bounded prompt cache; tokenization still uses the native Continuous Token builder."""

    def __init__(self, *args, cache_size=256, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache_size = cache_size
        self._prompt_cache = OrderedDict()

    def _cap_text_prompt_length(self, prompt_ids):
        if len(prompt_ids) > self.rollout_config.prompt_length:
            raise ValueError("Evaluation prompt exceeds its declared budget; truncation is disabled")
        return prompt_ids

    async def ct_build_initial_tokens(self, messages, **kwargs):
        key = json.dumps(messages, ensure_ascii=False, sort_keys=True)
        if key not in self._prompt_cache:
            value = await super().ct_build_initial_tokens(messages, **kwargs)
            self._prompt_cache[key] = tuple(value)
            if len(self._prompt_cache) > self.cache_size:
                self._prompt_cache.popitem(last=False)
        self._prompt_cache.move_to_end(key)
        return list(self._prompt_cache[key])


class NativeGenerator:
    """A bounded producer shared by frozen bootstrap and current-student evaluation.

    Questions contain id/prompt and optional domain metadata. This adapter does
    no reward computation and never sends full-vocab probabilities to storage.
    on_batch may feed the same completed rows to an independent CPU verifier.
    """

    def __init__(
        self,
        config,
        servers,
        input_model,
        *,
        engine_config,
        engine_model,
        max_model_len,
        concurrency=64,
        chunk_size=32,
        timeout=1800,
    ):
        self.config, self.servers, self.input_model = config, servers, input_model
        self.max_model_len, self.concurrency, self.chunk_size, self.timeout = (
            max_model_len,
            concurrency,
            chunk_size,
            timeout,
        )
        if not servers or min(max_model_len, concurrency, chunk_size) <= 0:
            raise ValueError("A live server and positive generation budgets are required")
        from verl.trainer.distillation.token_mapping import TokenIdMapping

        self.token_mapping = TokenIdMapping.build(input_model.tokenizer, engine_model.tokenizer)
        if (
            not self.token_mapping.is_identity
            and OmegaConf.select(
                config, "distillation.distillation_loss.full_vocab.input_id_mapping", default="identity"
            )
            != "token_semantics"
        ):
            raise ValueError("Generator input ID mapping was not explicitly enabled")
        self.engine_contract = engine_contract(engine_config, engine_model, input_model)
        self.engine_contract_id = identity(self.engine_contract)

    async def generate(
        self,
        destination,
        datasets,
        protocols,
        *,
        producer,
        model_identity,
        outer=0,
        expected_version=None,
        on_batch=None,
        progress=None,
        resume_from=None,
    ):
        """datasets is an iterable of (DatasetSpec, questions); all rows are explicit."""
        started_at = time.monotonic()
        specifications = list(datasets)
        expected = sum(len(questions) * spec.n for spec, questions in specifications)
        if expected == 0:
            raise ValueError("Empty generation request set")
        full_determinism = OmegaConf.select(self.config, "actor_rollout_ref.rollout.full_determinism", default=False)
        router = GlobalRequestLoadBalancer(
            {str(i): server for i, server in enumerate(self.servers)}, full_determinism=full_determinism
        )
        execution = {
            "scheduler": "global_bounded_v1",
            "routing": "deterministic" if full_determinism else "least_inflight",
            "replicas": len(self.servers),
            "concurrency": self.concurrency,
            "result_buffer": self.concurrency,
            "chunk_size": self.chunk_size,
        }
        store = ArtifactStore.create(
            destination,
            kind="generation",
            metadata={
                "producer": producer,
                "execution": execution,
                "model_identity": model_identity,
                "sample_identity_schema": 3,
                "outer": outer,
                "protocols": {k: v.metadata() for k, v in protocols.items()},
                "expected_samples": expected,
                "expected_version": expected_version,
                "engine_contract": self.engine_contract,
                "engine_contract_id": self.engine_contract_id,
                "question_set_sha256": identity(
                    [(spec.name, spec.n, spec.protocol, questions) for spec, questions in specifications]
                ),
            },
        )
        loops = {}
        nonce = uuid.uuid4().hex
        completed, shard = 0, 0
        reused, resume_metadata = set(), {}
        if resume_from is not None:
            from .generation_resume import replay_published

            reused, shard, resume_metadata = await settled_thread_call(
                replay_published, store, resume_from, specifications, protocols, on_batch=on_batch, progress=progress
            )
            completed = len(reused)

        for spec, questions in specifications:
            protocol = protocols[spec.protocol]
            if spec.protocol not in loops:
                loop_config = copy.deepcopy(self.config)
                with open_dict(loop_config.actor_rollout_ref.rollout):
                    loop_config.actor_rollout_ref.rollout.prompt_length = protocol.prompt_limit
                    loop_config.actor_rollout_ref.rollout.response_length = protocol.max_new_tokens
                data_config = copy.deepcopy(self.config.data)
                with open_dict(data_config):
                    data_config.apply_chat_template_kwargs = json.loads(protocol.template_json)
                    data_config.continuous_token.model_family = protocol.model_family
                client = CheckedClient(
                    self.config,
                    self.servers,
                    max_model_len=self.max_model_len,
                    prompt_limit=protocol.prompt_limit,
                    token_mapping=self.token_mapping,
                    router=router,
                )
                loops[spec.protocol] = CachedSingleTurn(
                    cache_size=sum(len(qs) for ds, qs in specifications if ds.protocol == spec.protocol),
                    trainer_config=DictConfigWrap(loop_config),
                    server_manager=client,
                    tokenizer=self.input_model.tokenizer,
                    processor=None,
                    dataset_cls=RLHFDataset,
                    data_config=DictConfigWrap(data_config),
                    hf_model_type=self.input_model.hf_config.model_type,
                )
            loop = loops[spec.protocol]
            # Preflight all protocols before admitting any generation.
            for question in questions:
                prompt = await loop.ct_build_initial_tokens(question["prompt"])
                if len(prompt) > protocol.prompt_limit or len(prompt) + protocol.max_new_tokens > self.max_model_len:
                    raise ValueError(f"Context budget does not cover {spec.name}/{question['id']}")

        preflight_seconds = time.monotonic() - started_at
        generated = completed
        response_tokens = 0
        publication_seconds = 0.0
        last_admission = last_completion = time.monotonic()

        async def one(spec, question, sample_index, ordinal):
            nonlocal generated, response_tokens, last_admission, last_completion
            protocol, loop = protocols[spec.protocol], loops[spec.protocol]
            tick = last_admission = time.monotonic()
            params = protocol.sampling(spec.name, question["id"], sample_index)
            output = await loop.run(params, raw_prompt=question["prompt"], priority=ordinal)
            versions = (
                output.extra_fields.get("min_global_steps"),
                output.extra_fields.get("max_global_steps"),
            )
            if expected_version is not None and versions != (expected_version, expected_version):
                raise RuntimeError(f"Stale current-student generation: {versions}, expected {expected_version}")
            if output.extra_fields.get("finish_reason") in {"abort", "aborted", "error"}:
                raise RuntimeError("Generation was aborted")
            prompt_text = self.input_model.tokenizer.decode(output.prompt_ids, skip_special_tokens=False)
            row = {
                "id": sample_identity(
                    model_identity,
                    outer,
                    spec.name,
                    question["id"],
                    sample_index,
                    protocol.protocol_id,
                    self.engine_contract_id,
                ),
                "source_id": question["id"],
                "outer": outer,
                "prompt_ids": output.prompt_ids,
                "response_ids": output.response_ids,
                "response_mask": output.response_mask,
                "rollout_key": f"{nonce}:{ordinal}",
                "min_global_steps": versions[0],
                "max_global_steps": versions[1],
                "data_source": spec.name,
                "finish_reason": output.extra_fields.get("finish_reason"),
                "bank": producer,
                "dataset": spec.name,
                "domain": spec.domain,
                "sample_index": sample_index,
                "protocol_id": protocol.protocol_id,
                "seed": protocol.sample_seed(spec.name, question["id"], sample_index),
                "producer": producer,
                "prompt_tokens": len(output.prompt_ids),
                "response_tokens": len(output.response_ids),
                "valid_tokens": sum(output.response_mask),
                "generation_seconds": time.monotonic() - tick,
                "thinking_open": prompt_text.rfind("<think>") > prompt_text.rfind("</think>"),
            }
            generated += 1
            response_tokens += len(output.response_ids)
            last_completion = time.monotonic()
            return row, question, spec, protocol

        requests = (
            (spec, question, index, ordinal)
            for ordinal, (spec, question, index) in enumerate(
                (spec, q, i) for spec, questions in specifications for q in questions for i in range(spec.n)
            )
            if (spec.name, question["id"], index) not in reused
        )

        async def abort():
            await asyncio.gather(*(s.abort_all_requests.remote() for s in self.servers))

        async def publish(outputs):
            nonlocal completed, shard, publication_seconds
            tick = time.monotonic()
            rows = [result[0] for result in outputs]
            await settled_thread_call(
                store.write_rows, f"generations/part-{shard:05d}.parquet", rows, schema=GENERATION
            )
            if on_batch is not None:
                await settled_thread_call(on_batch, outputs)
            completed += len(rows)
            shard += 1
            publication_seconds += time.monotonic() - tick
            if progress is not None:
                progress(
                    {
                        "phase": "generation",
                        "producer": producer,
                        "completed": completed,
                        "expected": expected,
                        "generated": generated,
                        "inflight": router.get_total_inflight(),
                    }
                )

        outputs = []
        async with aclosing(
            bounded_results(requests, one, concurrency=self.concurrency, timeout=self.timeout, abort=abort)
        ) as stream:
            async for result in stream:
                outputs.append(result)
                if len(outputs) >= self.chunk_size:
                    await publish(outputs)
                    outputs = []
            if outputs:
                await publish(outputs)
        if completed != expected:
            raise ValueError("Generation did not cover all requested samples")
        store.finish(
            metadata={
                "observed_samples": completed,
                "execution": {
                    **execution,
                    "wall_seconds": time.monotonic() - started_at,
                    "preflight_seconds": preflight_seconds,
                    "publication_seconds": publication_seconds,
                    "last_admission_to_completion_seconds": max(0.0, last_completion - last_admission),
                    "new_response_tokens": response_tokens,
                },
                **resume_metadata,
            }
        )
        return store
