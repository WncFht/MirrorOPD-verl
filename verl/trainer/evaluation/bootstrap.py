# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Prepare immutable banks using native standalone rollout replicas.

Run initial and teacher in separate local-Ray processes, then finalize. Each
producer owns its actors; this entrypoint never stops a shared Ray cluster.
"""

import argparse
import asyncio
import copy
import json
import os

import ray
from omegaconf import OmegaConf, open_dict

from verl.experimental.rollout_policy.validation import policy_engine_config
from verl.trainer.ppo.readout_callback import readout_options
from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config import HFModelConfig
from verl.workers.rollout.replica import get_rollout_replica_class

from .accuracy import AccuracyCollector
from .artifacts import ArtifactStore, file_sha256, identity
from .config import DatasetSpec, EvaluationConfig
from .coordinator import EvaluationCoordinator, read_questions
from .generation import NativeGenerator
from .generation_contract import engine_contract, require_current_generation
from .identity import bootstrap_contract, model_identity
from .trajectory import select_panel, trajectory_index


async def generate_bootstrap(config, producer, *, tensor_parallel=1, source_configs=(), resume_generation=None):
    require_current_generation(config)
    options = EvaluationConfig.parse(readout_options(config, EvaluationCoordinator))
    root = ArtifactStore(options.bootstrap)
    if not root.manifest_path.exists():
        root = ArtifactStore.create(root.root, kind="evaluation_bootstrap", metadata={})
    elif root.manifest()["status"] != "active":
        raise ValueError("Bootstrap is already immutable and complete")
    initial_model = omega_conf_to_dataclass(config.actor_rollout_ref.model, dataclass_type=HFModelConfig)
    teacher = config.distillation.teacher_models.teacher_model.model_path
    path = initial_model.path if producer in {"initial", "source"} else teacher
    model_config = copy.deepcopy(config.actor_rollout_ref.model)
    with open_dict(model_config):
        model_config.path = path
    model_config = omega_conf_to_dataclass(model_config, dataclass_type=HFModelConfig)
    if model_config.tokenizer.get_vocab() != initial_model.tokenizer.get_vocab():
        raise ValueError("Bootstrap requires verified shared token semantics")
    model_hash = model_identity(path)
    rollout = policy_engine_config(config.actor_rollout_ref.rollout)
    gpus = int(config.trainer.n_gpus_per_node)
    if tensor_parallel < 1 or gpus % tensor_parallel:
        raise ValueError("Bootstrap tensor parallelism must divide the allocated GPUs")
    with open_dict(rollout):
        rollout.tensor_model_parallel_size = tensor_parallel
        rollout.data_parallel_size = 1
        rollout.pipeline_model_parallel_size = 1
        rollout.load_format = "auto"
        rollout.max_model_len = max(p.prompt_limit + p.max_new_tokens for p in options.protocols.values())
    replicas = []
    collector = None
    try:
        for rank in range(gpus // tensor_parallel):
            replica = get_rollout_replica_class("vllm")(
                rank, rollout, model_config, gpus_per_node=gpus, name_suffix=f"eval_{producer}"
            )
            replicas.append(replica)
        await asyncio.gather(*(replica.init_standalone() for replica in replicas))
        servers = [r.server_handle for r in replicas]
        await asyncio.gather(
            *(s.collective_rpc.remote("configure_rollout_policy_sampler", args=(True,)) for s in servers)
        )
        generator = NativeGenerator(
            config,
            servers,
            initial_model,
            engine_config=rollout,
            engine_model=model_config,
            max_model_len=rollout.max_model_len,
            concurrency=options.generation_concurrency,
            chunk_size=options.generation_chunk_size,
            timeout=options.request_timeout,
        )
        progress = lambda event: print("EVALUATION_PROGRESS " + json.dumps(event), flush=True)
        if producer == "source":
            for source_config in (config, *source_configs):
                if source_config.actor_rollout_ref.model.path != initial_model.path:
                    raise ValueError("Shared source generation requires the same initial student")
                source_options = EvaluationConfig.parse(readout_options(source_config, EvaluationCoordinator))
                source_root = ArtifactStore(source_options.bootstrap)
                if not source_root.manifest_path.exists():
                    source_root = ArtifactStore.create(source_root.root, kind="evaluation_bootstrap", metadata={})
                if source_root.manifest()["status"] != "active":
                    raise ValueError("Cannot add a source to a completed bootstrap")
                source = DatasetSpec(
                    "source_heldout",
                    source_options.training_domain,
                    source_options.source_file,
                    "math",
                    protocol="source",
                )
                questions = read_questions(source)
                if len(questions) != source_options.source_count:
                    raise ValueError("Unexpected source heldout count")
                await generator.generate(
                    source_root.root / "source",
                    [(source, questions)],
                    source_options.protocols,
                    producer="initial_student",
                    model_identity=model_hash,
                    progress=progress,
                )
            return {}
        datasets = [(d, read_questions(d)) for d in options.datasets]
        accuracy_name = "initial-accuracy" if producer == "initial" else "teacher-accuracy"
        collector = AccuracyCollector(
            root.root / accuracy_name, datasets, initial_model.tokenizer, workers=options.verifier_workers
        )
        generated = await generator.generate(
            root.root / producer,
            datasets,
            options.protocols,
            producer=producer,
            model_identity=model_hash,
            on_batch=collector.submit,
            progress=progress,
            resume_from=resume_generation,
        )
        if producer == "initial":
            source = DatasetSpec(
                "source_heldout", options.training_domain, options.source_file, "math", protocol="source"
            )
            questions = read_questions(source)
            if len(questions) != options.source_count:
                raise ValueError("Unexpected source heldout count")
            await generator.generate(
                root.root / "source",
                [(source, questions)],
                options.protocols,
                producer="initial_student",
                model_identity=model_hash,
                progress=progress,
            )
        metrics = await asyncio.to_thread(collector.finish, generated)
        collector = None
        return metrics
    finally:
        if collector is not None:
            collector.close()
        for replica in replicas:
            # Kill only handles created above. The launcher owns the local-Ray session.
            for actor in replica.servers + replica.workers:
                ray.kill(actor, no_restart=True)
            if replica.resource_pool is not None:
                from ray.util import remove_placement_group

                for group in replica.resource_pool.get_placement_groups():
                    remove_placement_group(group)


def finalize_bootstrap(config):
    options = EvaluationConfig.parse(readout_options(config, EvaluationCoordinator))
    root = ArtifactStore(options.bootstrap)
    student = model_identity(config.actor_rollout_ref.model.path)
    teacher = model_identity(config.distillation.teacher_models.teacher_model.model_path)
    for child, expected in (("initial", student), ("source", student), ("teacher", teacher)):
        store = ArtifactStore(root.root / child)
        manifest = store.manifest(complete=True)
        if manifest["metadata"]["model_identity"] != expected:
            raise ValueError("Bootstrap producer weights differ")
        role_config = copy.deepcopy(config.actor_rollout_ref.model)
        with open_dict(role_config):
            role_config.path = (
                config.distillation.teacher_models.teacher_model.model_path
                if child == "teacher"
                else config.actor_rollout_ref.model.path
            )
        role_model = omega_conf_to_dataclass(role_config, dataclass_type=HFModelConfig)
        input_model = omega_conf_to_dataclass(config.actor_rollout_ref.model, dataclass_type=HFModelConfig)
        contract = engine_contract(policy_engine_config(config.actor_rollout_ref.rollout), role_model, input_model)
        if manifest["metadata"].get("engine_contract_id") != identity(contract):
            raise ValueError("Bootstrap bank does not use the declared effective generation contract")
        if identity(manifest["metadata"]["protocols"]) != identity(
            {k: v.metadata() for k, v in options.protocols.items()}
        ):
            raise ValueError("Bootstrap generation protocols differ")
        specs = (
            options.datasets
            if child != "source"
            else (
                DatasetSpec("source_heldout", options.training_domain, options.source_file, "math", protocol="source"),
            )
        )
        expected_questions = [(d.name, d.n, d.protocol, read_questions(d)) for d in specs]
        if manifest["metadata"].get("question_set_sha256") != identity(expected_questions):
            raise ValueError("Bootstrap question contents or sampling counts differ")
        refs = trajectory_index(store, bank=child)
        if child != "source":
            select_panel(refs, options.datasets)
        elif len(refs) != options.source_count:
            raise ValueError("Incomplete source bank")
    for child, generation in (("initial-accuracy", "initial"), ("teacher-accuracy", "teacher")):
        accuracy = ArtifactStore(root.root / child).manifest(complete=True)
        generated = ArtifactStore(root.root / generation)
        if accuracy["metadata"].get("generation_sha256") != file_sha256(generated.manifest_path):
            raise ValueError("Bootstrap accuracy is not bound to its immutable generation artifact")
    contract = bootstrap_contract(options, student=student, teacher=teacher)
    children = {
        name: file_sha256(ArtifactStore(root.root / name).manifest_path)
        for name in ("initial", "source", "teacher", "initial-accuracy", "teacher-accuracy")
    }
    root.finish(metadata={"contract": contract, "contract_sha256": identity(contract), "children_sha256": children})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Resolved native verl YAML including trainer.evaluation")
    parser.add_argument("--producer", choices=("initial", "teacher", "source", "finalize"), required=True)
    parser.add_argument(
        "--source-config", action="append", default=[], help="Additional sources using the same initial student"
    )
    parser.add_argument("--tensor-parallel", type=int, default=1)
    parser.add_argument("--resume-generation", help="Reuse published samples from a matching frozen-model generation")
    args = parser.parse_args()
    if args.source_config and args.producer != "source":
        parser.error("--source-config requires --producer source")
    if args.resume_generation and args.producer not in {"initial", "teacher"}:
        parser.error("--resume-generation requires initial or teacher")
    config = OmegaConf.load(args.config)
    if args.producer == "finalize":
        finalize_bootstrap(config)
        return
    # Refuse attaching to any pre-existing cluster through ambient RAY_ADDRESS.
    os.environ["RAY_ADDRESS"] = "local"
    ray.init(
        address="local",
        num_cpus=int(OmegaConf.select(config, "ray_kwargs.ray_init.num_cpus", default=32)),
        num_gpus=int(config.trainer.n_gpus_per_node),
        include_dashboard=False,
        _temp_dir=os.environ.get("SMOKE_RAY_TMPDIR"),
        object_store_memory=8 * 1024**3,
    )
    try:
        asyncio.run(
            generate_bootstrap(
                config,
                args.producer,
                tensor_parallel=args.tensor_parallel,
                source_configs=[OmegaConf.load(path) for path in args.source_config],
                resume_generation=args.resume_generation,
            )
        )
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
