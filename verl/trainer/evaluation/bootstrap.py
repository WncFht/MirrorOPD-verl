# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Prepare immutable banks using native standalone rollout replicas.

Run initial and teacher in separate local-Ray processes, then finalize. Each
producer owns its actors; this entrypoint never stops a shared Ray cluster.
"""

import argparse
import asyncio
import json
import os

import ray
from omegaconf import OmegaConf, open_dict

from verl.trainer.ppo.readout_callback import readout_options
from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config import HFModelConfig
from verl.workers.rollout.replica import get_rollout_replica_class

from .accuracy import AccuracyCollector
from .artifacts import ArtifactStore, file_sha256, identity
from .config import DatasetSpec, EvaluationConfig
from .coordinator import EvaluationCoordinator, read_questions
from .generation import NativeGenerator
from .generation_contract import engine_contract, prepared_generation_config, require_current_generation
from .identity import bootstrap_contract, model_identity
from .trajectory import select_panel, trajectory_index


async def generate_bootstrap(config, producer, *, tensor_parallel=1, source_configs=(), resume_generation=None):
    require_current_generation(config, independent_engine=True)
    options = EvaluationConfig.parse(readout_options(config, EvaluationCoordinator))
    if producer == "source" and not options.source_enabled:
        raise ValueError("Source evaluation is disabled; enable source_enabled to generate its bank")
    root = ArtifactStore(options.bootstrap)
    if not root.manifest_path.exists():
        root = ArtifactStore.create(root.root, kind="evaluation_bootstrap", metadata={})
    elif root.manifest()["status"] != "active":
        raise ValueError("Bootstrap is already immutable and complete")
    initial_model = omega_conf_to_dataclass(config.actor_rollout_ref.model, dataclass_type=HFModelConfig)
    teacher = config.distillation.teacher_models.teacher_model.model_path
    path = initial_model.path if producer in {"initial", "source"} else teacher
    model_config = (
        initial_model
        if producer in {"initial", "source"}
        else HFModelConfig(path=path, trust_remote_code=config.actor_rollout_ref.model.get("trust_remote_code", False))
    )
    from verl.trainer.distillation.compatibility import ModelPairContract

    if producer not in {"initial", "source"}:
        pair = ModelPairContract.build(
            initial_model.hf_config,
            model_config.hf_config,
            initial_model.tokenizer,
            model_config.tokenizer,
            distribution=config.distillation.distillation_loss.full_vocab.get("distribution", {}),
            allow_permutation=config.distillation.distillation_loss.full_vocab.get("input_id_mapping", "identity")
            == "token_semantics",
        )
        student_tokens = initial_model.tokenizer.get_vocab()
        if not set(model_config.tokenizer.get_vocab()) <= set(student_tokens):
            raise ValueError(
                "Teacher rollout support is not closed under shared student input; "
                "declare and validate a separate generation-support protocol"
            )
        for protocol in options.protocols.values():
            if protocol.prompt_limit + protocol.max_new_tokens > min(pair.student_context, pair.teacher_context):
                raise ValueError("Generation budget exceeds verified student/teacher context")
    model_hash = model_identity(path)
    rollout = prepared_generation_config(config.actor_rollout_ref.rollout, model_config, initial_model)
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
                if not source_options.source_enabled:
                    raise ValueError("Additional source configurations must enable source_enabled")
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
        if options.accuracy_enabled:
            collector = AccuracyCollector(
                root.root / accuracy_name, datasets, initial_model.tokenizer, workers=options.verifier_workers
            )
            collector.follow_generation(root.root / producer, options.protocols)
        generated = await generator.generate(
            root.root / producer,
            datasets,
            options.protocols,
            producer=producer,
            model_identity=model_hash,
            progress=progress,
            resume_from=resume_generation,
            on_batch=collector.check if collector is not None else None,
        )
        if producer == "initial" and options.source_enabled:
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
        metrics = await asyncio.to_thread(collector.finish, generated) if collector is not None else {}
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
    producers = {"initial": student, "teacher": teacher}
    if options.source_enabled:
        producers["source"] = student
    for child, expected in producers.items():
        store = ArtifactStore(root.root / child)
        manifest = store.manifest(complete=True)
        if manifest["metadata"]["model_identity"] != expected:
            raise ValueError("Bootstrap producer weights differ")
        input_model = omega_conf_to_dataclass(config.actor_rollout_ref.model, dataclass_type=HFModelConfig)
        role_model = (
            HFModelConfig(
                path=config.distillation.teacher_models.teacher_model.model_path,
                trust_remote_code=config.actor_rollout_ref.model.get("trust_remote_code", False),
            )
            if child == "teacher"
            else input_model
        )
        contract = engine_contract(
            prepared_generation_config(config.actor_rollout_ref.rollout, role_model, input_model),
            role_model,
            input_model,
        )
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
        if not options.accuracy_enabled:
            continue
        accuracy = ArtifactStore(root.root / child).manifest(complete=True)
        generated = ArtifactStore(root.root / generation)
        if accuracy["metadata"].get("generation_sha256") != file_sha256(generated.manifest_path):
            raise ValueError("Bootstrap accuracy is not bound to its immutable generation artifact")
    contract = bootstrap_contract(options, student=student, teacher=teacher)
    children = {name: file_sha256(ArtifactStore(root.root / name).manifest_path) for name in options.bootstrap_children}
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
    if (
        args.producer == "source"
        and not EvaluationConfig.parse(readout_options(config, EvaluationCoordinator)).source_enabled
    ):
        parser.error("Source evaluation is disabled; set source_enabled=true")
    if args.producer == "finalize":
        finalize_bootstrap(config)
        return
    from verl.trainer.distillation.compatibility import preflight_pair

    preflight_pair(config)
    # Refuse attaching to any pre-existing cluster through ambient RAY_ADDRESS.
    os.environ["RAY_ADDRESS"] = "local"
    from verl.utils.ray_utils import prewarm_local_ray, with_hf_modules_cache

    prewarm_local_ray()
    ray.init(
        address="local",
        num_cpus=int(OmegaConf.select(config, "ray_kwargs.ray_init.num_cpus", default=32)),
        num_gpus=int(config.trainer.n_gpus_per_node),
        include_dashboard=False,
        _temp_dir=os.environ.get("SMOKE_RAY_TMPDIR"),
        object_store_memory=8 * 1024**3,
        runtime_env=with_hf_modules_cache(
            OmegaConf.to_container(config.ray_kwargs.ray_init.get("runtime_env", {}), resolve=True)
        ),
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
