# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Compose the frozen four-domain protocol with a tested native serial execution profile.

This planner emits one resolved YAML; bootstrap and main_ppo consume it directly.
It neither reserves GPUs nor constructs an alternative training loop.
"""

import argparse
from pathlib import Path

from omegaconf import OmegaConf

from verl.trainer.evaluation.artifacts import ArtifactStore, file_sha256
from verl.trainer.evaluation.config import EvaluationConfig
from verl.trainer.evaluation.coordinator import EvaluationCoordinator
from verl.trainer.ppo.readout_callback import readout_options


def build(
    base,
    questions,
    output,
    *,
    domain="math",
    outer=50,
    batch_size=128,
    reuse=32,
    family="deepseek",
    source_enabled=False,
    history_enabled=False,
):
    output = Path(output).resolve()
    prepared = ArtifactStore(questions).manifest(complete=True)
    if prepared["kind"] != "question_sets":
        raise ValueError("Expected frozen prepared question tables")
    paths = prepared["metadata"]["paths"][domain]
    if prepared["metadata"]["split_reports"][domain]["training"] < outer * batch_size:
        raise ValueError("Prepared training pool cannot cover the budget without repeated questions")
    config = OmegaConf.load(base)
    template_kwargs = OmegaConf.to_container(
        OmegaConf.create(config.data.get("apply_chat_template_kwargs", {})), resolve=True
    )
    if config.trainer.v1.trainer_mode != "sync" or config.distillation.execution_mode != "shared_sync":
        raise ValueError("Base execution profile must use native synchronous serial distillation")
    if config.trainer.n_gpus_per_node > 6 or config.trainer.nnodes != 1:
        raise ValueError("This local recipe is restricted to at most six GPUs on one node")
    config.trainer.evaluation = {
        "_target_": "verl.trainer.evaluation.coordinator.EvaluationCoordinator",
        "output": str(output / "evaluation"),
        "bootstrap": str(output / "bootstrap"),
        "source_file": paths["source"],
        "source_enabled": source_enabled,
        "history_enabled": history_enabled,
        "training_domain": domain,
        "datasets": prepared["metadata"]["datasets"],
        "protocols": {
            "source": {
                "model_family": family,
                "max_new_tokens": 7168,
                "prompt_limit": 2048,
                "top_p": 1.0,
                "template_kwargs": template_kwargs,
            },
            "public": {
                "model_family": family,
                "max_new_tokens": 32768,
                "prompt_limit": 8192,
                "top_p": 0.95,
                "template_kwargs": template_kwargs,
            },
        },
        "source_count": 128,
        "history_size": 128,
        "generation_concurrency": 64,
        "generation_chunk_size": 64,
        "score": {"payload_bytes": 16 * 1024**3, "host_budget_bytes": 64 * 1024**3, "token_chunk": 256},
        "final_export": True,
        "verify_export": True,
    }
    config.data.train_files = [paths["train"]]
    config.data.val_files = []
    config.data.train_batch_size = batch_size
    config.data.max_prompt_length = 2048
    config.data.max_response_length = 7168
    config.data.shuffle = False
    config.data.filter_overlong_prompts = False
    config.data.truncation = "error"
    config.data.continuous_token.model_family = family
    actor = config.actor_rollout_ref.actor
    actor.ppo_mini_batch_size = batch_size
    actor.ppo_epochs = reuse
    actor.optim.lr = 1e-6
    actor.loss_agg_mode = "token-mean"
    actor.checkpoint.save_contents = ["model", "hf_model"]
    actor.checkpoint.load_contents = ["model"]
    actor.checkpoint.preserve_hf_master_weights = True
    actor.ppo_max_token_len_per_gpu = max(actor.ppo_max_token_len_per_gpu, 49152)
    rollout = config.actor_rollout_ref.rollout
    rollout.n = 1
    rollout.temperature = 1.0
    rollout.top_p = 1.0
    rollout.top_k = -1
    rollout.prompt_length = 2048
    rollout.response_length = 7168
    rollout.max_model_len = 40960
    rollout.log_prob_max_token_len_per_gpu = 49152
    config.distillation.shared_teacher.infer_max_token_len_per_gpu = 49152
    config.distillation.distillation_loss.use_task_rewards = False
    config.trainer.total_training_steps = outer
    config.trainer.total_epochs = 1
    config.trainer.test_freq = -1
    config.trainer.save_freq = -1
    config.trainer.val_before_train = True
    config.trainer.resume_mode = "disable"
    config.trainer.default_local_dir = str(output / "checkpoints")
    config.trainer.validation_data_dir = None
    config.trainer.experiment_name = output.name
    config.trainer.prepared_questions_manifest = str(ArtifactStore(questions).manifest_path)
    config.trainer.prepared_questions_sha256 = file_sha256(ArtifactStore(questions).manifest_path)
    config.ray_kwargs.ray_init.address = "local"
    config.ray_kwargs.ray_init._temp_dir = "${oc.env:SMOKE_RAY_TMPDIR}"
    env = config.ray_kwargs.ray_init.runtime_env.env_vars
    env.VERL_FILE_LOGGER_PATH = str(output / "metrics.jsonl")
    env.WANDB_DIR = str(output)
    env.VERL_ROLLOUT_IPC_NAMESPACE = "${oc.env:VERL_ROLLOUT_IPC_NAMESPACE}"
    env.LD_LIBRARY_PATH = "${oc.env:LD_LIBRARY_PATH}"
    env.PYTHONPATH = str(Path(__file__).resolve().parents[3])
    EvaluationConfig.parse(readout_options(config, EvaluationCoordinator))
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", required=True, help="Tested resolved native serial execution YAML")
    parser.add_argument("--questions", required=True, help="Complete prepared question artifact")
    parser.add_argument("--output", required=True, help="New run directory")
    parser.add_argument("--domain", choices=("math", "code", "science", "if"), default="math")
    parser.add_argument("--model-family", default="deepseek")
    parser.add_argument("--source-heldout", action="store_true", help="Enable optional source-heldout RKL")
    parser.add_argument("--training-history", action="store_true", help="Enable optional training-history RKL")
    args = parser.parse_args()
    config = build(
        args.base_config,
        args.questions,
        args.output,
        domain=args.domain,
        family=args.model_family,
        source_enabled=args.source_heldout,
        history_enabled=args.training_history,
    )
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=False)
    OmegaConf.save(config, destination / "run.yaml")
    print(destination / "run.yaml")


if __name__ == "__main__":
    main()
