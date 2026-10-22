# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Real FSDP2 teacher/actor phase test with tiny local Qwen2 models.

Run on idle GPUs:
    torchrun --standalone --nproc-per-node=2 tests/special_distributed/test_shared_teacher_fsdp2.py

Exercises TrainingWorker initialization, packed inference, repeated transfers,
distillation updates and teacher isolation. This does not start vLLM or Ray.
"""

import argparse
import shutil
import tempfile
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from tensordict import TensorDict
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast, Qwen2Config

from verl.trainer.distillation.losses import distillation_loss
from verl.trainer.ppo.v1.trainer_base import PPOTrainer
from verl.utils import tensordict_utils as tu
from verl.utils.device import get_device_name, get_torch_device
from verl.utils.distributed import initialize_global_process_group
from verl.workers.config import (
    DistillationConfig,
    DistillationLossConfig,
    FSDPEngineConfig,
    FSDPOptimizerConfig,
    HFModelConfig,
    TrainingWorkerConfig,
)
from verl.workers.engine_workers import TrainingWorker
from verl.workers.utils.padding import response_from_nested


class _ProgressTrainer(PPOTrainer):
    def on_step_end(self):
        pass

    def on_sample_end(self):
        pass


def nested(rows):
    """Build CPU jagged batches, as in the production TQ bridge."""
    return torch.nested.nested_tensor(rows, layout=torch.jagged)


def assert_on_cpu(worker):
    """Ensure parameters and gradients do not retain an active GPU model."""
    assert all(p.device.type == "cpu" for p in worker.engine.module.parameters())


def teacher_policy_loss(model_output, data, dp_group, *, config, distillation_config):
    """Use the production distillation objective with the engine loss signature."""
    return distillation_loss(config, distillation_config, model_output, data)


def make_worker(path, world_size, *, teacher, full_vocab=False):
    """Use the production worker and engine constructors."""
    model = HFModelConfig(
        path=str(path), enable_gradient_checkpointing=False, use_fused_kernels=False, enable_full_vocab=full_vocab
    )
    engine = FSDPEngineConfig(
        strategy="fsdp2",
        forward_only=teacher,
        forward_only_cpu_offload=False,
        model_dtype="bfloat16" if teacher else "float32",
        fsdp_size=world_size,
        param_offload=True,
        optimizer_offload=not teacher,
        use_torch_compile=False,
        infer_max_token_len_per_gpu=6,
        max_token_len_per_gpu=6,
        use_no_sync_for_gradient_accumulation=False,
    )
    optimizer = None if teacher else FSDPOptimizerConfig(lr=1e-3, total_training_steps=3)
    worker = TrainingWorker(
        TrainingWorkerConfig(
            model_type="language_model", model_config=model, engine_config=engine, optimizer_config=optimizer
        )
    )
    worker.reset()
    return worker


def main():
    """Check actual collective and transfer behavior across colocated engines."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--tie-word-embeddings", action="store_true")
    parser.add_argument(
        "--policy-loss-mode", choices=["vanilla", "token_ppo", "token_is", "reinforce"], default="vanilla"
    )
    parser.add_argument("--advantage-source", choices=["current", "rollout"], default="current")
    parser.add_argument("--ppo-epochs", type=int, default=1)
    args = parser.parse_args()
    assert args.ppo_epochs > 0
    assert get_device_name() == "cuda", "This integration test requires CUDA"
    _, rank, world_size = initialize_global_process_group(timeout_second=180)
    assert world_size >= 2
    paths = [tempfile.mkdtemp(prefix="shared-teacher-fsdp2-") if rank == 0 else None]
    dist.broadcast_object_list(paths)
    root = Path(paths[0])
    if rank == 0:
        vocab = {"[PAD]": 0, "[EOS]": 1, "[UNK]": 2, **{f"t{i}": i for i in range(3, 32)}}
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(WordLevel(vocab, unk_token="[UNK]")),
            pad_token="[PAD]",
            eos_token="[EOS]",
            unk_token="[UNK]",
        )
        for index, name in enumerate(("student", "teacher")):
            torch.manual_seed(100 + index)
            model = AutoModelForCausalLM.from_config(
                Qwen2Config(
                    vocab_size=32,
                    hidden_size=32,
                    intermediate_size=64,
                    num_hidden_layers=2,
                    num_attention_heads=4,
                    num_key_value_heads=2,
                    max_position_embeddings=64,
                    tie_word_embeddings=args.tie_word_embeddings,
                    eos_token_id=1,
                    pad_token_id=0,
                )
            )
            model.save_pretrained(root / name)
            tokenizer.save_pretrained(root / name)
    dist.barrier()
    actor = make_worker(root / "student", world_size, teacher=False)
    assert_on_cpu(actor)
    rows = [torch.tensor([3 + rank, 7, 8, 9, 1]), torch.tensor([4, 10, 11, 1])]
    data = TensorDict(
        {
            "input_ids": nested(rows),
            "position_ids": nested([torch.arange(len(r)) for r in rows]),
            "prompts": nested([r[:2] for r in rows]),
            "responses": nested([r[2:] for r in rows]),
            "response_mask": nested([torch.ones(len(r) - 2, dtype=torch.long) for r in rows]),
            "loss_mask": nested([torch.ones(len(r) - 2, dtype=torch.long) for r in rows]),
        },
        batch_size=[2],
    )
    tu.assign_non_tensor(data, temperature=1.0, compute_loss=False, calculate_entropy=False)
    before_teacher_init = actor.infer_batch(data)["log_probs"].values().clone()
    teacher = make_worker(root / "teacher", world_size, teacher=True)
    assert teacher.engine.optimizer is None
    assert not teacher.engine._uses_fsdp2_cpu_offload_policy
    assert_on_cpu(teacher)
    torch.testing.assert_close(actor.infer_batch(data)["log_probs"].values(), before_teacher_init, atol=0, rtol=0)
    loss_config = DistillationConfig(
        distillation_loss=DistillationLossConfig(
            loss_mode="k1",
            use_task_rewards=False,
            use_policy_gradient=True,
            loss_max_clamp=None,
            log_prob_min_clamp=None,
            policy_loss_mode=args.policy_loss_mode,
            advantage_source=args.advantage_source,
        )
    )
    actor.set_loss_fn(
        partial(
            teacher_policy_loss,
            config=SimpleNamespace(loss_agg_mode="token-mean", global_batch_info={}, loss_scale_factor=None),
            distillation_config=loss_config,
        )
    )
    initial = None
    progress = _ProgressTrainer.__new__(_ProgressTrainer)
    progress.actor_optimizer_step_attempts = progress.actor_optimizer_step = 0
    for iteration in range(3):
        teacher_data = data.select("input_ids", "position_ids", "loss_mask")
        tu.assign_non_tensor(
            teacher_data, temperature=1.0, compute_loss=False, calculate_entropy=False, disable_auto_offload=True
        )
        teacher.to("device", model=True, optimizer=False, grad=False)
        result = teacher.infer_batch(teacher_data)
        teacher.to("cpu", model=True, optimizer=False, grad=False)
        assert_on_cpu(teacher)
        assert all(p.grad is None for p in teacher.engine.module.parameters())
        scores = [r.clone() for r in result["log_probs"].unbind()]
        if initial is None:
            initial = [r.clone() for r in scores]
            reference = (
                AutoModelForCausalLM.from_pretrained(
                    root / "teacher", torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
                )
                .to(get_device_name())
                .eval()
            )
            with torch.no_grad():
                for tokens, actual in zip(rows, scores, strict=True):
                    logits = reference(tokens.to(get_device_name()).unsqueeze(0), use_cache=False).logits[0]
                    expected = (
                        logits[:-1]
                        .float()
                        .log_softmax(-1)
                        .gather(-1, tokens[1:].to(logits.device).unsqueeze(-1))
                        .squeeze(-1)
                        .cpu()
                    )
                    torch.testing.assert_close(actual[:-1].float(), expected, atol=0.03, rtol=0.01)
            del reference
            get_torch_device().empty_cache()
        else:
            for actual, expected in zip(scores, initial, strict=True):
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        for row in scores:
            row[-1] = 0
        data["teacher_logprobs"] = nested([r.float().unsqueeze(-1) for r in scores])
        old = actor.infer_batch(data)["log_probs"]
        assert_on_cpu(actor)
        data["old_log_probs"] = response_from_nested(old, data["response_mask"])
        # Fixed trajectories with cached behavior scores, independent of later
        # training forwards. This worker integration test does not run vLLM.
        data["rollout_log_probs"] = data["old_log_probs"].clone()
        tu.assign_non_tensor(
            data,
            global_batch_size=2 * world_size,
            mini_batch_size=2 * world_size,
            epochs=args.ppo_epochs,
            seed=42,
            dataloader_kwargs={"shuffle": False},
        )
        output = actor.train_mini_batch(data)
        assert tu.get(output, "optimizer_steps") == tu.get(output, "optimizer_step_attempts") == args.ppo_epochs
        progress._accumulate_actor_optimizer_steps(output)
        assert progress.actor_optimizer_step == (iteration + 1) * args.ppo_epochs
        metrics = tu.get(output, "metrics")
        if args.policy_loss_mode in ("token_ppo", "token_is"):
            assert sum(metrics["distillation/log_ratio_guard_count"]) == 0
        assert_on_cpu(actor)
        assert_on_cpu(teacher)
        updated = actor.infer_batch(data)["log_probs"]
        assert not torch.equal(updated.values(), old.values()), "Actor optimizer did not update the model"
        if rank == 0:
            print(
                f"shared teacher phase {iteration + 1}, optimizer steps={progress.actor_optimizer_step} passed",
                flush=True,
            )
    saved_scores = actor.infer_batch(data)["log_probs"].values().clone()
    actor.save_checkpoint(str(root / "checkpoint"), None, global_step=3)
    progress.global_steps = 3
    if rank == 0:
        progress._save_actor_optimizer_progress(root / "checkpoint")
    dist.barrier()
    tu.assign_non_tensor(
        data,
        global_batch_size=2 * world_size,
        mini_batch_size=2 * world_size,
        epochs=1,
        seed=42,
        dataloader_kwargs={"shuffle": False},
    )
    progress._accumulate_actor_optimizer_steps(actor.train_mini_batch(data))
    actor.load_checkpoint(str(root / "checkpoint"), del_local_after_load=False)
    progress._load_actor_optimizer_progress(root / "checkpoint")
    assert progress.actor_optimizer_step == 3 * args.ppo_epochs
    torch.testing.assert_close(actor.infer_batch(data)["log_probs"].values(), saved_scores, atol=0, rtol=0)
    # Resume with a different epoch count, checking saved optimizer state and progress.
    tu.assign_non_tensor(data, mini_batch_size=2 * world_size, epochs=2, global_batch_size=2 * world_size)
    progress._accumulate_actor_optimizer_steps(actor.train_mini_batch(data))
    assert progress.actor_optimizer_step == progress.actor_optimizer_step_attempts == 3 * args.ppo_epochs + 2
    assert_on_cpu(actor)
    assert_on_cpu(teacher)
    dist.barrier()
    if rank == 0:
        shutil.rmtree(root)
        print(
            f"test_shared_teacher_fsdp2 passed: {args.policy_loss_mode}/{args.advantage_source}, "
            f"optimizer steps after resume={progress.actor_optimizer_step}",
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
