import json
import os
import subprocess
import time
import uuid
from pathlib import Path

ROOT = Path("/mnt/tidalfs-bdsz01/dataset/llm_dataset/zkn_data/fht_data/MirrorOPD-verl")
ASSETS = ROOT.parent
BIN = Path("/root/.local/share/mirroropd/venvs/profile-cu128/bin")
RUN = ROOT / "outputs" / ("main-four-gpu-20261025-" + uuid.uuid4().hex[:8])
RUN.mkdir(parents=True)
Path("/tmp/main-four-gpu-run-path.txt").write_text(str(RUN) + "\n")
ENV = os.environ | {
    "CUDA_VISIBLE_DEVICES": "0,1,2,3",
    "PYTHONPATH": str(ROOT),
    "OMP_NUM_THREADS": "1",
    "TOKENIZERS_PARALLELISM": "false",
}
RESULTS = []


def execute(name, args, env=ENV):
    start = time.time()
    print("START", name, flush=True)
    with (RUN / (name + ".log")).open("w") as log:
        p = subprocess.run(args, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    result = {"name": name, "exit_code": p.returncode, "seconds": time.time() - start, "command": args}
    RESULTS.append(result)
    (RUN / "results.json").write_text(json.dumps(RESULTS, indent=2) + "\n")
    print("DONE", name, "exit", p.returncode, flush=True)
    if p.returncode:
        raise RuntimeError(f"{name} failed; see {RUN / name}.log")


common = [str(BIN / "torchrun"), "--standalone", "--nproc-per-node=4"]
execute(
    "full-qwen2",
    common
    + [
        "tests/special_distributed/test_full_vocab_fsdp2.py",
        "--backend",
        "precompute",
        "--teacher-extra-rows",
        "4",
        "--checkpoint",
        "--output",
        str(RUN / "full-qwen2.json"),
    ],
)
execute(
    "full-qwen3-fused",
    common
    + [
        "tests/special_distributed/test_full_vocab_fsdp2.py",
        "--backend",
        "precompute_fused",
        "--model-type",
        "qwen3",
        "--tied-student",
        "--tied-teacher",
        "--teacher-extra-rows",
        "4",
        "--teacher-hidden-size",
        "64",
        "--checkpoint",
        "--output",
        str(RUN / "full-qwen3-fused.json"),
    ],
)
execute(
    "sampled-current-is",
    common
    + [
        "tests/special_distributed/test_shared_teacher_fsdp2.py",
        "--policy-loss-mode",
        "token_is",
        "--advantage-source",
        "current",
        "--ppo-epochs",
        "4",
        "--tie-word-embeddings",
    ],
)
execute(
    "sampled-frozen-ppo",
    common
    + [
        "tests/special_distributed/test_shared_teacher_fsdp2.py",
        "--policy-loss-mode",
        "token_ppo",
        "--advantage-source",
        "rollout",
        "--ppo-epochs",
        "2",
    ],
)

# Use the same native environment to prepare isolated three-question validation data.
subsets = RUN / "validation-subsets"
execute(
    "prepare-validation",
    [
        str(BIN / "python"),
        "-c",
        "from pathlib import Path; import pyarrow.parquet as pq; "
        f"out=Path({str(subsets)!r}); out.mkdir(); root=Path({str(ASSETS / 'MD-OPD/data/math')!r}); "
        "[(pq.write_table(pq.read_table(root/n/'test.parquet').slice(0,1),out/(n+'.parquet'))) "
        "for n in ('MATH-500','AMC23','AIME25')]",
    ],
)

for mode in ["sampled", "full"]:
    run = RUN / ("real-" + mode)
    env = ENV | {
        "PYTHON_BIN": str(BIN / "python"),
        "VERL_USE_UV": "0",
        "MODEL_ROOT": str(ASSETS / "models"),
        "DATA_ROOT": str(ASSETS / "MD-OPD/data"),
        "STUDENT_GPUS": "4",
        "TEACHER_GPUS": "0",
        "DISTILLATION_EXECUTION_MODE": "shared_sync",
        "TRAIN_BATCH_SIZE": "8",
        "PPO_EPOCHS": "2",
        "LEARNING_RATE": "1e-6",
        "MAX_PROMPT_LENGTH": "1024",
        "MAX_RESPONSE_LENGTH": "512",
        "MAX_VAL_RESPONSE_LENGTH": "512",
        "VAL_N": "1",
        "VAL_BATCH_SIZE": "3",
        "TEST_FREQ": "2",
        "SAVE_FREQ": "-1",
        "PPO_MAX_TOKEN_LEN_PER_GPU": "2048",
        "TEACHER_FSDP_TOKEN_BUDGET": "2048",
        "ROLLOUT_TOKEN_BUDGET": "2048",
        "ROLLOUT_MAX_NUM_SEQS": "4",
        "ROLLOUT_GPU_MEMORY_UTILIZATION": "0.2",
        "ROLLOUT_ENFORCE_EAGER": "True",
        "TEACHER_ENFORCE_EAGER": "True",
        "FULL_KL_BACKEND": "precompute",
        "FULL_KL_ASYNC_WRITE": "True",
        "FULL_KL_PREFETCH": "True",
        "RAY_ADDRESS": "local",
        "RAY_USAGE_STATS_ENABLED": "0",
        "WANDB_MODE": "disabled",
        "NVTX_INJECTION64_PATH": "/nonexistent/verl-disabled-nvtx-injection.so",
        "EXPERIMENT_NAME": "main-four-gpu-" + mode,
        "RUN_DIR": str(run),
        "MATH_FILE": str(subsets / "MATH-500.parquet"),
        "AMC_FILE": str(subsets / "AMC23.parquet"),
        "AIME_FILE": str(subsets / "AIME25.parquet"),
    }
    script = "examples/mirroropd/run_math_opd_ablation_fsdp2.sh"
    args = [
        "bash",
        script,
        "custom", "--method", "full_kl" if mode == "full" else "sampled",
        "trainer.total_training_steps=2",
        "trainer.val_before_train=False",
        "data.train_max_samples=64",
        "data.shuffle=False",
        "data.filter_overlong_prompts_workers=1",
        "actor_rollout_ref.actor.use_torch_compile=False",
        "trainer.metrics_quantile_interval=1",
        "trainer.eval_metrics_resamples=16",
        "trainer.logger=[console,file]",
        "++ray_kwargs.ray_init.num_gpus=4",
        "++ray_kwargs.ray_init.num_cpus=24",
        "++ray_kwargs.ray_init.include_dashboard=False",
        "++ray_kwargs.ray_init.object_store_memory=2147483648",
        "++ray_kwargs.ray_init._temp_dir=/tmp/m4g-" + uuid.uuid4().hex[:8],
        "++ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH='" + str(ROOT) + "'",
        "++ray_kwargs.ray_init.runtime_env.env_vars.VERL_FILE_LOGGER_PATH='" + str(run / "metrics.jsonl") + "'",
    ]
    execute("real-" + mode, args, env)

print("ALL COMPLETE", RUN, flush=True)
