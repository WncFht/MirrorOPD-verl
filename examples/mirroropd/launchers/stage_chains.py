# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""BayesRL/Olmo3 and Light-R1 comparisons with declared common SFT protocols."""

from examples.mirroropd.launchers.mimo import options as native_options
from examples.mirroropd.launchers.mimo import preflight as native_preflight


def options(env, name, root, *, chain):
    if chain not in {"bayesrl", "light_r1"}:
        raise ValueError("Unknown post-training chain")
    is_light = chain == "light_r1"
    for key, value in {
        "ROLLOUT_MAX_NUM_SEQS": 4 if is_light else 8,
        "PPO_MAX_TOKEN_LEN_PER_GPU": 16384,
        "TEACHER_FSDP_TOKEN_BUDGET": 32768,
    }.items():
        if not env.get(key):
            env[key] = str(value)
    overrides = native_options(
        env,
        name,
        root,
        model_family="deepseek" if is_light else "default",
        stop_token_ids=(151643,) if is_light else (100257,),
        trust_remote_code=False,
        questions_key="STAGE_CHAIN_QUESTIONS",
        bootstrap_key="STAGE_CHAIN_BOOTSTRAP",
        run_namespace="stage-chain-native",
    )
    # Public readouts can contain one complete 32K trajectory. Training keeps
    # its smaller microbatch budget; both inference roles must admit that row.
    overrides.append("actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=32768")
    if is_light and int(env["STUDENT_GPUS"]) < 4:
        overrides.append("actor_rollout_ref.actor.fsdp_config.offload_policy=True")
    return overrides


def preflight(environment):
    native_preflight(environment, bootstrap_key="STAGE_CHAIN_BOOTSTRAP")
