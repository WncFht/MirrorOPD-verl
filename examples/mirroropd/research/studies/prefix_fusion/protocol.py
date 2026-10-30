# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Frozen v2 scientific identities, not executable launcher configuration.

Keep these original record fields stable: changing their spelling changes the
protocol hash of ongoing experiments. New launches use the unified model refs.
"""

from examples.mirroropd.research.core.artifacts import fingerprint

ARMS = {
    "S0": dict(mode="single", source="student", student="initial", teacher_weight=0.0, fusion_space="logit"),
    "S": dict(mode="trajectory_mixture", source="student", student="current", teacher_weight=0.0, fusion_space="logit"),
    "PM": dict(
        mode="token_mixture", source="student", student="current", teacher_weight=0.5, fusion_space="probability"
    ),
    "LM": dict(mode="token_mixture", source="student", student="current", teacher_weight=0.5, fusion_space="logit"),
    "TM": dict(
        mode="trajectory_mixture", source="student", student="current", teacher_weight=0.5, fusion_space="logit"
    ),
    "T": dict(mode="single", source="teacher", student="current", teacher_weight=1.0, fusion_space="logit"),
}
MODELS = {"S0": "DeepSeek-R1-Distill-Qwen-1.5B", "J": "JustRL-DeepSeek-1.5B", "P": "Polaris-7B-Preview"}
MAIN_GROUPS = ("M01", "M02", "M03", "M04", "Code01", "Code02", "IF01", "IF02")
LONG_GROUPS = ("M01", "M02")
PROTOCOL = {
    "schema": 1,
    "study": "prefix-fusion-scientific-v2",
    "pairs": ["J", "P"],
    "arms": ARMS,
    "training": dict(
        outers=40,
        reuse=16,
        batch=64,
        prompt_cap=4096,
        response_cap=7168,
        seed=42,
        shuffle=True,
        stop_ids=[151643],
        termination_token_ids=[],
        objective="native_full_vocab_reverse_kl",
        learning_rate=1e-6,
        optimizer="AdamW",
        betas=[0.9, 0.999],
        epsilon=1e-8,
        weight_decay=0.01,
        gradient_clip=1.0,
        scheduler="constant",
        warmup_steps=0,
        loss_aggregation="token-mean",
        master_parameters="float32",
        backbone_forward="bfloat16",
        gradient_reduction="float32",
        task_rewards_in_loss=False,
    ),
    "fixed_risk": dict(
        outers=list(range(0, 41, 5)),
        groups=list(MAIN_GROUPS),
        horizon=1024,
        sources=["S0", "paired_teacher"],
        metrics=["rkl", "js"],
        aggregation="per_prompt_sum_divided_by_fixed_H_then_mean",
        domain_weights={"math": 64, "code": 32, "if": 32},
    ),
    "long_risk": dict(
        outers=[0, 20, 40],
        groups=list(LONG_GROUPS),
        horizon=7168,
        windows=[[0, 1024], [1024, 4096], [4096, 7168]],
        prefix_policy="slice_one_full_bank_not_regenerate",
    ),
    "accuracy": dict(
        outers=[0, 40],
        samples=4,
        temperature=0.7,
        top_p=0.9,
        response_cap=31744,
        model_length=32768,
        seed=72000000,
        stop_ids=[151643],
        models="plain_student_checkpoints; original student and teacher baselines",
        primary="macro5_avg_at_4",
        intermediate_dev_outer=20,
        intermediate_dev_groups=["M01", "M02", "M03", "M04"],
    ),
    "banks": dict(
        generators=["S0", "J", "P"],
        temperature=1.0,
        top_p=1.0,
        top_k=-1,
        seed=52000000,
        stop_ids=[151643],
        math_horizon=7168,
        other_horizon=1024,
        support="shared_legal_token_ids; native scoring heads unchanged",
    ),
    "checkpoints": dict(
        persistent_weight_outers=[20, 40],
        resumable_every=20,
        resumable_keep=1,
        temporary_readouts_every=5,
        readout_dtype="float32_master",
        optimizer_in_readouts=False,
    ),
    "statistics": dict(
        training_seeds=[42],
        bootstrap_unit="prompt",
        bootstrap_draws=10000,
        bootstrap_seed=2026102712,
        scope="exploratory; no training-seed uncertainty estimate",
    ),
    "fresh": dict(
        outers=[0, 40],
        groups=["M01", "M02", "M03", "M04"],
        samples=2,
        temperature=1.0,
        top_p=1.0,
        response_cap=7168,
        seed=62000000,
        endpoint_cross_scoring="S0_plus_six_learner_endpoints_by_their_frozen_fresh_banks",
        metric_scope="conditional native-head risk on visited states, not asserted trajectory KL",
    ),
}


def protocol_identity():
    return fingerprint(PROTOCOL)
