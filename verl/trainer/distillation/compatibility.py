# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""CPU-only model-pair acceptance. Never infer compatibility from head size alone."""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from verl.models.transformers.full_vocab import linear_head_layout

from .distribution import DistributionSpec
from .token_mapping import TokenIdMapping


@dataclass(frozen=True)
class ModelPairContract:
    student_type: str
    teacher_type: str
    student_width: int
    teacher_width: int
    student_context: int
    teacher_context: int
    shared_input_ids: int
    student_eos: object
    teacher_eos: object
    student_tied: bool
    teacher_tied: bool
    distribution: dict
    tokenizer_sha256: str
    dynamic_date: bool = False
    input_mapping_sha256: str = "identity"
    input_protocol: str = "shared_student_prefix"

    @classmethod
    def build(
        cls,
        student_config,
        teacher_config,
        student_tokenizer,
        teacher_tokenizer,
        *,
        distribution=None,
        termination_ids=(),
        allow_permutation=False,
    ):
        linear_head_layout(student_config)
        linear_head_layout(teacher_config)
        sv, tv = student_tokenizer.get_vocab(), teacher_tokenizer.get_vocab()
        if not sv or max(sv.values()) >= student_config.vocab_size or max(tv.values()) >= teacher_config.vocab_size:
            raise ValueError("Tokenizer IDs exceed an input/output vocabulary")
        mapping = TokenIdMapping.build(student_tokenizer, teacher_tokenizer)
        if not set(termination_ids) <= set(sv.values()):
            raise ValueError("Termination groups must use actual tokenizer IDs")
        if not mapping.is_identity and not allow_permutation:
            raise ValueError("Input IDs differ; explicitly enable input_id_mapping=token_semantics")
        if not mapping.is_identity and DistributionSpec.parse(distribution).mode != "common_conditioned":
            raise ValueError("Permuted input IDs require explicit paired conditional output support")
        space = DistributionSpec.parse(distribution).compile(
            student_config.vocab_size, teacher_config.vocab_size, termination_ids
        )
        if space.spec.mode == "common_conditioned":
            reverse_s = {v: k for k, v in sv.items()}
            reverse_t = {v: k for k, v in tv.items()}
            for i, j in zip(space.student_ids, space.teacher_ids, strict=True):
                if reverse_s.get(i) != reverse_t.get(j) or (i != j and i not in reverse_s):
                    raise ValueError("Output action mapping does not preserve token meanings")
        return cls(
            student_config.model_type,
            teacher_config.model_type,
            student_config.vocab_size,
            teacher_config.vocab_size,
            student_config.max_position_embeddings,
            teacher_config.max_position_embeddings,
            len(sv),
            student_config.eos_token_id,
            teacher_config.eos_token_id,
            student_config.tie_word_embeddings,
            teacher_config.tie_word_embeddings,
            space.metadata,
            hashlib.sha256(student_tokenizer.backend_tokenizer.to_str().encode()).hexdigest(),
            "strftime_now" in str(student_tokenizer.chat_template),
            mapping.mapping_id,
        )

    def validate_prefix(self, token_ids):
        if not token_ids or len(token_ids) > min(self.student_context, self.teacher_context):
            raise ValueError("Shared prefix exceeds the verified model-pair context")
        limit = (
            self.student_width
            if self.input_mapping_sha256 != "identity"
            else min(self.student_width, self.teacher_width)
        )
        if min(token_ids) < 0 or max(token_ids) >= limit:
            raise ValueError("Shared prefix contains an unsupported embedding ID")

    def metadata(self):
        return asdict(self)


def inspect_pair(
    student, teacher, *, distribution=None, termination_ids=(), stop_token_ids=(), allow_permutation=False
):
    from safetensors import safe_open
    from transformers import AutoConfig, AutoTokenizer

    from verl.experimental.rollout_policy.identity import checkpoint_files

    effective_stops = []
    for path in (student, teacher):
        root = Path(path)
        config = AutoConfig.from_pretrained(path, local_files_only=True)
        generation = root / "generation_config.json"
        eos = (
            json.loads(generation.read_text()).get("eos_token_id", config.eos_token_id)
            if generation.exists()
            else config.eos_token_id
        )
        effective_stops.append(set([eos] if isinstance(eos, int) else eos or []))
        key = linear_head_layout(config)
        files = checkpoint_files(root)
        index = root / "model.safetensors.index.json"
        head = root / json.loads(index.read_text())["weight_map"][key] if index.exists() else root / "model.safetensors"
        if head.resolve() not in {f.resolve() for f in files}:
            raise ValueError("Output head is not in the verified safetensors checkpoint")
        with safe_open(head, framework="pt", device="cpu") as stream:
            if stream.get_slice(key).get_shape() != [config.vocab_size, config.hidden_size]:
                raise ValueError("Checkpoint output head does not match declared model structure")
            if "lm_head.bias" in stream.keys():
                raise ValueError("Biased checkpoint head is unsupported")
    source_tokenizer = AutoTokenizer.from_pretrained(student, local_files_only=True)
    target_tokenizer = AutoTokenizer.from_pretrained(teacher, local_files_only=True)
    mapping = TokenIdMapping.build(source_tokenizer, target_tokenizer)
    effective_stops[0].update(stop_token_ids or ())
    effective_stops[1].update(mapping.tokens(list(stop_token_ids or ())))
    if not set(termination_ids) <= effective_stops[0] or not set(mapping.tokens(termination_ids)) <= effective_stops[1]:
        raise ValueError("Termination equivalence requires common effective stopping actions for both roles")
    return ModelPairContract.build(
        AutoConfig.from_pretrained(student, local_files_only=True),
        AutoConfig.from_pretrained(teacher, local_files_only=True),
        source_tokenizer,
        target_tokenizer,
        distribution=distribution,
        termination_ids=termination_ids,
        allow_permutation=allow_permutation,
    )


def preflight_pair(config):
    """Run before Ray/GPU allocation for explicit hidden teacher-only RKL."""
    from omegaconf import OmegaConf

    from verl.workers.config.distillation_spec import resolve_distillation_spec

    if not OmegaConf.select(config, "distillation.enabled", default=False):
        return None
    loss = config.distillation.distillation_loss
    spec = resolve_distillation_spec(loss)
    if not spec.teacher_only_kl:
        return None
    full = loss.full_vocab
    teachers = config.distillation.teacher_models
    if len(teachers) != 1:
        raise ValueError("Model-pair acceptance requires exactly one teacher")
    pair = inspect_pair(
        config.actor_rollout_ref.model.path,
        next(iter(teachers.values())).model_path,
        distribution=full.get("distribution", {}),
        termination_ids=full.get("termination_token_ids", ()),
        stop_token_ids=config.actor_rollout_ref.rollout.get("stop_token_ids", ()),
        allow_permutation=full.get("input_id_mapping", "identity") == "token_semantics",
    )

    training_date = OmegaConf.select(config, "data.apply_chat_template_kwargs.date_string")
    if pair.dynamic_date and not training_date:
        raise ValueError("Dynamic-date templates require data.apply_chat_template_kwargs.date_string")
    evaluation = OmegaConf.select(config, "trainer.evaluation")
    if evaluation:
        from verl.trainer.evaluation.config import GenerationProtocol

        for name, protocol in evaluation.protocols.items():
            rendered = GenerationProtocol.parse(OmegaConf.to_container(protocol, resolve=True))
            if rendered.prompt_limit + rendered.max_new_tokens > min(pair.student_context, pair.teacher_context):
                raise ValueError(f"Evaluation protocol {name} exceeds the verified pair context")
            if pair.dynamic_date and json.loads(rendered.template_json).get("date_string") != training_date:
                raise ValueError(f"Evaluation protocol {name} must pin the same date_string as training")
    return pair


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student", required=True)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--common-vocab-size", type=int)
    parser.add_argument("--distribution-json", type=Path)
    parser.add_argument("--input-id-mapping", choices=["identity", "token_semantics"], default="identity")
    args = parser.parse_args()
    support = (
        {"mode": "common_conditioned", "common_vocab_size": args.common_vocab_size} if args.common_vocab_size else None
    )
    if args.distribution_json:
        if args.common_vocab_size:
            parser.error("Choose a distribution file or common-vocab-size")
        support = json.loads(args.distribution_json.read_text())
    print(
        json.dumps(
            inspect_pair(
                args.student,
                args.teacher,
                distribution=support,
                allow_permutation=args.input_id_mapping == "token_semantics",
            ).metadata(),
            indent=2,
        )
    )
