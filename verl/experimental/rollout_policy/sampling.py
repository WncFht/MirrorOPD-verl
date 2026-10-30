# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""One FP32 sampler for both models, with selected-token behavior probabilities."""

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SamplingConfig:
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    max_tokens: int = 128
    stop_ids: tuple[int, ...] = ()
    legal_ids: tuple[int, ...] = ()
    fusion_space: str = "logit"

    def __post_init__(self):
        if self.fusion_space not in {"logit", "probability"}:
            raise ValueError("fusion_space must be logit or probability")
        if not math.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("temperature must be finite and nonnegative")
        if not math.isfinite(self.top_p) or not 0 < self.top_p <= 1:
            raise ValueError("top_p must lie in (0, 1]")
        if type(self.top_k) is not int or self.top_k < -1:
            raise ValueError("top_k must be -1, 0, or a positive integer")
        if type(self.max_tokens) is not int or self.max_tokens < 1:
            raise ValueError("max_tokens must be positive")


def _probability_log_softmax(values):
    # Explicit FP32 logsumexp avoids the large-vocabulary CPU fused log-softmax
    # reduction error observed in native-logit replay. Shift first so the final
    # subtraction does not lose precision to a large common logit offset.
    shifted = values - values.amax(-1, keepdim=True)
    return shifted - shifted.logsumexp(-1, keepdim=True)


def probability_mixture_logits(student, teacher, alpha, *, legal_mask=None):
    """FP32 sampling logits for an arithmetic mixture on the common legal support.

    Each component is normalized BEFORE mixing. In particular, illegal/padded
    head rows must not change the effective mixture coefficient. The result is
    an input to the single shared temperature/filter/sampling stage.
    """
    if not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha must be finite and in [0, 1]")

    def view(values):
        values = values.float()
        return values.masked_fill(~legal_mask, -torch.inf) if legal_mask is not None else values.clone()

    if alpha == 0:
        return view(student)
    if alpha == 1:
        return view(teacher)
    if student.shape != teacher.shape:
        raise ValueError("Probability fusion requires aligned logits")
    student_logp = _probability_log_softmax(view(student))
    teacher_logp = _probability_log_softmax(view(teacher))
    return torch.logaddexp(student_logp + math.log1p(-alpha), teacher_logp + math.log(alpha))


def fusion_log_probs(student, teacher, alpha, config: SamplingConfig, *, legal_mask=None):
    if not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha must be finite and in [0, 1]")
    if config.legal_ids and legal_mask is None:
        active = teacher if alpha == 1 else student
        legal_mask = torch.zeros(active.shape[-1], dtype=torch.bool, device=active.device)
        legal_mask[list(config.legal_ids)] = True
    if config.fusion_space == "probability":
        logits = probability_mixture_logits(student, teacher, alpha, legal_mask=legal_mask)
    elif alpha == 0:
        logits = student.float().clone()
    elif alpha == 1:
        logits = teacher.float().clone()
    else:
        if student.shape != teacher.shape:
            raise ValueError("Fusion requires aligned logits")
        logits = student.float().mul(1 - alpha).add(teacher.float(), alpha=alpha)
    if logits.ndim != 2:
        raise ValueError("Expected [batch, vocabulary] logits")
    if config.legal_ids:
        logits.masked_fill_(~legal_mask, -torch.inf)
    if (torch.isnan(logits).any() | torch.isposinf(logits).any() | ~torch.isfinite(logits).any(-1).all()).item():
        raise ValueError("Fusion has invalid logits or empty common support")
    if config.temperature == 0:
        chosen = logits.argmax(-1, keepdim=True)
        return torch.full_like(logits, -torch.inf).scatter_(-1, chosen, 0.0)
    logits.div_(config.temperature)
    if config.top_k > 0:
        threshold = logits.topk(min(config.top_k, logits.shape[-1]), dim=-1).values[:, -1:]
        logits.masked_fill_(logits < threshold, -torch.inf)
    if config.top_p < 1:
        ordered, indices = logits.sort(dim=-1, descending=True)
        cumulative = ordered.softmax(-1).cumsum(-1)
        remove = cumulative > config.top_p
        remove[:, 1:] = remove[:, :-1].clone()
        remove[:, 0] = False
        logits.masked_fill_(torch.zeros_like(remove).scatter_(-1, indices, remove), -torch.inf)
    return _probability_log_softmax(logits) if config.fusion_space == "probability" else logits.log_softmax(-1)


def sample_tokens(log_probs, generators):
    # Independent streams survive microbatch splitting and request reordering.
    tokens = torch.stack(
        [torch.multinomial(row.exp(), 1, generator=g)[0] for row, g in zip(log_probs, generators, strict=True)]
    )
    return tokens, log_probs.gather(-1, tokens[:, None]).squeeze(-1)


def stop_event_probabilities(logits, alpha, stop_ids, native_ids, *, legal_mask=None, fusion_space="logit"):
    """Diagnostic event distributions at T=1 without filtering; never samples events.

    Native mapping sends each model's own EOS to STOP and leaves other IDs intact.
    Union mapping instead coarsens the complete raw stop set in both models.
    """
    if fusion_space not in {"logit", "probability"}:
        raise ValueError("Unknown fusion_space")
    component = {}
    result = {}
    queried = sorted(set(stop_ids) | set(native_ids))
    for name, values in logits.items():
        values = values.float()
        if legal_mask is not None:
            values = values.masked_fill(~legal_mask, -torch.inf)
        component[name] = _probability_log_softmax(values) if fusion_space == "probability" else values.log_softmax(-1)
        for token in queried:
            result[f"{name}/p_{token}"] = component[name][:, token].exp()

    def blend(student, teacher):
        if alpha == 0:
            return student
        if alpha == 1:
            return teacher
        if fusion_space == "probability":
            return torch.logaddexp(student + math.log1p(-alpha), teacher + math.log(alpha))
        return student * (1 - alpha) + teacher * alpha

    student, teacher = component.get("student"), component.get("teacher")
    mixed = blend(student, teacher)
    mixed_logp = _probability_log_softmax(mixed) if fusion_space == "probability" else mixed.log_softmax(-1)
    result["fusion/total_stop_mass"] = mixed_logp[:, list(stop_ids)].exp().sum(-1)
    for token in queried:
        result[f"fusion/p_{token}"] = mixed_logp[:, token].exp()

    def event_mass(student_ids, teacher_ids, removed):
        s = student[:, student_ids].logsumexp(-1) if student is not None else None
        t = teacher[:, teacher_ids].logsumexp(-1) if teacher is not None else None
        stop = blend(s, t)
        continuing = mixed.clone()
        continuing[:, list(removed)] = -torch.inf
        normalizer = torch.logaddexp(stop, continuing.logsumexp(-1))
        return (stop - normalizer).exp()

    s_eos, t_eos = native_ids
    removed = {s_eos} if alpha == 0 else ({t_eos} if alpha == 1 else {s_eos, t_eos})
    result["native_event/stop_mass"] = event_mass([s_eos], [t_eos], removed)
    result["union_event/stop_mass"] = event_mass(list(stop_ids), list(stop_ids), set(stop_ids))
    return result


def validate_tokenizers(student, teacher, student_vocab_size, teacher_vocab_size):
    """Check the mapping even when output head sizes happen to match."""
    vocabulary = student.get_vocab()
    if vocabulary != teacher.get_vocab():
        raise ValueError("Token fusion requires identical token-to-ID mappings")
    legal = tuple(sorted(set(vocabulary.values())))
    if not legal or legal[0] < 0 or legal[-1] >= min(student_vocab_size, teacher_vocab_size):
        raise ValueError("Tokenizer IDs fall outside one model's output head")
    return legal
