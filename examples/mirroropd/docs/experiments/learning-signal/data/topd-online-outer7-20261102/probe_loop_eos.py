"""Inspect real cached last-scored prefixes of the kappa0 outer7 rollout bank.

CPU diagnostic: BF16-quantized operands, FP32 GEMM, FP64 probabilities. Each
trajectory contributes one prefix before its final generated token. This is
not a token-average estimate, model update, or full-parameter causal effect.
"""

import hashlib
import json
from pathlib import Path

import torch
from probe_target_overlap import ETA, target_logs
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent
EOS = [151643, 151645]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for value in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(value)
    return h.hexdigest()


def identity(row):
    return hashlib.sha256(
        json.dumps({k: row[k] for k in ["request_id", "prompt_ids", "response_ids"]}, separators=(",", ":")).encode()
    ).hexdigest()


def main():
    torch.set_num_threads(4)
    out = ROOT / "loop-eos-diagnostic-20261102.json"
    assert not out.exists()
    folder = ROOT / "online/reuse-k0/outer07"
    with (folder / "bank.jsonl").open() as stream:
        rows = [json.loads(line) for line in stream]
    assert len(rows) == 64
    assert digest(folder / "bank.jsonl") == json.loads((folder / "bank-complete.json").read_text())["sha256"]
    protocol = json.loads((folder / "protocol.json").read_text())
    parent = json.loads((ROOT / "online/reuse-k0/outer06/complete.json").read_text())
    assert parent["status"] == "complete" and parent["updates"] == 96
    assert protocol["checkpoint_sha256"] == parent["checkpoint_sha256"]
    tokenizer = AutoTokenizer.from_pretrained(protocol["checkpoint"], local_files_only=True)
    raw_logs, provenance = {}, {}
    for role in ["anchor", "teacher"]:
        receipt = json.loads((folder / f"{role}-cache-complete.json").read_text())
        head_path = folder / f"states/{role}/head.pt"
        assert receipt["records"] == 64 and digest(head_path) == receipt["head_sha256"]
        payload = torch.load(head_path, mmap=True, map_location="cpu", weights_only=True)
        weight = payload["head"].to(torch.bfloat16).float()
        hidden, state_hashes = [], {}
        for row in rows:
            path = folder / f"states/{role}/{row['request_id']}.pt"
            state = torch.load(path, mmap=True, map_location="cpu", weights_only=True)
            assert state["identity"]["trajectory_sha256"] == identity(row)
            assert len(state["hidden"]) == len(row["response_ids"])
            hidden.append(state["hidden"][-1].to(torch.bfloat16).float())
            state_hashes[row["request_id"]] = digest(path)
        logits = torch.stack(hidden) @ weight.T
        assert torch.isfinite(logits).all()
        raw_logs[role] = logits.double().log_softmax(-1)
        provenance[role] = {
            "head_sha256": receipt["head_sha256"],
            "model_identity": payload["identity"],
            "state_sha256": state_hashes,
        }
        del payload, weight, hidden, logits
    keep = [i for i in range(raw_logs["anchor"].shape[1]) if i not in EOS]
    coarse = {
        role: torch.cat([logp[:, keep], logp[:, EOS].logsumexp(-1, keepdim=True)], dim=-1)
        for role, logp in raw_logs.items()
    }
    lq, lt = coarse["anchor"], coarse["teacher"]
    q, teacher = lq.exp(), lt.exp()
    target, _ = target_logs(lq, lt)
    gradients, max_errors = {}, {}
    for key, logm in target.items():
        kl = (q * (lq - logm)).sum(-1)
        analytic = q[:, -1] * (lq[:, -1] - logm[:, -1] - kl) / ETA
        bias = torch.zeros(len(rows), dtype=torch.float64, requires_grad=True)
        shift = torch.zeros_like(lq)
        shift[:, -1] = bias
        logp = (lq + shift).log_softmax(-1)
        objective = (logp.exp() * (logp - logm)).sum() / ETA
        autodiff = torch.autograd.grad(objective, bias)[0]
        torch.testing.assert_close(analytic, autodiff, atol=1e-11, rtol=1e-10)
        gradients[key] = analytic
        max_errors[key] = float((analytic - autodiff).abs().max())
    states = []
    for i, row in enumerate(rows):
        entry = {
            "request_id": row["request_id"],
            "finish_reason": row["finish_reason"],
            "position": len(row["response_ids"]) - 1,
            "generated_token": row["response_ids"][-1],
            "q_eos": float(q[i, -1]),
            "teacher_eos": float(teacher[i, -1]),
            "eos_log_teacher_over_q": float(lt[i, -1] - lq[i, -1]),
            "target_eos": {k: float(v[i, -1].exp()) for k, v in target.items()},
            "eos_bias_loss_derivative": {k: float(v[i]) for k, v in gradients.items()},
        }
        for name, probs in [("q", q), ("teacher", teacher)]:
            index = int(probs[i].argmax())
            token = None if index == len(keep) else keep[index]
            entry[name + "_top_class"] = {
                "token_id": token,
                "text": "<EOS class>" if token is None else tokenizer.decode([token]),
                "probability": float(probs[i, index]),
            }
        states.append(entry)
    groups = {}
    for name, subset in [
        ("capped", [s for s in states if s["finish_reason"] == "length"]),
        ("stopped", [s for s in states if s["finish_reason"] == "stop"]),
    ]:
        groups[name] = {
            "states": len(subset),
            "mean_q_eos": sum(s["q_eos"] for s in subset) / len(subset),
            "mean_teacher_eos": sum(s["teacher_eos"] for s in subset) / len(subset),
            "teacher_eos_ge_1percent": sum(s["teacher_eos"] >= 0.01 for s in subset),
            "teacher_top_is_eos": sum(s["teacher_top_class"]["token_id"] is None for s in subset),
            "mean_bias_derivative": {
                k: sum(s["eos_bias_loss_derivative"][k] for s in subset) / len(subset) for k in gradients
            },
        }
        print(name, groups[name], flush=True)
    report = {
        "scope": __doc__,
        "states": states,
        "groups": groups,
        "cache_identity": provenance,
        "bank_sha256": digest(folder / "bank.jsonl"),
        "parent_updates": 96,
        "autograd_max_abs_error": max_errors,
        "script_sha256": digest(Path(__file__)),
        "target_helper_sha256": digest(ROOT / "probe_target_overlap.py"),
        "interpretation": (
            "Negative loss derivative means descent would increase a common EOS logit bias. "
            "The model has no explicit such bias parameter."
        ),
        "limits": [
            "Conditioned on the student's existing prefix, not teacher generation from the original question.",
            "These are pre-final-action states, not the uncomputed state after a capped response.",
            "State groups are selected by stopping outcomes; no population or parameter-gradient inference.",
            "CPU normalization is FP64; no claim of bitwise GPU equivalence or improved task accuracy.",
        ],
    }
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
