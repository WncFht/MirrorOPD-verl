# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Freeze endpoint-visited states and cross-score native-head conditional risk.

The seven parameter/state labels are initial, S0, S, PM, LM, TM, T. Initial is
untrained; S0 is the trained learner whose training generator stayed frozen.
Generation masks shared legal IDs; scoring preserves both full native heads.
Consequently these risks are not asserted to equal sampling-trajectory KL.
"""

import argparse
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from examples.mirroropd.research.core.artifacts import complete, file_hash, fingerprint, receipt
from examples.mirroropd.research.core.io import dump, finite, read
from examples.mirroropd.research.studies.prefix_fusion.assets import verify_model_assets
from examples.mirroropd.research.studies.prefix_fusion.protocol import ARMS, PROTOCOL, protocol_identity

ENDPOINTS = ("initial", *ARMS)
SCOPE = PROTOCOL["fresh"]["metric_scope"]


def study_manifest(path):
    """Accept explicit run bindings or the expanded controller's authoritative assignments."""
    path = Path(path).resolve()
    digest = file_hash(path)
    raw = read(path)
    if file_hash(path) != digest:
        raise ValueError("Run assignments changed while reading their manifest")
    if raw.get("protocol") != protocol_identity():
        raise ValueError("Run manifest belongs to another scientific protocol")
    if "conditions" in raw:
        pairs = {pair: {} for pair in PROTOCOL["pairs"]}
        for condition in raw["conditions"]:
            pair, arm = condition["pair"], condition["arm"]
            if pair not in pairs or arm not in ARMS or arm in pairs[pair]:
                raise ValueError("Duplicate or foreign scientific condition")
            if condition["key"] != f"{pair}-{arm}" or condition["target_updates"] != 640:
                raise ValueError("Invalid expanded condition identity or budget")
            attempt = condition["attempt"]
            if type(attempt) is not int or attempt < 1:
                raise ValueError("Invalid expanded attempt")
            directory = path.parent / "attempts" / condition["key"] / f"a{attempt:02d}"
            pairs[pair][arm] = dict(run=condition["run"], audit=str(directory / "audit.json"))
    else:
        if raw.get("schema") != 1:
            raise ValueError("Unknown scientific run manifest schema")
        pairs = raw["pairs"]
    if set(pairs) != set(PROTOCOL["pairs"]) or any(set(pairs[p]) != set(ARMS) for p in pairs):
        raise ValueError("The manifest must explicitly bind all twelve conditions")
    for conditions in pairs.values():
        for condition in conditions.values():
            if not all(
                isinstance(condition.get(k), str) and Path(condition[k]).is_absolute() for k in ["run", "audit"]
            ):
                raise ValueError("Run and audit bindings must use explicit absolute paths")
    runs = [value["run"] for conditions in pairs.values() for value in conditions.values()]
    if len(set(runs)) != 12:
        raise ValueError("Scientific conditions share a run directory")
    return dict(schema=1, protocol=protocol_identity(), pairs=pairs, source_sha256=digest)


def code_identity():
    research = Path(__file__).resolve().parents[2]
    return {
        name: file_hash(research / name)
        for name in ["studies/prefix_fusion/fresh.py", "backends/hf_models.py", "backends/hf_rkl.py", "core/metrics.py"]
    }


def fresh_questions(root, smoke=False):
    prepared = Path(root) / "prepared-03"
    stored = read(prepared / "complete.json")
    if stored["inputs"]["protocol"] != protocol_identity():
        raise ValueError("Prepared questions belong to another protocol")
    path = prepared / "heldout-questions.json"
    if file_hash(path) != stored["artifacts"][path.name]:
        raise ValueError("Prepared held-out questions changed")
    rows = [r for r in read(path) if r["group"] in PROTOCOL["fresh"]["groups"]]
    if len(rows) != 64 or len({r["id"] for r in rows}) != 64 or any(r["domain"] != "math" for r in rows):
        raise ValueError("Fresh generation requires exactly the declared Math64 panel")
    return rows[:2] if smoke else rows


def request_rows(questions, endpoint, *, smoke=False):
    if endpoint not in ENDPOINTS:
        raise ValueError("Unknown parameter/state endpoint")
    cap = 32 if smoke else PROTOCOL["fresh"]["response_cap"]
    result = []
    for row in questions:
        question = row["id"]
        if type(question) is not int or question < 0:
            raise ValueError("Fresh question IDs must be nonnegative integers")
        for sample in range(PROTOCOL["fresh"]["samples"]):
            result.append(
                dict(
                    row,
                    id=2 * question + sample,
                    question_id=question,
                    sample_index=sample,
                    source=endpoint,
                    seed=PROTOCOL["fresh"]["seed"] + 100 * question + sample,
                    cap=cap,
                )
            )
    if len({r["id"] for r in result}) != len(result):
        raise ValueError("Duplicate fresh response slot")
    return result


def model_identity(root, pair, endpoint, manifest=None):
    """Only accepted final runs may contribute trained endpoint parameters."""
    if pair not in PROTOCOL["pairs"] or endpoint not in ENDPOINTS:
        raise ValueError("Unknown teacher or endpoint")
    root = Path(root)
    asset = read(root / "assets/complete.json")["models"]["S0"]
    original = read(Path(asset["directory"]) / "config.json")
    binding = dict(endpoint=endpoint, pair="shared" if endpoint == "initial" else pair)
    if endpoint == "initial":
        verify_model_assets(asset)
        directory = Path(asset["directory"])
        binding["original_model"] = asset["identity"]
    else:
        if manifest is None:
            raise ValueError("A trained endpoint requires an explicit accepted run binding")
        entry = study_manifest(manifest)["pairs"][pair][endpoint]
        audit = read(entry["audit"])
        if not audit.get("passed") or not audit.get("formal_complete") or audit.get("applied_updates") != 640:
            raise ValueError("Trained endpoint has not passed full run acceptance")
        if audit.get("pair") != pair or audit.get("arm") != endpoint:
            raise ValueError("Endpoint audit belongs to another condition")
        run = Path(entry["run"])
        dest = run / "science/outer-040"
        state = read(dest / "complete.json")
        if (state["outer"], state["optimizer_step"], state["protocol"], state["smoke"], state["retained_weights"]) != (
            40,
            640,
            protocol_identity(),
            False,
            True,
        ):
            raise ValueError("Endpoint is not the accepted FP32 outer40 readout")
        if state["fixed_receipt"] != file_hash(dest / "fixed/complete.json"):
            raise ValueError("Endpoint fixed-risk receipt changed")
        fixed = read(dest / "fixed/complete.json")
        teacher = read(root / "teacher-cache" / pair / "complete.json")
        if fixed["inputs"]["target"] != teacher["fingerprint"]:
            raise ValueError("Endpoint was trained/read out against another teacher")
        directory = dest / "weights/huggingface"
        binding.update(
            run=str(run), run_audit_sha256=file_hash(entry["audit"]), readout_sha256=file_hash(dest / "complete.json")
        )
    config = read(directory / "config.json")
    for field in [
        "model_type",
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "vocab_size",
    ]:
        if config[field] != original[field]:
            raise ValueError(f"Endpoint learner architecture changed: {field}")
    files = sorted(directory.glob("*.safetensors"))
    if not files:
        raise ValueError("Endpoint has no safetensors weights")
    weights = fingerprint({p.name: file_hash(p) for p in [directory / "config.json", *files]})
    if endpoint != "initial":
        if weights != fixed["inputs"]["weights"]:
            raise ValueError("Endpoint weights changed after the accepted readout")
        from safetensors import safe_open

        for path in files:
            with safe_open(path, framework="pt", device="cpu") as handle:
                if any(handle.get_slice(name).get_dtype() != "F32" for name in handle.keys()):
                    raise ValueError("Trained endpoint weights must preserve FP32 masters")
    return dict(binding, directory=str(directory), identity=weights)


def validate_bank_rows(rows, expected, legal):
    if len(rows) != len(expected) or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Incomplete or duplicate fresh generation slots")
    expected = {r["id"]: r for r in expected}
    legal = set(legal)
    for row in rows:
        original = expected.get(row["id"])
        if original is None or any(row.get(k) != v for k, v in original.items()):
            raise ValueError("Fresh response identity changed")
        tokens = row["response_ids"]
        if not 0 < len(tokens) <= row["cap"] or any(t not in legal for t in tokens):
            raise ValueError("Fresh response violates length or token support")
        if (
            row["finish_reason"] not in {"stop", "length"}
            or (row["finish_reason"] == "stop" and tokens[-1] != 151643)
            or (row["finish_reason"] == "length" and len(tokens) != row["cap"])
        ):
            raise ValueError("Fresh response has an invalid stopping event")
        probabilities = row["behavior_log_probs"]
        if len(probabilities) != len(tokens) or any(not math.isfinite(p) for p in probabilities):
            raise ValueError("Fresh response lacks finite selected-token probabilities")


def load_fresh_bank(root, path, *, allow_smoke=False):
    root, path = Path(root), Path(path)
    stored = read(path / "complete.json")
    inputs = stored["inputs"]
    if stored["fingerprint"] != fingerprint(inputs) or inputs["protocol"] != protocol_identity():
        raise ValueError("Fresh bank protocol or receipt changed")
    if inputs["smoke"] and not allow_smoke:
        raise ValueError("Short fresh probes cannot enter formal analysis")
    if file_hash(path / "bank.json") != stored["bank_sha256"]:
        raise ValueError("Fresh bank data changed")
    legal = read(root / "assets/legal-token-ids.json")
    if file_hash(root / "assets/legal-token-ids.json") != inputs["support"]:
        raise ValueError("Fresh sampling support changed")
    expected = request_rows(fresh_questions(root, inputs["smoke"]), inputs["model"]["endpoint"], smoke=inputs["smoke"])
    rows = read(path / "bank.json")
    questions = fresh_questions(root, inputs["smoke"])
    if inputs["questions"] != fingerprint(questions) or inputs["sampling"] != PROTOCOL["fresh"]:
        raise ValueError("Fresh bank question or sampling protocol changed")
    if stored["rows"] != len(expected) or stored["questions"] != len(questions) or stored["samples"] != 2:
        raise ValueError("Fresh bank completion counts changed")
    validate_bank_rows(rows, expected, legal)
    return rows, stored


def generate(root, pair, endpoint, out, manifest=None, *, smoke=False):
    from examples.mirroropd.research.studies.prefix_fusion.scoring import require_allocated_gpu

    require_allocated_gpu()
    from vllm import LLM, SamplingParams

    from verl.workers.rollout.vllm_fusion.compat import decode_graph_config

    root, out = Path(root), Path(out)
    model = model_identity(root, pair, endpoint, manifest)
    questions = fresh_questions(root, smoke)
    requests = request_rows(questions, endpoint, smoke=smoke)
    legal = read(root / "assets/legal-token-ids.json")
    inputs = dict(
        protocol=protocol_identity(),
        model=model,
        questions=fingerprint(questions),
        support=file_hash(root / "assets/legal-token-ids.json"),
        sampling=PROTOCOL["fresh"],
        smoke=smoke,
        code=code_identity(),
        metric_scope=SCOPE,
        execution=dict(
            dtype="bfloat16",
            backend="TRITON_ATTN",
            tp=1,
            max_model_len=11264,
            max_num_seqs=32,
            graph=True,
            prefix_caching=False,
            chunked_prefill=False,
        ),
    )
    out.mkdir(parents=True, exist_ok=False)
    dump(out / "inputs.json", inputs)
    config = read(Path(model["directory"]) / "config.json")
    if config["eos_token_id"] != 151643 or 151643 not in legal:
        raise ValueError("Fresh generator EOS is outside the frozen support")
    seqs, context = 32, 11264
    per_token = (
        4
        * config["num_hidden_layers"]
        * config["num_key_value_heads"]
        * (config.get("head_dim") or config["hidden_size"] // config["num_attention_heads"])
    )
    quantum = 256 * 1024**2
    kv = math.ceil(per_token * context * seqs / quantum) * quantum
    llm = LLM(
        model=model["directory"],
        dtype="bfloat16",
        generation_config="vllm",
        tensor_parallel_size=1,
        max_model_len=context,
        max_num_batched_tokens=context,
        max_num_seqs=seqs,
        enforce_eager=False,
        compilation_config=decode_graph_config(seqs),
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        kv_cache_memory_bytes=kv,
        attention_config={"backend": "TRITON_ATTN"},
        logprobs_mode="processed_logprobs",
        worker_extension_cls="verl.workers.rollout.vllm_rollout.utils.vLLMColocateWorkerExtension",
    )
    llm.collective_rpc("monkey_patch_model", args=(max(legal) + 1, sorted(set(range(max(legal) + 1)) - set(legal))))
    params = [
        SamplingParams(
            temperature=1.0,
            top_p=1.0,
            top_k=-1,
            max_tokens=r["cap"],
            seed=r["seed"],
            stop_token_ids=[151643],
            logprobs=0,
        )
        for r in requests
    ]
    outputs = llm.generate([{"prompt_token_ids": r["prompt_ids"]} for r in requests], params, use_tqdm=True)
    if len(outputs) != len(requests):
        raise ValueError("Fresh generation returned an incomplete batch")
    rows = []
    for request, result in zip(requests, outputs, strict=True):
        if list(result.prompt_token_ids) != request["prompt_ids"] or len(result.outputs) != 1:
            raise ValueError("Fresh generation returned another request")
        output = result.outputs[0]
        tokens = list(output.token_ids)
        rows.append(
            dict(
                request,
                response_ids=tokens,
                output=output.text,
                finish_reason=output.finish_reason,
                behavior_log_probs=[float(p[t].logprob) for t, p in zip(tokens, output.logprobs, strict=True)],
            )
        )
    validate_bank_rows(rows, requests, legal)
    dump(out / "bank.json", rows)
    if code_identity() != inputs["code"]:
        raise ValueError("Fresh generator implementation changed while running")
    receipt(
        out,
        inputs,
        rows=len(rows),
        questions=len(questions),
        samples=2,
        bank_sha256=file_hash(out / "bank.json"),
        response_tokens=sum(len(r["response_ids"]) for r in rows),
        kv_bytes=kv,
        physical_gpu=os.environ["CUDA_VISIBLE_DEVICES"],
    )


def cache_teacher(root, pair, bank, out):
    from examples.mirroropd.research.studies.prefix_fusion.scoring import require_allocated_gpu

    require_allocated_gpu()
    import torch

    from examples.mirroropd.research.backends import hf_models, hf_rkl

    root, bank, out = Path(root), Path(bank), Path(out)
    rows, bank_receipt = load_fresh_bank(root, bank, allow_smoke=True)
    asset = read(root / "assets/complete.json")["models"][pair]
    verify_model_assets(asset)
    inputs = dict(
        protocol=protocol_identity(),
        pair=pair,
        teacher=asset["identity"],
        bank=bank_receipt["bank_sha256"],
        smoke=bank_receipt["inputs"]["smoke"],
        code=code_identity(),
        contract="fresh-native-head-v1",
        precision="BF16 teacher backbone; FP32 projection; full native teacher head",
    )
    if complete(out, inputs):
        verify_cache(out, inputs)
        return
    out.mkdir(parents=True, exist_ok=True)
    if (out / "inputs.json").exists() and read(out / "inputs.json") != inputs:
        raise ValueError("Partial fresh teacher cache belongs to different inputs")
    dump(out / "inputs.json", inputs)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = hf_models.load(SimpleNamespace(directory=Path(asset["directory"])))
    head = model.get_output_embeddings()
    if head.weight.shape[0] != asset["vocab_size"]:
        raise ValueError("Teacher native vocabulary head was narrowed")
    hf_rkl.save_tensor(out / "head.pt", head.weight.detach().cpu())
    reconstruction = None
    with torch.no_grad():
        for index, row in enumerate(rows):
            path = out / "rows" / f"{row['id']}.pt"
            if path.exists():
                old = torch.load(path, map_location="cpu", weights_only=True)
                if old["row_identity"] != fingerprint(row) or old["teacher"] != asset["identity"]:
                    raise ValueError("Partial fresh teacher row changed")
                if index:
                    continue
            states = hf_models.hidden(model, [row], pad_id=151643)
            if len(states) != len(row["response_ids"]) or not torch.isfinite(states).all():
                raise ValueError("Invalid teacher hidden states")
            if index == 0:
                positions = sorted({0, len(states) // 2, len(states) - 1})
                absolute = torch.tensor([len(row["prompt_ids"]) - 1 + j for j in positions], device="cuda")

                def project(module, args, output):
                    with torch.autocast("cuda", enabled=False):
                        return torch.nn.functional.linear(args[0].float(), module.weight.float())

                hook = head.register_forward_hook(project)
                try:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        expected = (
                            model(
                                input_ids=torch.tensor([row["prompt_ids"] + row["response_ids"][:-1]], device="cuda"),
                                use_cache=False,
                                logits_to_keep=absolute,
                            )
                            .logits[0]
                            .float()
                        )
                finally:
                    hook.remove()
                actual = torch.nn.functional.linear(states[positions].float(), head.weight.float())
                error = float((actual - expected).abs().max())
                if not math.isfinite(error) or error >= 1e-4:
                    raise ValueError(f"Fresh teacher hidden/head reconstruction failed: {error}")
                reconstruction = dict(id=row["id"], positions=positions, max_logit_difference=error)
            if not path.exists():
                hf_rkl.save_tensor(
                    path, dict(row_identity=fingerprint(row), teacher=asset["identity"], hidden=states.cpu())
                )
    files = {str(path.relative_to(out)): file_hash(path) for path in sorted((out / "rows").glob("*.pt"))}
    if len(files) != len(rows):
        raise ValueError("Incomplete or foreign fresh cache rows")
    if code_identity() != inputs["code"]:
        raise ValueError("Fresh cache implementation changed while running")
    receipt(
        out,
        inputs,
        rows=len(rows),
        head_sha256=file_hash(out / "head.pt"),
        row_files=files,
        reconstruction=reconstruction,
        native_teacher_vocab=asset["vocab_size"],
    )


def verify_cache(path, expected_inputs=None):
    path = Path(path)
    stored = read(path / "complete.json")
    if stored["fingerprint"] != fingerprint(stored["inputs"]) or (
        expected_inputs and stored["inputs"] != expected_inputs
    ):
        raise ValueError("Fresh teacher cache receipt changed")
    if stored["inputs"]["protocol"] != protocol_identity() or stored["inputs"]["contract"] != "fresh-native-head-v1":
        raise ValueError("Unknown fresh teacher cache contract")
    if file_hash(path / "head.pt") != stored["head_sha256"]:
        raise ValueError("Fresh teacher head changed")
    for name, digest in stored["row_files"].items():
        if file_hash(path / name) != digest:
            raise ValueError("Fresh teacher hidden states changed")
    return stored


def score_row(root, pair, parameter, banks, caches, out, manifest=None, *, sources=None):
    """Load one FP32 learner once, then score the seven frozen state banks separately."""
    from examples.mirroropd.research.studies.prefix_fusion.scoring import require_allocated_gpu

    require_allocated_gpu()
    import torch

    from examples.mirroropd.research.backends import hf_models, hf_rkl

    root, banks, caches, out = map(Path, (root, banks, caches, out))
    selected = list(ENDPOINTS if sources is None else sources)
    if not selected or len(set(selected)) != len(selected) or not set(selected) <= set(ENDPOINTS):
        raise ValueError("Choose distinct known state endpoints for a matrix row")
    if manifest is None and (parameter != "initial" or any(source != "initial" for source in selected)):
        raise ValueError("Trained parameter/state endpoints require an explicit run manifest")
    point = model_identity(root, pair, parameter, manifest)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = hf_models.load(SimpleNamespace(directory=Path(point["directory"])), trainable=True)
    model.requires_grad_(False)
    model.eval()
    for source in selected:
        rows, bank = load_fresh_bank(root, banks / source, allow_smoke=True)
        cache_path = caches / source
        cache = verify_cache(cache_path)
        if cache["inputs"]["pair"] != pair or cache["inputs"]["bank"] != bank["bank_sha256"]:
            raise ValueError("Fresh teacher cache belongs to different states/teacher")
        if cache["inputs"]["teacher"] != read(root / "assets/complete.json")["models"][pair]["identity"]:
            raise ValueError("Fresh teacher identity changed")
        current_code = code_identity()
        if cache["inputs"]["code"] != current_code:
            raise ValueError("Fresh teacher cache uses a different scoring implementation")
        if bank["inputs"]["model"]["endpoint"] != source or bank["inputs"]["model"]["pair"] != (
            "shared" if source == "initial" else pair
        ):
            raise ValueError("Fresh state-bank alias points to another endpoint")
        if source != "initial":
            entry = study_manifest(manifest)["pairs"][pair][source]
            if Path(bank["inputs"]["model"]["run"]).resolve() != Path(entry["run"]).resolve() or bank["inputs"][
                "model"
            ]["run_audit_sha256"] != file_hash(entry["audit"]):
                raise ValueError("Fresh bank came from an unbound trained endpoint")
        inputs = dict(
            protocol=protocol_identity(),
            pair=pair,
            parameter=parameter,
            state=source,
            model=point,
            bank=bank["bank_sha256"],
            cache=cache["fingerprint"],
            smoke=bank["inputs"]["smoke"],
            horizon=32 if bank["inputs"]["smoke"] else 7168,
            code=current_code,
            metric_scope=SCOPE,
            precision="FP32 learner master/head; BF16 backbone; full native teacher head",
        )
        dest = out / source
        if complete(dest, inputs):
            if file_hash(dest / "per-question.json") != read(dest / "complete.json")["metrics_sha256"]:
                raise ValueError("Completed fresh scores changed")
            continue
        dest.mkdir(parents=True, exist_ok=False)
        head = torch.load(cache_path / "head.pt", map_location="cuda", weights_only=True).float()
        states = {}
        for row in rows:
            value = torch.load(cache_path / "rows" / f"{row['id']}.pt", map_location="cpu", weights_only=True)
            if value["row_identity"] != fingerprint(row) or value["teacher"] != cache["inputs"]["teacher"]:
                raise ValueError("Fresh cached row identity changed")
            states[row["id"]] = value["hidden"]
        scored = hf_rkl.score(
            model,
            head,
            states,
            rows,
            0 if parameter == "initial" else 640,
            SimpleNamespace(horizon=inputs["horizon"]),
            SimpleNamespace(microbatch=1, token_chunk=512),
            151643,
        )
        identities = {row["id"]: row for row in rows}
        for row in scored:
            original = identities[row["id"]]
            row.update(question_id=original["question_id"], sample_index=original["sample_index"])
        finite(scored)
        dump(dest / "per-question.json", scored)
        if code_identity() != current_code:
            raise ValueError("Fresh scorer changed during execution")
        receipt(
            dest,
            inputs,
            rows=len(scored),
            questions=len(scored) // 2,
            optimizer_updates=0,
            metrics_sha256=file_hash(dest / "per-question.json"),
        )
        del head, states


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["generate", "cache-teacher", "score-row"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--pair", choices=PROTOCOL["pairs"], required=True)
    parser.add_argument("--endpoint", choices=ENDPOINTS)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--bank", type=Path)
    parser.add_argument("--banks", type=Path)
    parser.add_argument("--caches", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--smoke", action="store_true", help="Short generate-only probe; scoring inherits the bank scope"
    )
    parser.add_argument("--states", choices=ENDPOINTS, nargs="+", help="Optional incremental score-row subset")
    args = parser.parse_args()
    if args.smoke and args.command != "generate":
        parser.error("--smoke applies to generation; downstream jobs inherit the frozen bank scope")
    if args.states and args.command != "score-row":
        parser.error("--states only applies to score-row")
    required = {"generate": ["endpoint"], "cache-teacher": ["bank"], "score-row": ["endpoint", "banks", "caches"]}[
        args.command
    ]
    if any(getattr(args, name) is None for name in required):
        parser.error("Missing command arguments: " + ",".join(required))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tag = args.out.parent / f"{args.out.name}-run-{time.time_ns()}"
    dump(
        str(tag) + "-source.json",
        dict(
            code=code_identity(),
            command=sys.argv,
            revision=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[5], text=True
            ).strip(),
        ),
    )
    monitor = Path(__file__).resolve().parents[3] / "benchmarks/fusion/monitor.py"
    observer = subprocess.Popen(
        [sys.executable, str(monitor), "--parent", str(os.getpid()), "--out", str(tag) + "-resources.jsonl"]
    )
    assert observer.pid > 0
    if args.command == "generate":
        generate(args.root, args.pair, args.endpoint, args.out, args.manifest, smoke=args.smoke)
    elif args.command == "cache-teacher":
        cache_teacher(args.root, args.pair, args.bank, args.out)
    else:
        score_row(
            args.root, args.pair, args.endpoint, args.banks, args.caches, args.out, args.manifest, sources=args.states
        )


if __name__ == "__main__":
    main()
