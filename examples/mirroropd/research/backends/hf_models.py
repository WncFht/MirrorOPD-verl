# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Buffer-preserving HF execution and response-aligned hidden states."""


def load(model, *, trainable=False, path=None):
    import torch
    from transformers import AutoModelForCausalLM

    value = AutoModelForCausalLM.from_pretrained(
        path or model.directory,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.float32 if trainable else torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map={"": "cuda:0"},
    ).eval()
    output = value.get_output_embeddings()
    if not isinstance(output, torch.nn.Linear) or output.bias is not None:
        raise ValueError("This adapter requires an audited bias-free linear vocabulary projection")
    if trainable and any(p.dtype != torch.float32 for p in value.parameters()):
        raise ValueError("Trainable master parameters must all remain FP32")
    value.requires_grad_(trainable)
    for name, buffer in value.named_buffers():
        if "inv_freq" in name:
            assert buffer.dtype == torch.float32, (name, buffer.dtype)
    return value


def training_mode(model, execution):
    import torch

    if execution.activation_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        model.gradient_checkpointing_disable()
    model.get_decoder().train()
    for layer in model.modules():
        if isinstance(layer, torch.nn.Dropout):
            layer.eval()


def hidden(model, rows, *, pad_id):
    """Concatenate response-aligned states; singleton path matches the original engine."""
    import torch

    sequences = [r["prompt_ids"] + r["response_ids"][:-1] for r in rows]
    device = model.get_input_embeddings().weight.device
    if len(rows) == 1:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model.get_decoder()(
                input_ids=torch.tensor([sequences[0]], device=device), use_cache=False
            ).last_hidden_state
        return output[0, len(rows[0]["prompt_ids"]) - 1 :]
    width = max(map(len, sequences))
    ids = torch.tensor([[pad_id] * (width - len(s)) + s for s in sequences], device=device)
    mask = torch.tensor([[0] * (width - len(s)) + [1] * len(s) for s in sequences], device=device)
    positions = (mask.cumsum(-1) - 1).masked_fill(mask == 0, 0)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = model.get_decoder()(
            input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False
        ).last_hidden_state
    return torch.cat(
        [
            output[i, width - len(s) + len(r["prompt_ids"]) - 1 :]
            for i, (s, r) in enumerate(zip(sequences, rows, strict=True))
        ]
    )
