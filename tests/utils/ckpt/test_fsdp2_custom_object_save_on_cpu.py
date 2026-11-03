# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""A portable custom-model export must contain model code, not FSDP internals."""

import importlib.util
import sys

from transformers import AutoConfig, AutoModelForCausalLM, Qwen2Config

from verl.utils.checkpoint.fsdp_checkpoint_manager import _save_custom_model_code
from verl.utils.fsdp_utils import FSDPModule


def test_fsdp2_export_bundles_original_model_source(tmp_path):
    module_path = tmp_path / "modeling_custom.py"
    module_path.write_text(
        "from transformers import Qwen2ForCausalLM\n"
        "class CustomLM(Qwen2ForCausalLM):\n"
        "    _auto_class = 'AutoModelForCausalLM'\n"
    )
    spec = importlib.util.spec_from_file_location("modeling_custom", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        original = module.CustomLM
        wrapped = type("FSDPCustomLM", (FSDPModule, original), {"__module__": FSDPModule.__module__})
        model = object.__new__(wrapped)
        config = Qwen2Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
        )
        config.auto_map = {"AutoModelForCausalLM": "modeling_custom.CustomLM"}
        destination = tmp_path / "export"
        destination.mkdir()
        _save_custom_model_code(model, destination, config)
        config.save_pretrained(destination)
        assert (destination / "modeling_custom.py").read_bytes() == module_path.read_bytes()
        assert sorted(p.name for p in destination.glob("*.py")) == ["modeling_custom.py"]
        assert AutoConfig.from_pretrained(destination).auto_map == config.auto_map
        reloaded = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(destination), trust_remote_code=True)
        assert type(reloaded).__name__ == "CustomLM"
        assert type(model) is wrapped
    finally:
        del sys.modules[spec.name]
