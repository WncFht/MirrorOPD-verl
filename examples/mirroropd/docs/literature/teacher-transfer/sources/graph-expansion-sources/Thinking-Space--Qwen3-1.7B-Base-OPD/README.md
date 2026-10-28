---
license: other
language:
- en
- zh
datasets:
- BytedTsinghua-SIA/DAPO-Math-17k
library_name: transformers
pipeline_tag: text-generation
tags:
- qwen
- qwen3
- math
- dapo
- opd
- on-policy-distillation
- knowledge-distillation
- reasoning
- safetensors
- arxiv:2604.13016
base_model: Qwen/Qwen3-1.7B-Base
base_model_relation: finetune
---

<h1 align="center">Qwen3-1.7B-Base-OPD</h1>

<div align="center" style="line-height: 1;">
  <a href="https://arxiv.org/abs/2604.13016" style="margin: 2px;">
    <img alt="Paper" src="https://img.shields.io/badge/paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://github.com/thunlp/OPD" style="margin: 2px;">
    <img alt="Github" src="https://img.shields.io/badge/OPD-000000?style=for-the-badge&logo=github&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://huggingface.co/papers/2604.13016" style="margin: 2px;">
    <img alt="HF Papers" src="https://img.shields.io/badge/HF--Paper-%23FFD14D?style=for-the-badge&logo=huggingface&logoColor=black" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://huggingface.co/lllyx/Qwen3-4B-Base-GRPO" style="margin: 2px;">
    <img alt="Teacher" src="https://img.shields.io/badge/Teacher-Qwen3--4B--Base--GRPO-blue?style=for-the-badge&logo=huggingface&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://x.com/HBX_hbx/status/2044464414829777354" style="margin: 2px;">
    <img alt="Twitter" src="https://img.shields.io/badge/Twitter-%23000000.svg?style=for-the-badge&logo=x&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>

<br>

Qwen3-1.7B-Base-OPD is an on-policy distillation (OPD) checkpoint initialized from [`Qwen3-1.7B-Base`](https://huggingface.co/Qwen/Qwen3-1.7B-Base). It is distilled from the teacher model [`Qwen3-4B-Base-GRPO`](https://huggingface.co/lllyx/Qwen3-4B-Base-GRPO) using the [`DAPO-Math-17k`](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k) dataset, and is intended for mathematical reasoning and problem-solving.

This model is associated with the paper:  
**Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe**  
Paper link: https://arxiv.org/abs/2604.13016

## Model Description

This model is obtained by applying on-policy distillation (OPD) to `Qwen3-1.7B-Base`, with `Qwen3-4B-Base-GRPO` serving as the teacher model. The OPD training uses DAPO math prompts/data and is designed to transfer the teacher's math-focused reasoning behavior into a smaller 1.7B-parameter student model.

### Key characteristics

- **Student/base model**: Qwen3-1.7B-Base
- **Teacher model**: lllyx/Qwen3-4B-Base-GRPO
- **Training data**: DAPO-Math-17k
- **Training stage**: On-Policy Distillation (OPD)
- **Training framework**: verl
- **Rollout engine**: vLLM
- **Primary domain**: Mathematical reasoning
- **Model architecture**: Qwen3ForCausalLM
- **Precision**: bfloat16
- **Context length**: 32768 tokens

## Training Details

### Training configuration

- **Base checkpoint**: `Qwen/Qwen3-1.7B-Base`
- **Teacher checkpoint**: [`lllyx/Qwen3-4B-Base-GRPO`](https://huggingface.co/lllyx/Qwen3-4B-Base-GRPO)
- **Training framework**: verl
- **Training method**: on-policy distillation with GRPO-style rollouts
- **Distillation loss mode**: `k1`
- **Policy-gradient term**: enabled
- **Training dataset**: `DAPO-Math-17k/DAPO-Math.parquet`
- **Primary task domain**: math reasoning
- **Chat template thinking mode**: disabled (`enable_thinking=False`)
- **Model type**: `qwen3`

### Rollout and optimization

- **Rollout engine**: vLLM
- **Responses per prompt**: 4
- **Prompt length**: 1024
- **Response length**: 7168
- **Max rollout model length**: 8193
- **Train batch size**: 64
- **PPO mini-batch size**: 16
- **PPO micro-batch size per GPU**: 1
- **Max PPO token length per GPU**: 8192
- **Actor learning rate**: `1e-6`
- **Total epochs**: 1
- **Save frequency**: every 20 steps

### Runtime setup

- **Distributed backend**: Ray
- **Number of nodes**: 1
- **GPUs per node**: 4
- **Teacher world size**: 4
- **Rollout tensor parallel size**: 1
- **Teacher tensor parallel size**: 1
- **Actor training**: FSDP with parameter and optimizer offload
- **Gradient checkpointing**: enabled
- **Padding removal**: enabled
- **Torch compile for actor**: enabled
- **Reward function**: rule-based math reward from `verl/recipe/r1_ascend/deepscaler.py::compute_score`

### Dataset

- **Training data**: [`BytedTsinghua-SIA/DAPO-Math-17k`](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k)
- **Teacher rollout/source model**: [`lllyx/Qwen3-4B-Base-GRPO`](https://huggingface.co/lllyx/Qwen3-4B-Base-GRPO)
- **Student initialization**: [`Qwen/Qwen3-1.7B-Base`](https://huggingface.co/Qwen/Qwen3-1.7B-Base)

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "lllyx/Qwen3-1.7B-Base-OPD"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
)
```

## Citation

If you use this model, please consider citing the related paper:

```bibtex
@article{li2026rethinking,
  title={Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe},
  author={Li, Yaxuan and Zuo, Yuxin and He, Bingxiang and Zhang, Jinqian and Xiao, Chaojun and Qian, Cheng and Yu, Tianyu and Gao, Huan-ang and Yang, Wenkai and Liu, Zhiyuan and Ding, Ning},
  journal={arXiv preprint arXiv:2604.13016},
  year={2026}
}
```
