---
license: other
language:
- en
- zh
library_name: transformers
pipeline_tag: text-generation
tags:
- qwen
- qwen3
- math
- grpo
- verl
- rl
- reinforcement-learning
- on-policy-distillation
- full-parameter-rl
- reasoning
- safetensors
- arxiv:2604.13016
base_model: Qwen/Qwen3-4B-Base
base_model_relation: finetune
---

<h1 align="center">Qwen3-4B-Base-GRPO</h1>

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
  <a href="https://x.com/HBX_hbx/status/2044464414829777354" style="margin: 2px;">
    <img alt="Twitter" src="https://img.shields.io/badge/Twitter-%23000000.svg?style=for-the-badge&logo=x&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>

<br>

Qwen3-4B-Base-GRPO is a post-RL checkpoint trained with the **verl** framework.
It starts from **Qwen3-4B-Base** and applies GRPO on the **DAPO-Math-17k-Processed** dataset for mathematical reasoning and problem-solving.

This model is associated with the paper:  
**Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe**  
Paper link: https://arxiv.org/abs/2604.13016

## Model Description

This model is obtained by applying GRPO reinforcement learning to `Qwen3-4B-Base` with verl. The training is intended to improve math-focused reasoning performance under the on-policy distillation setting.


### Key characteristics

- **Base model**: Qwen3-4B-Base
- **Training framework**: verl
- **Training stage**: Reinforcement Learning (GRPO)
- **Parameter update**: Full-parameter actor update
- **Primary domain**: Mathematical reasoning
- **Reward model**: Not used (`reward_model.enable: false`)
- **Rollout engine**: vLLM
- **Context length**: 32768 tokens
- **Responses per prompt**: 8

## Training Details

### Training configuration

- **Framework**: verl
- **Algorithm**: `grpo`
- **GRPO outcome weight**: `1.0`
- **Learned reward model**: disabled (`reward_model.enable: false`)
- **Reward source**: custom rule-based math reward function
- **Training dataset**: `DAPO-Math-17k-Processed`
- **Training file**: `datasets/DAPO-Math-17k-Processed/DAPO-Math.parquet`
- **Validation datasets**: `AIME25`, `AMC23`, `AIME24`
- **Prompt length**: `1024`
- **Response length**: `7168`
- **Validation response length**: `31744`
- **Max model length**: `32768`
- **Rollout temperature**: `1.0`
- **Repetition penalty**: `1.0`
- **KL loss**: disabled
- **Format reward**: disabled
- **Loss aggregation**: `token-mean`
- **Learning rate**: `1e-6`
- **PPO mini-batch size**: `64`
- **PPO micro-batch size per GPU**: `1`
- **Tensor parallel size**: `1`
- **Number of GPUs**: `8`
- **Number of epochs**: `1`
- **Save frequency**: every `20` steps
- **Test frequency**: every `20` steps

### Dataset

- **Training dataset**: `DAPO-Math-17k-Processed`
- **Validation datasets**: `AIME25`, `AMC23`, `AIME24`

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "lllyx/Qwen3-4B-Base-GRPO"

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
