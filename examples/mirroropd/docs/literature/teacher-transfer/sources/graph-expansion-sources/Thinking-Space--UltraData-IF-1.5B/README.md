---
license: other
language:
- en
- zh
library_name: transformers
pipeline_tag: text-generation
tags:
- qwen2
- deepseek
- instruction-following
- on-policy-distillation
- reinforcement-learning
- reasoning
base_model: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
base_model_relation: finetune
datasets:
- openbmb/UltraData-SFT-2605
---
---

<h1 align="center">UltraData-IF-1.5B</h1>

<div align="center" style="line-height: 1;">
  <a href="https://arxiv.org/pdf/2609.04172"><img src="https://img.shields.io/badge/paper-A42C25?style=for-the-badge&amp;logo=arxiv&amp;logoColor=white" alt="Paper"></a>&nbsp;&nbsp;
    <a href="https://github.com/Thinking-Space/One-Shot-OPD"><img src="https://img.shields.io/badge/One--Shot--OPD-000000?style=for-the-badge&amp;logo=github&amp;logoColor=white" alt="GitHub"></a>&nbsp;&nbsp;
    <a href="https://huggingface.co/papers/2609.04172"><img src="https://img.shields.io/badge/HF--Paper-%23FFD14D?style=for-the-badge&amp;logo=huggingface&amp;logoColor=black" alt="Hugging Face Paper"></a>&nbsp;&nbsp;
    <a href="https://huggingface.co/collections/Thinking-Space/one-shot-opd"><img src="https://img.shields.io/badge/Collection-%23FFD14D?style=for-the-badge&amp;logo=huggingface&amp;logoColor=black" alt="Hugging Face Collection"></a>
    <a href="https://x.com/HBX_hbx/status/2095716853196747175"><img src="https://img.shields.io/badge/Twitter-%23000000.svg?style=for-the-badge&logo=x&logoColor=white" alt="X Thread"></a>
</div>

UltraData-IF-1.5B is an **RL-trained instruction-following model** initialized from [DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B) and trained on the instruction-following subset of [openbmb/UltraData-SFT-2605](https://huggingface.co/datasets/openbmb/UltraData-SFT-2605).

This model is associated with the paper:
**Rethinking On-Policy Distillation of Large Language Models II: One Training Example**

## Model Description

UltraData-IF-1.5B serves as the teacher model for the instruction-following experiments in our paper. It is paired with DeepSeek-R1-Distill-Qwen-1.5B to study on-policy distillation in the instruction-following domain. The two models share the same architecture, tokenizer, and chat template, allowing token-level distillation without cross-model-family mismatch.

### Key characteristics

- **Role in the paper**: instruction-following teacher for OPD
- **Training method**: GRPO
- **Base model**: [DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B) （`R1-Distill-1.5B`）
- **Training data**: instruction-following subset of [openbmb/UltraData-SFT-2605](https://huggingface.co/datasets/openbmb/UltraData-SFT-2605)
- **Paired student in OPD**: [DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)
- **Primary domain**: Instruction following
- **Architecture**: `Qwen2ForCausalLM`
- **Parameters**: 1.78B

- **Precision**: bfloat16
- **Vocabulary size**: 151,936
- **Layers / hidden size**: 28 / 1536

- **Attention heads**: 12 query / 2 key-value (GQA)
- **Context length**: 131,072 positions in `config.json`; tokenizer `model_max_length` is 16,384

## Training Details

| Item                         | Value                                                        |
| ---------------------------- | ------------------------------------------------------------ |
| Training algorithm           | GRPO                                                         |
| Training set                 | [UltraData-SFT-2605](https://huggingface.co/datasets/openbmb/UltraData-SFT-2605) |
| Reward                       | Fraction of satisfied instruction constraints                |
| Prompt batch size            | 32                                                           |
| Rollouts per prompt          | 8                                                            |
| Rollouts per step            | 256                                                          |
| Learning rate                | 1e-6                                                         |
| Sampling temperature / top-p | 0.9 / 0.95                                                   |
| Max prompt / response length | 2,048 / 14,336                                               |
| KL regularization            | None                                                         |
| GRPO clipping                | 0.20 / 0.28                                                  |
| Entropy bonus                | 0.0                                                          |
| Loss aggregation             | `token-mean`                                                 |
| Training steps               | 600                                                          |
| Training framework           | [veRL](https://github.com/volcengine/verl)                   |
| Hardware                     | 8 GPUs                                                       |

## Evaluation

Scores for this checkpoint as the instruction-following teacher, alongside the student it is distilled into. Multi-IF is averaged over its eight languages; IFBench follows the standard prompt-level and instruction-level, strict and loose breakdown.

| Benchmark | DeepSeek-R1-Distill-Qwen-1.5B (Student) | **UltraData-IF-1.5B (teacher)** |
|---|---|---|
| Multi-IF, 8-lang, final turn | 20.84 | **28.58** |
| Multi-IF, 8-lang, 3-turn mean | 28.56 | **40.39** |
| Multi-IF, English, final turn | 24.71 | **35.54** |
| Multi-IF, English, 3-turn mean | 30.51 | **45.05** |
| IFBench, prompt-level strict | 10.33 | **17.00** |
| IFBench, prompt-level loose | 15.00 | **22.00** |
| IFBench, instruction-level strict | 12.79 | **18.90** |
| IFBench, instruction-level loose | 19.19 | **24.71** |

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Thinking-Space/UltraData-IF-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
)

messages = [{"role": "user", "content": "Write a haiku about distillation. Use exactly three lines."}]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
).to(model.device)

outputs = model.generate(inputs, max_new_tokens=2048)
print(tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True))
```

## Citation

If you use this model, please consider citing:

```bibtex
@article{fu2026rethinking,
  title={Rethinking on-policy distillation of large language models ii: One training example},
  author={Fu, Zixuan and He, Bingxiang and Zuo, Yuxin and Huang, Haohuan and Zhang, Jinqian and Xiao, Ruhang and Qian, Cheng and Luo, Qinyu and Gao, Huan-ang and Wang, Yudong and others},
  journal={arXiv preprint arXiv:2609.04172},
  year={2026}
}
```

