---
license: other
language:
- en
- zh
datasets:
- lllyx/OpenThought3-Qwen3-4B
pipeline_tag: text-generation
tags:
- qwen
- qwen3
- math
- sft
- supervised-fine-tuning
- full-finetuning
- reasoning
base_model: Qwen/Qwen3-1.7B-Base
---

<h1 align="center">Qwen3-1.7B-SFT</h1>

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
  <a href="https://huggingface.co/datasets/lllyx/OpenThought3-Qwen3-4B" style="margin: 2px;">
    <img alt="Dataset" src="https://img.shields.io/badge/Dataset-OpenThought3--Qwen3--4B-blue?style=for-the-badge&logo=huggingface&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://x.com/HBX_hbx/status/2044464414829777354" style="margin: 2px;">
    <img alt="Twitter" src="https://img.shields.io/badge/Twitter-%23000000.svg?style=for-the-badge&logo=x&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>

<br>

Qwen3-1.7B-SFT is a supervised fine-tuned model based on **Qwen3-1.7B-Base**, trained on the **OpenThought3-Qwen3-4B** dataset for mathematical reasoning and problem-solving.

This model is associated with the paper:  
**Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe**  
Paper link: https://arxiv.org/abs/2604.13016

## Model Description

This model is obtained by full-parameter supervised fine-tuning (SFT) from `Qwen3-1.7B-Base`.  
The training is designed to improve the model's performance on math-focused instruction-following and reasoning tasks.
This model is intended in **Section 5.1. Off-Policy Distillation from Teacher Rollouts as Cold Start**.

### Key characteristics

- **Base model**: Qwen3-1.7B-Base
- **Training stage**: Supervised Fine-Tuning (SFT)
- **Finetuning type**: Full finetuning
- **Primary domain**: Mathematical reasoning
- **Thinking mode**: Disabled during training (`enable_thinking: false`)
- **Context length**: 20480 tokens

## Training Details

### Training configuration

- **Framework**: LLaMA-Factory
- **Stage**: `sft`
- **Finetuning type**: `full`
- **DeepSpeed config**: `ds_z2_config.json`
- **Kernel optimization**: `enable_liger_kernel: True`
- **Precision**: `bf16`
- **Gradient checkpointing**: enabled
- **Learning rate**: `1e-5`
- **Scheduler**: cosine
- **Warmup ratio**: `0.1`
- **Number of epochs**: `2.0`
- **Per-device train batch size**: `16`
- **Gradient accumulation steps**: `1`
- **Validation split**: `0.01`
- **Evaluation strategy**: every `100` steps
- **Save strategy**: every `100` steps

### Dataset

- **Training dataset**: [`lllyx/OpenThought3-Qwen3-4B`](https://huggingface.co/datasets/lllyx/OpenThought3-Qwen3-4B)
- **Data source**: math-domain prompts selected from [`open-thoughts/OpenThoughts3-1.2M`](https://huggingface.co/datasets/open-thoughts/OpenThoughts3-1.2M), answered by **Qwen3-4B (Non-thinking)**.
- **Cleaning**: deduplication, removal of degenerate repetition/repeater-style outputs, and template checks on assistant responses.

## Training Hyperparameters

For reproducibility, the core configuration is summarized below:

```
model_name_or_path: ../model/Qwen3-1.7B-Base
trust_remote_code: true

stage: sft
do_train: true
finetuning_type: full
deepspeed: examples/deepspeed/ds_z2_config.json
enable_liger_kernel: true

dataset: openthought3_qwen3_4b
template: qwen3
enable_thinking: false
cutoff_len: 20480
preprocessing_num_workers: 64
dataloader_num_workers: 32

output_dir: ../model/Qwen3-1.7B-Base-SFT-OpenThought3-Qwen3-4B
logging_steps: 5
save_steps: 100
plot_loss: true
overwrite_output_dir: true
save_only_model: true
report_to: swanlab

per_device_train_batch_size: 16
gradient_accumulation_steps: 1
gradient_checkpointing: true
learning_rate: 1.0e-5
num_train_epochs: 2.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000

val_size: 0.01
per_device_eval_batch_size: 1
eval_strategy: steps
eval_steps: 100
```

## Citation

If you use this model, please consider citing the related paper:
```
@article{li2026rethinking,
  title={Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe},
  author={Li, Yaxuan and Zuo, Yuxin and He, Bingxiang and Zhang, Jinqian and Xiao, Chaojun and Qian, Cheng and Yu, Tianyu and Gao, Huan-ang and Yang, Wenkai and Liu, Zhiyuan and Ding, Ning},
  journal={arXiv preprint arXiv:2604.13016},
  year={2026}
}
```
Paper: https://arxiv.org/abs/2604.13016
