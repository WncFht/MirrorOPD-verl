---
base_model:
- deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
datasets:
- BytedTsinghua-SIA/DAPO-Math-17k
language:
- en
library_name: transformers
license: apache-2.0
pipeline_tag: text-generation
---

<div align="center">
<span style="font-family: default; font-size: 1.5em;">JustRL: Simplicity at Scale</span>
<div>
🚀 Competitive RL Performance Without Complex Techniques 🌟
</div>
</div>


<br>

<div align="center" style="line-height: 1;">
  <a href="https://github.com/thunlp/JustRL" style="margin: 2px;">
    <img alt="Code" src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://huggingface.co/collections/hbx/justrl" style="margin: 2px;">
    <img alt="Hugging Face" src="https://img.shields.io/badge/JustRL-fcd022?style=for-the-badge&logo=huggingface&logoColor=000&labelColor" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://relieved-cafe-fe1.notion.site/JustRL-Scaling-a-1-5B-LLM-with-a-Simple-RL-Recipe-24f6198b0b6b80e48e74f519bfdaf0a8" target="_blank" style="margin: 2px;">
    <img alt="Notion" src="https://img.shields.io/badge/Notion-%23000000.svg?style=for-the-badge&logo=notion&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://arxiv.org/abs/2512.16649" style="margin: 2px;">
    <img alt="Paper" src="https://img.shields.io/badge/Paper-2512.16649-b31b1b.svg" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://x.com/HBX_hbx/status/1988474153436090776" style="margin: 2px;">
    <img alt="Paper" src="https://img.shields.io/badge/X-000000?style=for-the-badge&logo=x&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>


</div>
</div>

## Overview

**JustRL** demonstrates that competitive reinforcement learning performance for small language models doesn't require complex multi-stage pipelines or dynamic schedules. Using a minimal recipe with single-stage training and fixed hyperparameters, we achieve state-of-the-art results on mathematical reasoning tasks.

We release two models:
- [**JustRL-DeepSeek-1.5B**](https://huggingface.co/hbx/JustRL-DeepSeek-1.5B): Trained from DeepSeek-R1-Distill-Qwen-1.5B
- [**JustRL-Nemotron-1.5B**](https://huggingface.co/hbx/JustRL-Nemotron-1.5B): Trained from OpenMath-Nemotron-1.5B

Both models use identical hyperparameters without per-model tuning, demonstrating the robustness of our approach.

![The AIME24 performance curve for scaling from a weak base DeekSeek-R1-Distill-Qwen-1.5B and a strong base OpenMath-Nemotron-1.5B over thousands of steps.](./assets/fig1_aime24_curves_added.png)

## Key Highlights

✨ **Simplicity**: Single-stage training with fixed hyperparameters, without multi-stage pipelines or dynamic schedules

📈 **Stability**: Smooth, monotonic improvement over 4,000+ training steps without collapses or oscillations

🎯 **Performance**: State-of-the-art results at 1.5B scale, matching or exceeding more complex approaches

💰 **Efficiency**: Comparable or better performance with 2× less compute than multi-stage methods

🔓 **Open**: Complete evaluation scripts, and model weights released

## Performance

### JustRL-DeepSeek-1.5B (Based on DeepSeek-R1-Distill-Qwen-1.5B)

| Model                    | AIME24 (@32) | AIME25 (@32) | AMC23 (@32) | MATH-500 (@4) | Minerva (@4) | OlympiadBench (@4) | HMMT25 (@32) | BRUMO25 (@32) | CMIMC25 (@32) | Avg       |
| ------------------------ | ------------ | ------------ | ----------- | ------------- | ------------ | ------------------ | ------------ | ------------- | ------------- | --------- |
| DeepSeek-R1-Distill-1.5B | 29.90        | 22.40        | 63.82       | 84.90         | 34.65        | 45.95              | 13.44        | 30.94         | 12.89         | 37.65     |
| DeepScaleR-1.5B-Preview  | 40.21        | 28.65        | 73.83       | 89.30         | 39.34        | 52.79              | 18.96        | 40.00         | 21.00         | 44.88     |
| ProRL-V2                 | 51.87        | 35.73        | 88.75       | 92.00         | 49.03        | 67.84              | 19.38        | 47.29         | **25.86**     | 53.08     |
| BroRL                    | **57.50**    | 36.88        | /           | **92.14**     | 49.08        | 61.54              | /            | /             | /             | /         |
| JustRL-DeepSeek-1.5B     | 52.60        | **38.75**    | **91.02**   | 91.65         | **51.47**    | **67.99**          | **21.98**    | **52.71**     | 25.63         | **54.87** |

Besides, the real question is whether our simplicity comes at a computational cost. It doesn't. We match half of ProRL-V2's compute budget while using a single-stage recipe with fixed hyperparameters. BroRL requires 4.9× more compute by increasing rollouts to 512 per example, essentially exhaustively exploring the solution space. Our approach achieves competitive performance without this computational overhead.

### JustRL-Nemotron-1.5B (Based on OpenMath-Nemotron-1.5B)

| Model                  | AIME24 (@32) | AIME25 (@32) | AMC23 (@32) | MATH-500 (@4) | Minerva (@4) | OlympiadBench (@4) | HMMT25 (@32) | BRUMO25 (@32) | CMIMC25 (@32) | Avg       |
| ---------------------- | ------------ | ------------ | ----------- | ------------- | ------------ | ------------------ | ------------ | ------------- | ------------- | --------- |
| OpenMath-Nemotron-1.5B | 58.75        | 48.44        | 90.55       | 92.40         | 26.93        | 71.70              | 30.10        | 61.67         | 30.08         | 56.74     |
| QUESTA-Nemotron-1.5B   | **71.56**    | 62.08        | 93.44       | 92.95         | **32.08**    | 72.28              | **40.94**    | **67.50**     | 41.48         | 63.81     |
| JustRL-Nemotron-1.5B   | 69.69        | **62.92**    | **96.02**   | **94.15**     | 30.24        | **76.59**          | 40.63        | 66.88         | **41.72**     | **64.32** |

We achieve 64.32% average, slightly outperforming QuestA's 63.81% and leading on five of nine benchmarks. The gap is narrow, which makes sense—both approaches are pushing the boundaries of what's achievable at 1.5B scale. The key difference is in how we get there. We use 2× less compute while achieving slightly better average performance without designing a complex curriculum as used in QuestA.

## Training Recipe

Our approach is deliberately minimal:

**Core Algorithm**: Standard GRPO with binary outcome rewards

- **Reward**: Simple DAPO verifier (string-matching, no SymPy)
- **Training**: Single-stage, no curriculum or stage transitions
- **Hyperparameters**: Fixed throughout (no adaptive schedules)
- **Data**: DAPO-Math-17k without filtering or dynamic sampling
- **Length Control**: 16K context cap (no explicit penalties)
- **Stabilization**: Only "clip higher" for gradient stability

Detail hyperparameters and comparisons on training techniques with other methods can refer to our [paper](https://arxiv.org/abs/2512.16649).

## Training Data

We train on [DAPO-Math-17k](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k), a curated dataset of mathematical problems. **No offline difficulty filtering or online dynamic sampling is used.**

## Usage

### Basic Inference
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "hbx/JustRL-Nemotron-1.5B"  # or JustRL-DeepSeek-1.5B
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = """<problem>

Please reason step by step, and put your final answer within \\boxed{}."""

messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer([text], return_tensors="pt").to(model.device)
outputs = model.generate(
    **inputs,
    max_new_tokens=16384,
    temperature=0.7,
    top_p=0.9,
    do_sample=True
)

response = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
print(response)
```

### Batch Inference with vLLM
```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="hbx/JustRL-Nemotron-1.5B",
    tensor_parallel_size=1,
    max_model_len=32768
)

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=16384,
)

problems = [...]  # Your list of problems
responses = llm.generate(problems, sampling_params)
```

## Reproduction

We provide evaluation scripts based on [POLARIS](https://github.com/ChenxinAn-fdu/POLARIS), the evaluation script is [here](https://github.com/thunlp/JustRL)


```bibtex
@article{he2025justrl,
  title={JustRL: Scaling a 1.5 B LLM with a Simple RL Recipe},
  author={He, Bingxiang and Qu, Zekai and Liu, Zeyuan and Chen, Yinghao and Zuo, Yuxin and Qian, Cheng and Zhang, Kaiyan and Chen, Weize and Xiao, Chaojun and Cui, Ganqu and others},
  journal={arXiv preprint arXiv:2512.16649},
  year={2025}
}
```