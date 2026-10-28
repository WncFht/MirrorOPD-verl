---
license: mit
library_name: transformers
datasets:
- PrimeIntellect/verifiable-coding-problems
- likaixin/TACO-verified
- livecodebench/code_generation_lite
language:
- en
base_model:
- deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
pipeline_tag: text-generation
---

<div align="center">
<span style="font-family: default; font-size: 1.5em;">DeepCoder-1.5B-Preview</span>
<div>
🚀 Democratizing Reinforcement Learning for LLMs (RLLM) 🌟
</div>
</div>
<br>
<div align="center" style="line-height: 1;">
  <a href="https://github.com/agentica-project/rllm" style="margin: 2px;">
    <img alt="Code" src="https://img.shields.io/badge/RLLM-000000?style=for-the-badge&logo=github&logoColor=000&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://pretty-radio-b75.notion.site/DeepCoder-A-Fully-Open-Source-14B-Coder-at-O3-mini-Level-1cf81902c14680b3bee5eb349a512a51" target="_blank" style="margin: 2px;">
    <img alt="Blog" src="https://img.shields.io/badge/Notion-%23000000.svg?style=for-the-badge&logo=notion&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://x.com/Agentica_" style="margin: 2px;">
    <img alt="X.ai" src="https://img.shields.io/badge/Agentica-white?style=for-the-badge&logo=X&logoColor=000&color=000&labelColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://huggingface.co/agentica-org" style="margin: 2px;">
    <img alt="Hugging Face" src="https://img.shields.io/badge/Agentica-fcd022?style=for-the-badge&logo=huggingface&logoColor=000&labelColor" style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>
</div>
</div>

## DeepCoder Overview
DeepCoder-1.5B-Preview is a code reasoning LLM fine-tuned from DeepSeek-R1-Distilled-Qwen-1.5B using distributed reinforcement learning (RL) to scale up to long context lengths.

## Data
Our training dataset consists of approximately 24K unique problem-tests pairs compiled from:
- Taco-Verified
- PrimeIntellect SYNTHETIC-1
- LiveCodeBench v5 (5/1/23-7/31/24)

## Training Recipe

Our training recipe relies on an improved version of GRPO (GRPO+) and iterative context lengthening, introduced in DeepScaleR.

### GRPO+

We enhance the original GRPO algorithm with insights from DAPO to enable more stable training:

- **Offline Difficulty Filtering:** DAPO employs online dynamic sampling, discarding both entirely correct and entirely incorrect samples on the fly. While this helps maintain a more stable effective batch size, it introduces significant runtime overhead due to rejection sampling. Instead, we perform offline difficulty filtering on a subset of coding problems to ensure the training dataset remains within a suitable difficulty range.
- **No Entropy Loss:** We observed that including an entropy loss term often led to instability, with entropy growing exponentially and ultimately collapsing training. To mitigate this, we eliminate the entropy loss entirely.
- **No KL Loss:** Eliminating KL loss prevents the LLM from staying within trust region of the original SFT model. This removal also obviates the need to compute log probabilities for the reference policy, thereby accelerating training.
- **Overlong Filtering** **(from DAPO):** To preserve long-context reasoning, we mask the loss for truncated sequences. This technique enables DeepCoder to generalize to 64K-context inference despite being trained with a 32K context.
- **Clip High (from DAPO):** By increasing the upper bound in GRPO/PPO’s surrogate loss, we encourage more exploration and more stable entropy.

### Iterative Context Lengthening

Our original `Deepscaler-1.5B-Preview` scaled long context training from 8K→16K→24K, achieving 33→38→43% on AIME respectively. Similarly, `Deepcoder-14B-Preview` is trained on 16K→32K, achieving 54→58% on LiveCodeBench (v5). `DeepCoder-14B-Preview` successfully generalizes to longer contexts when evaluated at 64K context, reaching 60.6%. 

DeepCoder generalizes better to long contexts than the base distilled model, due to DAPO's overlong filtering. However, it's longer responses are often truncated when the max length is capped at 16K, which can lower its scores.
| **Model** | **16K** | **32K** | **64K** |
| --- | --- | --- | --- |
| **DeepCoder-14B-Preview** | 45.6 | 57.9 | 60.6 |
| **DeepSeek-R1-Distill-Qwen-14B** | 50.2 | 53.0 | 53.0 |

A more detailed description of the training recipe can be found in our [blog post](https://pretty-radio-b75.notion.site/DeepCoder-A-Fully-Open-Source-14B-Coder-at-O3-mini-Level-1cf81902c14680b3bee5eb349a512a51).

## Evaluation

We evaluate `Deepcoder-1.5B-Preview` on various coding benchmarks, including LiveCodeBench (LCBv5), Codeforces, and HumanEval+. 

| **Model** | LCB (v5)(8/1/24-2/1/25) | Codeforces Rating | Codeforces Percentile | HumanEval+ |
| --- | --- | --- | --- | --- |
| **DeepCoder-1.5B-Preview** | **25.1** | **963** | **28.5** | **73.0** |
| **Deepseek-R1-Distill-Qwen-1.5B** | 16.9 | 615 | 1.9 | 58.3 |

## Serving DeepCoder
Our model can be served using popular high-performance inference systems:
- vLLM
- Hugging Face Text Generation Inference (TGI)
- SGLang
- TensorRT-LLM

All these systems support the OpenAI Chat Completions API format.

## License
This project is released under the MIT License, reflecting our commitment to open and accessible AI development.
We believe in democratizing AI technology by making our work freely available for anyone to use, modify, and build upon.
This permissive license ensures that researchers, developers, and enthusiasts worldwide can leverage and extend our work without restrictions, fostering innovation and collaboration in the AI community.

## Acknowledgement
- Our training experiments are powered by our heavily modified fork of [Verl](https://github.com/agentica-project/verl), an open-source post-training library.
- Notably, we train 1.5B with [verl pipeline](https://github.com/agentica-project/verl-pipeline), an extension of the original verl.
- Our model is trained on top of [`DeepSeek-R1-Distill-Qwen-1.5B`](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B).
- Our work is done as part of  [Berkeley Sky Computing Lab](https://skycomputing.berkeley.edu/) and [Berkeley AI Research](https://bair.berkeley.edu/).

## Citation

```bibtex
@misc{deepcoder2025,
  title={DeepCoder: A Fully Open-Source 14B Coder at O3-mini Level},
  author={Michael Luo, Sijun Tan, Roy Huang, Ameen Patel, Alpay Ariyak, Qingyang Wu, Xiaoxiang Shi, Rachel Xin, Colin Cai, Maurice Weber, Ce Zhang, Li Erran Li, Raluca Ada Popa, Ion Stoica},
  howpublished={\url{https://pretty-radio-b75.notion.site/DeepCoder-A-Fully-Open-Source-14B-Coder-at-O3-mini-Level-1cf81902c14680b3bee5eb349a512a51}},
  note={Notion Blog},
  year={2025}
}
```