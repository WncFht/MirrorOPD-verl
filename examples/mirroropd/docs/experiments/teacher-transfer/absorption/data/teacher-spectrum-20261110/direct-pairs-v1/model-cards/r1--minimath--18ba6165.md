---
tags:
- math
license: apache-2.0
datasets:
- oumi-ai/MetaMathQA-R1
language:
- en
metrics:
- accuracy
base_model:
- deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
pipeline_tag: text-generation
---

[![oumi logo](https://oumi.ai/logo_lockup_black.svg)](https://github.com/oumi-ai/oumi)
[![Made with Oumi](https://badgen.net/badge/Made%20with/Oumi/%23085CFF?icon=https%3A%2F%2Foumi.ai%2Flogo_dark.svg)](https://github.com/oumi-ai/oumi)

[![Documentation](https://img.shields.io/badge/Documentation-oumi-blue.svg)](https://oumi.ai/docs/en/latest/index.html)
[![Blog](https://img.shields.io/badge/Blog-oumi-blue.svg)](https://oumi.ai/blog)
[![Discord](https://img.shields.io/discord/1286348126797430814?label=Discord)](https://discord.gg/oumi)

# oumi-ai/MiniMath-R1-1.5B

<!-- Provide a quick summary of what the model is/does. -->
Supervised fine-tune of [DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B) using [oumi-ai/MetaMathQA-R1](https://huggingface.co/datasets/oumi-ai/MetaMathQA-R1).

Achieves **44.4% accuracy on MMLU-Pro-Math**, the **highest of any model with <=1.5B parameters**.

Improves the base model's accuracy by **+6 points**.


- **Developed by:** [Oumi AI](https://oumi.ai/)
- **Model type:** Small Language Model
- **Language(s) (NLP):** English
- **License:** [Apache 2.0](https://opensource.org/license/apache-2-0)
- **Finetuned from model:** [DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)
- **Demo:** [Fine-Tuning Notebook](https://github.com/oumi-ai/oumi/blob/307436bd98706cb9ce7b0bbf31204770af2b7c8c/notebooks/Oumi%20-%20MiniMath-R1-1.5B.ipynb)

## Uses

<!-- Address questions around how the model is intended to be used, including the foreseeable users of the model and those affected by the model. -->
Use as a conversational assistant for solving math problems with an exposed thought process.

## Out-of-Scope Use

<!-- This section addresses misuse, malicious use, and uses that the model will not work well for. -->
Smaller LLMs have limited capabilities and should be used with caution. Avoid using this model for purposes outside of mathematics.

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->
This model was finetuned with DeepSeek-R1 data on top of an R1-distill model, so any biases or risks associated with those models may be present.

## Training Details

### Training Data

<!-- This should link to a Dataset Card, perhaps with a short stub of information on what the training data is all about as well as documentation related to data pre-processing or additional filtering. -->
Training data: [oumi-ai/MetaMathQA-R1](https://huggingface.co/datasets/oumi-ai/MetaMathQA-R1)

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->
Training notebook: [Fine-Tuning Notebook](https://github.com/oumi-ai/oumi/blob/307436bd98706cb9ce7b0bbf31204770af2b7c8c/notebooks/Oumi%20-%20MiniMath-R1-1.5B.ipynb)

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->


## Environmental Impact

<!-- Total emissions (in grams of CO2eq) and additional considerations, such as electricity usage, go here. Edit the suggested text below accordingly -->

- **Hardware Type:** H100
- **Hours used:** 0.8 (0.1 * 8 GPUs)
- **Cloud Provider:** Google Cloud Platform
- **Compute Region:** us-east5
- **Carbon Emitted:** 0.07 kg

## Citation

<!-- If there is a paper or blog post introducing the model, the APA and Bibtex information for that should go in this section. -->

```
@misc{miniMathR1_2025,
  author = {Jeremiah Greer},
  title = {MiniMath-R1-1.5B},
  month = {February},
  year = {2025},
  url = {https://huggingface.co/oumi-ai/MiniMath-R1-1.5B}
}

@software{oumi2025,
  author = {Oumi Community},
  title = {Oumi: an Open, End-to-end Platform for Building Large Foundation Models},
  month = {January},
  year = {2025},
  url = {https://github.com/oumi-ai/oumi}
}
```