---
license: mit
library_name: transformers
pipeline_tag: text-generation
---

# TMLR-Group-HF/GT-Llama-3.2-3B-Instruct

This is the Llama-3.2-3B-Instruct model trained by GRPO Ground Truth method using MATH training set. This model is one of the checkpoints released in conjunction with the paper [Co-rewarding: Stable Self-supervised RL for Eliciting Reasoning in Large Language Models](https://huggingface.co/papers/2508.00410).

**Co-rewarding** is a novel self-supervised reinforcement learning (RL) framework designed to improve training stability by seeking complementary supervision from multiple views, addressing the training collapse issue in single-view self-rewarding methods. The framework is instantiated in two ways: Co-rewarding-I (data-side, using contrastive agreement) and Co-rewarding-II (model-side, using a slowly-updated reference teacher). Intuitively, such instantiations introduce different levels of discrepancy to increase the difficulty of training collapse on trivial reasoning solutions.

For more details on the Co-rewarding framework, training procedures, and other related models and datasets, please refer to the [official GitHub Repository](https://github.com/tmlr-group/Co-rewarding).

## Citation

```bibtex
@article{zhang2025coreward,
  title={Co-rewarding: Stable Self-supervised RL for Eliciting Reasoning in Large Language Models},
  author={Zhang, Zizhuo and Zhu, Jianing and Ge, Xinmu and Zhao, Zihua and Zhou, Zhanke and Li, Xuan and Feng, Xiao and Yao, Jiangchao and Han, Bo},
  journal={arXiv preprint arXiv:2508.00410},
  year={2025}
}
```