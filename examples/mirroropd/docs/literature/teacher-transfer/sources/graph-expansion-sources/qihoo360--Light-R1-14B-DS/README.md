---
base_model:
- deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
license: apache-2.0
pipeline_tag: text-generation
library_name: transformers
---

# Light-R1-14B-DS: SOTA 14B Math Model with RL

|Model|Trained From|Release Date|AIME24|AIME25|GPQA|
| ---- | ---- | ---- | ---- | ---- |---- |
|OpenThinker-32B|Qwen2.5-32B-Instruct|25.2.12|66.0|50.9|61.6|
|DeepSeek-R1-Distill-Qwen-14B|Qwen2.5-14B|25.1.20|69.7|50.2|59.1|
| [**Light-R1-14B-DS (ours)** 🤗](https://huggingface.co/qihoo360/Light-R1-14B-DS) |DeepSeek-R1-Distill-Qwen-14B|25.3.12|**74.0**|**60.2**|**61.7**|
| [Light-R1-32B (ours) 🤗](https://huggingface.co/qihoo360/Light-R1-32B) |Qwen2.5-32B-Instruct|25.3.4|76.6|64.6|61.8|

<p align="center">
  <img width="80%" src="14b-rl.png">
</p>

[technical report](https://arxiv.org/abs/2503.10460)

[GitHub page](https://github.com/Qihoo360/Light-R1)

[wandb log](https://api.wandb.ai/links/seek4-nus/4klmwpqs)

We introduce Light-R1-14B-DS, the first open-source successful RL attempt on already long-COT finetuned models of simialr sizes under **light** budget.
Light-R1-14B-DS is also the State-Of-The-Art 14B math model with AIME24 & 25 scores 74.0 & 60.2, outperforming many 32B models.

Recent RL works have successfully trained RL on base models (usually with *-zero* in their names), or on 1.5B models (with response length interestingly decreasing then increasing), or on QwQ-32B with presumably prohibitively **heavy** compute.

Light-R1-14B-DS marks one step further in reproducing and democratizing DeepSeek-R1.
We have finally seen expected behavior during RL training: simultaneous increase in response length and reward score *on an already long-COT finetuned model* (see [wandb log](https://api.wandb.ai/links/seek4-nus/4klmwpqs)).

Originated from DeepSeek-R1-Distill-Qwen-14B, Light-R1-14B-DS underwent our long-COT RL Post-Training and achieved a new State-Of-The-Art across 14B-Math models: 74.0 & 60.2 on AIME 24 & 25 respectively.
Light-R1-14B-DS also performed well on GPQA *without* any specific training.
We are excited to release this model along with the [technical report](https://arxiv.org/abs/2503.10460), and will continue to perfect our long-COT RL Post-Training.

## Usage
Same as DeepSeek-R1-Distill-Qwen-14B.

## Data Decontamination

We carefully evaluated data contamination of several open-sourced datasets.
While certain contamination may be [inevitable during pre-training](https://x.com/DimitrisPapail/status/1888325914603516214),
it is unacceptable for post-training to compare on benchmarks.
MATH-500 is somewhat compromised with tens of questions that are identical or only numbers changed. AIME 24 and 25 stay intact but we have to pay special attention when we incorporate AIME data up to 2023.

Light-R1 did thorough decontamination with exact matching (excluding digits) and N-gram (N=32) matching.

## Citation
```latex
@misc{lightr1proj,
      title={Light-R1: Curriculum SFT, DPO and RL for Long COT from Scratch and Beyond}, 
      author={Liang Wen, Yunke Cai, Fenrui Xiao, Xin He, Qi An, Zhenyu Duan, Yimin Du, Junchen Liu, Lifu Tang, Xiaowei Lv, Haosheng Zou, Yongchao Deng, Shousheng Jia, Xiangzheng Zhang},
      year={2025},
      eprint={},
      archivePrefix={},
      url={https://github.com/Qihoo360/Light-R1}, 
}
```