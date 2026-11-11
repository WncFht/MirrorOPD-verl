---
base_model:
- deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
license: apache-2.0
pipeline_tag: text-generation
library_name: transformers
---

# Light-R1-7B-DS: SOTA 7B Math Model with only 3K Data

|Model|Trained From|Release Date|AIME24|AIME25|GPQA|
| ---- | ---- | ---- | ---- | ---- | ---- |
|OpenThinker-7B|Qwen2.5-7B-Instruct|25.2.12|31.3|N/A|42.4|
|DeepSeek-R1-Distill-Qwen-7B|Qwen2.5-Math-7B|25.1.20|55.5|39.2|49.1|
| [**Light-R1-7B-DS (ours)** 🤗](https://huggingface.co/qihoo360/Light-R1-7B-DS) |DeepSeek-R1-Distill-Qwen-7B|25.3.12|**59.1**|**44.3**|**49.4**|
| [Light-R1-32B (ours) 🤗](https://huggingface.co/qihoo360/Light-R1-32B) |Qwen2.5-32B-Instruct|25.3.4|76.6|64.6|61.8|

[technical report](https://arxiv.org/abs/2503.10460)

[GitHub page](https://github.com/Qihoo360/Light-R1)


Light-R1-7B-DS is to the best of our knowledge the State-Of-The-Art open-source 7B math model with AIME24 & 25 scores 59.1 & 44.3.
Light-R1-7B-DS also performed well on GPQA *without* any specific training.

Originated from DeepSeek-R1-Distill-Qwen-7B, Light-R1-7B-DS is further trained with only [3K SFT data](https://huggingface.co/datasets/qihoo360/Light-R1-SFTData) as we've open-sourced, demonstrating the strong applicability of the released data.

We are excited to release this model along with the [technical report](https://arxiv.org/abs/2503.10460).

## Usage
Same as DeepSeek-R1-Distill-Qwen-7B.

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