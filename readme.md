# TKPO - Token-level Preference Self-Alignment Optimization for Multi-style Outline Controllable Generation

## Introduction

### TKPO adopts token-level preference self-alignment optimization for multi-style (concise vs. comprehensive; objective vs. literature) outline generation, as depicted in the toy example below.

<h1 align="center">Mulit-Style Outline Generation</h1>
<p align="center">
    <img src="https://github.com/Yuhanleeee/TKPO/blob/main/assets/toy_example.jpg" width="50%" style="max-width: 300px;">
</p>

Specifically, we extend the Bradley-Terry model from pair-wise to list-wise comparison, which is further applied at the token level for fine-grained preference signal utilization. In comparison to the representative methods, such as DPO, TKPO does not require response pairs; instead, we propose a controllable attributes-driven method to construct reject samples for self-alignment. 
Experiments demonstrate that TKPO outperforms DPO by up to 19.28% in performance while requiring only 56.25% in training time.

<h1 align="center">DPO vs. TKPO</h1>
<p align="center">
    <img src="https://github.com/Yuhanleeee/TKPO/blob/main/assets/framework.png" width="90%" style="max-width: 300px;">
</p>

Check out our papers to learn more: [Token-level Preference Self-Alignment Optimization for Multi-style Outline Controllable Generation]()


## Requirements

- python 3.10.11
- pytorch 2.0.1
- transformers 4.43.2
- deepspeed 0.14.4
- llamafactory 0.8.4.dev0
- cuda 11.7

## Data

We curate two datasets (level-of-detail and language style) in our paper for outline controllable generation, all of which are already included in the `data` directory of this repo. 

## Quick Start

All the experiments are conducted on 8 $\times$ Tesla V100-SXM2 32GB with Qwen2.5. Please kindly reload the `transformers/models/qwen2/modeling_qwen2.py` with our provided `modeling_qwen2.py`, run `llamafactory-cli train qwen_sft_tkpo.yaml` for SFT.


## Citation

If you find our code, data, models, or the paper useful, please cite the paper:
```

```

## Acknowledgements

This work benefits from [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) and [Qwen2.5](https://huggingface.co/spaces/Qwen/Qwen2.5). Thanks for their significant contributions to the community.

