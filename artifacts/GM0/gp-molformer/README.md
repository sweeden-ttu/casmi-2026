---
license: apache-2.0
library_name: transformers
pipeline_tag: text-generation
tags:
- chemistry
---

# GP-MoLFormer-Uniq

GP-MoLFormer is a class of models pretrained on SMILES string representations of 0.65-1.1B molecules from ZINC and PubChem.
This repository is for the model pretrained on all the _unique_ molecules from both datasets.

It was introduced in the paper [GP-MoLFormer: A Foundation Model For Molecular Generation](https://arxiv.org/abs/2405.04912) by Ross et al. and released in [this repository](https://github.com/IBM/gp-molformer).

## Model Details

### Model Description

GP-MoLFormer is a large-scale autoregressive chemical language model intended for molecule generation tasks. GP-MoLFormer employs the same architecture as MoLFormer-XL, including linear attention and rotary position embeddings, but uses decoder-only Transformer blocks trained with a causal language modeling objective. It is trained on up to 1.1B molecules in SMILES representation.

GP-MoLFormer was evaluated on _de novo_ generation (*at scale*), scaffold-constrained decoration, and molecular property optimization tasks.

## Intended use and limitations

The pretrained model may be used out-of-the-box for unconditional, _de novo_ molecule generation. It can also be prompted with a partial SMILES string to do scaffold completion/decoration. We also demonstrate it can be fine-tuned on a particular dataset to change the output distribution (e.g., more druglike) or tuned for molecular optimization using **pair-tuning**. For details, see the paper and GitHub repository.

This model is not tested for classification performance. It is also not tested for molecules larger than ~200 atoms (i.e., macromolecules). Furthermore, using invalid or noncanonical SMILES may result in worse performance.

## Example code

Use the code below to get started with the model.

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("ibm-research/GP-MoLFormer-Uniq", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("ibm-research/MoLFormer-XL-both-10pct", trust_remote_code=True)

outputs = model.generate(do_sample=True, top_k=None, max_length=202, num_return_sequences=3)
tokenizer.batch_decode(outputs, skip_special_tokens=True)
```

## Training Details

### Data

We trained GP-MoLFormer on a combination of molecules from the ZINC15 and PubChem datasets. This repository contains the version trained on all _unique_ molecules from both datasets.

Molecules were canonicalized with RDKit prior to training and isomeric information was removed. Also, molecules longer than 202 tokens were dropped.

### Hardware

- 16 x NVIDIA A100 80GB GPUs

## Evaluation

We evaluated GP-MoLFormer on various generation metrics. The tables below show the performance of GP-MoLFormer-Uniq compared to baseline models:

|                   | Val&#8593; | Uniq@10k&#8593; | Nov&#8593; | Frag&#8593; | Scaf&#8593; | SNN&#8593; | IntDiv&#8593; | FCD&#8595; |
|-------------------|------------|-----------------|------------|-------------|-------------|------------|---------------|------------|
| CharRNN           | 0.975      | 0.999           | 0.842      | **0.9998**  | 0.9242      | 0.6015     | 0.8562        | 0.0732     |
| VAE               | 0.977      | 0.998           | 0.695      | 0.9984      | **0.9386**  | **0.6257** | 0.8558        | 0.0990     |
| JT-VAE            | **1.000**  | **1.000**       | 0.914      | 0.9965      | 0.8964      | 0.5477     | 0.8551        | 0.3954     |
| LIMO              | **1.000**  | 0.976           | **1.000**  | 0.6989      | 0.0079      | 0.2464     | **0.9039**    | 26.78      |
| MolGen-7B         | **1.000**  | **1.000**       | 0.934      | **0.9999**  | 0.6538      | 0.5138     | 0.8617        | **0.0435** |
| GP-MoLFormer-Uniq | **1.000**  | 0.977           | 0.390      | **0.9998**  | 0.7383      | 0.5045     | 0.8655        | 0.0591     |

We report all metrics using the typical MOSES definitions on each model's respective test set. Note: novelty is with respect to each model's respective training set.

## Citation

```
@misc{ross2025gpmolformerfoundationmodelmolecular,
      title={GP-MoLFormer: A Foundation Model For Molecular Generation}, 
      author={Jerret Ross and Brian Belgodere and Samuel C. Hoffman and Vijil Chenthamarakshan and Jiri Navratil and Youssef Mroueh and Payel Das},
      year={2025},
      eprint={2405.04912},
      archivePrefix={arXiv},
      primaryClass={q-bio.BM},
      url={https://arxiv.org/abs/2405.04912}, 
}
```