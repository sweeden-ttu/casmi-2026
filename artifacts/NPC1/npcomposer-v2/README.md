---
library_name: transformers
license: apache-2.0
base_model: ibm-research/GP-MoLFormer-Uniq
tags:
- generated_from_trainer
model-index:
- name: NPComposer-v2
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

[<img src="https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-28.svg" alt="Visualize in Weights & Biases" width="200" height="32"/>](https://wandb.ai/npcomposer/npcomposer/runs/lpvwn86m)
# NPComposer

NPComposer is a conditional molecular generation model trained by fine-tuning GP-MoLFormer [ibm-research/GP-MoLFormer-Uniq](https://huggingface.co/ibm-research/GP-MoLFormer-Uniq) — a 46.8M parameter transformer decoder foundation model — on the COCONUT database containing over 700,000 experimentally validated natural products:
https://github.com/shawnralyn/NPComposer

By providing class labels and molecular property information as special tokens during model fine-tuning, NPComposer allows for conditional natural product generation based on: NP biosynthesis pathway (7 pathways including Alkaloids, Terpenoids, Shikimates and Phenylpropanoids, etc.), NP superclass (70+ superclasses), presence or absence of glycoside, number of aromatic rings (0–22), QED drug-likeness (0–1), and synthetic accessibility score (1–10).

## Model description

[More information needed](https://github.com/shawnralyn/NPComposer)

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 16
- eval_batch_size: 16
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- lr_scheduler_warmup_ratio: 0.03
- num_epochs: 8.0

### Training results

| Training Loss | Epoch | Step   | Validation Loss |
|:-------------:|:-----:|:------:|:---------------:|
| 0.4069        | 1.0   | 27763  | 0.3820          |
| 0.3507        | 2.0   | 55526  | 0.3371          |
| 0.3247        | 3.0   | 83289  | 0.3142          |
| 0.3074        | 4.0   | 111052 | 0.2995          |
| 0.291         | 5.0   | 138815 | 0.2888          |
| 0.2765        | 6.0   | 166578 | 0.2803          |
| 0.2677        | 7.0   | 194341 | 0.2750          |
| 0.2569        | 8.0   | 222104 | 0.2718          |


### Framework versions

- Transformers 4.44.2
- Pytorch 2.10.0+cu128
- Datasets 2.21.0
- Tokenizers 0.19.1
  
### Hardware
1 x NVIDIA A40 48GB GPU
