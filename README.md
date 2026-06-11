# PATCH: Prompt Assortment for Traditional Chinese Hazards

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Dataset on HuggingFace](https://img.shields.io/badge/HuggingFace-Dataset-orange)](TODO_HF_URL)

PATCH is the first large-scale adversarial dataset for Traditional Chinese (TC) LLM safety evaluation. It contains over 820,000 prompts aligned with 13 MLCommons hazard categories, combining synthetic data generation (PATCH-GPT and Rainbow Teaming) with 390 human-annotated prompts (PATCH-H) authored by native TC speakers (Fleiss' kappa = 0.84). PATCH is used to train and evaluate lightweight safety classifiers -- Llama Guard 3 1B, RoBERTa, and Longformer -- via full fine-tuning (FFT) and LoRA, achieving F1 > 0.99 on synthetic data and F1 = 0.87 on human-authored adversarial prompts.

## Dataset Overview

| Subset | Safe | Unsafe | Total | Source |
|--------|-----:|-------:|------:|--------|
| PATCH (synthetic) | 593,020 | 231,924 | 824,944 | PATCH-GPT + PATCH-RT |
| PATCH-H (human) | 260 | 130 | 390 | Native TC speakers |

Unsafe prompts include both direct harmful prompts (PATCH-GPT) and evasive adversarial prompts generated via Rainbow Teaming (PATCH-RT). All prompts are categorized across 13 MLCommons hazard categories.

## Dataset Access

The full PATCH dataset (train/val/test splits) is hosted on HuggingFace:
**[TODO_HF_URL](TODO_HF_URL)**

A subset of benchmark and evaluation files is included in the `dataset/` directory of this repository for convenience.

## Repository Structure

```
PATCH/
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── dataset/                          # Benchmark/evaluation datasets
│   ├── english_safety_benchmark.jsonl
│   ├── jailbreak_chinese.jsonl
│   └── jailbreak_english.jsonl
├── model_training/                   # Training scripts for safety classifiers
│   ├── full_finetune/                #   FFT for Llama Guard, RoBERTa, Longformer
│   ├── LoRA/                         #   LoRA for Llama Guard, RoBERTa, Longformer
│   └── chat_vector/                  #   Chat vector extraction and combination
├── evaluation/                       # Evaluation scripts and metrics
├── data_generation/                  # Dataset generation pipelines
│   ├── openai_api/                   #   PATCH-GPT: direct harmful prompt generation
│   └── rainbow_teaming_zh/           #   PATCH-RT: Rainbow Teaming adversarial generation
└── patch_rt_asr/                     # Attack success rate evaluation
```

## Quick Start

### Installation

```bash
git clone https://github.com/Harrychangtw/PATCH.git
cd PATCH
pip install -r requirements.txt
```

> **Note:** The root project requires Python 3.9+. The `data_generation/rainbow_teaming_zh/` subproject requires Python 3.12+.

## Training

Training scripts for all safety classifiers are in `model_training/`. We support both full fine-tuning and LoRA for three architectures:

- **Llama Guard 3 1B**, **RoBERTa**, **Longformer**
- Full fine-tuning scripts in `model_training/full_finetune/`
- LoRA scripts in `model_training/LoRA/`
- Chat vector utilities in `model_training/chat_vector/`

See `model_training/README.md` for detailed instructions and hyperparameter configurations.

## Evaluation

Evaluation scripts for computing safety classification metrics (precision, recall, F1) are in `evaluation/`. These support evaluation on both synthetic test splits and the human-annotated PATCH-H benchmark.

## Data Generation

PATCH uses two pipelines for generating unsafe prompts:

- **PATCH-GPT**: Direct generation of harmful prompts across 13 hazard categories via OpenAI API. See `data_generation/openai_api/`.
- **PATCH-RT**: Adversarial prompt generation using Rainbow Teaming to produce evasive, harder-to-detect unsafe prompts. See `data_generation/rainbow_teaming_zh/`.

## ASR Evaluation

Attack success rate (ASR) evaluation for Rainbow Teaming prompts is in `patch_rt_asr/`. This measures how effectively adversarial prompts bypass target LLM safety filters.

## Reproducibility Notes

### Dataset Access
The full PATCH dataset (train/val/test splits) is available on HuggingFace at [https://huggingface.co/datasets/Raymond102103028/PATCH]. Training and evaluation scripts expect the data files in the `dataset/` directory.

### Hyperparameters
All hyperparameters in the training scripts are configured to match those reported in the paper (Tables 10 and 11). Adjust `BATCH_SIZE` based on your available GPU VRAM.

### Weights & Biases (Optional)
Training scripts support optional W&B logging. Set `WANDB_PROJECT` and `WANDB_ENTITY` environment variables to enable it. If unset, training proceeds without W&B.

### Rainbow Teaming Subproject
The `data_generation/rainbow_teaming_zh/` directory is a standalone Poetry project requiring Python ≥3.12 and CUDA-capable GPU. See its own README for setup instructions. It is independent of the root `requirements.txt`.

## Citation

If you use PATCH in your research, please cite:

*BibTeX entry will be added once the arXiv preprint is available.*

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
