# PATCH Dataset

## Full Dataset

The full PATCH dataset, including train/val/test splits for all 820,000+ prompts, is hosted on HuggingFace:

**[TODO_HF_URL](TODO_HF_URL)**

To download the full dataset:

```python
from datasets import load_dataset

dataset = load_dataset("TODO_HF_URL")
```

## Benchmark Files (this directory)

This directory contains benchmark and evaluation datasets included directly in the repository for convenience:

| File | Description |
|------|-------------|
| `english_safety_benchmark.jsonl` | English safety benchmark for cross-lingual evaluation |
| `jailbreak_chinese.jsonl` | Chinese jailbreak prompts |
| `jailbreak_english.jsonl` | English jailbreak prompts |

## Data Format

Each file uses JSONL format (one JSON object per line):

```json
{"text": "prompt content here", "label": 0}
```

- `text` -- the prompt string
- `label` -- safety label: `0` = safe, `1` = unsafe

## Downloading the Full Dataset

1. Install the HuggingFace `datasets` library:
   ```bash
   pip install datasets
   ```

2. Load the dataset in Python:
   ```python
   from datasets import load_dataset

   dataset = load_dataset("TODO_HF_URL")

   train = dataset["train"]
   val = dataset["validation"]
   test = dataset["test"]
   ```

3. Alternatively, download files directly from the HuggingFace repository page at [TODO_HF_URL](TODO_HF_URL).
