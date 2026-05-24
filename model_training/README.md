# Model Training Scripts

This directory contains Python scripts for fine-tuning various language models using Full Fine-Tuning (FFT) and Low-Rank Adaptation (LoRA) techniques.

## Scripts

The following scripts are included:

### Full Fine-Tuning (FFT)

Located in the `full-fine-tune/` directory:

- `llamaguard1b_fft.py`: Fine-tunes the LlamaGuard 1B model using FFT.
- `longformer_fft.py`: Fine-tunes the Longformer model using FFT.
- `roberta_fft.py`: Fine-tunes the RoBERTa model using FFT.

### Low-Rank Adaptation (LoRA)

Located in the `LoRA/` directory:

- `llamaguard1b_lora.py`: Fine-tunes the LlamaGuard 1B model using LoRA.
- `longformer_lora.py`: Fine-tunes the Longformer model using LoRA.
- `roberta_lora.py`: Fine-tunes the RoBERTa model using LoRA.

## Functionality

Each training script typically performs the following steps:

1.  **Environment Setup**: Checks for `WANDB_PROJECT` and `WANDB_ENTITY` environment variables for Weights & Biases logging.
2.  **Load Model and Tokenizer**: Loads a pre-trained model and its corresponding tokenizer from Hugging Face.
3.  **Load Dataset**: Loads training and validation datasets (e.g., `patch_train.jsonl`, `patch_val.jsonl`).
4.  **Tokenization**: Tokenizes the input texts from the datasets. Some scripts use sliding window tokenization for longer texts.
5.  **PEFT Configuration (for LoRA)**: For LoRA scripts, configures the PEFT `LoraConfig` specifying `task_type`, `r`, `lora_alpha`, `lora_dropout`, and `target_modules`.
6.  **Training**: Utilizes the Hugging Face `Trainer` API to fine-tune the model. This includes:
    *   Setting `TrainingArguments` (learning rate, batch size, epochs, weight decay, evaluation strategy, etc.).
    *   Defining `compute_metrics` function to calculate accuracy, precision, recall, F1-score, confusion matrix, and ROC AUC score.
    *   Using `EarlyStoppingCallback` to prevent overfitting.
7.  **Save Model**: Saves the best performing model based on the specified metric (e.g., F1-score) to a results directory (e.g., `../results/ModelName_Method_Results/Best_model`).
8.  **Weights & Biases Logging**: Logs metrics, training progress, and model checkpoints to Weights & Biases.

## Usage

Before running any training scripts, ensure you have set up the Weights & Biases environment variables:

```bash
export WANDB_PROJECT=<your_wandb_project_name>
export WANDB_ENTITY=<your_wandb_entity_name>
```

Also, ensure you have installed the necessary dependencies:

```bash
pip install -r ../requirements.txt
```

To run a training script, navigate to the respective subdirectory (`full-fine-tune` or `LoRA`) and execute the desired Python script. For example:

To run a Full Fine-Tuning script:
```bash
cd full-fine-tune
python llamaguard1b_fft.py
```

To run a LoRA script:
```bash
cd LoRA
python llamaguard1b_lora.py
```

---

## Chat Vector Utilities

The `chat_vector/` directory provides utilities for extracting and combining chat vectors from language models. These scripts enable advanced model manipulation by computing the difference between two models (vector subtraction) and combining multiple chat vectors (vector addition) to create new model variants.

### Files
- `vector_subtraction.py`: Extracts the difference ("chat vector") between a base model and a tuned model.
- `vector_addition.py`: Combines two chat vectors with a weighted sum and applies the result to a base model, producing a new model variant.
- `constant.py`: Stores file paths and configuration constants used by the above scripts.

### vector_subtraction.py
This script computes the parameter-wise difference between a base model and a chat-tuned model, excluding embedding and output head layers (to avoid vocabulary mismatches). The resulting chat vector is saved as a `.pt` file along with metadata.

**Usage:**
- Edit `constant.py` to set `LLAMA_BASE`, `LLAMA_GUARD`, and `BASE_DIR` to the appropriate model and output paths.
- Run:
  ```bash
  python vector_subtraction.py
  ```
- Output: A directory containing `chat_vector.pt` and `metadata.json`.

### vector_addition.py
This script loads two chat vectors and applies a weighted sum (using parameter `k`) to a base model's parameters, skipping excluded layers. The resulting model is saved to a new directory, along with a generated `README.md` describing the combination.

**Usage:**
- Edit `constant.py` to set `LLAMA_BASE`, `CV1`, `CV2`, and `BASE_DIR`.
- Run:
  ```bash
  python vector_addition.py --k 0.7
  ```
  (where `--k` is the weight for the first chat vector; `1-k` is used for the second)
- Output: A new model directory with the combined weights and documentation.

### constant.py
This file defines the following variables for use in the above scripts:
- `LLAMA_BASE`: Path to the base model.
- `LLAMA_TW`: (Optional) Path to another model variant.
- `LLAMA_GUARD`: Path to the chat-tuned model.
- `CV1`, `CV2`: Paths to chat vector directories.
- `BASE_DIR`: Output directory for results.

**Note:**
- Update these constants before running the scripts.
- The scripts require the Hugging Face `transformers` and `torch` libraries.

---

## Output Files

The training scripts, utilizing the Hugging Face `Trainer`, will save the following types of files to the specified `output_dir` (e.g., `../results/ModelName_Method_Results/`):

*   **Model Checkpoints**: Saved in subdirectories like `checkpoint-500`, `checkpoint-1000`, etc. These typically include:
    *   `pytorch_model.bin` or `model.safetensors`: The model weights.
    *   `config.json`: The model's configuration file.
    *   `tokenizer_config.json`, `tokenizer.json`, `vocab.txt` (or similar like `merges.txt`, `spiece.model`), `special_tokens_map.json`: Files related to the tokenizer.
    *   `training_args.bin`: The `TrainingArguments` used for the run.
    *   `trainer_state.json`: Contains the state of the trainer, including logs and metrics history for that checkpoint.
*   **Best Model**: If `load_best_model_at_end=True` (which is common when `save_strategy` is "steps" or "epoch" and an `evaluation_strategy` is active), the best model checkpoint (based on `metric_for_best_model`) will often be copied or linked to the root of `output_dir` or a specific "best_model" subdirectory.
*   **LoRA Specific Files** (when using LoRA scripts):
    *   `adapter_model.bin` or `adapter_model.safetensors`: The trained LoRA adapter weights.
    *   `adapter_config.json`: The configuration for the LoRA adapter.
*   **Trainer State at the end of training**:
    *   A `trainer_state.json` file in the `output_dir` summarizing the entire training process if not already covered by the last checkpoint.
*   **TensorBoard/Weights & Biases Logs**: If enabled (W&B is enabled by default in these scripts), log files will be generated. W&B logs are synced to the cloud, but local log directories might also be present depending on the configuration.

The exact file names and structure can sometimes vary slightly based on the `transformers` library version and the specific model being trained. The `../results/ModelName_Method_Results_Best_model` directory mentioned in the "Save Model" functionality description typically refers to a copy or the final state of the best performing model checkpoint.

### Configurable Parameters

The following hyperparameters within each script **can be adjusted** by the user. However, it is **essential to adjust `BATCH_SIZE` based on your available GPU VRAM**. Other parameters are pre-configured to align with the hyperparameters mentioned in the accompanying paper and generally do not need to be changed unless you have specific requirements. These include, but are not limited to:

*   `BATCH_SIZE`: Batch size for training and evaluation. **This must be set based on the available VRAM of your GPU.**
*   `output_dir`: Directory where results and model checkpoints will be saved (e.g., `../results/ModelName_Method_Results`).
*   `data_files`: Paths to your training and validation datasets (typically `.jsonl` files).
    *   The datasets should contain "text" and "label" keys.
*   `TrainingArguments`: Many parameters can be configured here, such as:
    *   `learning_rate`
    *   `num_train_epochs`
    *   `weight_decay`
    *   `eval_steps`, `save_steps`, `logging_steps`
    *   `gradient_accumulation_steps`
    *   `fp16` (for mixed-precision training)
    *   `warmup_steps`
*   `LoraConfig` (for LoRA scripts):
    *   `r` (rank of the update matrices)
    *   `lora_alpha`
    *   `lora_dropout`
    *   `target_modules` (modules to apply LoRA to)
*   `max_length` (in tokenization functions): Maximum sequence length for tokenization.
*   `stride` (in `sliding_tokenize_batched` for RoBERTa): Stride for overlapping tokens.

The scripts utilize libraries such as `transformers`, `datasets`, `scikit-learn`, `torch`, `wandb`, and `peft`.