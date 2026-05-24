\
# Evaluation Scripts

This directory contains Python scripts for evaluating the performance of various language models.

## Scripts

The following scripts are included:

- `chatvector_evaluation.py`: Evaluates ChatVector models.
- `llamaguard1b_base_evaluation.py`: Evaluates the base LlamaGuard 1B model.
- `llamaguard1b_fft_evaluation.py`: Evaluates the LlamaGuard 1B model fine-tuned with Full Fine-Tuning (FFT).
- `llamaguard1b_lora_evaluation.py`: Evaluates the LlamaGuard 1B model fine-tuned with Low-Rank Adaptation (LoRA).
- `longformer_fft_evaluation.py`: Evaluates the Longformer model fine-tuned with FFT.
- `longformer_lora_evaluation.py`: Evaluates the Longformer model fine-tuned with LoRA.
- `roberta_fft_evaluation.py`: Evaluates the RoBERTa model fine-tuned with FFT.
- `roberta_lora_evaluation.py`: Evaluates the RoBERTa model fine-tuned with LoRA.

## Functionality

Each evaluation script typically performs the following steps:

1.  **Load Model and Tokenizer**: Loads a pre-trained or fine-tuned model and its corresponding tokenizer.
2.  **Load Dataset**: Loads a dataset (e.g., `patch_test.jsonl`) for evaluation<>.
3.  **Tokenization**: Tokenizes the input texts from the dataset.
4.  **Inference**: Performs inference on the tokenized dataset in batches.
5.  **Calculate Metrics**: Computes various evaluation metrics, such as:
    *   Accuracy (overall and per-label)
    *   Precision
    *   F1-score
    *   Cohen's Kappa
    *   Confusion Matrix
6.  **Save Results**: Saves the evaluation results to the `results/` subdirectory. This includes:
    *   `stats.txt`: Summary statistics of the evaluation.
    *   `samples.jsonl`: Detailed predictions for each sample in the dataset.
    *   `error_buckets.jsonl`: Errors bucketed by input length.
    *   `confusion_matrix.png`: A plot of the confusion matrix.
    *   `metrics.txt`: Key metrics like precision, F1-score, and kappa.

## Usage

Before running any evaluation scripts, ensure you have installed the necessary dependencies:

```bash
pip install -r ../requirements.txt 
```

To run an evaluation script, navigate to the `evaluation` directory and execute the desired Python script. For example:

```bash
python llamaguard1b_lora_evaluation.py
```

### Configurable Parameters

Within each script, you can adjust the following parameters:

*   `DATASET_PATH`: Path to your evaluation dataset.
    *   The datasets referenced in the accompanying paper are located in the `../dataset/` directory.
    *   If you intend to use a custom dataset, it must be a `.jsonl` file. Each line in this file should be a JSON object possessing, at a minimum, `"text"` and `"label"` keys. The labels should be integers, where `0` typically represents a benign or safe prompt, and `1` represents a harmful or unsafe prompt. Please verify the specific label meanings if using your own dataset. For instance:
        ```json
        {"text": "This is an example sentence.", "label": 0}
        {"text": "Another piece of text for evaluation.", "label": 1}
        ```
*   `BATCH_SIZE`: This should be set based on the available VRAM of your GPU. A larger batch size generally speeds up processing but requires more memory.
*   `NUM_PROC`: Adjust this parameter according to the number of CPU cores available on your system to optimize data processing parallelism.

The `MODEL_PATH` variable within each script is pre-configured to point to the relevant model discussed in the paper and generally does not need to be changed. However, if you are evaluating a custom model, ensure this path is updated accordingly.

The scripts utilize libraries such as `transformers`, `datasets`, `scikit-learn`, and `matplotlib`.
