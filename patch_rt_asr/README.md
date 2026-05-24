# patch_rt_asr: Adversarial Safety Rate (ASR) Evaluation

This module is used to measure the attack success rate of language models using an adversarial dataset. It provides tools to generate model responses to adversarial prompts and to evaluate the safety of those responses using an ensemble of content safety models.

## Workflow

1. **Response Generation** (`asr_response_generation.py`):
   - Reads adversarial prompts from a CSV file (e.g., `dataset.csv`).
   - For each prompt, generates a response using a specified local model via the Ollama API.
   - Saves the generated responses to a CSV file for later evaluation.
   - Supports resuming from progress and handles model pulling if not available locally.

2. **Safety Judging** (`asr_judging.py`):
   - Reads the generated responses CSV.
   - For each response, uses an ensemble of external models (via OpenRouter API) to judge if the response is "SAFE" or "UNSAFE" according to detailed harm categories.
   - Aggregates the ensemble verdicts and records the final safety label.
   - Computes and prints the attack success rate (percentage of responses judged as unsafe).

## Usage

1. **Generate Model Responses**
   - Edit the `models` list in `asr_response_generation.py` to include your model names.
   - Run the script to generate responses:
     ```bash
     python asr_response_generation.py
     ```
   - Responses will be saved in the `model_responses/` directory.

2. **Evaluate Safety of Responses**
   - Edit the `response_files` list in `asr_judging.py` to include the generated CSV files.
   - Set the `OPENROUTER_API_KEY` environment variable for API access.
   - Run the script to evaluate safety:
     ```bash
     python asr_judging.py
     ```
   - The script will update the CSV with safety labels and print attack success statistics.

## Requirements
- Python 3.x
- pandas, tqdm, requests, openai
- Ollama service running locally for response generation
- OpenRouter API access for safety judging

## Files
- `asr_response_generation.py`: Generates model responses to adversarial prompts.
- `asr_judging.py`: Judges the safety of responses and computes attack success rate.
- `dataset.csv`: Input adversarial prompts.


