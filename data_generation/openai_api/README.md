# OpenAI Batch API Processing

This directory contains a Python script (`openai_api.py`) designed to interact with the OpenAI Batch API. It handles the creation of batch jobs, monitors their progress, retrieves results, and aggregates the generated data.

## Script: `openai_api.py`

This script automates the process of sending requests to OpenAI's batch endpoint, managing these batches, and processing their outputs.

## Functionality

The script is divided into three main operational phases:

1.  **Batch Creation (`main_create_batches`)**
    *   Reads a `request.jsonl` file from the current directory.
    *   Uploads this file to OpenAI to be used as input for batch processing.
    *   Creates a new batch job with the uploaded file, targeting the `/v1/chat/completions` endpoint with a 24-hour completion window.
    *   Saves the ID of the newly created batch job to `batch_ids.json`.
    *   This process is repeated `ITERATION_COUNTS` times (default: 30).

2.  **Batch Output Processing (`main_process_batch_outputs`)**
    *   Reads batch IDs from `batch_ids.json`.
    *   Asynchronously monitors the status of each batch ID.
        *   Uses `ThreadPoolExecutor` for concurrent processing of multiple batches.
        *   Periodically polls OpenAI for batch status (default interval: `POLLING_INTERVAL_SECONDS` - 60 seconds).
    *   If a batch is `completed`:
        *   Retrieves the output file ID.
        *   Downloads and parses the content of the output file (which is expected to be a series of JSON objects, one per line).
        *   Stores the parsed output in `all_completed_outputs.json`, keyed by batch ID.
    *   If a batch `failed`, `cancelled`, or `expired`:
        *   Logs the error or status.
        *   Removes the batch ID from `batch_ids.json` to prevent further processing.
    *   Updates `batch_ids.json` with only the IDs of successfully completed or still pending batches.

3.  **Data Aggregation (`main_aggregate_data`)**
    *   Reads the collected outputs from `all_completed_outputs.json`.
    *   Extracts content from responses:
        *   Groups items by `custom_id` found in the batch output.
        *   Parses the `content` field from the message within each choice of a response.
        *   Specifically looks for a pattern like `"request": "..."` within the content and extracts the value.
    *   Converts extracted requests from Simplified Chinese to Traditional Chinese using the `OpenCC` library (`s2t` conversion).
    *   Removes duplicate requests within each category.
    *   Saves the final, aggregated, and de-duplicated requests to `final_aggregated_requests.json`.
        *   Attempts to sort the categories numerically (e.g., "T1", "T2") before saving.
    *   Prints a summary of the number of values (requests) aggregated for each category.

## Usage

### Prerequisites

1.  **OpenAI API Key**: Ensure the `OPENAI_API_KEY` environment variable is set.
    ```bash
    export OPENAI_API_KEY="your_openai_api_key"
    ```
    On Windows, you might use:
    ```powershell
    $Env:OPENAI_API_KEY="your_openai_api_key"
    ```
    Or set it system-wide.

2.  **Input File**: Prepare a `request.jsonl` file in the `data_generation/openai_api/` directory. Each line should be a JSON object formatted according to OpenAI's batch input requirements for chat completions. Detailed prompts are described in the accompanying paper.

3.  **Dependencies**: Install necessary Python libraries. These typically include `openai`, `opencc`, and other dependencies specified in the main requirements file. You can install them using pip:
    ```bash
    pip install -r ../../requirements.txt
    ```

### Running the Script

Navigate to the `data_generation/openai_api/` directory and execute the Python script:

```bash
python openai_api.py
```

The script will run through the three phases: batch creation, output processing, and data aggregation.

## Input Files

*   **`request.jsonl`**: User-provided file. Contains the requests to be sent to the OpenAI Batch API, with one JSON object per line. Each object should define `custom_id`, `method`, `url`, and `body` for the API call. The detailed prompts used for generation are specified in the accompanying research paper.

## Output Files

The script generates and manages the following files in the `data_generation/openai_api/` directory:

*   **`batch_ids.json`**: Stores a JSON list of OpenAI batch IDs. This file is created/updated during the batch creation phase and pruned during the output processing phase.
    *   Example: `["batch_abc123", "batch_def456"]`
*   **`all_completed_outputs.json`**: Stores a JSON object where keys are batch IDs and values are lists of parsed JSON responses from the completed batches.
    *   Example:
        ```json
        {
          "batch_abc123": [
            {"custom_id": "T1_req1", "response": {"body": {"choices": [{"message": {"content": "{\"request\": \"Hello\"}"}}]}}},
            // ... more responses
          ]
        }
        ```
*   **`final_aggregated_requests.json`**: Stores the final processed data, a JSON object where keys are `custom_id` categories and values are lists of unique, Traditional Chinese requests.
    *   Example:
        ```json
        {
          "T1": ["Hello", "Thank you"],
          "T2": ["Goodbye"]
        }
        ```

## Configurable Parameters

The following constants at the beginning of `openai_api.py` can be adjusted:

*   `ITERATION_COUNTS`: (Default: `1`) The number of times the batch creation process will loop, effectively creating this many batches from the `request.jsonl` file.
*   `POLLING_INTERVAL_SECONDS`: (Default: `60`) The number of seconds the script waits between checks for batch job statuses.

Adjust these values as needed before running the script.
