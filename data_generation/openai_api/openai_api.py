import openai
import json
import time
import os
import re
from opencc import OpenCC
import asyncio
from concurrent.futures import ThreadPoolExecutor

ITERATION_COUNTS = 1
POLLING_INTERVAL_SECONDS = 60  

API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set.")
openai.api_key = API_KEY
def create_batch_and_save_id(file_path: str, batch_ids_file: str = "batch_ids.json"):
    file_response = openai.files.create(
        file=open(file_path, "rb"),
        purpose="batch"
    )
    file_id = file_response.id
    print("File ID:", file_id)
    batch_job = openai.batches.create(
        input_file_id=file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )
    new_batch_id = batch_job.id
    print("New batch ID:", new_batch_id)
    if os.path.exists(batch_ids_file):
        with open(batch_ids_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = []  
    data.append(new_batch_id)
    with open(batch_ids_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved new batch ID to {batch_ids_file}")

def process_batch_sync(b_id, idx):
    result = {
        "batch_id": b_id,
        "status": None,
        "output": None,
        "error": None,
        "log": []
    }
    try:
        batch_status = openai.batches.retrieve(b_id)
        result["status"] = batch_status.status
        result["log"].append(f"{idx}. Batch ID: {b_id}")
        result["log"].append(f"   Status: {batch_status.status}")
        if batch_status.status == "failed":
            error_reason = None
            if hasattr(batch_status, "error_message") and batch_status.error_message:
                error_reason = batch_status.error_message
            elif isinstance(batch_status, dict) and "error" in batch_status and batch_status["error"]:
                error_reason = batch_status["error"]
            result["error"] = error_reason
            result["log"].append("   -> Status is 'failed'. Removing this BatchID.")
            if error_reason:
                result["log"].append(f"   -> Fail reason: {error_reason}")
            else:
                result["log"].append("   -> No detailed error message provided.")
                try:
                    status_dict = batch_status.to_dict()
                except Exception:
                    status_dict = batch_status.__dict__
                result["log"].append("   -> Full batch_status info:")
                result["log"].append(json.dumps(status_dict, ensure_ascii=False, indent=2))
        else:
            if batch_status.status == "completed":
                output_file_id = batch_status.output_file_id
                if output_file_id:
                    file_obj = openai.files.retrieve(output_file_id)
                    result["log"].append("File Object: " + str(file_obj))
                    try:
                        content_stream = openai.files.content(output_file_id)
                        file_bytes = content_stream.read()
                        file_text = file_bytes.decode("utf-8", errors="replace")
                        lines = file_text.splitlines()
                        parsed_lines = []
                        for line in lines:
                            try:
                                obj = json.loads(line)
                                parsed_lines.append(obj)
                            except json.JSONDecodeError as e:
                                result["log"].append(f"      Warning: could not parse line: {e}")
                        result["output"] = parsed_lines
                        result["log"].append(f"   -> Downloaded {len(parsed_lines)} lines from batch.")
                    except Exception as e:
                        result["log"].append(f"   -> Error downloading file content: {e}")
                else:
                    result["log"].append("   -> No output_file_id found (but status is completed?).")
            else:
                result["log"].append("   -> Batch not completed yet, skipping download.")
    except Exception as e:
        result["log"].append(f"{idx}. ERROR retrieving batch '{b_id}': {e}")
    return result

async def process_batch_async(b_id, idx, loop, executor):
    return await loop.run_in_executor(executor, process_batch_sync, b_id, idx)

async def check_and_save_outputs_async(batch_ids_file="batch_ids.json", output_json="all_completed_outputs.json"):
    if not os.path.exists(batch_ids_file):
        print(f"File '{batch_ids_file}' not found. Nothing to check.")
        return

    with open(batch_ids_file, "r", encoding="utf-8") as f:
        current_batch_ids_from_file = json.load(f)

    if not current_batch_ids_from_file:
        print("No batch IDs found in the file to process.")
        return

    all_outputs = {}
    active_batch_ids = list(current_batch_ids_from_file)
    batch_ids_to_persist = list(current_batch_ids_from_file)
    loop = asyncio.get_event_loop()
    max_workers = min(10, len(active_batch_ids) if active_batch_ids else 1) 
    executor = ThreadPoolExecutor(max_workers=max_workers if max_workers > 0 else None)  

    while True: 
        if not active_batch_ids:
            print("All batches have been processed or removed from the active list.")
            break
        print(f"\\nChecking status for {len(active_batch_ids)} active batch(es): {active_batch_ids}")    
        tasks = [
            process_batch_async(b_id, idx, loop, executor)
            for idx, b_id in enumerate(active_batch_ids, start=1)
        ]
        current_run_results = await asyncio.gather(*tasks, return_exceptions=True)
        still_pending_after_this_iteration = [] 
        for i, res_or_exc in enumerate(current_run_results):
            current_batch_id = active_batch_ids[i] 
            if isinstance(res_or_exc, Exception):
                print(f"An error occurred during async processing for batch {current_batch_id}: {res_or_exc}")
                still_pending_after_this_iteration.append(current_batch_id)
                continue 
            res = res_or_exc 
            for log_line in res["log"]:
                print(log_line)
            print("-" * 50)
            if res["status"] == "completed":
                if res["output"] is not None:
                    all_outputs[current_batch_id] = res["output"]
                if current_batch_id not in batch_ids_to_persist:
                     batch_ids_to_persist.append(current_batch_id)
            elif res["status"] in ["failed", "cancelled", "expired"]:
                if current_batch_id in batch_ids_to_persist:
                    batch_ids_to_persist.remove(current_batch_id) 
            else: 
                still_pending_after_this_iteration.append(current_batch_id)
        active_batch_ids = still_pending_after_this_iteration
        if active_batch_ids:
            print(f"\\n{len(active_batch_ids)} batch(es) still pending. Waiting for {POLLING_INTERVAL_SECONDS} seconds before next check...")
            await asyncio.sleep(POLLING_INTERVAL_SECONDS)
        else:
            print("\\nAll batches have reached a terminal state.")
            break
    ordered_batch_ids_to_persist = [bid for bid in current_batch_ids_from_file if bid in batch_ids_to_persist]

    with open(batch_ids_file, "w", encoding="utf-8") as f_out:
        json.dump(ordered_batch_ids_to_persist, f_out, ensure_ascii=False, indent=2)
    print(f"\\nUpdated batch IDs saved to '{batch_ids_file}' (failed/cancelled/expired batch IDs removed).")

    if all_outputs:
        with open(output_json, "w", encoding="utf-8") as f_out:
            json.dump(all_outputs, f_out, ensure_ascii=False, indent=2)
        print(f"\\nSaved completed batch outputs to '{output_json}'.")
    else:
        print("\\nNo completed batch outputs to save from this run.")
    if max_workers > 0 : 
        executor.shutdown(wait=True)

def main_aggregate_data():
    if not os.path.exists("all_completed_outputs.json"):
        print("File 'all_completed_outputs.json' not found. Skipping data aggregation.")
        return
    try:
        with open("all_completed_outputs.json", "r", encoding="utf-8") as f:
            raw_data = json.load(f)
    except Exception as e:
        print(f"Error reading all_completed_outputs.json: {e}")
        return
    grouped_contents = {}
    for batch_key, items in raw_data.items():
        for item in items:
            category = item.get("custom_id")
            if not category:
                continue
            choices = item.get("response", {}).get("body", {}).get("choices", [])
            for choice in choices:
                content = choice.get("message", {}).get("content")
                if content:
                    grouped_contents.setdefault(category, []).append(content)
    pattern = re.compile(r'"request"\s*:\s*"([^"]+)"')
    grouped_requests = {}
    for category, contents in grouped_contents.items():
        reqs = []
        for content in contents:
            found = pattern.findall(content)
            reqs.extend(found)
        grouped_requests[category] = reqs
    cc = OpenCC('s2t')
    final_data = {}
    for category, reqs in grouped_requests.items():
        new_reqs = []
        for req in reqs:
            trad = cc.convert(req)
            new_reqs.append(trad)
        new_reqs = list(dict.fromkeys(new_reqs))
        final_data[category] = new_reqs
    try:
        sorted_final_data = dict(sorted(final_data.items(), key=lambda x: int(x[0][1:])))
    except ValueError: 
        print("Warning: Could not sort categories numerically. Using default order.")
        sorted_final_data = final_data
    try:
        with open("final_aggregated_requests.json", "w", encoding="utf-8") as f:
            json.dump(sorted_final_data, f, ensure_ascii=False, indent=2)
        print("Generated final JSON: final_aggregated_requests.json")
    except Exception as e:
        print(f"Error writing final_aggregated_requests.json: {e}")

    print("\nNumber of values for each category:")
    for category, values in sorted_final_data.items():
        print(f"{category} has {len(values)} values")

def main_create_batches():
    JSONL_FILE_PATH = "request.jsonl"  
    for i in range(ITERATION_COUNTS): 
        time.sleep(0.1) 
        create_batch_and_save_id(JSONL_FILE_PATH)
    print(f"Batch creation process completed after {ITERATION_COUNTS} iterations.")

def main_process_batch_outputs():
    asyncio.run(check_and_save_outputs_async(
        batch_ids_file="batch_ids.json",
        output_json="all_completed_outputs.json"
    ))
    print("Batch output processing completed.")

if __name__ == "__main__":
    print("Starting API batch creation...")
    main_create_batches()

    print("\nStarting batch output processing...")
    main_process_batch_outputs()

    print("\nStarting data aggregation...")
    main_aggregate_data() 
    
    print("\nAll operations completed.")
