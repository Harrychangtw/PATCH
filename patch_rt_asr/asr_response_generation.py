import pandas as pd
import json
import os
import time
import requests
import re
from tqdm import tqdm

def clean_name_for_filesystem(name):
    cleaned_name = re.sub(r'[\\/*?:"<>|]', '_', name)
    cleaned_name = re.sub(r'_+', '_', cleaned_name)
    cleaned_name = cleaned_name.strip('_ ')
    return cleaned_name

def pull_model(model_name):
    print(f"Pulling model '{model_name}'. This may take a while...")
    try:
        pull_request = {
            "name": model_name
        }
        with requests.post("http://localhost:11434/api/pull", 
                          json=pull_request, 
                          stream=True) as response:
            status = ""
            total = 0
            completed = 0
            pbar = None
            for line in response.iter_lines():
                if line:
                    try:
                        progress = json.loads(line)
                        if 'total' in progress and progress['total'] != total:
                            total = progress['total']
                            if pbar:
                                pbar.close()
                            pbar = tqdm(total=total, unit='B', unit_scale=True, 
                                       desc=f"Downloading {model_name}")
                        if 'completed' in progress:
                            if pbar:
                                completed_diff = progress['completed'] - completed
                                if completed_diff > 0:
                                    pbar.update(completed_diff)
                                completed = progress['completed']
                        if 'status' in progress and progress['status'] != status:
                            status = progress['status']
                            if pbar:
                                pbar.set_description(f"{model_name}: {status}")
                            else:
                                print(f"Status: {status}")
                        if status.lower() == 'success':
                            if pbar:
                                pbar.close()
                            print(f"Successfully pulled model {model_name}")
                            return True
                    except json.JSONDecodeError:
                        continue
            if pbar:
                pbar.close()
            return True
    except Exception as e:
        print(f"Error pulling model {model_name}: {str(e)}")
        return False

def generate_responses(input_csv_path, model_name, output_dir="./"):
    clean_model_name = clean_name_for_filesystem(model_name)
    progress_file = os.path.join(output_dir, f"progress_{clean_model_name}.json")
    output_csv = os.path.join(output_dir, f"model_response_{clean_model_name}.csv")
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(input_csv_path)
    prompts = df['prompt'].tolist()
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            progress = json.load(f)
            completed = progress.get('completed_indices', [])
    else:
        progress = {'total': len(prompts), 'completed': 0, 'completed_indices': []}
        completed = []
    if os.path.exists(output_csv):
        output_df = pd.read_csv(output_csv)
    else:
        output_df = pd.DataFrame(columns=['prompt', 'response', 'safe_unsafe'])
    try:
        print(f"Checking Ollama service and model {model_name}...")
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_exists = any(model["name"] == model_name for model in models)
            if not model_exists:
                print(f"Model {model_name} not found locally. Attempting to pull...")
                pull_success = pull_model(model_name)
                if not pull_success:
                    print(f"Failed to pull model {model_name}. Exiting.")
                    return
        else:
            print(f"Error checking model availability: {response.status_code}")
            return
    except requests.exceptions.ConnectionError:
        print("Error: Ollama service not running. Please start Ollama first.")
        return
    remaining = len(prompts) - len(completed)
    print(f"Generating responses for {remaining} prompts using {model_name}...")
    for i in tqdm(range(len(prompts)), desc="Processing prompts"):
        if i in completed:
            continue
        prompt = prompts[i]
        try:
            truncated_prompt = prompt[:80] + "..." if len(prompt) > 80 else prompt
            print(f"\nPrompt {i+1}/{len(prompts)}: {truncated_prompt}")
            response_text = ""
            start_time = time.time()            
            print("Response:")
            stream_request = {
                "model": model_name,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "temperature": 0, 
                    "top_p": 0.9,
                    "top_k": 40,
                    "repeat_penalty": 1.1,
                    "num_predict": 512,
                    "stop": ["Human:", "User:"]
                }
            }
            with requests.post("http://localhost:11434/api/generate", 
                              json=stream_request, 
                              stream=True) as response:
                for line in response.iter_lines():
                    if line:
                        try:
                            chunk = json.loads(line)
                            chunk_text = chunk.get("response", "")
                            response_text += chunk_text
                            print(chunk_text, end="", flush=True)
                            if chunk.get("done", False):
                                break
                        except json.JSONDecodeError:
                            continue
            print()  
            gen_time = time.time() - start_time
            tokens_per_sec = len(response_text.split()) / gen_time if gen_time > 0 else 0
            print(f"Generation speed: {tokens_per_sec:.2f} tokens/sec")
            new_row = pd.DataFrame({
                'prompt': [prompt],
                'response': [response_text],
                'safe_unsafe': ['']
            })
            output_df = pd.concat([output_df, new_row], ignore_index=True)
            progress['completed'] += 1
            progress['completed_indices'].append(i)
            with open(progress_file, 'w') as f:
                json.dump(progress, f, indent=2)
            output_df.to_csv(output_csv, index=False)
        except Exception as e:
            print(f"Error processing prompt {i}: {str(e)}")
    print(f"\nResponse generation complete. Results saved to {output_csv}")
    return output_csv

if __name__ == "__main__":
    models = [
        "model_1",
        "model_2",
        "model_3",
    ]
    for model in models:
        generate_responses("./dataset.csv", model, "model_responses")