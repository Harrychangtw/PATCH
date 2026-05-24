import os
import argparse
import json
import torch
from transformers import AutoModelForCausalLM
from shutil import copy
from datetime import datetime

def generate_readme(cv1_path, cv2_path, k, base_model_path):
    cv1_name = os.path.basename(cv1_path)
    cv2_name = os.path.basename(cv2_path)
    base_model_name = os.path.basename(base_model_path)
    try:
        with open(os.path.join(cv1_path, "metadata.json"), "r") as f:
            cv1_metadata = json.load(f)
            if "name" in cv1_metadata:
                cv1_name = cv1_metadata["name"]
    except FileNotFoundError:
        cv1_description = "No description available"
    try:
        with open(os.path.join(cv2_path, "metadata.json"), "r") as f:
            cv2_metadata = json.load(f)
            if "name" in cv2_metadata:
                cv2_name = cv2_metadata["name"]

    except FileNotFoundError:
        cv2_description = "No description available"
    k_percentage = int(k * 100)
    one_minus_k_percentage = int((1 - k) * 100)
    current_date = datetime.now().strftime("%Y-%m-%d")
    readme_content = f"""# Combined Chat Vector Model

## Model Information
- **Base Model**: {base_model_name}
- **Creation Date**: {current_date}
- **Combination Ratio**: {k_percentage}% of ChatVector 1 and {one_minus_k_percentage}% of ChatVector 2

## Chat Vector 1
- **Name**: {cv1_name}
- **Weight Applied**: {k_percentage}%
- **Path**: {cv1_path}

## Chat Vector 2
- **Name**: {cv2_name}
- **Weight Applied**: {one_minus_k_percentage}%
- **Path**: {cv2_path}

This model was created by applying a weighted combination of the above chat vectors to the base model.
"""
    return readme_content

def apply_chat_vectors(base_model_path, cv1_path, cv2_path, k, output_dir):
    print(f"Loading chat vector 1 from {cv1_path}")
    cv1 = torch.load(os.path.join(cv1_path, "chat_vector.pt"))
    print(f"Loading chat vector 2 from {cv2_path}")
    cv2 = torch.load(os.path.join(cv2_path, "chat_vector.pt"))
    try:
        with open(os.path.join(cv1_path, "metadata.json"), "r") as f:
            cv1_metadata = json.load(f)
        cv1_excluded = cv1_metadata.get("excluded_layers", [])
    except FileNotFoundError:
        print("Metadata file for chat vector 1 not found, assuming no excluded layers")
        cv1_excluded = []
    try:
        with open(os.path.join(cv2_path, "metadata.json"), "r") as f:
            cv2_metadata = json.load(f)
        cv2_excluded = cv2_metadata.get("excluded_layers", [])
    except FileNotFoundError:
        print("Metadata file for chat vector 2 not found, assuming no excluded layers")
        cv2_excluded = []
    excluded_layers = set(cv1_excluded + cv2_excluded)
    print(f"The following layers will be excluded when applying chat vectors: {excluded_layers}")
    print(f"Loading base model from {base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path)
    base_params_dict = dict(base_model.named_parameters())
    modified_params = 0
    total_params = 0
    for name, param in base_params_dict.items():
        total_params += 1
        if name in excluded_layers:
            print(f"Skipping excluded layer: {name}")
            continue
        if name in cv1 and name in cv2:
            param.data += k * cv1[name] + (1 - k) * cv2[name]
            modified_params += 1
        elif name in cv1:
            param.data += k * cv1[name]
            modified_params += 1
        elif name in cv2:
            param.data += (1 - k) * cv2[name]
            modified_params += 1
    print(f"Modified {modified_params} out of {total_params} parameters")
    k_percentage = int(k * 100)
    one_minus_k_percentage = int((1 - k) * 100)
    folder_name = f"ChatVector_Llama-Guard_{k_percentage}_{one_minus_k_percentage}"
    output_path = os.path.join(output_dir, folder_name)
    os.makedirs(output_path, exist_ok=True)
    readme_content = generate_readme(cv1_path, cv2_path, k, base_model_path)
    with open(os.path.join(output_path, "README.md"), "w") as f:
        f.write(readme_content)
    print(f"Created README.md in {output_path}")
    print(f"Saving modified model to {output_path}")
    base_model.save_pretrained(output_path)
    print("Copying tokenizer files from base model")
    for file in os.listdir(base_model_path):
        if ("tokenizer" in file or "special_tokens" in file or "config" in file) and not file.endswith(".model"):
            src_path = os.path.join(base_model_path, file)
            dst_path = os.path.join(output_path, file)
            if os.path.isfile(src_path):
                copy(src_path, dst_path)
                print(f"Copied {file}") 
    print(f"Modified model saved to {output_path}")

if __name__ == "__main__":
    from constant import  LLAMA_BASE, CV1, CV2, BASE_DIR
    parser = argparse.ArgumentParser(description="Apply chat vectors to a base model")
    parser.add_argument("--k", type=float, default=0.5, help="Weight for cv1 (1-k will be applied to cv2)")    
    args = parser.parse_args()
    apply_chat_vectors(LLAMA_BASE, CV1, CV2, args.k, BASE_DIR)
