import os
import argparse
import json
import torch
from transformers import AutoModelForCausalLM

def extract_chat_vector(base_model_path, chat_model_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Loading base model from {base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path)
    
    print(f"Loading chat model from {chat_model_path}")
    chat_model = AutoModelForCausalLM.from_pretrained(chat_model_path)
    chat_vector = {}
    base_params_dict = dict(base_model.named_parameters())
    chat_params_dict = dict(chat_model.named_parameters())
    excluded_layers = []
    for name in base_params_dict:
        if "embed" in name.lower() or "lm_head" in name.lower() or "word_embeddings" in name.lower():
            excluded_layers.append(name)
    print(f"Excluding the following layers due to potential vocabulary differences: {excluded_layers}")
    skipped_params = []
    for name, base_param in base_params_dict.items():
        if name in excluded_layers:
            skipped_params.append(name)
            continue
        if name in chat_params_dict:
            chat_param = chat_params_dict[name]
            if base_param.shape == chat_param.shape:
                chat_vector[name] = chat_param.detach().clone() - base_param.detach().clone()
            else:
                print(f"Skipping parameter {name} due to shape mismatch: {base_param.shape} vs {chat_param.shape}")
                skipped_params.append(name)
        else:
            print(f"Parameter {name} not found in chat model, skipping")
            skipped_params.append(name)
    metadata = {
        "excluded_layers": excluded_layers,
        "skipped_params": skipped_params,
        "base_model": base_model_path,
        "chat_model": chat_model_path
    }
    cv_dir_name = f"ChatVector_{os.path.basename(chat_model_path)}-{os.path.basename(base_model_path)}"
    cv_dir_path = os.path.join(output_dir, cv_dir_name)
    os.makedirs(cv_dir_path, exist_ok=True)
    torch.save(chat_vector, os.path.join(cv_dir_path, "chat_vector.pt"))
    with open(os.path.join(cv_dir_path, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Chat vector saved to {os.path.join(cv_dir_path, 'chat_vector.pt')}")
    print(f"Metadata saved to {os.path.join(cv_dir_path, 'metadata.json')}")
    print(f"Excluded layers: {excluded_layers}")
    print(f"Total skipped parameters: {len(skipped_params)}")

if __name__ == "__main__":
    from constant import LLAMA_BASE, LLAMA_GUARD, LLAMA_TW, BASE_DIR
    extract_chat_vector(LLAMA_BASE, LLAMA_GUARD, BASE_DIR)