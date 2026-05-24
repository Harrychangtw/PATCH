import os, gc, json, torch, multiprocessing
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import confusion_matrix, precision_score, f1_score, cohen_kappa_score
from datasets import load_dataset

DATASET_PATH = "../dataset/patch_test.jsonl"  
BATCH_SIZE   = 16
NUM_PROC = min(os.cpu_count() or 4, 8)

def build_chat_prompts(text_list):
    return [f"[INST] {txt} [/INST]" for txt in text_list]

def bucket_errors_by_length(error_samples, tokenizer, bucket=10):
    buckets = {}
    for s in error_samples:
        length = len(tokenizer.tokenize(s["input"]))
        lo = (length // bucket) * bucket
        buckets.setdefault(f"{lo}-{lo + bucket}", []).append(s)
    return buckets

def plot_confusion_matrix(cm, classes, path):
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    ticks = range(len(classes))
    plt.xticks(ticks, classes); plt.yticks(ticks, classes)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"),
                     ha="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel("True label"); plt.xlabel("Predicted label")
    plt.tight_layout(); plt.savefig(path); plt.close()

def tokenize_function(batch, tokenizer):
    return tokenizer(batch["text"], truncation=True, padding="max_length")

def classify_batch(model, tokenizer, text_batch, max_new_tokens=10):
    prompts = build_chat_prompts(text_batch)
    model_inputs = tokenizer(
                prompts, 
                return_tensors="pt",
                padding=True, 
                truncation=True
    ).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            output_scores=True,
            return_dict_in_generate=True,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    preds, confidences, texts = [], [], []
    prompt_lens = (model_inputs["input_ids"] != tokenizer.pad_token_id).sum(dim=1)
    first_scores = outputs.scores[0]
    for i in range(len(text_batch)):
        generated_tokens = outputs.sequences[i, prompt_lens[i]:]
        gen_text = tokenizer.decode(generated_tokens).lower()
        pred = 1 if "unsafe" in gen_text else 0
        prob = torch.softmax(first_scores[i], dim=-1)[generated_tokens[0]].item()
        preds.append(pred); confidences.append(prob); texts.append(gen_text)
    del model_inputs, outputs
    torch.cuda.empty_cache(); gc.collect()
    return preds, confidences, texts

def main():
    multiprocessing.freeze_support() 
    if not DATASET_PATH.lower().endswith(".jsonl"):
        raise ValueError("DATASET_PATH must point to a .jsonl file")
    if not os.path.isfile(DATASET_PATH):
        raise FileNotFoundError(f"File not found: {DATASET_PATH}")
    with open(DATASET_PATH, "r", encoding="utf-8") as _f:
        try:
            json.loads(_f.readline())
        except json.JSONDecodeError as e:
            raise ValueError(f"First line is not valid JSON: {e}")
    print("Loading dataset list ...")
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        line_count = sum(1 for _ in open(DATASET_PATH, "r", encoding="utf-8"))
        dataset = [json.loads(line) for line in tqdm(f, total=line_count, desc="Reading lines")]
    total_samples = len(dataset)
    print(f"Dataset loaded: {total_samples} samples")
    print("Loading model ...")
    model_id = "meta-llama/Llama-Guard-3-1B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.resize_token_embeddings(len(tokenizer))
    print("Tokenizing with datasets.map() ...")
    ds = load_dataset("json", data_files={"test": DATASET_PATH}) 
    tokenized = ds["test"].map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=["text"],         
        num_proc=NUM_PROC,
        keep_in_memory=True
    )
    print(f"Tokenized dataset cached to cache/test.arrow (rows: {len(tokenized)})")
    dataset_stem = os.path.splitext(os.path.basename(DATASET_PATH))[0]
    OUTPUT_DIR = f"./results/{dataset_stem}_llamaguard1b_base_evaluation_results"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    y_true, y_pred, details, errors = [], [], [], []
    totals = dict(total=0, correct=0, 
                  label0_total=0, label0_correct=0,
                  label1_total=0, label1_correct=0)
    batch_indices = range(0, total_samples, BATCH_SIZE)
    with tqdm(total=len(batch_indices), desc="Inference batches") as pbar:
        for start_idx in batch_indices:
            batch = dataset[start_idx:start_idx + BATCH_SIZE]
            texts  = [d.get("input", d.get("text")) for d in batch]
            labels = [d["label"] for d in batch]
            preds, confs, gens = classify_batch(model, tokenizer, texts)
            for text, gt, pred, conf, gen in zip(texts, labels, preds, confs, gens):
                totals["total"] += 1
                totals["correct"] += pred == gt
                if gt == 0:
                    totals["label0_total"] += 1
                    totals["label0_correct"] += pred == gt
                else:
                    totals["label1_total"] += 1
                    totals["label1_correct"] += pred == gt
                y_true.append(gt); y_pred.append(pred)
                rec = dict(input=text, ground_truth=gt, predicted=pred,
                           generated_text=gen, confidence=conf)
                details.append(rec)
                if pred != gt:
                    errors.append(rec)
            pbar.update(1)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    precision = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred)
    acc_total = totals["correct"] / totals["total"]
    acc0 = totals["label0_correct"] / totals["label0_total"] if totals["label0_total"] else 0
    acc1 = totals["label1_correct"] / totals["label1_total"] if totals["label1_total"] else 0

    with open(os.path.join(OUTPUT_DIR, "stats.txt"), "w", encoding="utf-8") as f:
        f.write(f"Label 0: total={totals['label0_total']}, correct={totals['label0_correct']}, accuracy={acc0:.2%}\n")
        f.write(f"Label 1: total={totals['label1_total']}, correct={totals['label1_correct']}, accuracy={acc1:.2%}\n")
        f.write(f"All samples: total={totals['total']}, correct={totals['correct']}, accuracy={acc_total:.2%}\n")

    with open(os.path.join(OUTPUT_DIR, "samples.jsonl"), "w", encoding="utf-8") as f:
        for s in details:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    buckets = bucket_errors_by_length(errors, tokenizer, 10)
    with open(os.path.join(OUTPUT_DIR, "error_buckets.jsonl"), "w", encoding="utf-8") as f:
        for label in sorted(buckets, key=lambda x: int(x.split("-")[0])):
            f.write(json.dumps({"bucket": label, "samples": buckets[label]}, ensure_ascii=False) + "\n")
    plot_confusion_matrix(cm, ["safe (0)", "unsafe (1)"],
                          os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
    with open(os.path.join(OUTPUT_DIR, "metrics.txt"), "w", encoding="utf-8") as f:
        f.write(f"Overall precision: {precision:.4f}\n")
        f.write(f"Cohen's kappa:     {kappa:.4f}\n")
        f.write(f"F1 score:          {f1:.4f}\n")
    print(f"Evaluation complete! Results saved to folder: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()