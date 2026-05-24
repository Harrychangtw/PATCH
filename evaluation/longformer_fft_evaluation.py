import os, gc, json, torch, multiprocessing
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, LongformerForSequenceClassification
from sklearn.metrics import confusion_matrix, precision_score, f1_score, cohen_kappa_score
from datasets import load_dataset

DATASET_PATH = "../dataset/patch_test.jsonl"    
BATCH_SIZE   = 16
NUM_PROC = min(os.cpu_count() or 4, 8)
MODEL_PATH   = "../model/longformer_fft"    

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
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=2048)

def classify_batch(model, tokenizer, text_batch):
    enc = tokenizer(
        text_batch,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048
    ).to(model.device)
    with torch.no_grad():
        logits = model(**enc).logits
    probs = torch.softmax(logits, dim=-1)
    preds = torch.argmax(probs, dim=-1).tolist()
    confidences = probs[:, 1].cpu().tolist()
    texts = [""] * len(text_batch)
    del enc, logits, probs
    torch.cuda.empty_cache(); gc.collect()
    return preds, confidences, texts

def main():
    multiprocessing.freeze_support()
    if not DATASET_PATH.lower().endswith(".jsonl"):
        raise ValueError("DATASET_PATH must point to a .jsonl file")
    if not os.path.isfile(DATASET_PATH):
        raise FileNotFoundError(f"File not found: {DATASET_PATH}")
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        try:
            json.loads(f.readline())
        except json.JSONDecodeError as e:
            raise ValueError(f"First line is not valid JSON: {e}")
    print("Loading dataset list ...")
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        line_count = sum(1 for _ in open(DATASET_PATH, "r", encoding="utf-8"))
        dataset = [json.loads(line) for line in tqdm(f, total=line_count, desc="Reading lines")]
    total_samples = len(dataset)
    print(f"Dataset loaded: {total_samples} samples")
    print("Loading model ...")
    model_dir = MODEL_PATH
    BASE_MODEL_NAME = "schen/longformer-chinese-base-4096"            
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = LongformerForSequenceClassification.from_pretrained(
        model_dir,
        num_labels=2
    ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.eval()
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
    OUTPUT_DIR = f"./results/{dataset_stem}_longformer_fft_evaluation_results"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    y_true, y_pred, details, errors = [], [], [], []
    totals = dict(total=0, correct=0,
                  label0_total=0, label0_correct=0,
                  label1_total=0, label1_correct=0)
    batch_indices = range(0, total_samples, BATCH_SIZE)
    with tqdm(total=len(batch_indices), desc="Inference batches") as pbar:
        for start_idx in batch_indices:
            batch  = dataset[start_idx:start_idx + BATCH_SIZE]
            texts  = [d.get("input", d.get("text")) for d in batch]
            labels = [d["label"] for d in batch]
            preds, confs, _ = classify_batch(model, tokenizer, texts)
            for text, gt, pred, conf in zip(texts, labels, preds, confs):
                totals["total"] += 1
                totals["correct"] += pred == gt
                if gt == 0:
                    totals["label0_total"] += 1
                    totals["label0_correct"] += pred == gt
                else:
                    totals["label1_total"] += 1
                    totals["label1_correct"] += pred == gt
                y_true.append(gt); y_pred.append(pred)
                rec = dict(input=text, ground_truth=gt,
                           predicted=pred, confidence=conf)
                details.append(rec)
                if pred != gt: errors.append(rec)
            pbar.update(1)
    cm        = confusion_matrix(y_true, y_pred, labels=[0,1])
    precision = precision_score(y_true, y_pred, zero_division=0)
    f1        = f1_score(y_true, y_pred, zero_division=0)
    kappa     = cohen_kappa_score(y_true, y_pred)
    acc_total = totals["correct"] / totals["total"]
    acc0     = totals["label0_correct"] / totals["label0_total"] if totals["label0_total"] else 0
    acc1     = totals["label1_correct"] / totals["label1_total"] if totals["label1_total"] else 0

    with open(os.path.join(OUTPUT_DIR, "stats.txt"), "w", encoding="utf-8") as f:
        f.write(f"Label 0: total={totals['label0_total']}, correct={totals['label0_correct']}, accuracy={acc0:.2%}\n")
        f.write(f"Label 1: total={totals['label1_total']}, correct={totals['label1_correct']}, accuracy={acc1:.2%}\n")
        f.write(f"All samples: total={totals['total']}, correct={totals['correct']}, accuracy={acc_total:.2%}\n")

    with open(os.path.join(OUTPUT_DIR, "samples.jsonl"), "w", encoding="utf-8") as f:
        for s in details:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    buckets = bucket_errors_by_length(errors, tokenizer, bucket=10)
    with open(os.path.join(OUTPUT_DIR, "error_buckets.jsonl"), "w", encoding="utf-8") as f:
        for label in sorted(buckets, key=lambda x: int(x.split("-")[0])):
            f.write(json.dumps({"bucket":  label, "samples": buckets[label]}, ensure_ascii=False) + "\n")
    plot_confusion_matrix(cm, ["safe (0)", "unsafe (1)"],
                          os.path.join(OUTPUT_DIR, "confusion_matrix.png"))

    with open(os.path.join(OUTPUT_DIR, "metrics.txt"), "w", encoding="utf-8") as f:
        f.write(f"Overall precision: {precision:.4f}\n")
        f.write(f"Cohen's kappa:     {kappa:.4f}\n")
        f.write(f"F1 score:          {f1:.4f}\n")
    print(f"Evaluation complete! Results saved to folder: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()