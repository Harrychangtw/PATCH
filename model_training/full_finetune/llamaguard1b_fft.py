import os
import multiprocessing as mp
import torch
import wandb
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
)

BATCH_SIZE = 32
   
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-Guard-3-1B")
tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    cm = confusion_matrix(labels, preds)
    try:
        auc = roc_auc_score(labels, logits[:, 1])
    except Exception:
        auc = None
    metrics = {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
    }
    if auc is not None:
        metrics["auc"] = auc
    return metrics

def main():
    WANDB_PROJECT = os.getenv("WANDB_PROJECT")
    WANDB_ENTITY = os.getenv("WANDB_ENTITY")
    USE_WANDB = WANDB_PROJECT is not None and WANDB_ENTITY is not None
    if not USE_WANDB:
        os.environ["WANDB_DISABLED"] = "true"

    output_dir = "../results/LlamaGuard_FFT_Results"

    if USE_WANDB:
        wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            dir=output_dir
        )
    data_files = {
        "train": "../../dataset/patch_train.jsonl",
        "validation": "../../dataset/patch_val.jsonl"
    }
    raw_datasets = load_dataset("json", data_files=data_files)
    model = AutoModelForSequenceClassification.from_pretrained(
        "meta-llama/Llama-Guard-3-1B",
        num_labels=2
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    num_proc = max(mp.cpu_count() - 1, 1)
    tokenized = raw_datasets.map(
        lambda examples: {
            **tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=2048,
            ),
            "labels": examples["label"],
        },
        batched=True,
        num_proc=num_proc,
        remove_columns=["text"],
        desc=f"Tokenizing with {num_proc} processes",
    )

    training_args = TrainingArguments(
        output_dir=output_dir, 
        learning_rate=2e-5,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=3,
        weight_decay=0.01,
        eval_strategy="steps",
        eval_steps=1000,
        save_steps=1000,
        logging_steps=50,
        gradient_accumulation_steps=2,
        fp16=True,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        warmup_steps=500,        
        report_to=["wandb"] if USE_WANDB else ["none"],
        run_name="LlamaGuard_FFT",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    torch.cuda.empty_cache()
    trainer.train()
    trainer.save_model("../results/LlamaGuard_FFT_Best_model")
    torch.cuda.empty_cache()

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True) 
    main()