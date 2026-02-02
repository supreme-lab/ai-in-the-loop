import pandas as pd
import torch
import numpy as np
import json
import os
import utils
import random
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    auc,
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = PARENT_DIR.rsplit("/", 2)[0]

"""
    This script is used to train and evaluate transformer models for binary classification tasks, specifically for scam detection.
    We leverage the pre-trained models, BERT-base, BERT-large, RoBERTa-large, and DistilBERT-base, to fine-tune them on our scam detection dataset.
    It includes functions for loading datasets, tokenizing text, training models, and evaluating performance metrics.
    The evaluation results for the datasets masc, sasc, ssc and ssd are added in the paper (i.e. Table 1) with respect to F1, FPR, FNR, AUPRC.
"""

# ------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = torch.argmax(torch.tensor(logits), dim=1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    return {"accuracy": acc, "f1": f1}


def compute_binary_metrics(eval_pred):
    """
    Computes Acc, F1, FPR, FNR, AUPRC.

    Args:
        eval_pred: tuple of (logits, labels)

    Returns:
        dict: Dictionary with evaluation metrics
    """
    logits, labels = eval_pred
    labels = np.array(labels)

    probs = torch.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()  # class 1 probs
    preds = np.argmax(logits, axis=1)

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)

    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    precision, recall, _ = precision_recall_curve(labels, probs)
    auprc = auc(recall, precision)

    return {
        "Acc": acc,
        "F1": f1,
        "FPR": fpr,
        "FNR": fnr,
        "AUPRC": auprc,
    }


# ------------------------------------------------------------------
# Data helpers
# ------------------------------------------------------------------

def prepare_batch_data(data):
    input_data = []
    for item in data:
        input_data.append(item["input"])
    return "\n".join(input_data)


def load_training_dataframe(parent_dir: str) -> pd.DataFrame:
    json_data = []

    data_dir = f"{parent_dir}/ai-in-the-loop/data/classification/all_eval_data/zero-shot"
    for dataset_name in os.listdir(data_dir):
        file_path = os.path.join(data_dir, dataset_name)
        # with open(file_path, "r") as f:
        #     dataset = [json.loads(line) for line in f if line.strip()]

        dataset = utils.load_json(file_path)
        if isinstance(dataset, pd.DataFrame):
            dataset = dataset.to_dict("records")

        random.shuffle(dataset)
        dataset = dataset[: int(0.7 * len(dataset))]

        for entry in dataset:
            data = entry["eval_scam_risk"]
            label = entry["output"]
            input_data = prepare_batch_data(data)
            json_data.append({"text": input_data, "label": label})

    input_file = f"{parent_dir}/ai-in-the-loop/data/multi_task_train/multi-task_conversation_train_data.jsonl"
    # with open(input_file, "r") as f:
    #     dataset = json.load(f)
    dataset = utils.load_json(input_file)
    if isinstance(dataset, pd.DataFrame):
        dataset = dataset.to_dict("records")

    for entry in dataset:
        if "Scam Risk Score" in entry["output"]:
            label = 1 if float(entry["output"].split(":")[1].strip()) >= 0.5 else 0
            json_data.append({"text": entry["input"], "label": label})

    df = pd.DataFrame(json_data)
    df["label"] = df["label"].astype(int)
    return df


def build_hf_datasets(df: pd.DataFrame):
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df["label"],
        random_state=42,
    )
    train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
    test_dataset = Dataset.from_pandas(test_df.reset_index(drop=True))
    return train_dataset, test_dataset


# ------------------------------------------------------------------
# Training / evaluation
# ------------------------------------------------------------------

def run_model_training_loop(train_dataset: Dataset, test_dataset: Dataset, parent_dir: str):
    for model_name in ["bert-base-uncased", "roberta-large", "distilbert-base-uncased"]:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        def tokenize_fn(example):
            encodings = tokenizer(
                example["text"],
                truncation=True,
                padding=True,
                max_length=512,
            )
            if "token_type_ids" in encodings and model_name.startswith("roberta"):
                encodings.pop("token_type_ids")
            return encodings

        mapped_train = train_dataset.map(tokenize_fn, batched=True)
        mapped_test = test_dataset.map(tokenize_fn, batched=True)

        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

        training_args = TrainingArguments(
            output_dir=f"{parent_dir}/ai-in-the-loop/logs",
            eval_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            gradient_accumulation_steps=2,
            num_train_epochs=3,
            logging_steps=100,
            save_steps=500,
            weight_decay=0.01,
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="F1",
            logging_dir=f"{parent_dir}/ai-in-the-loop/logs",
            fp16=True,
            report_to="none",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=mapped_train,
            eval_dataset=mapped_test,
            processing_class=tokenizer,  # instead of tokenizer=tokenizer
            data_collator=DataCollatorWithPadding(tokenizer),
            compute_metrics=compute_binary_metrics,
        )

        trainer.train()

        save_path = f"{parent_dir}/ai-in-the-loop/results/fine-tuned/classification"
        trainer.model.save_pretrained(os.path.join(save_path, model_name + "-tuned"))
        tokenizer.save_pretrained(os.path.join(save_path, model_name + "-tuned"))

        metrics = trainer.evaluate()
        print("Evaluation Metrics:", metrics)

        preds = trainer.predict(mapped_test)
        y_pred = torch.argmax(torch.tensor(preds.predictions), dim=1)
        y_true = preds.label_ids

        cm = classification_report(y_true, y_pred, digits=4)
        print("Classfication Report: ", cm)

# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def main():
    df = load_training_dataframe(PARENT_DIR)
    # Passing only first 10 samples, if you pass only df instead of selected_samples that will train the entire dataset
    selected_samples = df[0:500]
    train_dataset, test_dataset = build_hf_datasets(selected_samples)
    run_model_training_loop(train_dataset, test_dataset, PARENT_DIR)
    print("-----> Training and Evaluation completed!!")

if __name__ == "__main__":
    main()

# CUDA_VISIBLE_DEVICES=1 nohup python transformer_model_tuning.py > ai-in-the-loop/logs/transformer.log 2>&1 &
