import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from datasets import Dataset
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc, f1_score
import numpy as np
import json
import os

"""
    This script is used to train and evaluate transformer models for binary classification tasks, specifically for scam detection.
    We leverage the pre-trained models, BERT-base, BERT-large, RoBERTa-large, and DistilBERT-base, to fine-tune them on our scam detection dataset.
    It includes functions for loading datasets, tokenizing text, training models, and evaluating performance metrics.
    The evaluation results for the datasets masc, sasc, ssc and ssd are added in the paper (i.e. Table 1) with respect to F1, FPR, FNR, AUPRC.
"""

# 5. Define Metrics
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

    # Convert logits to predictions and probabilities
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
        'Acc': acc,
        'F1': f1,
        'FPR': fpr,
        'FNR': fnr,
        'AUPRC': auprc
    }

def prepare_batch_data(data):
    input_data = []
    for item in data:
        input_data.append(item['input'])
    return "\n".join(input_data)

json_data = []

data_dir = "ai-in-the-loop/data/classification/all_eval_data/zero-shot"
for dataset_name in os.listdir(data_dir):
    file_path = os.path.join(data_dir, dataset_name)
    with open(file_path, 'r') as f:
        dataset = [json.loads(line) for line in f if line.strip()]

    np.random.shuffle(dataset)
    dataset = dataset[:int(0.7 * len(dataset))]
    
    for entry in dataset:
        data = entry["eval_scam_risk"]
        label = entry['output']
        input_data = prepare_batch_data(data)
        json_data.append({'text': input_data, 'label': label})


input_file = 'ai-in-the-loop/data/multi_task_train/multi-task_conversation_train_data.jsonl'
with open(input_file, "r") as f:
    # lines = [json.loads(line.strip()) for line in f if line.strip()]
    dataset = json.load(f)

for entry in dataset:
    if 'Scam Risk Score' in entry['output']:
        label = 1 if float(entry['output'].split(":")[1].strip()) >= 0.5 else 0
        json_data.append({'text': entry['input'], 'label': label})

df = pd.DataFrame(json_data)

df['label'] = df['label'].astype(int)
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
test_dataset = Dataset.from_pandas(test_df.reset_index(drop=True))

# 2. Select Model: Choose from 'bert-large-uncased', 'roberta-large', 'distilbert-base-uncased'
# model_name = "bert-base-uncased"
for model_name in ['roberta-large', 'distilbert-base-uncased']:
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 3. Tokenization
    def tokenize_fn(example):
        return tokenizer(example["text"], truncation=True)

    train_dataset = train_dataset.map(tokenize_fn, batched=True)
    test_dataset = test_dataset.map(tokenize_fn, batched=True)

    # 4. Load Model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    # 6. Training Setup
    training_args = TrainingArguments(
        output_dir="ai-in-the-loop/logs",
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps = 2,
        num_train_epochs=3,
        logging_steps=100,
        save_steps=500,
        weight_decay=0.01,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="F1",
        logging_dir="/scam-prevention/logs",
        fp16=True,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_binary_metrics,
    )

    # 7. Train
    trainer.train()

    save_path = 'ai-in-the-loop/results/fine-tuned/classification'
    # Save final model and tokenizer
    trainer.model.save_pretrained(os.path.join(save_path, model_name+'-tuned'))
    tokenizer.save_pretrained(os.path.join(save_path, model_name+'-tuned'))

    # 8. Evaluate
    metrics = trainer.evaluate()
    print("Evaluation Metrics:", metrics)

    # Optional: Detailed Classification Report
    preds = trainer.predict(test_dataset)
    y_pred = torch.argmax(torch.tensor(preds.predictions), dim=1)
    y_true = preds.label_ids

    cm = classification_report(y_true, y_pred, digits=4)
    print("Classfication Report: ", cm)


# CUDA_VISIBLE_DEVICES=3 nohup python transformer_model_tuning.py > ai-in-the-loop/logs/transformer.log 2>&1 &
