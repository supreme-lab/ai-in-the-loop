import os
import json
import random
import numpy as np
import pandas as pd
import utils
import random

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score,
    confusion_matrix,
    average_precision_score,
)

from datasets import Dataset

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


# ------------------------------------------------------------------
# Environment & Reproducibility
# ------------------------------------------------------------------

torch.cuda.empty_cache()
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

SEED = 2
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------

PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = PARENT_DIR.rsplit("/", 2)[0]


"""
    This script is used to train and evaluate transformer models for binary classification tasks, specifically for scam detection.
    We leverage the models like BiLSTM and BiGRU to fine-tune them on our scam detection dataset.
    Evaluation results are reported with respect to F1, FPR, FNR, and AUPRC.
"""


# ------------------------------------------------------------------
# Data preparation utilities
# ------------------------------------------------------------------

def prepare_batch_data(data):
    inputs = []
    for item in data:
        inputs.append(item["input"])
    return "\n".join(inputs)


def load_training_dataframe(parent_dir: str) -> pd.DataFrame:
    json_data = []
    data_dir = f"{parent_dir}/ai-in-the-loop/data/classification/all_eval_data/zero-shot"

    for dataset_name in os.listdir(data_dir):
        file_path = os.path.join(data_dir, dataset_name)
        dataset = utils.load_json(file_path) #pd.read_json(file_path, lines=True).to_dict('records')
        if isinstance(dataset, pd.DataFrame):
            dataset = dataset.to_dict("records")

        random.shuffle(dataset)

        for entry in dataset:
            input_data = prepare_batch_data(entry["eval_scam_risk"])
            label = entry["output"]
            json_data.append({"text": input_data, "label": label})

    input_file = f"{parent_dir}/ai-in-the-loop/data/multi_task_train/multi-task_conversation_train_data.jsonl"
    dataset = utils.load_json(input_file) #pd.read_json(input_file, lines=True).to_dict('records')
    if isinstance(dataset, pd.DataFrame):
        dataset = dataset.to_dict("records")

    for entry in dataset:
        if "Scam Risk Score" in entry["output"]:
            label = 1 if float(entry["output"].split(":")[1].strip()) >= 0.5 else 0
            json_data.append({"text": entry["input"], "label": label})

    df = pd.DataFrame(json_data)
    df["label"] = df["label"].astype(int)
    return df


# ------------------------------------------------------------------
# Tokenization
# ------------------------------------------------------------------

def tokenize_datasets(df: pd.DataFrame, tokenizer):
    train_df, valid_df = train_test_split(
        df,
        test_size=0.3,
        stratify=df["label"],
        random_state=42,
    )

    train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
    valid_dataset = Dataset.from_pandas(valid_df.reset_index(drop=True))

    def tokenize_batch(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=128,
        )

    train_dataset = train_dataset.map(tokenize_batch, batched=True)
    valid_dataset = valid_dataset.map(tokenize_batch, batched=True)

    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    valid_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    return train_dataset, valid_dataset


# ------------------------------------------------------------------
# Model definition
# ------------------------------------------------------------------

class RNNClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim,
        hidden_dim,
        output_dim,
        model_type="bilstm",
        num_layers=1,
        dropout=0.3,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=tokenizer.pad_token_id,
        )

        if model_type.lower() == "bilstm":
            self.rnn = nn.LSTM(
                embed_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
                bidirectional=True,
            )
        elif model_type.lower() == "bigru":
            self.rnn = nn.GRU(
                embed_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
                bidirectional=True,
            )
        else:
            raise ValueError("model_type must be 'bilstm' or 'bigru'")

        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        outputs, _ = self.rnn(embedded)
        last_hidden = outputs[:, -1, :]
        return self.fc(self.dropout(last_hidden))


# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------

def train_model(model, train_loader, val_loader, epochs, lr, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                labels = batch["label"].to(device)
                outputs = model(input_ids)
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        acc = correct / total
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Acc: {acc:.4f}")


# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------

def evaluate_metrics(model, test_loader, device):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["label"].to(device)
            outputs = model(input_ids)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    f1 = f1_score(all_labels, all_preds)
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)
    auprc = average_precision_score(all_labels, all_preds)

    print(f"F1: {f1:.4f}, FPR: {fpr:.4f}, FNR: {fnr:.4f}, AUPRC: {auprc:.4f}")

    return {
        "F1": f1,
        "FPR": fpr,
        "FNR": fnr,
        "AUPRC": auprc,
    }


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def main():
    global tokenizer  # used inside model

    model_checkpoint = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    df = load_training_dataframe(PARENT_DIR)
    # Passing only first 10 samples, if you pass only df instead of selected_samples that will train the entire dataset
    selected_samples = df[0:500]
    train_dataset, valid_dataset = tokenize_datasets(selected_samples, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=4)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_type = "bigru" # bilstm or bigru
    print(f"-----> {model_type} is being trained now!\n")

    model = RNNClassifier(
        vocab_size=len(tokenizer),
        embed_dim=128,
        hidden_dim=64,
        output_dim=len(df["label"].unique()),
        model_type=model_type,
        num_layers=1,
        dropout=0.3,
    )

    print("## Training Started!")
    train_model(model, train_loader, valid_loader, epochs=5, lr=1e-3, device=device)
    print("## Training Finished!")

    print("## Evaluation Started!")
    data_dir = f"{PARENT_DIR}/ai-in-the-loop/data/classification/all_eval_data/zero-shot"

    for dataset_name in os.listdir(data_dir):
        file_path = os.path.join(data_dir, dataset_name)
        print("Dataset:", dataset_name)

        with open(file_path, "r") as f:
            dataset = [json.loads(line) for line in f if line.strip()]

        np.random.shuffle(dataset)

        json_data = []
        for entry in dataset:
            input_data = prepare_batch_data(entry["eval_scam_risk"])
            label = entry["output"]
            json_data.append({"text": input_data, "label": label})

        df_test = pd.DataFrame(json_data)
        df_test["label"] = df_test["label"].astype(int)
        _, test_df = train_test_split(
            df_test,
            test_size=0.3,
            stratify=df_test["label"],
            random_state=42,
        )

        test_dataset = Dataset.from_pandas(test_df.reset_index(drop=True))
        test_dataset = test_dataset.map(
            lambda x: tokenizer(
                x["text"],
                padding="max_length",
                truncation=True,
                max_length=128,
            ),
            batched=True,
        )
        test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

        test_loader = DataLoader(test_dataset, batch_size=4)
        metrics = evaluate_metrics(model, test_loader, device)
        metrics["ds_name"] = dataset_name.split("_")[0]

        save_path = f"{PARENT_DIR}/ai-in-the-loop/results/reports/classification/bigru_evaluation.json"
        with open(save_path, "a") as f:
            f.write(json.dumps(metrics) + "\n")

    print("## Evaluation Done!")


if __name__ == "__main__":
    main()
