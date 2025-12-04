import os
import json
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import f1_score, confusion_matrix, average_precision_score
torch.cuda.empty_cache()
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
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

"""
    This script is used to train and evaluate transformer models for binary classification tasks, specifically for scam detection.
    We leverage the models like- BiLSTM, BiGRU, to fine-tune them on our scam detection dataset.
    It includes functions for loading datasets, tokenizing text, training models, and evaluating performance metrics.
    The evaluation results for the datasets masc, sasc, ssc and ssd are added in the paper (i.e. Table 1) with respect to F1, FPR, FNR, AUPRC.
"""

# -------------------------
# 1. Your Data Preparation
# -------------------------
def prepare_batch_data(data):
    input_data = []
    for item in data:
        input_data.append(item['input'])
    return "\n".join(input_data)

json_data = []
data_dir = "./scam-prevention/dataset/classification/all_eval_data/zero-shot"

# Load zero-shot datasets
for dataset_name in os.listdir(data_dir):
    file_path = os.path.join(data_dir, dataset_name)
    with open(file_path, 'r') as f:
        dataset = [json.loads(line) for line in f if line.strip()]

    np.random.shuffle(dataset)
    # dataset = dataset[:int(0.7 * len(dataset))]

    for entry in dataset:
        data = entry["eval_scam_risk"]
        label = entry['output']
        input_data = prepare_batch_data(data)
        json_data.append({'text': input_data, 'label': label})

# Load multi-task dataset
input_file = "./scam-prevention/dataset/multi-task_balanced_scam_types_data_diverse.jsonl"
with open(input_file, "r") as f:
    dataset = json.load(f)

for entry in dataset:
    if 'Scam Risk Score' in entry['output']:
        label = 1 if float(entry['output'].split(":")[1].strip()) >= 0.5 else 0
        json_data.append({'text': entry['input'], 'label': label})

# DataFrame + split
df = pd.DataFrame(json_data)
df['label'] = df['label'].astype(int)
train_df, valid_df = train_test_split(df, test_size=0.3, stratify=df['label'], random_state=42)

# Convert to HuggingFace datasets
train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
valid_dataset = Dataset.from_pandas(valid_df.reset_index(drop=True))

# -------------------------
# 2. Tokenization
# -------------------------
model_checkpoint = "bert-base-uncased"  # can switch to other tokenizers
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def tokenize_batch(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

train_dataset = train_dataset.map(tokenize_batch, batched=True)
valid_dataset = valid_dataset.map(tokenize_batch, batched=True)

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
valid_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# -------------------------
# 3. PyTorch Dataset Loader
# -------------------------
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=4)

# -------------------------
# 4. BiLSTM / BiGRU Model
# -------------------------
class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, model_type="bilstm", num_layers=1, dropout=0.3):
        super(RNNClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=tokenizer.pad_token_id)

        if model_type.lower() == "bilstm":
            self.rnn = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, 
                               batch_first=True, dropout=dropout, bidirectional=True)
        elif model_type.lower() == "bigru":
            self.rnn = nn.GRU(embed_dim, hidden_dim, num_layers=num_layers, 
                              batch_first=True, dropout=dropout, bidirectional=True)
        else:
            raise ValueError("model_type must be 'bilstm' or 'bigru'")

        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        outputs, _ = self.rnn(embedded)
        last_hidden = outputs[:, -1, :]  # last timestep
        out = self.fc(self.dropout(last_hidden))
        return out

# -------------------------
# 5. Training Loop
# -------------------------
def train_model(model, train_loader, val_loader, epochs, lr, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
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

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                labels = batch["label"].to(device)
                outputs = model(input_ids)
                predictions = torch.argmax(outputs, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Acc: {accuracy:.4f}")

# -------------------------
# 6. Run Training
# -------------------------
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

model_type = "bigru"  # change to "bigru" for GRU and bilstm for BiLSTM
model = RNNClassifier(vocab_size=len(tokenizer), embed_dim=128, hidden_dim=64, 
                      output_dim=len(df["label"].unique()), model_type=model_type, num_layers=1, dropout=0.3)

print("##Training Started!")
train_model(model, train_loader, valid_loader, epochs=5, lr=1e-3, device=device)
print("##Training Started!")

def evaluate_metrics(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # F1 Score
    f1 = f1_score(all_labels, all_preds)

    # Confusion Matrix → TN, FP, FN, TP
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()

    # False Positive Rate (FPR) = FP / (FP + TN)
    fpr = fp / (fp + tn)

    # False Negative Rate (FNR) = FN / (FN + TP)
    fnr = fn / (fn + tp)

    # AUPRC (Average Precision Score)
    auprc = average_precision_score(all_labels, all_preds)

    print(f"F1 Score: {f1:.4f}")
    print(f"False Positive Rate (FPR): {fpr:.4f}")
    print(f"False Negative Rate (FNR): {fnr:.4f}")
    print(f"AUPRC: {auprc:.4f}")

    return {
        "F1": f1,
        "FPR": fpr,
        "FNR": fnr,
        "AUPRC": auprc
    }

print("##Evaluation Started!")
# After training:

data_dir = "./scam-prevention/dataset/classification/all_eval_data/zero-shot"

# Load zero-shot datasets
for dataset_name in os.listdir(data_dir):
    file_path = os.path.join(data_dir, dataset_name)
    print("Dataset: ", dataset_name)
    with open(file_path, 'r') as f:
        dataset = [json.loads(line) for line in f if line.strip()]

    np.random.shuffle(dataset)
    # dataset = dataset[:int(0.7 * len(dataset))]
    json_data = []
    for entry in dataset:
        data = entry["eval_scam_risk"]
        label = entry['output']
        input_data = prepare_batch_data(data)
        json_data.append({'text': input_data, 'label': label})

    # DataFrame + split
    df = pd.DataFrame(json_data)
    df['label'] = df['label'].astype(int)
    _, test_df = train_test_split(df, test_size=0.3, stratify=df['label'], random_state=42)

    # Convert to HuggingFace datasets
    # train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
    test_dataset = Dataset.from_pandas(test_df.reset_index(drop=True))

    test_dataset = test_dataset.map(tokenize_batch, batched=True)
    test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    test_loader = DataLoader(test_dataset, batch_size=4)

    metrics = evaluate_metrics(model, test_loader, device)
    metrics['ds_name'] = dataset_name.split("_")[0]

    with open('./scam-prevention/results/reports/classification/bigru_evaluation.json', 'a') as f:
        f.write(json.dumps(metrics) +"\n")

print("##Evaluation Done!")
