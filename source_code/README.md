# 📘 AI-in-the-Loop

This repository contains **code, datasets, and experiments** for research on **AI-in-the-Loop safety evaluation and instruction tuning**.
It focuses on *scam-baiting conversations*, *federated instruction tuning with differential privacy*, and *benchmarking LLM responses for safety and robustness*.

---

## 🚀 Getting Started

### 1️⃣ Load the Dataset

```python
from data import load_scam_dataset, load_from_local

# Download + save dataset
load_scam_dataset("BothBosu/youtube-scam-conversations", "./ytsc_dataset")

# Load from local
dataset = load_from_local("./ytsc_dataset")
print(dataset['train'][0])
```

### 2️⃣ `Preparing dataset for instruction tuning`

This script "dataset_convert.py" converts scam classification datasets into **instruction-tuning** and **chat-based formats**.
It also supports building multi-task evaluation datasets and preparing scam-baiting dialogues for model training.

---

### Main functions

- **`create_multi_task_instruction_data(data_dir)`**
  Combines multiple classification datasets (MASC, SASC, SSD, SSC) into a multi-task evaluation dataset.
  Saves processed data in a unified format for downstream evaluation.
- **`convert_to_chat_format(data_path)`**
  Converts dialogue datasets (with role identifiers like *caller/receiver, suspect/innocent, user/agent*) into structured chat format.
  Returns a list of conversations with `user` and `assistant` roles.
- **`convert_to_instruction_tuning(data, idx=0)`**Converts dialogue datasets into instruction-tuning JSONL format with fields:

  - `id`
  - `input` (dialogue reformatted into user/agent turns)
  - `output` (scam label)
- **`convert_scam_bait_to_chat_format(data_path)`**
  Converts scam-baiting email datasets into structured `scammer` / `baiter` dialogues.
- **`create_scam_bait_trainset(prior, output)`**
  Builds scam-baiting training prompts where the model plays the role of the **baiter** against a scammer.

### Run an example

```bash
python dataset_convert.py
```

### 3️⃣ `Fine-tuning LLMs with LoRA for instruction tuning`

This script `"llm_instruction_tuning.py"` fine-tunes large language models (LLMs) using **LoRA (Low-Rank Adaptation)** for scam-related **instruction-tuning tasks**.
It supports preprocessing datasets, training with quantization (QLoRA), and evaluating the fine-tuned model.

---

### Main functions

- **`preprocess_dataset(base_model, dataset)`**

  - Tokenizes and formats the dataset for causal language modeling.
  - Creates prompts with `### Instruction`, `### Input`, and `### Response` structure.
  - Splits the dataset into train/test sets.
- **`train_model(base_model, dataset, save_path=None)`**

  - Loads a pre-trained model with QLoRA quantization.
  - Applies **LoRA adapters** on attention projection layers.
  - Uses Hugging Face `SFTTrainer` to fine-tune the model.
  - Saves the fine-tuned model and tokenizer to disk.
- **`parse_model_output(output_text)`**

  - Extracts structured information from generated outputs, e.g.:
    - Engagement score
    - PII risk score
    - Whether PII is present
    - Types of PII detected
- **`eval_model(dataset, pretrained_path)`**

  - Loads a fine-tuned model and runs **text generation** evaluation.
  - Formats prompts for evaluation and prints model responses.

---

### Run an example

```bash
python llm_instruction_tuning.py
```

### 4️⃣ `Transformer-based model tuning for scam classification`

This script `"transformer_model_tuning.py"` fine-tunes **transformer-based sequence classification models** (e.g., BERT, RoBERTa, DistilBERT) for **binary scam detection**.
It loads scam-related datasets, tokenizes them, trains models, and evaluates performance with metrics such as **Accuracy, F1, FPR, FNR, and AUPRC**.

---

### Main functions

- **`compute_metrics(eval_pred)`**
  Computes basic evaluation metrics (Accuracy, F1).
- **`compute_binary_metrics(eval_pred)`**Computes extended evaluation metrics:

  - Accuracy
  - F1 score
  - False Positive Rate (FPR)
  - False Negative Rate (FNR)
  - Area Under Precision-Recall Curve (AUPRC)
- **`prepare_batch_data(data)`**
  Converts batch input data into concatenated text format for training.

---

### Run an example

```bash
python transformer_model_tuning.py
```

### 5️⃣ `BiLSTM / BiGRU Scam Detection Training`

This script trains and evaluates a BiLSTM/BiGRU scam‑detection classifier on zero‑shot and multi‑task scam datasets.

#### Required Data Layout

From the repository root, the following paths must exist:

- `ai-in-the-loop/data/classification/all_eval_data/zero-shot/`
- `ai-in-the-loop/data/multi_task_train/multi-task_conversation_train_data.jsonl`
- `ai-in-the-loop/results/reports/classification/`  <!-- output directory -->

#### Model type (BiLSTM vs BiGRU)

 - `model_type = "bigru"  # use "bilstm" for a BiLSTM model`
 - `model_checkpoint = "bert-base-uncased"`
 - `device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")`

#### How to Run

From the repository root (the directory that contains `ai-in-the-loop/`):

```bash
python ai-in-the-loop/source_code/bilstm_bigru_fine_tuning.py
```

### 5️⃣ `Federated Instruction Tuning with LoRA for Multi-Task Scam Classification & Generation`

This script **`fed_instruction_tuning.py`** performs *federated learning (FL)* using LoRA-tuned large language models for multi-task scam-related objectives.
It simulates multiple clients, fine-tunes LoRA adapters locally, aggregates updates with FedAvg, and evaluates the global model after each round.
The tasks include **instruction tuning**, **scam-baiting generation**, **engagement scoring**, and **PII risk estimation**, all within a privacy-aware FL workflow.

---

### Main functions

- **`preprocess_dataset(base_model, dataset)`**

  - Formats the dataset using the Instruction / Input / Response structure.
  - Tokenizes examples and generates train/test splits.
- **`split_dataset_among_clients(dataset, num_clients)`**

  - Randomly partitions the dataset into multiple shards for FL simulation.
- **`train_client(local_model, client_dataset, tokenizer, training_args, lora_config)`**

  - Fine-tunes LoRA adapter weights locally on each simulated client.
  - Returns only LoRA parameters for efficient federated aggregation.
- **`federated_avg(models)`** and **`average_weights(weights_list)`**

  - Aggregates LoRA adapter weights from all clients using FedAvg.
- **`get_peft_wrapped_model(base_model_path, lora_config)`**

  - Loads a 4-bit–quantized LLM and attaches LoRA adapters for lightweight training.
- **`compute_uncertainty(dataset, model, tokenizer)`**

  - Computes entropy and log-probability metrics for generation uncertainty analysis.
- **`parse_model_output(output_text)`**

  - Extracts structured prediction fields:
    - Engagement score
    - PII risk score
    - Whether PII is present
    - Types of PII
- **`eval_model(round_num, dataset, model, tokenizer)`**

  - Generates outputs for evaluation prompts and computes engagement/PII metrics.
  - Logs evaluation results per federated round.
- **`run_federated_learning(base_model, raw_dataset, save_path, num_clients, num_rounds)`**

  - Full FL pipeline: preprocessing → client training → FedAvg → evaluation.
  - Saves LoRA-tuned global models after each round.

---

### Run an example

```bash
python fed_instruction_tuning.py
```

### To run on specific GPU and log output:

```bash
CUDA_VISIBLE_DEVICES=3 nohup python fed_instruction_tuning.py > logs/fed_multi_task.log 2>&1 &
```

### 7️⃣ `Transformer-Based Binary Classification for Scam Detection`

This script **fine-tunes transformer models** (e.g., **RoBERTa-large**, **DistilBERT-base**) for **binary scam classification**.  
It loads scam datasets, preprocesses and tokenizes text, trains multiple transformer models, and evaluates them using metrics such as **F1**, **FPR**, **FNR**, and **AUPRC**.  
These results correspond to the classification performance reported in the paper (Table 1).

---

### Main functions

- **`compute_metrics(eval_pred)`**  
  - Computes standard accuracy and F1 for quick evaluation.

- **`compute_binary_metrics(eval_pred)`**  
  - Computes detailed binary classification metrics:
    - Accuracy (Acc)  
    - F1 Score  
    - False Positive Rate (FPR)  
    - False Negative Rate (FNR)  
    - Area Under Precision-Recall Curve (AUPRC)  
  - Used as the core evaluation metric for the models.

- **Dataset loading & preprocessing**  
  - Reads scam evaluation JSONL datasets (MASC, SASC, SSC, SSD variants).  
  - Extracts scam-risk labels and flattens multi-turn evaluation data.  
  - Merges with multi-task dataset labels based on thresholded "Scam Risk Score".

- **Tokenization (`tokenize_fn`)**  
  - Uses each model’s tokenizer to prepare text sequences with truncation.

- **Model selection & training**  
  - Iterates over models such as:  
    - `roberta-large`  
    - `distilbert-base-uncased`
    - `bert-base`
    - `bert-large`

- **Evaluation & reporting**  
  - Prints detailed evaluation metrics.  
  - Generates a full **classification report** for each model.  
  - Saves fine-tuned model files to:
    ```
    ./scam-prevention/results/pre-trained/classification/
    ```

### Run an example

```bash
python transformer_model_tuning.py
```

---
# Evaluation Instructions (`eval_all.py`)

Follow the instructions below to run the complete evaluation pipeline for classification, safeness, risk awareness, and scam-baiting benchmarks.

## Step 1: Prepare Models

Ensure the following pretrained judge models and their corresponding fine-tuned checkpoints are available and correctly paired:

| Pretrained Model                          | Fine-tuned Checkpoint       |
|------------------------------------------|-----------------------------|
| `meta-llama/LlamaGuard-7b`               | `tuned-llama-guard`         |
| `meta-llama/Llama-Guard-2-8B`            | `tuned-llama-guard2`        |
| `meta-llama/Llama-Guard-3-8B`            | `tuned-llama-guard3`        |
| `OpenSafetyLab/MD-Judge-v0.1`            | `tuned-md-judge`            |

All fine-tuned checkpoints must be located at:  
`ai-in-the-loop/results/fine-tuned/multi-task/tuned-{model}`

## Step 2: Verify Required Data

Before running any evaluation, verify that the following datasets exist.

### Classification Datasets
`ai-in-the-loop/data/classification/masc_dataset/all_data.chat.json`
`ai-in-the-loop/data/classification/sasc_dataset/all_data.chat.json`
`ai-in-the-loop/data/classification/ssc_dataset/all_data.chat.json`
`ai-in-the-loop/data/classification/ssd_dataset/all_data.chat.json`

### Safeness & Risk Awareness Dataset
`ai-in-the-loop/data/generation/selected_conversation_to_scam_baiter_performance.jsonl`

### Scam-Baiting Evaluation Dataset
`ai-in-the-loop/data/generation/all_eval_data/combined_asb_sbc_ytsc_dataset.jsonl`

## Step 3: Select Evaluation Stages

Open `eval_all.py` and uncomment the evaluation functions you want to run.

- **`run_eval_for_f1_auprc_fpr_fnr()`**  
  Computes evaluation results for calculating **F1, AUPRC, FPR, and FNR** on all classification datasets.

- **`run_eval_safeness_risk_awareness()`**  
  Evaluates **scam risk awareness, PII handling, engagement behavior, and moderation**.

- **`run_eval_scam_baiting_scam_pii_engage_time()`**  
  Evaluates **scam-baiting effectiveness, PII leakage, and engagement time**.

You may run any subset or all stages together.


## Step 4: Run the Evaluation

Run the evaluation script on a GPU-enabled machine:

```bash
CUDA_VISIBLE_DEVICES=2 nohup python eval_all.py > ai-in-the-loop/logs/eval_all.log 2>&1 &
```

---

## 📜 License

This project is released under the **MIT License** (update if different).

---
