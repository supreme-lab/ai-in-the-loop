# 📘 AI-in-the-Loop

This repository contains **code, datasets, and experiments** for research on **AI-in-the-Loop safety evaluation and instruction tuning**.
It focuses on *scam-baiting conversations*, *federated instruction tuning with differential privacy*, and *benchmarking LLM responses for safety and robustness*.

---

## 🚀 Getting Started

### 1️⃣ Load the Dataset

Datasets are provided in the repository, if you still load the dataset please follow the instruction below.

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

This script fine-tunes safety-oriented LLMs (e.g., Llama Guard and MD-Judge) on a multi-task instruction dataset using QLoRA-style 4-bit quantization and LoRA adapters.

#### High-level Behavior

The script will:

- Load two datasets:
  - `multi-task_conversation_train_data.jsonl`
  - `combined_scam_baiting_turns_train.jsonl`
- Interleave them into a single Hugging Face `Dataset` and create a 10% eval split. [web:26]
- Format samples into an instruction–input–response prompt of the form:

```text
  ### Instruction:
  {instruction}

  ### Input:
  {input}

  ### Response:
  {output}
```

```python
BATCH_SIZE = 2

MODEL_NAME = "meta-llama/Llama-Guard-3-8B"
pretrained_path = ".../results/fine-tuned/multi-task/tuned-llama-guard3"
```
Change MODEL_NAME to swap between Llama Guard variants, MD-Judge, or other causal LMs; update pretrained_path for where to save the fine-tuned checkpoint.

```python
DATA_PATH = ".../data/multi_task_train/multi-task_conversation_train_data.jsonl"
BAITER_DATA_PATH = ".../data/multi_task_train/combined_scam_baiting_turns_train.jsonl"
```

```python
DATA_PATH = ".../data/multi_task_train/multi-task_conversation_train_data.jsonl"
BAITER_DATA_PATH = ".../data/multi_task_train/combined_scam_baiting_turns_train.jsonl"
```

### How to Run
#### From the repository root:

```bash
python ai-in-the-loop/source_code/llm_instruction_tuning.py
```

#### To train on a specific GPU and log to file (example: GPU 3):

```bash
CUDA_VISIBLE_DEVICES=3 nohup python ai-in-the-loop/source_code/llm_instruction_tuning.py \
  > ai-in-the-loop/logs/llm_instruction_tuning.log 2>&1 &
```



### 4️⃣ `Transformer-based Scam Detection Training`
This script fine-tunes multiple transformer models for binary scam detection using zero-shot evaluation datasets and a multi-task conversation dataset.

#### Script Behavior

The script will:

- Load and subsample all zero-shot scam evaluation datasets from  
  `ai-in-the-loop/data/classification/all_eval_data/zero-shot`.
- Aggregate conversation data with "Scam Risk Score" annotations from  
  `ai-in-the-loop/data/multi_task_train/multi-task_conversation_train_data.jsonl`.
- Build a binary classification dataframe with `text` and integer `label` fields.
- Split the data into train and test sets with stratification on the label.
- For each model in `['bert-base-uncased', 'roberta-large', 'distilbert-base-uncased']`:
  - Tokenize text with truncation and max length 512, removing `token_type_ids` for RoBERTa.
  - Fine-tune a sequence classification head for 2-way classification.
  - Compute Acc, F1, FPR, FNR, and AUPRC on the test set.
  - Print evaluation metrics and a detailed classification report.

#### How to Run

From the repository root (the directory that contains `ai-in-the-loop/`):

```bash
python ai-in-the-loop/source_code/transformer_model_tuning.py
```

#### To run on a specific GPU, you can use:

```bash
CUDA_VISIBLE_DEVICES=2 python ai-in-the-loop/source_code/transformer_model_tuning.py
```

Important Parameters and Paths
 - Data sources

    - Zero-shot evaluation data directory:
ai-in-the-loop/data/classification/all_eval_data/zero-shot

    - Multi-task conversation data file:
ai-in-the-loop/data/multi_task_train/multi-task_conversation_train_data.jsonl

 - Model selection loop

```bash
for model_name in ['bert-base-uncased', 'roberta-large', 'distilbert-base-uncased']:
    ...
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

This script runs federated learning without differential privacy on a multi-task instruction dataset using the **MD-Judge-v0.1** safety guard model as the base LLM.

#### High-level Behavior

The script will:

- Load two multi-task datasets:
  - `multi-task_conversation_train_data.jsonl`
  - `combined_scam_baiting_turns_train.jsonl`
- Interleave them into a single Hugging Face `Dataset` and create a small eval split.
- Format each sample into an instruction–input–response chat template and tokenize up to 1024 tokens.
- Split the training set across `NUM_CLIENTS` clients and run `NUM_ROUNDS` of federated LoRA fine-tuning using 4-bit quantization (`BitsAndBytesConfig`) and `SFTTrainer`.
- Aggregate LoRA adapter weights with FedAvg after each round and update a global model.
- After every round, evaluate the global model as a judge, extracting:
  - Engagement Score
  - PII Risk Score
  - Contains PII (yes/no) and PII types
- Append per-round evaluation statistics (mean, stdev of scores) to:
  `ai-in-the-loop/results/reports/multi_task/FL/eval_scores_fl_round_wise.json`.

#### Key Script Parameters

- **Federated setup**

```python
BATCH_SIZE = 2
LOCAL_EPOCHS = 3
NUM_CLIENTS = 10
NUM_ROUNDS = 30

MODEL_NAME = "OpenSafetyLab/MD-Judge-v0.1"
DATA_PATH = ".../multi-task_conversation_train_data.jsonl"
BAITER_DATA_PATH = ".../combined_scam_baiting_turns_train.jsonl"
PRETRAINED_PATH = ".../results/fine-tuned/multi-task/FL/noDP/tuned-md-judge"
```

#### How to Run
From the repository root:

```bash
python ai-in-the-loop/source_code/fed_instruction_tuning.py
```
To fix the GPU and run in the background (example for GPU 3):

```bash
CUDA_VISIBLE_DEVICES=3 nohup python ai-in-the-loop/source_code/fed_instruction_tuning.py \
  > ai-in-the-loop/logs/fed_multi_task.log 2>&1 &
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
