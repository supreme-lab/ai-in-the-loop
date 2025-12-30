# 📘 AI-in-the-Loop

This repository contains **code, datasets, and experiments** for research on **AI-in-the-Loop safety evaluation and instruction tuning**.
It focuses on *scam-baiting conversations*, *federated instruction tuning with differential privacy*, and *benchmarking LLM responses for safety and robustness*.

🚨 **Threat Model:**

![1758568572392](images/README/threat-model.png)

📡 🛡️ ⚙️ **Overview of the proposed real-time scam prevention system architecture:**

![1758568612809](images/README/system-overview.png)

---

## 📂 Repository Structure

```
ai-in-the-loop/
├── data/                   
│   ├── classification/        # Data for classification tasks
│   ├── generation/            # Data for dataset generation
│   └── multi_task_train/      # Multi-task training datasets
│
├── logs/                      # Training and evaluation logs
│
├── source_code/               # Source code for experiments
│   ├── analyzer.ipynb                   # Jupyter notebook for analysis
│   ├── bilstm_bigru_fine_tuning.py      # BiLSTM-BiGRU model fine-tuning
│   ├── data.py                          # Dataset loading utilities
│   ├── dataset_preparation.py               # Dataset preprocessing & conversion
│   ├── qualitative_evaluation.py        # Qualitative evaluation of scam-baiting turns
│   ├── calculate_perplexity.py          # Calculate Perplexity for the AI response
│   ├── eval_for_f1_auprc_fpr_fnr.py     # Evaluation for F1, AURPC, FPR, FNR
│   ├── eval_safeness_risk_awareness.py‎  # PII, Engagement, and Moderation evaluation
│   ├── fed_dp_instruction_tuning.py     # Federated DP instruction tuning
│   ├── fed_evaluation_wo_dp.py          # Federated scam-baiting evaluation
│   ├── fed_instruction_tuning.py        # Federated instruction tuning
│   ├── fed_scam_baiting_dp.py           # Federated scam-bait generation
│   ├── grid_search.ipynb                # Hyperparameter search experiments
│   ├── llm_instruction_tuning.py        # LLM instruction tuning
│   ├── prompt_util.py                   # Prompt construction utilities
│   ├── sammer_scam_baiter_conversation.py # Synthetic scam-baiting convos
│   ├── eval_scam_baiting_scam_pii_engage_time.py‎        # Scam-bait response generation and evaluating scam, pii, engagement, and calculating conversation time
│   ├── transformer_model_tuning.py      # Transformer fine-tuning
│   └── utils.py                         # Helper functions
```

---

## 🚀 Getting Started

### 1️⃣ Clone the repository

```bash
git clone https://github.com/supreme-lab/ai-in-the-loop.git
cd ai-in-the-loop
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run an example

```bash
python source_code/dataset_preparation.py
```

### Data Preparation

The instruction fine-tuning datasets located in  
`ai-in-the-loop/data/multi_task_train/`  
are used directly for training.

All required training datasets are already available in this directory; therefore, **there is no need to run the `dataset_preparation.py` script**.

The directory contains two types of JSON files:

- **Multi-task conversational dataset**  
  Each JSON file contains conversational turns along with the corresponding:
  - scam risk score  
  - PII risk score  
  - engagement score  

- **Scam-baiting dataset**  
  A separate JSON file containing scam-baiting conversations.

⚠️ **Path Note:**  
All paths should be referenced starting with `ai-in-the-loop/`.  
If the paths do not resolve correctly, prepend the parent directory corresponding to the location where the GitHub repository was cloned.

---
### Instruction for Fine-tuning
All python script ends with `_tuning.py` are written for the fine-tuning. These include-
`bilstm_bigru_fine_tuning.py`, `fed_dp_instruction_tuning.py`, `fed_instruction_tuning.py`, `llm_instruction_tuning.py`, `transformer_model_tuning.py`

Before running any of these scripts please use the following command:
`CUDA_VISIBLE_DEVICES=X nohup python llm_instruction_tuning.py > /scam-prevention/logs/multi_task_tuning.log 2>&1 &`
Here, you can replace 'X' with the available cuda device and if you continue with the background process you can keep the `nohup` otherwise you can skip it and finally if
you want to see the logs, can keep `ai-in-the-loop/logs/llm_instruction_tuning.log 2>&1 &`

Note: without enough gpu space (40GB memory is required), you may face cude out of memory issue. 

### Instruction for Evaluation

All python script starts with `eval_` are written for the evaluation task. These include-
`eval_for_f1_auprc_fpr_fnr.py`,`eval_safeness_risk_awareness.py`, `eval_scam_baiting_scam_pii_engage_time.py`.
The results of evaluation task will be stored `ai-in-the-loop/results/reports/` directory. You may change it or if you see file not found error, you should create this directory first before evaluation task.

---

## 📊 Research Features

- **Scam-Baiting Datasets**

  - Legitimate scam-baiting conversations
  - Multi-task training resources for safety research
- **Federated + Differential Privacy**

  - Implementation of **federated instruction tuning**
  - Secure aggregation with Differential Privacy (DP) guarantees
- **Model Training**

  - LlamaGuard, LlamaGuard2, LlamaGuard3, and MD-Judge
  - BiLSTM-BiGRU and Transformer architectures (Bert-base, Bert-large, Distil-bert, RoBerta)
  - Hyperparameter search utilities
- **Evaluation Pipelines**

  - Multi-turn evaluation of scam-baiting responses
  - PII engagement & AI safety benchmarks

---

## 📜 Citation

If you use this repository in your research, please cite our work:

```bibtex
@inproceedings{hossain2025aiintheloop,
  title={AI-in-the-Loop: Benchmarking Scam-Baiting Conversations for Safety Evaluation},
  author={Hossain, Ismail, Puppala, Sai, Talukder, Sajedul.},
  booktitle={[Arxiv](https://arxiv.org/pdf/2509.05362)},
  year={2025}
}
```

---

## 📌 TODO

- [ ]  Add Hugging Face dataset links
- [ ]  Release pretrained checkpoints
- [ ]  Provide evaluation leaderboard

---

## 📜 License

This project is released under the **MIT License** (update if different).

---
