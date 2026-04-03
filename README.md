# 📘 AI-in-the-Loop

This repository contains **code, datasets, and experiments** for research on **AI-in-the-Loop safety evaluation and instruction tuning**.
It focuses on *scam-baiting conversations*, *federated instruction tuning with differential privacy*, and *benchmarking LLM responses for safety and robustness*.

**Demo:**
[▶ Watch Demo](demo.webm)

<video src="demo.mp4" controls width="100%"></video>

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
Please go through the ARTIFACT_APPENDIX.md file to get the python package installation packages through conda environment

### 3️⃣ Run an example

```bash
cd source_code
CUDA_VISIBLE_DEVICES=0 nohup python run_all.py > <parent_path>/ai-in-the-loop/logs/full_pipeline.log 2>&1 &
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
### Before and After the Evaluation
When you fine-tune RNN models (lstm, gru), Transformer models (BERT variants), LLMs (LlamaGurad, LlamaGuard2, LlamaGuard3 and MD-Judge) the fine-tuned models will be
saved under the `results/` directory. Your can specify the folder under this directory based on the model type.
For multi task instruction tuning when no federated learning applied just running `llm_instruction_tuning.py` the fine-tuned model will be saved e.g. `tuned-llama-guard2`, `tuned-llama-guard3` and so on. When it is baiter model it would be like- `tuned-llama-guard2-baiter`, `tuned-llama-guard3-baiter` and so on.
These models then used for evaluation to measure the performance with respect to Engagement Score, PII risk score, Scam score and generating scam-baiting response.

When you run the `fed_dp_instruction_tuning.py` and `fed_instruction_tuning.py`, the only difference is that with differential privacy and without differential privacy. The fine-tuned models are the saved during the rounds of model update, so for the 30 rounds there will be total of 30 checkpoints of fine-tuned models.
These models were then used to evalaute and show the round-wise model performance.

When you have all fine-tuned models, use them for evaluation. You should check the below how the results (Figures/Tables) are mapped with the evalutions' scripts.
You will have the idea how to do that. The directories `ai-in-the-loop/data/classification/all_eval_data` or `ai-in-the-loop/data/generation/all_eval_data` are used for evaluation. The script `selected_conversation_to_scam_baiter_performance.jsonl` is used to evaluated when conversations are selected randomly to show the results of different metrics and comparisions of the fined-tuned models' performance.

When evaluation is done! Then the results are used to generate figures and tabular data for the paper. So, for that `analyzer.ipynb` notebook is used for all type of data visualization, plot and tabular result generation.

---

### 📌 Script–Table/Figure Mapping

### **1️⃣ `eval_for_f1_auprc_fpr_fnr.py`**

**Used for:** *Table 2, Figure 4*
This script computes core classification metrics including F1, AUPRC, FPR, and FNR across all model variants. It evaluates performance on multi-task scam datasets and outputs both tabular summary statistics (Table 2) and plots/curves used to produce Figure 4.

Later on the python script for calculating F1 score, AUPRC, FPR, FNR was written based on the generated data by `eval_for_f1_auprc_fpr_fnr.py`

---

### **2️⃣ `calculate_perplexity.py` + `analyzer.ipynb`**

**Used for:** *Table 3*
The perplexity script computes token-level log-likelihood and perplexity scores for all evaluated models. The notebook aggregates the results, formats them, and generates the final table summarizing model perplexity comparisons.

---

### **3️⃣ `eval_scam_baiting_response_scam_pii_engage_time.py`**

**Used for:** *Table 5, Figure 7*
This script evaluates scam-baiting responses for engagement quality, PII risk, task success, and response time. It computes numerical measures (Table 5) and produces the distributions/plots used to generate Figure 7.

---

### **4️⃣ `qualitative_evaluation.py` + `fed_evaluation_dp.py` + `fed_evaluation_wo_dp.py`**

**Used for:** *Table 6*
These scripts handle qualitative scoring and round-wise evaluation of conversational outputs under DP and non-DP federated learning. They extract example responses, safety attributes, and qualitative judgments that form the entries shown in Table 6.

---

### **5️⃣ `eval_safeness_risk_awareness.py`**

**Used for:** *Table 7, Table 14, Table 15, Table 16*
This script evaluates safety alignment, risk awareness, harmfulness avoidance, and refusal patterns across models. It computes the numerical safety metrics populating Tables 7, 14, 15, and 16.

---

### **6️⃣ `analyzer.ipynb`**

**Used for:** *Table 8, Table 9, Figure 10*
The analysis notebook loads evaluation outputs, performs data aggregation, and generates summary tables. It also produces the visualization used to construct Figure 10.

---

### **7️⃣ `grid_search.ipynb`**

**Used for:** *Figure 9*
This notebook runs hyperparameter grid search experiments (e.g., thresholds, utility weights) and visualizes their performance impact. The resulting heatmaps/plots form the basis for Figure 9.

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
