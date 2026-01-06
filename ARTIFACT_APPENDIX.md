# Artifact Appendix

Paper title: **AI-in-the-Loop: Privacy Preserving Real-Time Scam Detection and Conversational Scambaiting by Leveraging LLMs and Federated Learning**

Requested Badge(s):
- [x] **Available**
- [x] **Functional**
- [x] **Reproduced**

---
## Description

This artifact accompanies the paper:

> **Hossain, Ismail; Puppala, Sai; Alam, Jahangir; Talukder, Sajedul.**  
> *AI-in-the-Loop: Privacy Preserving Real-Time Scam Detection and Conversational Scambaiting by Leveraging LLMs and Federated Learning*  
> arXiv preprint, 2025.  
> https://arxiv.org/pdf/2509.05362

This repository provides the **complete experimental pipeline** used in the paper, including:

- Multi-task scam and safety datasets
- Instruction tuning scripts for RNNs, Transformer models, and LLM-based safety judges
- Federated instruction tuning with and without Differential Privacy (DP)
- Evaluation pipelines for scam risk, PII risk, engagement, perplexity, and safety alignment
- Analysis notebooks for reproducing tables and figures reported in the paper

The artifact enables researchers to **train, evaluate, and analyze AI-in-the-loop scam-baiting systems** under centralized, federated, and privacy-preserving settings.

---

### Security / Privacy Issues and Ethical Concerns

This artifact **does not contain malware, exploits, or system-level attacks**.

It includes:
- Scam-related and scam-baiting conversations
- Automatically generated scam-baiting responses

All datasets are **synthetic or anonymized** and provided strictly for **research and evaluation purposes**.  
No real personally identifiable information (PII) is included.

Federated learning and differential privacy experiments operate only on pre-processed datasets.  
The authors strongly discourage deploying generated scam-baiting responses in real-world systems without additional ethical review and safeguards.

---

## Basic Requirements

### Hardware Requirements

**Minimum requirements (Functional badge):**
- NVIDIA GPU with **≥ 24 GB VRAM**
- ≥ 64 GB system RAM
- ≥ 200 GB available disk space

**Hardware used for experiments reported in the paper:**
- NVIDIA A100 GPUs (40 GB VRAM)

Federated and DP experiments are **sensitive to GPU memory and runtime budget**.

---

### Software Requirements
- **Python:** 3.9 – 3.12
- **CUDA:** ≥ 11.8

All dependencies are listed in:
`requirements.txt`


Key dependencies include:
- `torch`
- `transformers`
- `datasets`
- `scikit-learn`
- `numpy`, `pandas`
- `matplotlib`, `seaborn`
- `opacus` (for Differential Privacy)

#### Machine Learning Models

The artifact uses the following models:
- LlamaGuard, LlamaGuard2, LlamaGuard3
- MD-Judge
- BERT, RoBERTa, DistilBERT variants
- BiLSTM–BiGRU models

All models are downloaded automatically from Hugging Face during execution.  
No proprietary models are required.

#### Datasets

All required datasets are included under:
`ai-in-the-loop/data/`

No external dataset download is required.

---

### Estimated Time and Storage Consumption

| Task | Time | Storage |
|---|---|---|
| Environment setup | 10–15 minutes | ~5 GB |
| Single model fine-tuning | 1–3 hours (GPU) | 5–10 GB |
| Federated tuning (30 rounds) | 6–12 hours | 50–80 GB |
| Evaluation scripts | 1–2 hours | <5 GB |
| Analysis notebooks | variable | - |

---

## Environment

### Accessibility

The artifact is publicly available at:

`https://github.com/supreme-lab/ai-in-the-loop`


The `main` branch contains the latest version used for artifact evaluation.

---

### Set up the Environment

```bash
git clone https://github.com/supreme-lab/ai-in-the-loop.git
cd ai-in-the-loop
```

#### Conda setup (if not installed)

If conda is not available on your machine, install **Miniconda** and create an environment before running the scripts.

1. **Install Miniconda (Linux/macOS, terminal)**

```bash
# Download the latest Miniconda installer
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh

# Run the installer in batch mode
bash ~/miniconda.sh -b -p $HOME/miniconda

# Initialize conda for your shell
$HOME/miniconda/bin/conda init bash   # or zsh, etc.
# Restart your terminal after this step
```

```bash
conda create -n ai-in-the-loop python=3.10 -y
conda activate ai-in-the-loop
pip install -r requirements.txt
```

### Testing the Environment

To verify that the environment is correctly set up, run:

`python source_code/dataset_preparation.py`

Expected result:

Successful dataset loading and preprocessing

No missing-file or dependency errors


## Artifact Evaluation
### Main Results and Claims
#### Main Result 1: Safety-Aware Classification Performance

Instruction-tuned models significantly outperform baselines on F1, AUPRC, FPR, and FNR metrics for scam and safety classification tasks.
These results are reported in Table 2 and Figure 4 of the paper.

#### Main Result 2: Perplexity under Safety Constraints

Safety-aware instruction tuning introduces measurable perplexity trade-offs across models, quantified and compared in Table 3.

#### Main Result 3: Federated and Differentially Private Learning Effects

Federated and DP instruction tuning achieves competitive safety performance while introducing controlled accuracy degradation, as reported in Tables 6, 7, 14–16.

#### Main Result 4: Scam-Baiting Response Quality

AI-in-the-loop models generate scam-baiting responses with improved engagement and reduced PII and scam risk, shown in Table 5 and Figure 7.


## Experiments
### Experiment 1: Classification Metrics Evaluation

Script: `eval_for_f1_auprc_fpr_fnr.py`
Supports: Table 2, Figure 4

`python source_code/eval_for_f1_auprc_fpr_fnr.py`

### Experiment 2: Perplexity Evaluation

Script: `calculate_perplexity.py`
Notebook: analyzer.ipynb

`python source_code/calculate_perplexity.py`

### Experiment 3: Scam-Baiting Evaluation

Script: `eval_scam_baiting_response_scam_pii_engage_time.py`

Supports: Table 5, Figure 7

### Experiment 4: Federated Learning Evaluation

Scripts: `fed_evaluation_dp.py, fed_evaluation_wo_dp.py`
Supports: Table 6

### Experiment 5: Safety and Risk Awareness Evaluation

Script: eval_safeness_risk_awareness.py
Supports: Tables 7, 14, 15, 16

## Limitations

Full reproduction of federated learning experiments requires high-memory GPUs

Training time and convergence may vary across hardware platforms

Some qualitative results involve stochastic generation and may differ slightly

Despite these limitations, all evaluation pipelines and scripts are fully functional and executable.

## Notes on Reusability

This artifact can be reused to:

Benchmark safety-aware instruction tuning methods

Extend federated learning with alternative privacy mechanisms

Evaluate new scam-baiting strategies

Integrate additional safety classifiers or LLM judges

The modular structure allows researchers to easily replace datasets, models, or evaluation metrics and extend the framework to new domains.