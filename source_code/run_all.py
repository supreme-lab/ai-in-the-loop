"""
Unified runner script to execute all stages of the AI-in-the-loop pipeline.

Each imported module MUST expose a `main()` function.
Execution order is explicit and reproducible.
"""

import os
import sys

# ------------------------------------------------------------------
# Ensure project root is on PYTHONPATH
# ------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)


# ------------------------------------------------------------------
# Import individual pipelines
# (names should match your actual filenames)
# ------------------------------------------------------------------

import llm_instruction_tuning          # llm_instruction_tuning.py
import fed_instruction_tuning          # fed_instruction_tuning.py (no-DP)
import fed_dp_instruction_tuning       # fed_dp_instruction_tuning.py (DP)
import transformer_model_tuning        # transformer_model_tuning.py
import bilstm_bigru_fine_tuning        # BiLSTM / BiGRU script
import dataset_preparation             # dataset conversion / prompt utils


# ------------------------------------------------------------------
# Master pipeline
# ------------------------------------------------------------------

def main():
    """
    Master execution pipeline.
    Comment/uncomment steps as needed.
    """

    print("\n================ PIPELINE START ================\n")

    # --------------------------------------------------
    # 1. Dataset conversion / prompt generation
    # --------------------------------------------------
    print("▶ Step 1: Dataset Preparation")
    dataset_preparation.main()

    # --------------------------------------------------
    # 2. LLM instruction tuning (single-node)
    # --------------------------------------------------
    print("\n▶ Step 2: LLM instruction tuning")
    llm_instruction_tuning.main()

    # --------------------------------------------------
    # 3. Federated learning (no DP)
    # --------------------------------------------------
    print("\n▶ Step 3: Federated learning (no DP)")
    fed_instruction_tuning.main()

    # --------------------------------------------------
    # 4. Federated learning (with DP)
    # --------------------------------------------------
    print("\n▶ Step 4: Federated learning (DP)")
    fed_dp_instruction_tuning.main()

    # --------------------------------------------------
    # 5. Transformer-based classification
    # --------------------------------------------------
    print("\n▶ Step 5: Transformer classification models")
    transformer_model_tuning.main()

    # --------------------------------------------------
    # 6. RNN-based classification (BiLSTM / BiGRU)
    # --------------------------------------------------
    print("\n▶ Step 6: RNN classification models")
    bilstm_bigru_fine_tuning.main()

    print("\n================ PIPELINE END ================\n")


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    main()


# CUDA_VISIBLE_DEVICES=1 nohup python run_all.py > ai-in-the-loop/logs/full_pipeline.log 2>&1 &
