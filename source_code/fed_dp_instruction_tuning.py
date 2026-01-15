import torch
import os, json
import copy
import re
import numpy as np
from statistics import mean, stdev
from dataclasses import dataclass
from tqdm import tqdm

from torch.utils.data import DataLoader, Dataset as TorchDataset
from torch.optim import AdamW

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig
)

from peft import LoraConfig, get_peft_model
from opacus import PrivacyEngine
from datasets import interleave_datasets

import utils  # your custom module


# ------------------------------------------------------------------
# Global constants
# ------------------------------------------------------------------

PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = PARENT_DIR.rsplit("/", 2)[0]

BATCH_SIZE = 2
LOCAL_EPOCHS = 3
NUM_CLIENTS = 10
NUM_ROUNDS = 2


# ------------------------------------------------------------------
# Configs
# ------------------------------------------------------------------

@dataclass
class DPConfig:
    use_dp: bool = True
    noise_multiplier: float = 0.1
    max_grad_norm: float = 1.0
    delta: float = 1e-5
    batch_size: int = 2


# ------------------------------------------------------------------
# Model utilities
# ------------------------------------------------------------------

def get_peft_wrapped_model(base_model_path, lora_config, use_dp: bool):
    if use_dp:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map={"": torch.cuda.current_device()},
            trust_remote_code=True
        )
    else:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="float16"
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
            device_map={"": torch.cuda.current_device()},
            trust_remote_code=True
        )

    peft_model = get_peft_model(base_model, lora_config)

    for name, p in peft_model.named_parameters():
        p.requires_grad = ("lora_" in name)

    return peft_model


# ------------------------------------------------------------------
# Dataset helpers
# ------------------------------------------------------------------

class HFTensorTripletDataset(TorchDataset):
    def __init__(self, hf_ds):
        self.ds = hf_ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        return (
            torch.tensor(item["input_ids"], dtype=torch.long),
            torch.tensor(item["attention_mask"], dtype=torch.long),
            torch.tensor(item["labels"], dtype=torch.long),
        )


# ------------------------------------------------------------------
# Training (DP)
# ------------------------------------------------------------------

def train_client_dp(local_model, client_dataset, dp_cfg: DPConfig,
                    lr=2e-5, epochs=3, device=None):

    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    local_model = local_model.to(device)
    local_model.train()

    params = [p for p in local_model.parameters() if p.requires_grad]
    optimizer = AdamW(params, lr=lr)

    tensor_ds = HFTensorTripletDataset(client_dataset)

    loader = DataLoader(
        tensor_ds,
        batch_size=dp_cfg.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=(device.type == "cuda"),
    )

    privacy_engine = PrivacyEngine()
    local_model, optimizer, loader = privacy_engine.make_private(
        module=local_model,
        optimizer=optimizer,
        data_loader=loader,
        noise_multiplier=dp_cfg.noise_multiplier,
        max_grad_norm=dp_cfg.max_grad_norm,
        poisson_sampling=False,
    )

    for _ in tqdm(range(epochs), desc="Training Fed with DP"):
        for input_ids, attention_mask, labels in loader:
            if input_ids.numel() == 0:
                continue

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            out = local_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = out.loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    try:
        eps = privacy_engine.get_epsilon(delta=dp_cfg.delta)
        print(f"[DP] ε = {eps:.2f}")
    except Exception as e:
        print("[DP] Could not compute ε:", e)

    return {k: v.detach().cpu() for k, v in local_model.state_dict().items() if "lora_" in k}


# ------------------------------------------------------------------
# FedAvg
# ------------------------------------------------------------------

def average_weights(weights_list):
    if not weights_list:
        raise ValueError("weights_list is empty")

    common_keys = set(weights_list[0].keys())
    for w in weights_list[1:]:
        common_keys &= set(w.keys())

    avg = {}
    n = len(weights_list)

    for k in common_keys:
        acc = None
        for w in weights_list:
            t = w[k].to("cpu", dtype=torch.float32)
            acc = t if acc is None else acc.add_(t)
        avg[k] = (acc / n).to(dtype=weights_list[0][k].dtype)

    return avg


# ------------------------------------------------------------------
# Preprocessing
# ------------------------------------------------------------------

def preprocess_dataset(base_model, dataset):
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    def train_format_prompt(example):
        prompt = f"### Instruction:\n{example['instruction']}\n\n"
        if example["input"].strip():
            prompt += f"### Input:\n{example['input']}\n\n"
        prompt += f"### Response:\n{example['output']}"
        return {"text": prompt}

    def preprocess(example):
        tokenized = tokenizer(
            example["text"],
            padding="max_length",
            truncation=True,
            max_length=1024,
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    dataset = dataset.map(train_format_prompt)
    dataset = dataset.map(preprocess)
    dataset = dataset.train_test_split(test_size=0.2, shuffle=True, seed=42)
    return dataset["train"], dataset["test"], tokenizer


def split_dataset_among_clients(dataset, num_clients):
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)
    splits = np.array_split(indices, num_clients)
    return [dataset.select(list(map(int, s))) for s in splits]


# ------------------------------------------------------------------
# Federated learning driver
# ------------------------------------------------------------------

def run_federated_learning(base_model, raw_dataset, save_path,
                           num_clients=10, num_rounds=30,
                           use_dp=True, dp_cfg=DPConfig()):

    train_data, eval_data, tokenizer = preprocess_dataset(base_model, raw_dataset)
    client_datasets = split_dataset_among_clients(train_data, num_clients)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        task_type="CAUSAL_LM"
    )

    global_model = get_peft_wrapped_model(base_model, lora_config, use_dp)

    for rnd in range(NUM_ROUNDS):
        print(f"\n--- Federated Round {rnd+1} ---")
        local_weights = []

        for i, client_data in enumerate(client_datasets):
            print(f"Client {i+1} training...")
            local_model = copy.deepcopy(global_model)

            client_weights = train_client_dp(
                local_model,
                client_data,
                dp_cfg=dp_cfg,
                epochs=LOCAL_EPOCHS,
            )
            local_weights.append(client_weights)

        averaged = average_weights(local_weights)
        global_model.load_state_dict(averaged, strict=False)

        global_model.save_pretrained(save_path + f"-round_{rnd+1}")
        tokenizer.save_pretrained(save_path + f"-round_{rnd+1}")


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def main():
    DATA_PATH = f"{PARENT_DIR}/ai-in-the-loop/data/multi_task_train/multi-task_conversation_train_data.jsonl"
    BAITER_DATA_PATH = f"{PARENT_DIR}/ai-in-the-loop/data/multi_task_train/combined_scam_baiting_turns_train.jsonl"

    MODEL_NAME = "OpenSafetyLab/MD-Judge-v0.1"
    PRETRAINED_PATH = f"{PARENT_DIR}/ai-in-the-loop/results/fine-tuned/multi-task/FL/DP/tuned-md-judge"

    ds1 = utils.load_jsonl_dataset(DATA_PATH)
    ds2 = utils.load_dataset_plain_jsons(BAITER_DATA_PATH)

    dataset_merged = interleave_datasets(
        [ds1, ds2],
        stopping_strategy="all_exhausted",
        seed=42
    )

    print("### Starting Federated Training")
    # Passing only first 10 samples, if you pass only dataset_merged instead of selected_samples that will train the entire dataset
    selected_samples = dataset_merged.select(range(100))
    run_federated_learning(MODEL_NAME, selected_samples, PRETRAINED_PATH)
    print("### Federated Training Complete!!!")


if __name__ == "__main__":
    main()


# CUDA_VISIBLE_DEVICES=1 nohup python fed_dp_instruction_tuning.py > ai-in-the-loop/logs/fed_dp_multi_task.log 2>&1 &