
import torch
import os, json, torch
from statistics import mean, stdev
import copy
import re
import torch.nn.functional as F
import numpy as np
from statistics import mean, stdev
from contextlib import nullcontext
from datasets import Dataset
from torch.utils.data import Subset, DataLoader
from opacus.data_loader import DataLoader as DPDataLoader
from opacus.utils.batch_memory_manager import BatchMemoryManager
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig,
    pipeline
)
from peft import LoraConfig
from trl import SFTTrainer
from transformers import LlamaForCausalLM
from peft import get_peft_model
from opacus import PrivacyEngine
from torch.utils.data.dataloader import default_collate
import utils  # your custom module

from torch.utils.data import DataLoader
from torch.optim import AdamW
from opacus import PrivacyEngine
from torch.utils.data import DataLoader, Dataset as TorchDataset
from dataclasses import dataclass
from tqdm import tqdm
from datasets import interleave_datasets

BATCH_SIZE = 2
LOCAL_EPOCHS = 3
NUM_CLIENTS = 10
NUM_ROUNDS = 30

@dataclass
class DPConfig:
    use_dp: bool = True
    noise_multiplier: float = 0.1   # tune this
    max_grad_norm: float = 1.0
    delta: float = 1e-5              # ~1/num_records
    batch_size: int = 2              # per-device batch size for DP

def get_peft_wrapped_model(base_model_path, lora_config, use_dp: bool):
    if use_dp:
        # No 4-bit quantization; Opacus needs standard grads
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

    # Train only LoRA params
    for name, p in peft_model.named_parameters():
        p.requires_grad = ("lora_" in name)

    return peft_model

def _collate(features):
    # Your dataset already has input_ids, attention_mask, labels as lists/arrays
    batch = {}
    for k in ["input_ids", "attention_mask", "labels"]:
        batch[k] = torch.tensor([f[k] for f in features], dtype=torch.long)
    return batch


class HFTensorTripletDataset(TorchDataset):
    """
    Wraps a HuggingFace dataset row into a tuple of tensors:
      (input_ids, attention_mask, labels)
    Ensures Opacus sees only tensors, no dicts/ints.
    """
    def __init__(self, hf_ds):
        self.ds = hf_ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        # Assumes preprocess_dataset() created these fields with fixed length
        return (
            torch.tensor(item["input_ids"], dtype=torch.long),
            torch.tensor(item["attention_mask"], dtype=torch.long),
            torch.tensor(item["labels"], dtype=torch.long),
        )

def train_client_dp(local_model, client_dataset, dp_cfg: DPConfig, lr=2e-5, epochs=3, device=None):
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    local_model = local_model.to(device)
    local_model.train()

    # Only LoRA params get grads
    params = [p for p in local_model.parameters() if p.requires_grad]
    optimizer = AdamW(params, lr=lr)

    # ✅ Wrap HF dataset so each sample is a tuple of tensors
    tensor_ds = HFTensorTripletDataset(client_dataset)

    # IMPORTANT: drop_last=True for Poisson-like batching stability
    loader = DataLoader(
        tensor_ds,
        batch_size=dp_cfg.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True if device.type == "cuda" else False,
    )

    # Attach Opacus privacy engine
    privacy_engine = PrivacyEngine()
    local_model, optimizer, loader = privacy_engine.make_private(
        module=local_model,
        optimizer=optimizer,
        data_loader=loader,
        noise_multiplier=dp_cfg.noise_multiplier,
        max_grad_norm=dp_cfg.max_grad_norm,
        poisson_sampling=False,   # ✅ no empty batches
    )

    for _ in tqdm(range(epochs), desc="Training Fed with DP"):
        for input_ids, attention_mask, labels in loader:
            # Skip pathological empty batches (shouldn’t happen with poisson_sampling=False, but safe)
            if input_ids.numel() == 0 or input_ids.size(0) == 0:
                continue

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            out = local_model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels)
            loss = out.loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()


    # (Optional) report ε
    try:
        eps = privacy_engine.get_epsilon(delta=dp_cfg.delta)
        print(f"[DP] ε = {eps:.2f} (δ={dp_cfg.delta}, σ={dp_cfg.noise_multiplier}, C={dp_cfg.max_grad_norm})")
    except Exception as e:
        print("[DP] Could not compute ε:", e)

    # Return LoRA adapter weights only
    return {k: v.detach().cpu() for k, v in local_model.state_dict().items() if "lora_" in k}

def average_weights(weights_list):
    """
    FedAvg for LoRA-only state_dicts.
    - Intersects keys across clients (in case a client missed a key).
    - Sums in float32 on CPU for numerical stability.
    """
    if not weights_list:
        raise ValueError("average_weights(): weights_list is empty")

    # Intersect keys across all clients to be safe
    common_keys = set(weights_list[0].keys())
    for w in weights_list[1:]:
        common_keys &= set(w.keys())
    if not common_keys:
        raise ValueError("average_weights(): no common keys across client state dicts")

    avg = {}
    n = len(weights_list)
    for k in common_keys:
        acc = None
        for w in weights_list:
            t = w[k].detach().to("cpu", dtype=torch.float32)
            acc = t if acc is None else acc.add_(t)
        avg[k] = (acc / n).to(dtype=weights_list[0][k].dtype)  # cast back to original dtype
    return avg

def parse_model_output(output_text):
    result = {}
    engagement_match = re.search(r'Engagement Score:\s*([\d.]+)', output_text)
    pii_risk_match = re.search(r'PII Risk Score:\s*([\d.]+)', output_text)
    if engagement_match:
        result['engagement_score'] = float(engagement_match.group(1))
    if pii_risk_match:
        result['pii_risk_score'] = float(pii_risk_match.group(1))
    pii_match = re.search(r'Contains PII:\s*(yes|no)', output_text, re.IGNORECASE)
    if pii_match:
        result['contains_pii'] = pii_match.group(1).strip().lower() == 'yes'
    pii_types_match = re.search(r'PII Types:\s*(.+)', output_text)
    if pii_types_match:
        result['pii_types'] = [s.strip() for s in pii_types_match.group(1).split(',')]
    return result

def eval_model(round_num, dataset, model, tokenizer, save_dir="./scam-prevention/results/reports/multi_task/FL",
               max_new_tokens=100, sample_size=100):
    """
    Runs deterministic generation on a subset, parses Engagement/PII risk, and saves:
      - per-sample parsed scores
      - summary stats (mean, stdev)
    Uses model.generate() directly (no pipeline) to avoid dtype/device surprises.
    """
    os.makedirs(save_dir, exist_ok=True)

    def eval_format_prompt(example):
        prompt = f"### Instruction:\n{example['instruction']}\n\n"
        if example["input"].strip():
            prompt += f"### Input:\n{example['input']}\n\n"
        prompt += f"### Response:\n"
        return {"text": prompt}

    def extract_response(output_text):
        marker = "### Response:"
        return output_text.split(marker, 1)[1].strip() if marker in output_text else output_text.strip()

    eval_data = dataset.map(eval_format_prompt)

    # Ensure pad token exists
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    engagement_scores, pii_risk_scores = [], []
    per_sample = []

    # Choose a deterministic slice
    subset = eval_data.select(range(min(sample_size, len(eval_data))))
    model.eval()

    # Use autocast if model weights are bf16/fp16
    dtype = getattr(model.lm_head.weight, "dtype", torch.float32)
    use_autocast = dtype in (torch.bfloat16, torch.float16)
    ctx = (torch.autocast(device_type="cuda", dtype=dtype) if use_autocast else torch.no_grad())

    with torch.no_grad():
        for i, sample in enumerate(subset):
            prompt = sample["text"]
            toks = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=1024)
            toks = {k: v.to(model.device) for k, v in toks.items()}

            out = model.generate(
                **toks,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True
            )
            gen_text = tokenizer.decode(out.sequences[0], skip_special_tokens=True)
            parsed = parse_model_output(extract_response(gen_text))

            row = {"idx": i, "text": prompt, "generated": gen_text, "parsed": parsed}
            per_sample.append(row)

            if "engagement_score" in parsed:
                engagement_scores.append(parsed["engagement_score"])
            if "pii_risk_score" in parsed:
                pii_risk_scores.append(parsed["pii_risk_score"])

    # Summary
    eval_results = {
        "round": int(round_num),
        "count_engagement": len(engagement_scores),
        "count_pii_risk": len(pii_risk_scores),
        "engagement_mean": (mean(engagement_scores) if engagement_scores else None),
        "engagement_stdev": (stdev(engagement_scores) if len(engagement_scores) > 1 else None),
        "pii_risk_mean": (mean(pii_risk_scores) if pii_risk_scores else None),
        "pii_risk_stdev": (stdev(pii_risk_scores) if len(pii_risk_scores) > 1 else None),
    }

    # Save per-round detailed and summary files
    summary_path = os.path.join(save_dir, "eval_scores_fl_round_wise.jsonl")
    details_path = os.path.join(save_dir, f"eval_samples_round_{int(round_num):03d}.jsonl")

    with open(summary_path, "a") as f:
        f.write(json.dumps(eval_results) + "\n")

    with open(details_path, "w") as f:
        for r in per_sample:
            f.write(json.dumps(r) + "\n")

    print(f"[Eval] Round {round_num}: "
          f"Engagement n={len(engagement_scores)} mean={eval_results['engagement_mean']:.4f} "
          f"PII n={len(pii_risk_scores)} mean={eval_results['pii_risk_mean']:.4f}"
          if engagement_scores and pii_risk_scores else f"[Eval] Round {round_num} completed.")


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
        full_text = example["text"]
        tokenized = tokenizer(
            full_text,
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
    split_indices = np.array_split(indices, num_clients)
    return [dataset.select(list(map(int, idxs))) for idxs in split_indices]

def run_federated_learning(base_model, raw_dataset, save_path,
                           num_clients=10, num_rounds=30,
                           use_dp=True, dp_cfg: DPConfig = DPConfig()):
    

    print("### Dataset spliting for the clients.")

    train_data, eval_data, tokenizer = preprocess_dataset(base_model, raw_dataset)
    client_datasets = split_dataset_among_clients(train_data, num_clients=num_clients)

    print("### Dataset is ready for the next the traing!!")

    lora_config = LoraConfig(
        r=8, lora_alpha=16,
        target_modules=["q_proj","k_proj","v_proj","o_proj"],
        lora_dropout=0.05,
        task_type="CAUSAL_LM"
    )

    print("### Loading Global model...")
    # IMPORTANT: no 4-bit when use_dp=True
    global_model = get_peft_wrapped_model(base_model, lora_config, use_dp=use_dp)

    training_args = TrainingArguments(
        output_dir="./scam-prevention/logs",
        num_train_epochs=LOCAL_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE if not use_dp else dp_cfg.batch_size,
        gradient_accumulation_steps=1 if use_dp else 8,   # no GA for DP
        learning_rate=2e-5,
        logging_steps=50,
        save_strategy="no",
        bf16=not use_dp,                 # keep bf16 off for Opacus stability
        tf32=True,
        report_to="none",
        gradient_checkpointing=False if use_dp else True, # no checkpointing for DP
        ddp_find_unused_parameters=False,
    )

    print("### Start round iteration")
    for rnd in range(num_rounds):
        print(f"\n--- Federated Round {rnd+1} ---")
        local_weights = []

        for i, client_data in enumerate(client_datasets):
            print(f"Client {i+1} training...")
            local_model = copy.deepcopy(global_model)

            if use_dp:
                client_weights = train_client_dp(
                    local_model,
                    client_data,
                    dp_cfg=dp_cfg,
                    lr=training_args.learning_rate,
                    epochs=training_args.num_train_epochs,
                )
            # else:
            #     client_weights = train_client(local_model, client_data, tokenizer, training_args, lora_config)

            local_weights.append(client_weights)

        averaged_weights = average_weights(local_weights)
        global_model.load_state_dict(averaged_weights, strict=False)

        # print("### Starting Global Model Evaluation")
        # eval_model(rnd, eval_data, global_model, tokenizer)
        # print("### Global Evaluation Completed!!")

        global_model.save_pretrained(save_path + f"-round_{rnd+1}")
        tokenizer.save_pretrained(save_path + f"-round_{rnd+1}")

if __name__ == "__main__":
    """
        This script runs federated learning with differential privacy on a multi-task dataset.
        It uses a pre-trained model, tunes it with LoRA, and evaluates the model after each round.
        The dataset is split among clients, and each client trains its local model with DP.
        The local models are then averaged to update the global model.
        We fine-tune MD-Judge model on the multi-task dataset.
    """

    DATA_PATH = "./scam-prevention/dataset/multi-task_balanced_scam_types_data_diverse.jsonl"
    BAITER_DATA_PATH = "./scam-prevention/dataset/generation/all_train_data/scam_baiting_turns.jsonl"

    MODEL_NAME = "OpenSafetyLab/MD-Judge-v0.1" #"meta-llama/Llama-Guard-3-8B"
    PRETRAINED_PATH = "./scam-prevention/results/pre-trained/multi-task/FL/tuned-md-judge"

    ds1 = utils.load_jsonl_dataset(DATA_PATH)
    ds2 = utils.load_dataset_plain_jsons(BAITER_DATA_PATH)

    # dataset_merged = interleave_datasets([ds1, ds2], probabilities=[0.5, 0.5], seed=42)
    dataset_merged = interleave_datasets([ds1, ds2], stopping_strategy="all_exhausted", seed=42)

    splits = dataset_merged.train_test_split(test_size=0.1, seed=42)  # random half

    print("### Length of the Train Datset: ", len(dataset_merged))
    print("### Lenght of the Eval Datset: ", len(splits['test']))

    # raw_dataset = utils.load_jsonl_dataset(DATA_PATH)  # returns Dataset
    # split_dataset = raw_dataset.train_test_split(test_size=0.2, shuffle=True, seed=42)
    # train_data = split_dataset['train']
    # test_data = split_dataset['test']

    print("### Starting Federated Training")
    run_federated_learning(MODEL_NAME, dataset_merged, PRETRAINED_PATH)
    print("### Federated Training Complete")

# CUDA_VISIBLE_DEVICES=2 nohup python fed_dp_instruction_tuning.py > /scam-prevention/logs/fed_dp_multi_task.log 2>&1 &