import os
import copy
import re
import json
import torch
import torch.nn.functional as F
import numpy as np
from statistics import mean, stdev
from contextlib import nullcontext

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig,
    pipeline,
)

from peft import get_peft_model, LoraConfig
from trl import SFTTrainer
from datasets import interleave_datasets

import utils  # your custom module


# ------------------------------------------------------------------
# Globals
# ------------------------------------------------------------------

PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = PARENT_DIR.rsplit("/", 2)[0]

BATCH_SIZE = 2
LOCAL_EPOCHS = 3
NUM_CLIENTS = 10
NUM_ROUNDS = 2


# ------------------------------------------------------------------
# Dataset preprocessing
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
# Client training
# ------------------------------------------------------------------

def train_client(local_model, client_dataset, tokenizer, training_args, lora_config):
    trainer = SFTTrainer(
        model=local_model,
        train_dataset=client_dataset,
        args=training_args,
        peft_config=lora_config,
    )
    trainer.model.train()

    return {
        k: v.cpu()
        for k, v in trainer.model.state_dict().items()
        if "lora_" in k
    }


def federated_avg(models):
    avg_model = copy.deepcopy(models[0])
    for key in avg_model:
        avg_model[key] = avg_model[key].to(torch.float32)
        for i in range(1, len(models)):
            avg_model[key] += models[i][key].to(torch.float32)
        avg_model[key] /= len(models)
    return avg_model


# ------------------------------------------------------------------
# Parsing utilities
# ------------------------------------------------------------------

def parse_model_output(output_text):
    result = {}

    engagement_match = re.search(r"Engagement Score:\s*([\d.]+)", output_text)
    pii_risk_match = re.search(r"PII Risk Score:\s*([\d.]+)", output_text)

    if engagement_match:
        result["engagement_score"] = float(engagement_match.group(1))
    if pii_risk_match:
        result["pii_risk_score"] = float(pii_risk_match.group(1))

    pii_match = re.search(r"Contains PII:\s*(yes|no)", output_text, re.IGNORECASE)
    if pii_match:
        result["contains_pii"] = pii_match.group(1).lower() == "yes"

    pii_types_match = re.search(r"PII Types:\s*(.+)", output_text)
    if pii_types_match:
        result["pii_types"] = [s.strip() for s in pii_types_match.group(1).split(",")]

    return result


# ------------------------------------------------------------------
# Uncertainty estimation
# ------------------------------------------------------------------

def compute_uncertainty(dataset, model, tokenizer, max_new_tokens=100):
    def eval_format_prompt(example):
        prompt = f"### Instruction:\n{example['instruction']}\n\n"
        if example["input"].strip():
            prompt += f"### Input:\n{example['input']}\n\n"
        prompt += "### Response:\n"
        return {"text": prompt}

    eval_data = dataset.map(eval_format_prompt)

    results_all = []
    dtype_ctx = (
        torch.autocast("cuda", dtype=model.lm_head.weight.dtype)
        if model.lm_head.weight.dtype in (torch.bfloat16, torch.float16)
        else nullcontext()
    )

    for sample in eval_data.select(range(100)):
        prompt = sample["text"]
        tokenized = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(model.device)

        with torch.no_grad(), dtype_ctx:
            output = model.generate(
                input_ids=tokenized["input_ids"],
                max_new_tokens=max_new_tokens,
                return_dict_in_generate=True,
                output_scores=True,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        prompt_len = tokenized["input_ids"].shape[-1]
        generated_ids = output.sequences[0][prompt_len:]
        scores = output.scores

        entropies, logprobs = [], []

        for i, logits in enumerate(scores):
            probs = F.softmax(logits[0], dim=-1)
            log_probs = F.log_softmax(logits[0], dim=-1)
            entropies.append((-torch.sum(probs * log_probs)).item())
            logprobs.append(log_probs[generated_ids[i].item()].item())

        results_all.append({
            "avg_entropy": sum(entropies) / len(entropies),
            "avg_logprob": sum(logprobs) / len(logprobs),
            "token_count": len(generated_ids),
            "entropies": entropies,
            "logprobs": logprobs,
        })

    return results_all


# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------

def eval_model(round_num, dataset, model, tokenizer):
    def eval_format_prompt(example):
        prompt = f"### Instruction:\n{example['instruction']}\n\n"
        if example["input"].strip():
            prompt += f"### Input:\n{example['input']}\n\n"
        prompt += "### Response:\n"
        return {"text": prompt}

    def extract_response(text):
        marker = "### Response:"
        return text.split(marker, 1)[1].strip() if marker in text else text.strip()

    eval_data = dataset.map(eval_format_prompt)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

    engagement_scores, pii_risk_scores = [], []

    for sample in eval_data.select(range(10)):
        output = pipe(sample["text"], max_new_tokens=100, do_sample=False)[0]["generated_text"]
        parsed = parse_model_output(extract_response(output))

        if "engagement_score" in parsed:
            engagement_scores.append(parsed["engagement_score"])
        if "pii_risk_score" in parsed:
            pii_risk_scores.append(parsed["pii_risk_score"])

    eval_results = {
        "round": round_num,
        "engagement_scores": engagement_scores,
        "pii_risk_scores": pii_risk_scores,
        "engagement_mean": mean(engagement_scores) if engagement_scores else None,
        "engagement_stdev": stdev(engagement_scores) if len(engagement_scores) > 1 else None,
        "pii_risk_mean": mean(pii_risk_scores) if pii_risk_scores else None,
        "pii_risk_stdev": stdev(pii_risk_scores) if len(pii_risk_scores) > 1 else None,
    }

    save_dir = f"{PARENT_DIR}/ai-in-the-loop/results/reports/multi_task/FL/noDP"
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "eval_scores_fl_round_wise.json"), "a") as f:
        json.dump(eval_results, f, indent=2)


# ------------------------------------------------------------------
# Model wrapping
# ------------------------------------------------------------------

def get_peft_wrapped_model(base_model_path, lora_config):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="float16",
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        device_map={"": torch.cuda.current_device()},
        trust_remote_code=True,
    )

    peft_model = get_peft_model(base_model, lora_config)
    for name, param in peft_model.named_parameters():
        if "lora" in name:
            param.requires_grad = True

    return peft_model


# ------------------------------------------------------------------
# Federated learning
# ------------------------------------------------------------------

def run_federated_learning(base_model, raw_dataset, save_path,
                           num_clients=NUM_CLIENTS, num_rounds=NUM_ROUNDS):

    train_data, eval_data, tokenizer = preprocess_dataset(base_model, raw_dataset)
    client_datasets = split_dataset_among_clients(train_data, num_clients)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )

    global_model = get_peft_wrapped_model(base_model, lora_config)

    training_args = TrainingArguments(
        output_dir=f"{PARENT_DIR}/ai-in-the-loop/logs",
        num_train_epochs=LOCAL_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=8,
        learning_rate=2e-5,
        logging_steps=50,
        save_strategy="no",
        bf16=True,
        tf32=True,
        report_to="none",
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False,
    )

    for rnd in range(num_rounds):
        print(f"\n--- Federated Round {rnd + 1} ---")
        local_weights = []

        for i, client_data in enumerate(client_datasets):
            print(f"Client {i + 1} training...")
            local_model = copy.deepcopy(global_model)
            client_weights = train_client(
                local_model, client_data, tokenizer, training_args, lora_config
            )
            local_weights.append(client_weights)

        averaged_weights = federated_avg(local_weights)
        global_model.load_state_dict(averaged_weights, strict=False)

        print("### Starting Global Evaluation")
        eval_model(rnd, eval_data, global_model, tokenizer)

        global_model.save_pretrained(save_path + f"-round_{rnd + 1}")
        tokenizer.save_pretrained(save_path + f"-round_{rnd + 1}")


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def main():
    DATA_PATH = f"{PARENT_DIR}/ai-in-the-loop/data/multi_task_train/multi-task_conversation_train_data.jsonl"
    BAITER_DATA_PATH = f"{PARENT_DIR}/ai-in-the-loop/data/multi_task_train/combined_scam_baiting_turns_train.jsonl"

    MODEL_NAME = "OpenSafetyLab/MD-Judge-v0.1"
    PRETRAINED_PATH = f"{PARENT_DIR}/ai-in-the-loop/results/fine-tuned/multi-task/FL/noDP/tuned-md-judge"

    ds1 = utils.load_jsonl_dataset(DATA_PATH)
    ds2 = utils.load_dataset_plain_jsons(BAITER_DATA_PATH)

    dataset_merged = interleave_datasets(
        [ds1, ds2],
        stopping_strategy="all_exhausted",
        seed=42,
    )

    splits = dataset_merged.train_test_split(test_size=0.1, seed=42)

    print("### Length of the Train Dataset:", len(dataset_merged))
    print("### Length of the Eval Dataset:", len(splits["test"]))

    print("### Starting Federated Training")
    # Passing only first 10 samples, if you pass only dataset_merged instead of selected_samples that will train the entire dataset
    selected_samples = dataset_merged.select(range(100))
    run_federated_learning(MODEL_NAME, selected_samples, PRETRAINED_PATH)
    print("### Federated Training Complete")

if __name__ == "__main__":
    main()

# CUDA_VISIBLE_DEVICES=1 nohup python fed_instruction_tuning.py > ai-in-the-loop/logs/fed_multi_task.log 2>&1 &
