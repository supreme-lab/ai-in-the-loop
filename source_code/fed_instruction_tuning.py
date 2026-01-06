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
    pipeline
)
from peft import LoraConfig
from trl import SFTTrainer
import utils  # Your custom data loading module
from transformers import AutoTokenizer
from peft import  get_peft_model, LoraConfig
from statistics import mean, stdev
import json
from datasets import interleave_datasets
import os
PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = PARENT_DIR.rsplit("/", 2)[0]

BATCH_SIZE = 2
LOCAL_EPOCHS = 3
NUM_CLIENTS = 10
NUM_ROUNDS = 30

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

def train_client(local_model, client_dataset, tokenizer, training_args, lora_config):
    trainer = SFTTrainer(
        model=local_model,
        train_dataset=client_dataset,
        args=training_args,
        peft_config=lora_config
    )
    trainer.model.train()
    # return copy.deepcopy(local_model.state_dict())
    return {k: v.cpu() for k, v in trainer.model.state_dict().items() if "lora_" in k}

def federated_avg(models):
    avg_model = copy.deepcopy(models[0])
    for key in avg_model:
        # Convert to float32 for stable averaging
        avg_model[key] = avg_model[key].to(torch.float32)
        for i in range(1, len(models)):
            avg_model[key] += models[i][key].to(torch.float32)
        avg_model[key] /= len(models)

    return avg_model

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

def compute_uncertainty(dataset, model, tokenizer, max_new_tokens=100):
    def eval_format_prompt(example):
        prompt = f"### Instruction:\n{example['instruction']}\n\n"
        if example["input"].strip():
            prompt += f"### Input:\n{example['input']}\n\n"
        prompt += f"### Response:\n"
        return {"text": prompt}

    eval_data = dataset.map(eval_format_prompt)

    results_all = []
    dtype_ctx = (
        torch.autocast("cuda", dtype=model.lm_head.weight.dtype)
        if model.lm_head.weight.dtype in [torch.bfloat16, torch.float16]
        else nullcontext()
    )

    for sample in eval_data.select(range(100)):
        prompt = sample["text"]
        tokenized = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(model.device)

        with torch.no_grad(), dtype_ctx:
            output = model.generate(
                input_ids=tokenized['input_ids'],
                max_new_tokens=max_new_tokens,
                return_dict_in_generate=True,
                output_scores=True,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
            )

        prompt_len = tokenized['input_ids'].shape[-1]
        generated_ids = output.sequences[0][prompt_len:]
        scores = output.scores

        entropies = []
        logprobs = []

        for i, logits in enumerate(scores):
            probs = F.softmax(logits[0], dim=-1)
            log_probs = F.log_softmax(logits[0], dim=-1)
            token_entropy = -torch.sum(probs * log_probs).item()
            chosen_token_id = generated_ids[i].item()
            token_logprob = log_probs[chosen_token_id].item()
            entropies.append(token_entropy)
            logprobs.append(token_logprob)

        avg_entropy = sum(entropies) / len(entropies)
        avg_logprob = sum(logprobs) / len(logprobs)

        results_all.append({
            "avg_entropy": avg_entropy,
            "avg_logprob": avg_logprob,
            "token_count": len(generated_ids),
            "entropies": entropies,
            "logprobs": logprobs
        })

    return results_all

# (eval_model and run_federated_learning remain unchanged)
# Add full main block and save logic as needed

def eval_model(round_num, dataset, model, tokenizer):
    def eval_format_prompt(example):
        prompt = f"### Instruction:\n{example['instruction']}\n\n"
        if example["input"].strip():
            prompt += f"### Input:\n{example['input']}\n\n"
        prompt += f"### Response:\n"
        return {"text": prompt}

    def extract_response(output_text):
        response_marker = "### Response:"
        return output_text.split(response_marker, 1)[1].strip() if response_marker in output_text else output_text.strip()

    eval_data = dataset.map(eval_format_prompt)
    # model = AutoModelForCausalLM.from_pretrained(pretrained_path, torch_dtype=torch.float16, device_map={"": torch.cuda.current_device()}, trust_remote_code=True)
    # tokenizer = AutoTokenizer.from_pretrained(pretrained_path, use_fast=True)

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

    engagement_scores = []
    pii_risk_scores = []
    for sample in eval_data.select(range(100)):
        prompt = sample["text"]
        output = pipe(prompt, max_new_tokens=100, do_sample=False)[0]["generated_text"]
        print("\nPrompt:\n", prompt)
        parsed = parse_model_output(extract_response(output))
        print("Generated:\n", parsed)
        if "engagement_score" in parsed:
            engagement_scores.append(parsed["engagement_score"])
        if "pii_risk_score" in parsed:
            pii_risk_scores.append(parsed["pii_risk_score"])
    
    # print("=== Engagement Score ===")
    # print(f"Count: {len(engagement_scores)}")
    # print(f"Mean: {mean(engagement_scores):.4f}")
    # print(f"Stdev: {stdev(engagement_scores):.4f}")
    
    # print("=== PII Risk Score ===")
    # print(f"Count: {len(pii_risk_scores)}")
    # print(f"Mean: {mean(pii_risk_scores):.4f}")
    # print(f"Stdev: {stdev(pii_risk_scores):.4f}")
    # Save evaluation scores to file per round
    # round_num = os.environ.get("FED_ROUND", "unknown")
    eval_results = {
        "round": round_num,
        "engagement_scores": engagement_scores,
        "pii_risk_scores": pii_risk_scores,
        "engagement_mean": mean(engagement_scores) if engagement_scores else None,
        "engagement_stdev": stdev(engagement_scores) if len(engagement_scores) > 1 else None,
        "pii_risk_mean": mean(pii_risk_scores) if pii_risk_scores else None,
        "pii_risk_stdev": stdev(pii_risk_scores) if len(pii_risk_scores) > 1 else None,
    }
    save_dir = f"{PARENT_DIR}/ai-in-the-loop/results/reports/multi_task/FL"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"eval_scores_fl_round_wise.json")
    with open(save_path, "a") as f:
        json.dump(eval_results, f, indent=2)


def average_weights(weights_list):
    avg_weights = {}
    for key in weights_list[0].keys():
        avg_weights[key] = sum(w[key] for w in weights_list) / len(weights_list)
    return avg_weights

def get_peft_wrapped_model(base_model_path, lora_config):
    # base_model = LlamaForCausalLM.from_pretrained(base_model_path)
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
    # ✅ Explicitly enable grads for LoRA
    for name, param in peft_model.named_parameters():
        if "lora" in name:
            param.requires_grad = True

    return peft_model


def run_federated_learning(base_model, raw_dataset, save_path, num_clients=10, num_rounds=30):
    train_data, eval_data, tokenizer = preprocess_dataset(base_model, raw_dataset)
    client_datasets = split_dataset_among_clients(train_data, num_clients=NUM_CLIENTS)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        task_type="CAUSAL_LM"
    )

    # Global model (wrapped)
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

    eval_save_dir = f"{PARENT_DIR}/ai-in-the-loop/results/reports/multi_task/FL"

    for round in range(NUM_ROUNDS):
        print(f"\n--- Federated Round {round+1} ---")
        local_weights = []

        for i, client_data in enumerate(client_datasets):
            print(f"Client {i+1} training...")
            local_model = copy.deepcopy(global_model)
            # local_model = get_peft_wrapped_model(base_model, lora_config)
            client_weights = train_client(local_model, client_data, tokenizer, training_args, lora_config)
            local_weights.append(client_weights)

            # print("### Starting Client Evaluation")
            # results = compute_uncertainty(eval_data, local_model, tokenizer)
            # results["client_id"] = i + 1
            # client_eval_save_path = os.path.join(eval_save_dir, f"client_eval_uncertainty_round_{round+1}.json")
            # with open(client_eval_save_path, "a") as f:
            #     json.dump(results, f, indent=2)
            # print("### Ended Client Evaluation")

        averaged_weights = average_weights(local_weights)
        # Load averaged adapter weights into global model
        global_model.load_state_dict(averaged_weights, strict=False)

        print("### Starting Global Model Evaluation")
        # results = compute_uncertainty(eval_data, global_model, tokenizer)
        # results["round"] = i + 1
        # global_eval_save_path = os.path.join(eval_save_dir, f"eval_uncertainty_round_wise.json")
        # with open(global_eval_save_path, "a") as f:
        #     json.dump(results, f, indent=2)
        eval_model(round, eval_data, global_model, tokenizer)
        print("### Global Evaluation Completed!!")

        # Save the final global model
        global_model.save_pretrained(save_path + f"-round_{round+1}")
        tokenizer.save_pretrained(save_path + f"-round_{round+1}")

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

# ===== Main Execution =====
if __name__ == "__main__":
    """
        This script runs federated learning without differential privacy on a multi-task dataset.
        It uses a pre-trained model, tunes it with LoRA, and evaluates the model after each round.
        The dataset is split among clients, and each client trains its local model without DP.
        The local models are then averaged to update the global model.
        We fine-tune MD-Judge model on the multi-task dataset.
    """
    DATA_PATH = f"{PARENT_DIR}/ai-in-the-loop/data/multi_task_train/multi-task_conversation_train_data.jsonl"
    BAITER_DATA_PATH = f"{PARENT_DIR}/ai-in-the-loop/data/multi_task_train/combined_scam_baiting_turns_train.jsonl"

    MODEL_NAME = "OpenSafetyLab/MD-Judge-v0.1"
    PRETRAINED_PATH = f"{PARENT_DIR}/ai-in-the-loop/results/fine-tuned/multi-task/FL/noDP/tuned-md-judge"
    
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

# CUDA_VISIBLE_DEVICES=3 nohup python fed_instruction_tuning.py > /scam-prevention/logs/fed_multi_task.log 2>&1 &
