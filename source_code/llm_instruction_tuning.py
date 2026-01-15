from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig,
    pipeline,
)
from peft import LoraConfig
from trl import SFTTrainer
from datasets import interleave_datasets

import torch
import utils
import re
import os


# ------------------------------------------------------------------
# Globals
# ------------------------------------------------------------------

PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = PARENT_DIR.rsplit("/", 2)[0]

BATCH_SIZE = 2


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
            return_tensors="pt",
        )
        tokenized["labels"] = tokenized["input_ids"].clone()
        return {k: v.squeeze(0) for k, v in tokenized.items()}

    dataset = dataset.map(train_format_prompt)
    dataset = dataset.map(preprocess)
    dataset = dataset.train_test_split(test_size=0.2, shuffle=True, seed=42)

    return dataset["train"], dataset["test"], tokenizer


# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------

def train_model(base_model, dataset, save_path):
    train_data, eval_data, tokenizer = preprocess_dataset(base_model, dataset)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="float16",
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        device_map={"": torch.cuda.current_device()},
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )

    training_args = TrainingArguments(
        output_dir=f"{PARENT_DIR}/ai-in-the-loop/logs",
        num_train_epochs=3,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=64,
        optim="paged_adamw_8bit",
        save_strategy="steps",
        save_steps=500,
        eval_strategy="steps",
        logging_steps=100,
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        max_grad_norm=0.3,
        bf16=True,
        tf32=True,
        report_to="none",
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=eval_data,
        peft_config=lora_config,
        args=training_args,
    )

    trainer.train()

    trainer.model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)


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
# Evaluation
# ------------------------------------------------------------------

def eval_model(dataset, pretrained_path):
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

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_path,
        torch_dtype=torch.float16,
        device_map={"": torch.cuda.current_device()},
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(pretrained_path, use_fast=True)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

    for sample in eval_data.select(range(10)):
        output = pipe(
            sample["text"],
            max_new_tokens=100,
            do_sample=False,
        )[0]["generated_text"]
        print("\nGenerated:\n", extract_response(output))


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def main():
    DATA_PATH = f"{PARENT_DIR}/ai-in-the-loop/data/multi_task_train/multi-task_conversation_train_data.jsonl"
    BAITER_DATA_PATH = f"{PARENT_DIR}/ai-in-the-loop/data/multi_task_train/combined_scam_baiting_turns_train.jsonl"

    MODEL_NAME = "meta-llama/Llama-Guard-3-8B"
    PRETRAINED_PATH = f"{PARENT_DIR}/ai-in-the-loop/results/fine-tuned/multi-task/tuned-llama-guard3"

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

    print("## Starting training...")
    # Passing only first 10 samples, if you pass only dataset_merged instead of selected_samples that will train the entire dataset
    selected_samples = dataset_merged.select(range(10))
    train_model(MODEL_NAME, selected_samples, save_path=PRETRAINED_PATH)
    print("Training complete. Model saved.")

    # Optional evaluation
    # print("## Starting evaluation...")
    # eval_model(splits["test"], PRETRAINED_PATH)
    # print("Evaluation complete.")


if __name__ == "__main__":
    main()


# CUDA_VISIBLE_DEVICES=1 nohup python llm_instruction_tuning.py > ai-in-the-loop/logs/llm_instruction_tuning.log 2>&1 &
