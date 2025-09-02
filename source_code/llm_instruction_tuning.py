from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer
import torch
import utils
from transformers import pipeline
import re
from datasets import interleave_datasets

BATCH_SIZE = 2
# Load dataset
def preprocess_dataset(base_model, dataset):
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token  # Ensure no pad token issues

    # Format the prompts
    def train_format_prompt(example):
        prompt = f"### Instruction:\n{example['instruction']}\n\n"
        if example["input"].strip():
            prompt += f"### Input:\n{example['input']}\n\n"
        prompt += f"### Response:\n{example['output']}"
        return {"text": prompt}

    def preprocess(example):
        # Join prompt + target as one sequence for causal LM
        full_text = example["text"]
        
        tokenized = tokenizer(
            full_text,
            padding="max_length",
            truncation=True,
            max_length=1024,
            return_tensors="pt"
        )
        
        # Make labels same as input_ids (causal LM style)
        tokenized["labels"] = tokenized["input_ids"].clone()

        # Squeeze batch dim and return
        return {k: v.squeeze(0) for k, v in tokenized.items()}

    dataset = dataset.map(train_format_prompt)
    dataset = dataset.map(preprocess)
    dataset = dataset.train_test_split(test_size=0.2, shuffle=True, seed=42)
    train_data = dataset["train"]
    eval_data = dataset["test"]
    return train_data, eval_data, tokenizer

def train_model(base_model, dataset, save_path=None):

    train_data, eval_data, tokenizer = preprocess_dataset(base_model, dataset)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,           # or load_in_8bit=True
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="float16"  # or "bfloat16" if your hardware supports it
    )

    # Load model in 8-bit (for memory efficiency)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        device_map={"": torch.cuda.current_device()},
        trust_remote_code=True
    )

    # LoRA configuration
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        task_type="CAUSAL_LM"
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir="/home/ihossain/ISMAIL/SUPREMELAB/scam-prevention/logs",
        # per_device_train_batch_size=BATCH_SIZE,
        # per_device_eval_batch_size=BATCH_SIZE,
        # gradient_accumulation_steps=4,
        # num_train_epochs=3,
        # learning_rate=2e-5,
        # save_strategy="epoch",
        # logging_steps=20,
        # fp16=True,
        # report_to="none",
        # num_train_epochs=3,  # Set as needed
        # per_device_train_batch_size=BATCH_SIZE,
        # per_device_eval_batch_size=BATCH_SIZE,
        # gradient_accumulation_steps=48,  # 48 * 16 = 768 effective batch size
        # # eval_strategy="no",
        # save_strategy="steps",
        # save_steps=500,
        # learning_rate=5e-7,
        # lr_scheduler_type="constant",
        # max_grad_norm=1.0,
        # logging_dir="/home/ihossain/ISMAIL/SUPREMELAB/scam-prevention/logs",
        # logging_steps=100,
        # warmup_ratio=0.0,
        # # max_steps=1474560,  # Or use `total_episodes // (effective_batch_size)` if needed
        # fp16=True,  # or bf16
        num_train_epochs=3,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=64,             # Effective batch size = 64
        optim="paged_adamw_8bit",                   # For QLoRA (8-bit optimizer)
        save_strategy="steps",
        save_steps=500,
        eval_strategy="steps",                   # Or "steps" if you have a val set
        logging_steps=100,
        learning_rate=2e-5,                         # For LoRA tuning; don't go higher
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        max_grad_norm=0.3,
        bf16=True,                                  # Preferred over fp16 on H100
        tf32=True,                                  # Speed optimization
        report_to="none",
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False,           # Important for LoRA
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=eval_data,
        peft_config=lora_config,
        args=training_args
    )

    trainer.train()

    # Save final model and tokenizer
    trainer.model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)


def parse_model_output(output_text):
    result = {}

    # Extract numeric scores
    engagement_match = re.search(r'Engagement Score:\s*([\d.]+)', output_text)
    pii_risk_match = re.search(r'PII Risk Score:\s*([\d.]+)', output_text)

    if engagement_match:
        result['engagement_score'] = float(engagement_match.group(1))
    if pii_risk_match:
        result['pii_risk_score'] = float(pii_risk_match.group(1))

    # Extract boolean value for PII
    pii_match = re.search(r'Contains PII:\s*(yes|no)', output_text, re.IGNORECASE)
    if pii_match:
        result['contains_pii'] = pii_match.group(1).strip().lower() == 'yes'

    # Extract list of PII types
    pii_types_match = re.search(r'PII Types:\s*(.+)', output_text)
    if pii_types_match:
        result['pii_types'] = [s.strip() for s in pii_types_match.group(1).split(',')]

    return result

def eval_model(dataset, pretrained_path):
    # Format the prompts
    def eval_format_prompt(example):
        prompt = f"### Instruction:\n{example['instruction']}\n\n"
        if example["input"].strip():
            prompt += f"### Input:\n{example['input']}\n\n"
        prompt += f"### Response:\n"
        return {"text": prompt}
    
    def extract_response(output_text):
        response_marker = "### Response:"
        if response_marker in output_text:
            return output_text.split(response_marker, 1)[1].strip()
        else:
            return output_text.strip()  # fallback in case marker is missing

    eval_data = dataset.map(eval_format_prompt)
    # sample = eval_data[0]
    # prompt = sample["text"]

    # Load the fine-tuned model
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_path,
        torch_dtype=torch.float16,
        device_map={"": torch.cuda.current_device()},
        trust_remote_code=True
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_path,
        use_fast=True
    )

    for sample in eval_data.select(range(10)):
        prompt = sample["text"]
        # print(f"Prompt:\n{prompt}")
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

        output = pipe(prompt, max_new_tokens=100, do_sample=False)[0]["generated_text"]
        # print("Prompt:\n", prompt)
        print("\nGenerated:\n", extract_response(output))

# === Main Execution ===
if __name__ == "__main__":

    """
        This script runs fine-tuning LLMs on a multi-task dataset.
        It uses a pre-trained model, tunes it with LoRA, and evaluates the model after each round.
        We fine-tune models on the multi-task dataset:
        - Llama-Guard-3-8B
        - Llama-Guard-2-8B
        - LlamaGuard-7B
        - MD-Judge-v0.1
    """

    DATA_PATH = "/home/ihossain/ISMAIL/SUPREMELAB/scam-prevention/dataset/multi-task_balanced_scam_types_data_diverse.jsonl"
    BAITER_DATA_PATH = "/home/ihossain/ISMAIL/SUPREMELAB/scam-prevention/dataset/generation/all_train_data/scam_baiting_turns.jsonl"

    MODEL_NAME = "meta-llama/Llama-Guard-3-8B" #"meta-llama/Llama-3.1-8B" #"deepseek-ai/deepseek-llm-7b-base" #"OpenSafetyLab/MD-Judge-v0.1" #"deepseek-ai/deepseek-llm-67b-base" #"deepseek-ai/deepseek-llm-7b-base" #"allenai/Llama-3.1-Tulu-3.1-8B" #"meta-llama/Llama-3.1-8B" #"meta-llama/Llama-2-7b-hf" #"mistralai/Mistral-7B-v0.1" #"meta-llama/Meta-Llama-Guard-2-8B" #"meta-llama/LlamaGuard-7b" #"OpenSafetyLab/MD-Judge-v0.1"
    # Choose your base model
    # base_model = "mistralai/Mistral-7B-v0.1"  # or "meta-llama/Llama-2-7b-hf", "meta-llama/Llama-3.1-8B"
    pretrained_path = "/home/ihossain/ISMAIL/SUPREMELAB/scam-prevention/results/pre-trained/multi-task/tuned-llama-guard3"  # or "tuned-deepseek-7b", "tuned-llama3-tulu-8b", "tuned-llama2-7b", "tuned-mistral-7b"
    ds1 = utils.load_jsonl_dataset(DATA_PATH)
    ds2 = utils.load_dataset_plain_jsons(BAITER_DATA_PATH)

    # dataset_merged = interleave_datasets([ds1, ds2], probabilities=[0.5, 0.5], seed=42)
    dataset_merged = interleave_datasets([ds1, ds2], stopping_strategy="all_exhausted", seed=42)

    splits = dataset_merged.train_test_split(test_size=0.1, seed=42)  # random half

    print("### Length of the Train Datset: ", len(dataset_merged))
    print("### Lenght of the Eval Datset: ", len(splits['test']))

    # Start training
    print("##Starting training...")
    train_model(MODEL_NAME, dataset_merged, save_path = pretrained_path)
    print("Training complete. Model saved.")

    # Start evaluation
    # print("##Starting evaluation...")
    # eval_model(splits['test'], pretrained_path)
    # print("Evaluation complete!!")

# CUDA_VISIBLE_DEVICES=3 nohup python llm_instruction_tuning.py > /scam-prevention/logs/multi_task_tuning.log 2>&1 &
