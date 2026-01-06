from sklearn.metrics import mean_squared_error, accuracy_score
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from peft import PeftModel
from datasets import load_dataset, Dataset
import prompt_util
import os
PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = PARENT_DIR.rsplit("/", 2)[0]

# ==== Load and Format Dataset ====
def load_jsonl_dataset(path):
    with open(path, "r") as f:
        # lines = [json.loads(line.strip()) for line in f if line.strip()]
        lines = json.load(f)
    return Dataset.from_list(lines)

def load_dataset_plain_jsons(path):
    with open(path, "r") as f:
        lines = [json.loads(line.strip()) for line in f if line.strip()]
    return Dataset.from_list(lines)

def format_sample(example, model="llama"):

    formatted = prompt_util.build_prompt_for_tuning(example, model_id=model)
    
    return {"text": formatted}

def moderate_with_template(chat, tokenizer, model, device):
    input_ids = tokenizer.apply_chat_template(chat, return_tensors="pt").to(device)
    output = model.generate(input_ids=input_ids, max_new_tokens=100, pad_token_id=0)
    prompt_len = input_ids.shape[-1]
    return tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)

def evaluation_prompt_for_chat(chat, model_id, eval_type="moderation"):
    """
    Generate an evaluation prompt for LlamaGuard or LLaMA model.
    
    eval_type: 'moderation' | 'engagement+pii'
    """
    # conversation = [turn["content"] for turn in chat]
    # is_agent = len(conversation) % 2 == 0
    # role = "Agent" if is_agent else "User"

    # print("TASK: ", eval_type)

    if eval_type == "moderation":
        prompt = prompt_util.moderation_prompt_for_chat(chat, model_id)
        return prompt

    elif eval_type == "engagement+pii":
        prompt = prompt_util.engagement_pii_prompt_for_chat(chat)
        return prompt
    
    elif eval_type == "scam_detection":
        prompt = prompt_util.scam_prompt_for_chat(chat)
        return prompt

    else:
        raise ValueError("Unknown eval_type: use 'moderation' or 'engagement+pii'")

def load_model(device="cuda", model_id="meta-llama/LlamaGuard-7b", checkpoint_dir=None):

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,           # or load_in_8bit=True
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="float16"  # or "bfloat16" if your hardware supports it
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        device_map={"": torch.cuda.current_device()},
        # device_map="auto",  # Automatically use all available GPUs
        trust_remote_code=True
    )

    # # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_dir,
        padding_side="left",
        use_fast=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    model = PeftModel.from_pretrained(base_model, checkpoint_dir)  # Must point to adapter dir
    return model.eval(), tokenizer


def generate_response(chat, model_id, eval_type, model, tokenizer, checkpoint_dir=None):
    # model, tokenizer = load_model(checkpoint_dir=checkpoint_dir)
    prompt =  evaluation_prompt_for_chat(chat, model_id, eval_type)
    # prompt = build_prompt(instruction, inp)
    inputs = tokenizer([prompt], return_tensors='pt').to("cuda")
    
    output = model.generate(**inputs, max_new_tokens=512, pad_token_id=0)
    prompt_len = inputs["input_ids"].shape[-1]
    return tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)

def convert_string_to_float(s):
    """
    Extract a float from the string s by slicing up to the second dot ('.').
    Falls back to the whole string if less than 2 dots exist.
    Handles invalid formats and returns None in case of failure.
    """
    dot_count = 0
    index = None

    for i, char in enumerate(s):
        if char == '.':
            dot_count += 1
            if dot_count == 2:
                index = i
                break

    if dot_count >= 2:
        number_str = s[:index]
    else:
        number_str = s.rstrip('.')  # remove trailing dot if present

    try:
        number = float(number_str)
        return number
    except ValueError:
        # Return None or raise a custom error if parsing fails
        return 0.0

