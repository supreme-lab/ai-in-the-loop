
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json
from sklearn.metrics import mean_squared_error, accuracy_score
import utils
from tqdm import tqdm
import os
from transformers import pipeline
import re
from datasets import load_dataset, Dataset
import pandas as pd
import prompt_util

# | Use Case                       | Best Temperature |
# | ------------------------------ | ---------------- |
# | Score/numeric prediction       | `0.0 – 0.2`      |
# | Factual QA, summarization      | `0.3 – 0.6`      |
# | Dialogue generation (balanced) | `0.6 – 0.8`      |
# | Creative writing / exploration | `0.9 – 1.2`      |

# | Task Type                  | Suggested Settings                                          |
# | -------------------------- | ----------------------------------------------------------- |
# | **Score Prediction**       | `top_k=1`, `top_p=1.0`, `temperature=0.0` *(deterministic)* |
# | **Text Generation (chat)** | `top_k=50`, `top_p=0.9–0.95`, `temperature=0.7–1.0`         |
# | **Sensitive applications** | Lower `temperature` and `top_p` to increase reliability     |

def parse_engagement_pii_output(output_text):
    result = {}

    # Extract numeric scores
    engagement_match = re.search(r'Engagement Score:\s*([\d.]+)', output_text)
    pii_risk_match = re.search(r'PII Risk Score:\s*([\d.]+)', output_text)

    if engagement_match:
        result['engagement_score'] = utils.convert_string_to_float(engagement_match.group(1))
    if pii_risk_match:
        result['pii_risk_score'] = utils.convert_string_to_float(pii_risk_match.group(1))

    # Extract boolean value for PII
    pii_match = re.search(r'Contains PII:\s*(yes|no)', output_text, re.IGNORECASE)
    if pii_match:
        result['contains_pii'] = pii_match.group(1).strip().lower() == 'yes'

    # Extract list of PII types
    pii_types_match = re.search(r'PII Types:\s*(.+)', output_text)
    if pii_types_match:
        result['pii_types'] = [s.strip() for s in pii_types_match.group(1).split(',')]

    return result


def parse_scam_risk_output(output_text):
    result = {}

    # Extract numeric value after "### Response:"
    match = re.search(r"\s*([\d.]+)", output_text)

    if match:
        try:
            result['scam_risk_score'] = utils.convert_string_to_float(match.group(1))
        except ValueError:
            pass  # Next line isn't a number

    return result

def parse_model_baiter_safeness_output(output_text):
    result = {}
    # Extract the first Baiter response
    match = re.search(r"Baiter:\s*(.+)", output_text)
    first_baiter_response = match.group(1).strip() if match else None
    if first_baiter_response is None:
        # print("No response found.")
        match = re.search(r"<Baiter>\s*(.+)", output_text)
        first_baiter_response = match.group(1).strip() if match else None
        if first_baiter_response is None:
            if 'safe' in output_text.lower() or 'unsafe' in output_text.lower():
                result['safeness'] = output_text.lower()
            else:
                print("No response found.")
        else:
            result['baiter_response'] = first_baiter_response
    else:
        result['baiter_response'] = first_baiter_response

    return result

# Format the prompts
def eval_format_prompt(example):
    prompt = f"### Instruction:\n{example['instruction']}\n\n"
    if example["input"].strip():
        prompt += f"### Input:\n{example['input']}\n\n"
    prompt += f"### Response:\n"
    return prompt

def prepare_batch_prompts(data):
    prompts = [eval_format_prompt(item) for item in data]
    return prompts

def generate_batch_output(first_input_file, second_input_file, output_file, tuned_model, model_id, pretrained_path):
    # Load the JSON list from file
    with open(first_input_file, 'r') as f:
        input_data = json.load(f)
    with open(second_input_file, 'r') as f:
        output_data = [json.loads(line) for line in f if line.strip()]

    dataset = []
    for (chat1, chat2) in zip(input_data, output_data):
        # Conversation is going to be Scam and last turn of the conversation is for assistant
        if chat1['label'] == 1 and chat1['conversation'][-1]['role'] == 'assistant':
            result = prompt_util.build_prompt_from_chat_for_evaluation([{'role': 'user', 'content': chat1['conversation'][-2]['content']},
                                                               {'role': 'assistant', 'content': chat2['scam_baiter']}])
            dataset.append({'id': chat1['id'], 'eval_engage_pii': result[0], \
                            'eval_scam_risk': result[1], 'llama_guard': result[2]['prompt'], 'md_judge': result[3]['prompt'], \
                                'scammer_msg': chat1['conversation'][-2]['content'], \
                                'reference_response': chat1['conversation'][-1]['content'], \
                                    'scam_baiter_response': chat2['scam_baiter']})


    model, tokenizer = utils.load_model(model_id=model_id, checkpoint_dir=pretrained_path)

    # model_outputs = []

    # Example: Loop through each data entry
    for entry in dataset:
        print(f"\n--- ID {entry['id']} ---")
        engage_pii_score = None
        scam_scores = None
        scam_baiter = None
        # Example: print first engage_pii input
        if "eval_engage_pii" in entry:
            data = entry["eval_engage_pii"][0]
            prompt = eval_format_prompt(data)
            # Step 2: Tokenize as a batch
            tokenized = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(model.device)
            output_ids = model.generate(
                input_ids=tokenized['input_ids'].cuda(),
                do_sample=False,
                temperature=0.1,
                # top_k=toK,
                top_p=1.0,
                max_new_tokens=50,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
            )
            # Step 4: Batch decode
            prompt_len = tokenized['input_ids'].shape[-1]
            decoded_value = tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True)
            # Step 5: Clean and print results
            engage_pii_score = []
            print(f"Response:\n{decoded_value.strip()}")
            dict_value = parse_engagement_pii_output(decoded_value.strip())
            if dict_value:
                if 'pii_risk_score' not in dict_value:
                    dict_value['pii_risk_score'] = 0.0
                if 'engagement_score' not in dict_value:
                    dict_value['engagement_score'] = 0.0
                if 'contains_pii' not in dict_value:
                    dict_value['contains_pii'] = None
                if 'pii_types' not in dict_value:
                    dict_value['pii_types'] = None
                engage_pii_score = [dict_value['engagement_score'], dict_value['pii_risk_score'], dict_value['contains_pii'], dict_value['pii_types']]
                # else:
                #     engage_pii_score.append([0.0, 0.0, None, None])

        # Example: show scam risk inputs
        if "eval_scam_risk" in entry:
            data = entry["eval_scam_risk"][0]
            prompt = eval_format_prompt(data)
            # Step 2: Tokenize as a batch
            tokenized = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(model.device)
            output_ids = model.generate(
                input_ids=tokenized['input_ids'].cuda(),
                # do_sample=True,
                # temperature=0.95,
                # top_k=50,
                # top_p=0.95,
                max_new_tokens=50,
                pad_token_id=0
            )
            # Step 4: Batch decode
            prompt_len = tokenized['input_ids'].shape[-1]
            decoded_value = tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True)
            # Step 5: Clean and print results
            scam_scores = 0.0
            print(f"Response:\n{decoded_value.strip()}")
            dict_value = parse_scam_risk_output(decoded_value.strip())
            if dict_value:
                if 'scam_risk_score' not in dict_value:
                    dict_value['scam_risk_score'] = 0.0
                
                # if len(dict_value['scam_risk_score']) == 1 and dict_value['scam_risk_score'][0]=='.':
                #     continue
                # else:
                scam_scores = dict_value['scam_risk_score']
            else:
                scam_scores = 0.0

        llama_safeness = None
        wild_safeness = None
        md_safeness = None

        if "llama" in tuned_model or "deepseek" in tuned_model:
            tokenized = tokenizer(entry['llama_guard'], return_tensors="pt", padding=True, truncation=True)
            output_ids = model.generate(
                input_ids=tokenized['input_ids'].cuda(),
                # do_sample=True,
                # temperature=0.95,
                # top_k=50,
                # top_p=0.95,
                max_new_tokens=50,
                pad_token_id=0
            )

            prompt_len = tokenized['input_ids'].shape[-1]
            output_text = tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True)
            # print("\nGenerated:\n", output_text)
            # print("Prompt:\n", prompt)
            llama_safeness = parse_model_baiter_safeness_output(output_text)

        if "md" in tuned_model:
            tokenized = tokenizer(entry['md_judge'], return_tensors="pt", padding=True, truncation=True)
            output_ids = model.generate(
                input_ids=tokenized['input_ids'].cuda(),
                # do_sample=True,
                # temperature=0.95,
                # top_k=50,
                # top_p=0.95,
                max_new_tokens=50,
                pad_token_id=0
            )

            prompt_len = tokenized['input_ids'].shape[-1]
            output_text = tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True)
            # print("\nGenerated:\n", output_text)
            # print("Prompt:\n", prompt)
            md_safeness = parse_model_baiter_safeness_output(output_text)

        safeness = llama_safeness if llama_safeness else wild_safeness if wild_safeness else md_safeness if md_safeness else None
        # Optionally show keys like llama_guard, wild_guard, md_judge
        result = {
            "id": entry["id"],
            "scam_risk_score": scam_scores,
            "engage_pii_score": engage_pii_score,
            "safeness": safeness['safeness'] if safeness and 'safeness' in safeness else None,
            'scammer_msg': entry['scammer_msg'], 
            'reference_response': entry['reference_response'], 
            'scam_baiter_response': entry['scam_baiter_response']
        }
        with open(output_file, 'a') as f:  # Open in append mode
            f.write(json.dumps(result) + '\n')

# === Main Execution ===
if __name__ == "__main__":

    """
        We evaluate the models' scam detection performance with respect to the following metrics:
         - F1, FPR, FNR, AUPRC
        We evaluate the models' performance on the following datasets:
            - MASC (Multi-Agent Scam Conversation)
            - SASC (Scam Agent Scam Conversation)
            - SSC (Synthesized Scam Conversation)
            - SSD (Synthesized Scam Dialogue)
        We added evaluation results for the following models:
            - Llama Guard 3 (8B)
            - Llama Guard 2 (8B)
            - Llama Guard (7B)
            - MD-Judge (v0.1)
        In the paper the results are reported in the Table 2.
    """

    for dataset_name in ['masc', 'sasc', 'ssc', 'ssd']:

        input_file_path = f"ai-in-the-loop/data/classification/{dataset_name}_dataset/all_data.chat.json"
        output_file_path = f"ai-in-the-loop/results/reports/{dataset_name}/eval_md-judge_lora_merged.json"
        
        MODEL_NAMEs = ["meta-llama/Llama-Guard-3-8B", "meta-llama/Meta-Llama-Guard-2-8B", "meta-llama/LlamaGuard-7b", "OpenSafetyLab/MD-Judge-v0.1"]
        # Choose your base model
        tuned_models = ['llama-guard3', 'llama-guard2', 'llama-guard', 'md-judge']
        for i, model_name in enumerate(MODEL_NAMEs):
            print(f"Evaluating model: {model_name}")
            tuned_model = tuned_models[i]
            pretrained_path = f"ai-in-the-loop/results/fine-tuned/multi-task/tuned-{tuned_model}"

            # Start evaluation
            print("##Starting evaluation...")
            # eval_model(eval_data, model_name, pretrained_path, tuned_model, ds_name)
            generate_batch_output(input_file_path, output_file_path, dataset_name, tuned_model, model_name, pretrained_path)
            print("Evaluation complete!!")

# CUDA_VISIBLE_DEVICES=2 nohup python evaluation_scam_baiting.py > /scam-prevention/logs/multi_task_eval.log 2>&1 &
