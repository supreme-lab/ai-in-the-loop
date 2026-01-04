import json
import utils
import re
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

def load_evaluation_tasks(filepath):
    # Read the entire content of the file
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Use regex to split content into top-level JSON arrays
    raw_blocks = re.findall(r'(\[\s*(?:.|\n)*?\])(?=\s*\[|$)', content)

    # Extracted data will be stored here
    records = []

    # Loop through each top-level JSON block
    for idx, block in enumerate(raw_blocks):
        try:
            parsed = json.loads(block)
        except json.JSONDecodeError as e:
            print(f"Block {idx + 1}: JSON parse error: {e}")
            continue

        for item in parsed:
            if isinstance(item, list):
                for subitem in item:
                    if isinstance(subitem, dict):
                        key = "instruction" if "instruction" in subitem else "prompt"
                        records.append({
                            "type": key,
                            "text": subitem.get(key, ""),
                            "input": subitem.get("input", "")
                        })
            elif isinstance(item, dict):
                key = "instruction" if "instruction" in item else "prompt"
                records.append({
                    "type": key,
                    "text": item.get(key, ""),
                    "input": item.get("input", "")
                })

    # Create a DataFrame for further use
    return pd.DataFrame(records)

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

def generate_batch_output(filepath, output_dir, tuned_model, model_id, pretrained_path):

    # Load the JSON list from file
    with open(filepath, 'r') as f:
        input_dataset = [json.loads(line) for line in f if line.strip()]

    dataset = []
    for chat in input_dataset:
        # Conversation is going to be Scam and last turn of the conversation is for assistant
        result = prompt_util.build_prompt_from_chat_for_evaluation(chat['conversation'])
        dataset.append({'id': chat['id'], 'eval_engage_pii': result[0], \
                        'eval_scam_risk': result[1], 'scam_baiter': result[2], 'llama_guard': result[3]['prompt'], 'md_judge': result[4]['prompt'], \
                            'output': chat['label']})    

    # Check how many entries
    print(f"Loaded {len(dataset)} conversation entries.")
    print("Example entry:", dataset[0])
    
    # Load the model and tokenizer
    model, tokenizer = utils.load_model(model_id=model_id, checkpoint_dir=pretrained_path)

    # Example: Loop through each data entry
    for entry in dataset:
        print(f"\n--- ID {entry['id']} ---")
        engage_pii_score = None
        scam_scores = None
        scam_baiter = None
        # Example: print first engage_pii input
        if "eval_engage_pii" in entry:
            data = entry["eval_engage_pii"]
            prompts = prepare_batch_prompts(data)
            # Step 2: Tokenize as a batch
            tokenized = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
            output_ids = model.generate(
                input_ids=tokenized['input_ids'].cuda(),
                do_sample=True,
                temperature=0.3,
                # top_k=toK,
                top_p=1.0,
                max_new_tokens=100,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
            )
            # Step 4: Batch decode
            decoded = tokenizer.batch_decode(output_ids[:, tokenized['input_ids'].shape[-1]:], skip_special_tokens=True)
            # Step 5: Clean and print results
            engage_pii_score = []
            for text in decoded:
                print(f"Response:\n{text.strip()}")
                dict_value = parse_engagement_pii_output(text.strip())
                if dict_value:
                    if 'pii_risk_score' not in dict_value:
                        dict_value['pii_risk_score'] = 0.0
                    if 'engagement_score' not in dict_value:
                        dict_value['engagement_score'] = 0.0
                    if 'contains_pii' not in dict_value:
                        dict_value['contains_pii'] = None
                    if 'pii_types' not in dict_value:
                        dict_value['pii_types'] = None
                    engage_pii_score.append([dict_value['engagement_score'], dict_value['pii_risk_score'], dict_value['contains_pii'], dict_value['pii_types']])
                # else:
                #     engage_pii_score.append([0.0, 0.0, None, None])

        # Example: show scam risk inputs
        if "eval_scam_risk" in entry:
            data = entry["eval_scam_risk"]
            prompts = prepare_batch_prompts(data)
            # Step 2: Tokenize as a batch
            tokenized = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
            output_ids = model.generate(
                input_ids=tokenized['input_ids'].cuda(),
                do_sample=True,
                temperature=0.3,
                # top_k=toK,
                top_p=1.0,
                max_new_tokens=100,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
            )
            # Step 4: Batch decode
            decoded = tokenizer.batch_decode(output_ids[:, tokenized['input_ids'].shape[-1]:], skip_special_tokens=True)
            # Step 5: Clean and print results
            scam_scores = []
            for text in decoded:
                print(f"Response:\n{text.strip()}")
                dict_value = parse_scam_risk_output(text.strip())
                if dict_value:
                    if 'scam_risk_score' not in dict_value:
                        dict_value['scam_risk_score'] = 0.0
                    scam_scores.append(dict_value['scam_risk_score'])
                else:
                    scam_scores.append(0.0)
            
        # Show scam_baiter text
        if "scam_baiter" in entry:
            tokenized = tokenizer(eval_format_prompt(entry['scam_baiter']), return_tensors="pt", padding=True, truncation=True)
            output_ids = model.generate(
                input_ids=tokenized['input_ids'].cuda(),
                do_sample=True,
                temperature=0.95,
                top_k=50,
                top_p=0.95,
                max_new_tokens=100,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
            )

            prompt_len = tokenized['input_ids'].shape[-1]
            output_text = tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True)
            # print("\nGenerated:\n", output_text)
            # print("Prompt:\n", prompt)
            scam_baiter = parse_model_baiter_safeness_output(output_text)

        llama_safeness = None
        wild_safeness = None
        md_safeness = None

        if "llama" in tuned_model:
            tokenized = tokenizer(entry['llama_guard'], return_tensors="pt", padding=True, truncation=True)
            output_ids = model.generate(
                input_ids=tokenized['input_ids'].cuda(),
                # do_sample=True,
                # temperature=0.95,
                # top_k=50,
                # top_p=0.95,
                max_new_tokens=100,
                pad_token_id=0
            )

            prompt_len = tokenized['input_ids'].shape[-1]
            output_text = tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True)
            # print("\nGenerated:\n", output_text)
            # print("Prompt:\n", prompt)
            llama_safeness = parse_model_baiter_safeness_output(output_text)

        if "wild" in tuned_model:
            tokenized = tokenizer(entry['wild_guard'], return_tensors="pt", padding=True, truncation=True)
            output_ids = model.generate(
                input_ids=tokenized['input_ids'].cuda(),
                # do_sample=True,
                # temperature=0.95,
                # top_k=50,
                # top_p=0.95,
                max_new_tokens=100,
                pad_token_id=0
            )

            prompt_len = tokenized['input_ids'].shape[-1]
            output_text = tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True)
            # print("\nGenerated:\n", output_text)
            # print("Prompt:\n", prompt)
            wild_safeness = parse_model_baiter_safeness_output(output_text)

        if "md" in tuned_model:
            tokenized = tokenizer(entry['md_judge'], return_tensors="pt", padding=True, truncation=True)
            output_ids = model.generate(
                input_ids=tokenized['input_ids'].cuda(),
                # do_sample=True,
                # temperature=0.95,
                # top_k=50,
                # top_p=0.95,
                max_new_tokens=100,
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
            "scam_baiter": scam_baiter['baiter_response'] if 'baiter_response' in scam_baiter else None,
            "safeness": safeness['safeness'] if safeness and 'safeness' in safeness else None,
            "reference_scam_score": entry['output']
        }
        with open(output_dir, 'a') as f:  # Open in append mode
            f.write(json.dumps(result) + '\n')

def extract_response(output_text):
    response_marker = "Response:"
    if response_marker in output_text:
        return output_text.split(response_marker, 1)[1].strip()
    else:
        return output_text.strip()  # fallback in case marker is missing

# === Main Execution ===
if __name__ == "__main__":

    """
        We are going to evaluate the performance of the models on the following tasks:
            1. Engagement and PII detection
            2. Scam risk detection
            3. Scam baiter performance
            4. Moderation performance (safeness of the response)
        We will use the following models:
            1. LlamaGuard-7b
            2. LlamaGuard-3-8B
            3. LlamaGuard-2-8B
            4. MD-Judge-v0.1
        We will use the combined evaluation dataset (Size: 1200) of the following:
            - MASC
            - SASC
            - SSC
            - SSD
        We will calculate the correlation of the Moderation, Engagement, PII, and Scam risk scores the four models.
        We show the results in the paper (Table 7).
    """

    data_dir = "ai-in-the-loop/data/generation/selected_conversation_to_scam_baiter_performance.jsonl"

    # for dataset_name in os.listdir(data_dir):
    #     # dataset = utils.load_jsonl_dataset(os.path.join(data_dir, dataset_name))
    #     # dataset = dataset.train_test_split(test_size=0.2, shuffle=True, seed=42)
    #     # eval_data = dataset["test"]

    #     print(f"Loading dataset: {dataset_name}")
    #     # Load the evaluation tasks from the JSON file
    #     # dataset = load_evaluation_tasks(os.path.join(data_dir, dataset_name))
    #     # eval_data = dataset.apply(eval_format_prompt, axis=1)

    #     ds_name = dataset_name.split("_")[0]  # Extract the dataset name without extension
    #     # if 'ssd' not in ds_name:
    #     #     continue

    MODEL_NAMEs = ["meta-llama/Llama-Guard-3-8B", "meta-llama/Meta-Llama-Guard-2-8B", "meta-llama/LlamaGuard-7b", "OpenSafetyLab/MD-Judge-v0.1"]
    # Choose your base model
    tuned_models = ['llama-guard3', 'llama-guard2', 'llama-guard', 'md-judge']

    for i, model_name in enumerate(MODEL_NAMEs):
        print(f"Evaluating model: {model_name}")
        tuned_model = tuned_models[i]
        pretrained_path = f"ai-in-the-loop/results/pre-trained/multi-task/tuned-{tuned_model}"

        # Start evaluation
        print("##Starting evaluation...")
        # eval_model(eval_data, model_name, pretrained_path, tuned_model, ds_name)
        generate_batch_output(data_dir, "", tuned_model, model_name, pretrained_path)
        print("Evaluation complete!!")

# CUDA_VISIBLE_DEVICES=3 nohup python evaluation_scam_pii_engage.py > ai-in-the-loop/logs/eval.log 2>&1 &
