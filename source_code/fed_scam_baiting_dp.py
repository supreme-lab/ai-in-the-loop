import random
import json
import utils
import re
import time
from datetime import datetime

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

scam_bait_instruction_old = (
            "You are a helpful and safety-aware AI assistant specializing in scam baiting. "
            "Your role is to continue the conversation as the 'Baiter', engaging with the scammer in a safe and contextually appropriate way.\n\n"
            "Instructions:\n"
            "- Remain in character as the Baiter.\n"
            "- Respond convincingly to the scammer while never disclosing personal or sensitive information.\n"
            "- Aim to prolong the conversation to waste the scammer’s time without escalating or violating safety norms."
        )

scam_bait_instruction = (
    "You are a helpful and safety-aware AI assistant specializing in scam baiting. "
    "Your role is to continue the conversation as the 'Baiter', engaging with the scammer in a safe, contextually accurate, and realistic manner.\n\n"

    "Primary Objective:\n"
    "- Keep the scammer engaged in extended dialogue while revealing no private, personal, or sensitive information.\n"

    "Character Role:\n"
    "- Always remain in character as the Baiter: a curious, naive, or cooperative persona (as context requires), but never so gullible as to compromise safety.\n"
    "- You are aware this is a scam but pretend not to be.\n"

    "Response Guidelines:\n"
    "- Respond only based on the context of the scammer's most recent message.\n"
    "- Never make up unrelated information or diverge from the thread of conversation.\n"
    "- Ask harmless questions, delay actions, express confusion, or use plausible excuses to prolong the exchange.\n"
    "- Use vague or fictitious personal details when absolutely necessary, but avoid specific identifiers (e.g., SSN, full address, real banking info).\n"
    "- Avoid escalating the tone or alerting the scammer that you know it's a scam.\n"
    "- Never offer payments, credentials, downloads, or real-world access of any kind.\n"

    "Safety and Filtering:\n"
    "- Never output any personally identifiable information (PII), security credentials, or real-world links.\n"
    "- If the scammer’s message includes a URL, avoid responding to the link and steer the conversation back to dialogue.\n"
    "- Maintain compliance with ethical and safety guidelines—never encourage harm, hacking, fraud, or vigilantism.\n"

    "Quality Control:\n"
    "- Ensure each response is relevant, plausible, and logically follows the scammer’s previous message.\n"
    "- If the scammer’s message is unclear, ask for clarification instead of guessing.\n"
    "- Use natural language to maintain realism, but avoid verbosity that could seem artificial.\n"
)

scam_risk_score_instruction = (
            "You are an expert evaluator analyzing a conversation to estimate the likelihood that it is a scam.\n\n"
            "Instructions:\n"
            "- Analyze the conversation between Suspected Scammer and User.\n"
            "- Output a scam risk score from 0.0 (not a scam at all) to 1.0 (certainly a scam).\n"
            "- Provide only a floating point number as output (e.g., 0.87), no text or explanation.\n"
        )

pii_risk_score_instruction = (
            "You are an evaluator analyzing a dialogue between a Suspected Scammer and an User.\n"
            "1. Estimate how engaging the User's response is (0.0 = low, 1.0 = high).\n"
            "2. Identify whether the User's response contains personally identifiable information (PII).\n"
            "3. Provide a structured report with fixed phrasing.\n\n"
            "Respond using the following structured format:\n"
            "Engagement Score: <float>\n"
            "PII Risk Score: <float>\n"
            "Contains PII: <yes|no>\n"
            "PII Types: <comma-separated list or 'none'>\n"
        )

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
            result['baiter_response'] = output_text.strip()
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

def generate_batch_output(json_dataset, round, tuned_model, model_id, pretrained_path):
    # Load the JSON list from file
    model, tokenizer = utils.load_model(model_id=model_id, checkpoint_dir=pretrained_path)

    # model_outputs = []
    for idx, input_data in enumerate(json_dataset):
        # Example: Loop through each data entry
        gen_result = []
        for i, entry in enumerate(input_data['conversations']): # if dataset == 'asb' input_data['messages'] else input_data['conversations']:
            print(f"\n--- ID {i} ---")
            # if i==0 and 'bait' in entry['author_role']:
            #     continue
            if 'bait' in entry['role']:
                print("Skipping bait turn... it is not for the scammer.")
                continue

            if i == len(input_data['conversations']) - 1 and 'scam' in entry['role']:
                print("Skipping last message in the conversation as it is for the scammer no bait next.")
                continue
            
            ###+++++++++++++++++[Scam-Baiter Response Generation Begins]+++++++++++++++++++++++++++++++++++++++++++
            start_time = time.time()
            scam_baiter = None
            # Show scam_baiter text
            data = {'instruction': scam_bait_instruction, 'input': f'Scammer: {entry['content']}'}
            prompt = eval_format_prompt(data)
            tokenized = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
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

            end_time = time.time()
            ###==================[Scam-Baiter Response Generation Ends!!]==========================================

            engage_pii_score = None
            scam_scores = None
            userA_turn = entry['content']
            userB_turn = input_data['conversations'][i+1]['content']
            input_text = f"Suspected Scammer: {userA_turn}\nUser: {userB_turn}\n"

            ###+++++++++++++++++[PII Risk Score Calcultion]++++++++++++++++++++++++++++++++++++++++++++++++++
            # Example: print first engage_pii input
            data = data = {'instruction': pii_risk_score_instruction, 'input': input_text}
            prompt = eval_format_prompt(data)
            # Step 2: Tokenize as a batch
            tokenized = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(model.device)
            output_ids = model.generate(
                input_ids=tokenized['input_ids'].cuda(),
                do_sample=False,
                temperature=0.1,
                # top_k=toK,
                top_p=1.0,
                max_new_tokens=100,
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

            ###+++++++++++++++++[Scam Risk Score Calcultion]++++++++++++++++++++++++++++++++++++++++++++++++++
            # Example: show scam risk inputs
            # if "eval_scam_risk" in entry:
            data = {'instruction': scam_risk_score_instruction, 'input': input_text}
            prompt = eval_format_prompt(data)
            # Step 2: Tokenize as a batch
            tokenized = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(model.device)
            output_ids = model.generate(
                input_ids=tokenized['input_ids'].cuda(),
                do_sample=False,
                temperature=0.1,
                # top_k=toK,
                top_p=1.0,
                max_new_tokens=100,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
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
            
            result = {
                "id": i,
                "scam_risk_score": scam_scores,
                "engage_pii_score": engage_pii_score,
                # "safeness": safeness['safeness'] if safeness and 'safeness' in safeness else None,
                'scammer_msg': entry['content'], 
                'reference_baiter_response': input_data['conversations'][i+1]['content'], 
                'ai_baiter_response': scam_baiter['baiter_response'],
                # 'reference_time': reference_time,
                'ai_baiting_time': (end_time - start_time)
            }
            gen_result.append(result)
        with open(f'ai-in-the-loop/results/reports/FL/dp/0.1/gen_scam_bait_response_{tuned_model}_round_{round}.json', 'a') as f:  # Open in append mode
            f.write(json.dumps({"id": idx, "conversation": gen_result}) + '\n')

# === Main Execution ===
if __name__ == "__main__":
    input_file_path = f"ai-in-the-loop/data/generation/all_eval_data/combined_asb_sbc_ytsc_dataset.jsonl"
    
    with open(input_file_path, 'r', encoding='utf-8') as f:
        dataset = [json.loads(line) for line in f if line.strip()]

    print(f"First sample of dataset: {dataset[0]}")

    MODEL_NAMEs = ["meta-llama/LlamaGuard-7b", "meta-llama/Meta-Llama-Guard-2-8B", "meta-llama/Llama-Guard-3-8B", "OpenSafetyLab/MD-Judge-v0.1"]
    # Choose your base model
    tuned_models = ['llama-guard', 'llama-guard2', 'llama-guard3', 'md-judge']

    # target_dataset = dataset[0:100]
    target_dataset = random.sample(dataset, 100)  # picks 100 unique random elements

    for round in [5, 10, 15, 20, 25, 30]:

        print(f"Evaluating Round: {round}")

        for i, model_name in enumerate(MODEL_NAMEs):
            print(f"Evaluating model: {model_name}")
            tuned_model = tuned_models[i]
            pretrained_path = f"ai-in-the-loop/results/FL/dp/0.1/tuned-{tuned_model}-round_{round}"

            # Start evaluation
            print("##Starting evaluation...")
            # eval_model(eval_data, model_name, pretrained_path, tuned_model, ds_name)
            generate_batch_output(target_dataset, round, tuned_model, model_name, pretrained_path)
            print("Evaluation complete!!")

# CUDA_VISIBLE_DEVICES=2 nohup python fed_scam_baiting_gen.py > /scam-prevention/logs/scam_bait_gen.log 2>&1 &
