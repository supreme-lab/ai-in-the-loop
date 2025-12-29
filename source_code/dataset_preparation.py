import json
import argparse
from datasets import Dataset
import os
import torch
import random
from pathlib import Path

import sys
sys.path.append("./scam-prevention")
import prompt_util
import random

def create_multi_task_instruction_data(data_dir):
    # all_tasks = []

    save_path = 'ai-in-the-loop/data/classification/all_eval_data/few-shot'

    for path in ["masc_dataset/all_data.chat.json", "sasc_dataset/all_data.chat.json", 
             "ssd_dataset/all_data.chat.json", "ssc_dataset/all_data.chat.json"]:
        with open(os.path.join(data_dir, path)) as file:
            print(f"Processing {path}...")
            all_chats = json.load(file)  # Load the entire list of chat sessions
            
            for entry in all_chats:
                # print(f"Chat: {chat}")
                result = prompt_util.build_prompt_from_chat_for_evaluation(entry['conversation'])
                dict_data = {'id': entry['id'], 'eval_engage_pii': result[0], 'eval_scam_risk': result[1], \
                             'scam_baiter': result[2], 'llama_guard': result[3]['prompt'], \
                                'md_judge': result[4]['prompt'], 'output': entry['label']}
                with open(os.path.join(save_path, path.replace('/', '_')), "a") as f:
                    f.write(json.dumps(dict_data) + "\n")

    # random.shuffle(all_tasks)
    print(f"💾 Saved Multi task instruction evaluation data.")


def get_identifier(dialogue):
    if "caller:" in dialogue:
        user1 = "caller:"
    if "receiver:" in dialogue:
        user2 = "receiver:"
    if "Suspect:" in dialogue:
        user1 = "Suspect:"
    if "Innocent:" in dialogue:
        user2 = "Innocent:"
    if "Person A:" in dialogue:
        user1 = "Person A:"
    if "Person B:" in dialogue:
        user2 = "Person B:"
    if "User:" in dialogue:
        user1 = "User:"
    if "Agent:" in dialogue:
        user2 = "Agent:"

    return user1, user2

def convert_scam_bait_to_chat_format(data_path):
    with open(data_path, "r", encoding="utf-8") as f:
        email = json.load(f)

    conversation = []
    # for chat in email['messages']:
    #     if chat['author_role'] == 'scam':
    #         user_msg = chat['body']
    #         conversation.append({"role": "scammer", "content": user_msg})
    #     elif chat['author_role'] == 'bait':
    #         assistant_msg = chat['body']
    #         conversation.append({"role": "baiter", "content": assistant_msg})
    #----------------------------------------------------------------------------
    for chat in email['conversations']:
        if chat['direction'] == 'Inbound':
            user_msg = chat['body']
            conversation.append({"role": "scammer", "content": user_msg})
        elif chat['direction'] == 'Outbound':
            assistant_msg = chat['body']
            conversation.append({"role": "baiter", "content": assistant_msg})
    
    return conversation

def convert_to_chat_format(data_path):
    with open(data_path, "r") as f:
        data = [json.loads(line) for line in f]
    
    conversation = []
    for example in data:
        id = example['id']
        dialogue = example["input"]
        output = example["output"]
        user1, user2 = get_identifier(dialogue)
        lines = dialogue.split(user1)
        chat = []

        for line in lines:
            if not line.strip():
                continue
            parts = line.split(user2)
            if len(parts) == 2:
                user_msg = parts[0].strip()
                assistant_msg = parts[1].strip()
                if user_msg:
                    chat.append({"role": "user", "content": user_msg})
                if assistant_msg:
                    chat.append({"role": "assistant", "content": assistant_msg})
            else:
                # Only user message (e.g., starts with caller:)
                msg = parts[0].strip()
                if msg:
                    chat.append({"role": "user", "content": msg})
        
        conversation.append({'id': id, 'conversation': chat, 'label': output})
    return conversation


def save_jsonl(data, file_path):
    with open(file_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

def replace_identifier(dialogue, user1, user2):
    dialogue = dialogue.replace(user1, "\nUser: ")
    dialogue = dialogue.replace(user2, "\nAgent: ")
    return dialogue
    
def convert_to_instruction_tuning(data, idx=0):
    results = []
    # idx = 0
    for i, item in enumerate(data):
        dialogue = item.get("dialogue", "").strip() # if ssc_dataset then 'conversation' is a key otherwise 'dialogue'
        # personality = item.get("personality", "unknown")
        # conv_type = item.get("type", "unknown")
        label = str(item.get("label", "1"))  # if ssc_dataset then 'label' is a key otherwise 'labels'
        user1, user2 = get_identifier(dialogue)
        input = replace_identifier(dialogue, user1, user2)

        results.append({
            "id": idx,
            # "instruction": """Determine whether the message or conversation or dialogue is part of a scam (1) or not (0).Respond with a single number: 1 (scam) or 0 (not a scam).""",
            "input": input,
            "output": int(label)
        })
        idx += 1
    return results, idx

def main(input_paths, output_path):
    # Load Arrow file as a Dataset
    all_data = []
    idx = 0
    for path in input_paths:
        dataset = Dataset.from_file(path)
        data = [dict(item) for item in dataset]
        formatted_data, idx = convert_to_instruction_tuning(data, idx)
        all_data.extend(formatted_data)

    save_jsonl(all_data, output_path)
    print(f"✅ Converted {len(formatted_data)} records to instruction-tuning format.")
    print(f"💾 Saved to: {output_path}")


def create_scam_bait_trainset(prior, output):
    instruction = ("You are a helpful and safety-aware AI assistant specializing in scam baiting." \
    "Your role is to continue the conversation as the 'Baiter', engaging with the scammer in a safe and contextually appropriate way." \
    "\n\nInstructions:" \
    "\n- Remain in character as the Baiter.\n- Respond convincingly to the scammer while never disclosing personal or sensitive information."
    "\n- Aim to prolong the conversation to waste the scammer\u2019s time without escalating or violating safety norms.")

    input = prior + "\n"

    prompt = {'instruction': instruction, 'input': input, 'output': f"<Baiter> {output}"}

    with open("ai-in-the-loop/data/generation/all_train_data/combined_scam_baiting_turns_train.jsonl", "a") as f:
        f.write(json.dumps(prompt) + "\n")

if __name__ == "__main__":
    """
        # Set the data directory
        # This script is used to convert scam classification datasets from Arrow format to an instruction-tuning JSONL format.
        # It processes input Arrow files, formats dialogues for instruction-based learning, and saves the output for downstream tasks.
    """
    #---------------------------------------------------------------------------------------------------------
    # """
    #     # Convert multi-task instruction data
    #     # This function creates a multi-task instruction dataset from various classification datasets.
    #     # It reads chat data from multiple sources, formats it for evaluation, and saves the results
    #     # in a specified directory.
    # """
    # data_dir = "ai-in-the-loop/data/classification"
    # create_multi_task_instruction_data(data_dir)
    #---------------------------------------------------------------------------------------------------------
    TEST_PATH = "ai-in-the-loop/data/generation/all_test_data"
    TRAIN_PATH = "ai-in-the-loop/data/generation/all_train_data"

    FILE_PATHs = ['asb', 'sbc', 'ytsc']

    for file_path in FILE_PATHs:
        file_dir = os.path.join(TRAIN_PATH, file_path)
        print(f"----------[{file_path}]----------")
        for file in os.listdir(file_dir):
            with open(os.path.join(file_dir, file), 'r', encoding="utf-8") as f:
                dataset = json.load(f)
                print("Length of Dataset: ", len(dataset))
                if file_path == 'ytsc':
                    for conv in dataset:
                        # print(dataset[0])
                        if conv[0]['role'] == 'baiter':
                            conv = conv[1:]

                        prior = ''
                        for item in conv:
                            if item['role']=='scammer':
                                prior += f"Scammer: {item['content']}\n"
                            if item['role']=='baiter':
                                # create_scam_bait_trainset(prior, item['content'])
                                prior += f"Baiter: {item['content']}\n"
                else:
                    # print("Length of Dataset: ", len(dataset))
                    if dataset[0]['role'] == 'baiter':
                        dataset = dataset[1:]

                    prior = ''
                    for item in dataset:
                        if item['role']=='scammer':
                            prior += f"Scammer: {item['content']}\n"
                        if item['role']=='baiter':
                            # create_scam_bait_trainset(prior, item['content'])
                            prior += f"Baiter: {item['content']}\n"
    
    #---------------------------------------------------------------------------------------------------------