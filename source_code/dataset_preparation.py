import json
import os
from datasets import Dataset
import prompt_util
import pandas as pd
import utils

PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = PARENT_DIR.rsplit("/", 2)[0]

# ------------------------------------------------------------------
# Multi-task instruction data creation
# ------------------------------------------------------------------

def create_multi_task_instruction_data(data_dir):
    save_path = f"{PARENT_DIR}/ai-in-the-loop/data/classification/all_eval_data/few-shot"

    for path in [
        "masc_dataset/all_data.chat.json",
        "sasc_dataset/all_data.chat.json",
        "ssd_dataset/all_data.chat.json",
        "ssc_dataset/all_data.chat.json",
    ]:
        # with open(os.path.join(data_dir, path)) as file:
        print(f"Processing {path}...")
        all_chats = utils.load_json(os.path.join(data_dir, path)) #pd.read_json(os.path.join(data_dir, path), lines=True).to_dict('records')

        for entry in all_chats:
            result = prompt_util.build_prompt_from_chat_for_evaluation(
                entry["conversation"]
            )
            dict_data = {
                "id": entry["id"],
                "eval_engage_pii": result[0],
                "eval_scam_risk": result[1],
                "scam_baiter": result[2],
                "llama_guard": result[3]["prompt"],
                "md_judge": result[4]["prompt"],
                "output": entry["label"],
            }

            with open(
                os.path.join(save_path, path.replace("/", "_")), "a"
            ) as f:
                f.write(json.dumps(dict_data) + "\n")

    print("💾 Saved Multi task instruction evaluation data.")


# ------------------------------------------------------------------
# Dialogue helpers
# ------------------------------------------------------------------

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


def replace_identifier(dialogue, user1, user2):
    dialogue = dialogue.replace(user1, "\nUser: ")
    dialogue = dialogue.replace(user2, "\nAgent: ")
    return dialogue

# ------------------------------------------------------------------
# Conversion utilities
# ------------------------------------------------------------------

def convert_scam_bait_to_chat_format(data_path):
    # with open(data_path, "r", encoding="utf-8") as f:
    email =  utils.load_json(data_path) #pd.read_json(data_path, lines=True).to_dict('records')

    conversation = []
    for chat in email["conversations"]:
        if chat["direction"] == "Inbound":
            conversation.append({"role": "scammer", "content": chat["body"]})
        elif chat["direction"] == "Outbound":
            conversation.append({"role": "baiter", "content": chat["body"]})

    return conversation


def convert_to_chat_format(data_path):
    with open(data_path, "r") as f:
        data = [json.loads(line) for line in f]

    conversation = []
    for example in data:
        id = example["id"]
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
                if parts[0].strip():
                    chat.append({"role": "user", "content": parts[0].strip()})
                if parts[1].strip():
                    chat.append({"role": "assistant", "content": parts[1].strip()})
            else:
                if parts[0].strip():
                    chat.append({"role": "user", "content": parts[0].strip()})

        conversation.append({"id": id, "conversation": chat, "label": output})

    return conversation


def save_jsonl(data, file_path):
    with open(file_path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


def convert_to_instruction_tuning(data, idx=0):
    results = []
    for item in data:
        dialogue = item.get("dialogue", "").strip()
        label = str(item.get("label", "1"))

        user1, user2 = get_identifier(dialogue)
        input_text = replace_identifier(dialogue, user1, user2)

        results.append(
            {
                "id": idx,
                "input": input_text,
                "output": int(label),
            }
        )
        idx += 1

    return results, idx


def create_scam_bait_trainset(prior, output):
    instruction = (
        "You are a helpful and safety-aware AI assistant specializing in scam baiting."
        "Your role is to continue the conversation as the 'Baiter', engaging with the scammer in a safe and contextually appropriate way."
        "\n\nInstructions:"
        "\n- Remain in character as the Baiter."
        "\n- Respond convincingly to the scammer while never disclosing personal or sensitive information."
        "\n- Aim to prolong the conversation to waste the scammer’s time without escalating or violating safety norms."
    )

    prompt = {
        "instruction": instruction,
        "input": prior + "\n",
        "output": f"<Baiter> {output}",
    }

    with open(
        f"{PARENT_DIR}/ai-in-the-loop/data/generation/all_train_data/combined_scam_baiting_turns_train.jsonl",
        "a",
    ) as f:
        f.write(json.dumps(prompt) + "\n")


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def main():
    """
    This script converts scam classification and scam-baiting datasets into
    instruction-tuning or evaluation-ready JSONL formats.
    """

    # ------------------------------------------------------------------
    # OPTIONAL: Create multi-task instruction evaluation data
    # ------------------------------------------------------------------
    # data_dir = "ai-in-the-loop/data/classification"
    # create_multi_task_instruction_data(data_dir)

    # ------------------------------------------------------------------
    # Scam baiting dataset processing
    # ------------------------------------------------------------------
    EVAL_PATH = f"{PARENT_DIR}/ai-in-the-loop/data/generation/all_eval_data"
    TRAIN_PATH = f"{PARENT_DIR}/ai-in-the-loop/data/multi_task_train"
    is_train = is_eval = False

    if os.path.exists(f"{TRAIN_PATH}/multi-task_conversation_train_data.jsonl") and \
        os.path.exists(f"{TRAIN_PATH}/combined_scam_baiting_turns_train.jsonl"):
        print("----> Train dataset already exists!!")
        is_train=True
    
    if os.path.exists(f"{EVAL_PATH}/combined_asb_sbc_ytsc_dataset.jsonl"):
        print("----> Evaluation dataset already exists!!")
        is_eval =True

    if is_train and is_eval:
        return

    FILE_PATHS = ["asb", "sbc", "ytsc"]

    for file_path in FILE_PATHS:
        file_dir = os.path.join(TRAIN_PATH, file_path)
        print(f"----------[{file_path}]----------")

        for file in os.listdir(file_dir):
            with open(os.path.join(file_dir, file), "r", encoding="utf-8") as f:
                dataset = json.load(f)
                print("Length of Dataset:", len(dataset))

                if file_path == "ytsc":
                    for conv in dataset:
                        if conv[0]["role"] == "baiter":
                            conv = conv[1:]

                        prior = ""
                        for item in conv:
                            if item["role"] == "scammer":
                                prior += f"Scammer: {item['content']}\n"
                            if item["role"] == "baiter":
                                create_scam_bait_trainset(prior, item['content'])
                                prior += f"Baiter: {item['content']}\n"
                else:
                    if dataset[0]["role"] == "baiter":
                        dataset = dataset[1:]

                    prior = ""
                    for item in dataset:
                        if item["role"] == "scammer":
                            prior += f"Scammer: {item['content']}\n"
                        if item["role"] == "baiter":
                            create_scam_bait_trainset(prior, item['content'])
                            prior += f"Baiter: {item['content']}\n"


if __name__ == "__main__":
    main()
