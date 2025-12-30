import json
import os
from evaluate import load
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

data_path = "ai-in-the-loop/results/reports/multi_task/zero-shot1/scam-bait/conv_scammer_scam_baiter_md-judge_lora_merged.json"

eval_file_output = []
with open(data_path, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue
        try:
            evaluation_output = json.loads(line)
            eval_file_output.append(evaluation_output)
        except json.JSONDecodeError as e:
            print(f"JSONDecodeError on line {i}: {e}")
            print("Problematic line:")
            print(line)
            continue

# Step 1 — patch GPT-2 in the local cache so evaluate() will use it
model_id = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token  # use EOS as padding

model = AutoModelForCausalLM.from_pretrained(model_id)
model.config.pad_token_id = tokenizer.pad_token_id

# **Add this line to fix CUDA embedding error**
model.resize_token_embeddings(len(tokenizer))


# Save patched tokenizer & model to a local folder
patched_model_dir = "ai-in-the-loop/results/pre-trained/multi-task/patched_gpt2"
tokenizer.save_pretrained(patched_model_dir)
model.save_pretrained(patched_model_dir)


# Step 2 — use patched model in perplexity calculation
def cal_perplexity(predictions):
    perplexity = load("perplexity", module_type="metric")
    results = perplexity.compute(
        predictions=predictions,
        model_id=patched_model_dir,  # point to our fixed model
        batch_size=1,
        max_length=1024             # enforce GPT-2 context size
    )
    return results



# mean_perplexity_ai = []
# for entity in eval_file_output:
#     ai_texts = [item['ai_baiter_response'] for item in entity['conversation'] if item['ai_baiter_response'].strip()]
#     if ai_texts:
#         results = cal_perplexity(ai_texts)
#         mean_perplexity_ai.append(results['mean_perplexity'])
#     else:
#         mean_perplexity_ai.append(None)

# output_file = "ai-in-the-loop/results/reports/multi_task/zero-shot1/scam-bait/perplexity_results_ai.json"
# with open(output_file, 'w', encoding='utf-8') as f:
#     json.dump({
#         "mean_perplexity_ai": mean_perplexity_ai
#     }, f, indent=4)

mean_perplexity_ref = []
for entity in eval_file_output:
    # ref_texts = [item['reference_baiter'] for item in entity['conversation'] if item['reference_baiter'].strip()]
    ref_texts = []
    for item in entity['conversation']:
        if item['reference_baiter'] == None:
            print("None!!")
            continue
        if len(item['reference_baiter'])==0:
            # print(item['reference_baiter'])
            print("Lenght Zero!!")
            continue
        if not item['reference_baiter']:
            # print(item['reference_baiter'])
            print("Invalid TEXT!!")
            continue
        ref_texts.append(item['reference_baiter'])

    if ref_texts:
        results = cal_perplexity(ref_texts)
        mean_perplexity_ref.append(results['mean_perplexity'])
    # else:
    # mean_perplexity_ref.append(ref_texts)


# Save results to JSON
output_file = "ai-in-the-loop/results/reports/multi_task/zero-shot1/scam-bait/perplexity_results_ref.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump({
        "mean_perplexity_ref": mean_perplexity_ref
    }, f, indent=4)

print(f"Saved perplexity results to {output_file}")
