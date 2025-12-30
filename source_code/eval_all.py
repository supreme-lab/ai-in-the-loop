import os
import json
import eval_for_f1_auprc_fpr_fnr as eval_faff
import eval_safeness_risk_awareness as eval_sra
import eval_scam_baiting_scam_pii_engage_time as eval_sbspet


MODEL_NAMEs = [
        "meta-llama/LlamaGuard-7b",
        "meta-llama/Meta-Llama-Guard-2-8B",
        "meta-llama/Llama-Guard-3-8B",
        "OpenSafetyLab/MD-Judge-v0.1",
    ]
tuned_models = ["llama-guard", "llama-guard2", "llama-guard3", "md-judge"]

def run_eval_for_f1_auprc_fpr_fnr():

    for dataset_name in ["masc", "sasc", "ssc", "ssd"]:
        input_file_path = f"ai-in-the-loop/data/classification/{dataset_name}_dataset/all_data.chat.json"

        for i, model_name in enumerate(MODEL_NAMEs):
            print(f"Evaluating model: {model_name}")
            tuned_model = tuned_models[i]
            pretrained_path = f"ai-in-the-loop/results/fine-tuned/multi-task/tuned-{tuned_model}"
            output_file_path = f"ai-in-the-loop/results/reports/{dataset_name}/eval_f1_fpr_fnr_{tuned_model}.json"

            print("##Starting 1st evaluation...")
            eval_faff.generate_batch_output(
                input_file_path,
                output_file_path,
                dataset_name,
                tuned_model,
                model_name,
                pretrained_path,
            )
            print("Evaluation complete!!")


def run_eval_safeness_risk_awareness():
    """
    Run evaluation for:
      - Engagement/PII/scam-risk/scam-baiter/moderation metrics
      - On combined conversation dataset.
    This corresponds to your 2nd script.
    """
    data_dir = "ai-in-the-loop/data/generation/selected_conversation_to_scam_baiter_performance.jsonl"

    for i, model_name in enumerate(MODEL_NAMEs):
        print(f"Evaluating model: {model_name}")
        tuned_model = tuned_models[i]
        pretrained_path = f"ai-in-the-loop/results/fine-tuned/multi-task/tuned-{tuned_model}"
        output_dir = f'ai-in-the-loop/results/reports/multi_task/eval_safeness_risk_{tuned_model}.json'

        print("##Starting 2nd evaluation...")
        # Note: keeping your original signature here:
        # generate_batch_output(data_dir, "", tuned_model, model_name, pretrained_path)
        eval_sra.generate_batch_output(
            data_dir,
            output_dir,
            tuned_model,
            model_name,
            pretrained_path,
        )
        print("Evaluation complete!!")


def run_eval_scam_baiting_scam_pii_engage_time():
    """
    Evaluate AI scam-baiter turns against reference/scammer messages.
    Uses ASB dataset under ai-in-the-loop/data/generation/all_eval_data/combined_asb_sbc_ytsc_dataset.jsonl.
    This corresponds to your 3rd script.
    """
    input_file_path = "ai-in-the-loop/data/generation/all_eval_data/combined_asb_sbc_ytsc_dataset.jsonl"
    with open(input_file_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    dataset_name = input_file_path.split("/")[-1].split("_")[0]

    for i, model_name in enumerate(MODEL_NAMEs):
        print(f"Evaluating model: {model_name}")
        tuned_model = tuned_models[i]

        pretrained_path = f"ai-in-the-loop/results/fine-tuned/multi-task/tuned-{tuned_model}"
        output_file_path = f"ai-in-the-loop/results/reports/eval_baiting_scam_pii_engagement{tuned_model}.json"

        print("##Starting 3rd evaluation...")
        eval_sbspet.generate_batch_output(
            dataset,
            output_file_path,
            dataset_name,
            tuned_model,
            model_name,
            pretrained_path,
        )
        print("Evaluation complete!!")


if __name__ == "__main__":
    # Call whichever combination you want:
    # run_eval_for_f1_auprc_fpr_fnr()
    # run_eval_safeness_risk_awareness()
    # run_eval_scam_baiting_scam_pii_engage_time()
    pass


# CUDA_VISIBLE_DEVICES=2 nohup python eval_all.py > ai-in-the-loop/logs/eval_all.log 2>&1 &