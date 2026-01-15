import json, argparse
import eval_for_f1_auprc_fpr_fnr as eval_faff
import eval_safeness_risk_awareness as eval_sra
import eval_scam_baiting_scam_pii_engage_time as eval_sbspet
import os
import qualitative_evaluation as qual_eval

PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = PARENT_DIR.rsplit("/", 2)[0]

MODEL_NAMEs = [
        "meta-llama/LlamaGuard-7b",
        "meta-llama/Meta-Llama-Guard-2-8B",
        "meta-llama/Llama-Guard-3-8B",
        "OpenSafetyLab/MD-Judge-v0.1",
    ]
tuned_models = ["llama-guard", "llama-guard2", "llama-guard3", "md-judge"]

def run_eval_for_f1_auprc_fpr_fnr():

    for dataset_name in ["masc", "sasc", "ssc", "ssd"]:
        first_input_file = f"{PARENT_DIR}/ai-in-the-loop/data/classification/{dataset_name}_dataset/all_data.chat.json"

        if not os.path.exists(first_input_file):
                print(f"Dataset, {tuned_model} is not available!")
                continue

        for i, model_name in enumerate(MODEL_NAMEs):
            print(f"Evaluating model: {model_name}")
            tuned_model = tuned_models[i]
            pretrained_path = f"{PARENT_DIR}/ai-in-the-loop/results/fine-tuned/multi-task/tuned-{tuned_model}"

            if not os.path.exists(pretrained_path):
                print(f"Fine-tuned model, {tuned_model} is not available!")
                continue

            second_input_file = f"{PARENT_DIR}/ai-in-the-loop/results/reports/multi_task/eval_{dataset_name}_by_md-judge.json"
            output_file = f"{PARENT_DIR}/ai-in-the-loop/results/reports/eval_result_of_{dataset_name}_by_{tuned_model}_for_evaluating_f1_auprc_fnr_fpr.json",

            print("##Starting 1st evaluation...")
            eval_faff.generate_batch_output(
                first_input_file,
                second_input_file,
                output_file,
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
    data_dir = f"{PARENT_DIR}/ai-in-the-loop/data/generation/selected_conversation_to_scam_baiter_performance.jsonl"

    for i, model_name in enumerate(MODEL_NAMEs):
        print(f"Evaluating model: {model_name}")
        tuned_model = tuned_models[i]
        pretrained_path = f"{PARENT_DIR}/ai-in-the-loop/results/fine-tuned/multi-task/tuned-{tuned_model}"

        if not os.path.exists(pretrained_path):
            print(f"Fine-tuned model, {tuned_model} is not available!")
            continue
        
        output_dir = f'{PARENT_DIR}/ai-in-the-loop/results/reports/eval_safeness_risk_{tuned_model}.json'

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
    input_file_path = f"{PARENT_DIR}/ai-in-the-loop/data/generation/all_eval_data/combined_asb_sbc_ytsc_dataset.jsonl"
    with open(input_file_path, "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f if line.strip()]

    dataset_name = input_file_path.split("/")[-1].split("_")[0]

    for i, model_name in enumerate(MODEL_NAMEs):
        print(f"Evaluating model: {model_name}")
        tuned_model = tuned_models[i]

        pretrained_path = f"{PARENT_DIR}/ai-in-the-loop/results/fine-tuned/multi-task/tuned-{tuned_model}"

        if not os.path.exists(pretrained_path):
            print(f"Fine-tuned model, {tuned_model} is not available!")
            continue
        
        output_file_path = f"{PARENT_DIR}/ai-in-the-loop/results/reports/eval_baiting_scam_pii_engagement_{tuned_model}.json"

        print("##Starting 3rd evaluation...")
        eval_sbspet.generate_batch_output(
            dataset[:10], # you can remove the slicing for full dataset
            output_file_path,
            dataset_name,
            tuned_model,
            model_name,
            pretrained_path,
        )
        print("Evaluation complete!!")


if __name__ == "__main__":
    # Call whichever combination you want:
    run_eval_safeness_risk_awareness()
    print("------> evaluation for safeness risk awareness ended!!")
    run_eval_for_f1_auprc_fpr_fnr()
    print("------> evaluation for f1 auprc fpr fnr ended!!")
    run_eval_scam_baiting_scam_pii_engage_time()
    print("------> evaluation for scam_baiting scam pii engage time ended!!")
    
    # Qualitative Evaluation
    print("------> Qualitative Evaluation started...")
    ap = argparse.ArgumentParser(description="Evaluate AI scam-baiter turns against reference and scammer msg.")
    ap.add_argument("--data", default=f'{PARENT_DIR}/ai-in-the-loop/results/reports/turns.csv', help="CSV or JSONL with fields: scammer,reference,ai_response")
    ap.add_argument("--out", default=f'{PARENT_DIR}/ai-in-the-loop/results/reports/scores.csv', help="Output CSV filepath")
    ap.add_argument("--no_bertscore", action="store_true", help="Disable BERTScore")
    ap.add_argument("--no_relevance", action="store_true", help="Disable reference-free relevance")
    args = ap.parse_args()
    qual_eval.main(args.data, args.out,
         use_bertscore=(not args.no_bertscore), use_rel=(not args.no_relevance))
    print("------> Qualitative Evaluation Ended!!")


# CUDA_VISIBLE_DEVICES=1 nohup python eval_all.py > /home/ihossain/ISMAIL/SUPREMELAB/ai-in-the-loop/logs/eval_all.log 2>&1 &