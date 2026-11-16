import os
import re
import json
import argparse
import wandb

def parse_task_log(log_path):
    """Extract (samples_seen, loss, speed) from task log lines."""
    records = []
    if not os.path.exists(log_path):
        return records
    pattern = re.compile(r"Progress:\s*([\d,]+)/[\d,]+.*Loss:\s*([\d.]+)\s*\|\s*Speed:\s*([\d.]+)")
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                samples = int(match.group(1).replace(",", ""))
                loss = float(match.group(2))
                speed = float(match.group(3))
                records.append({
                    "samples_seen": samples,
                    "train/loss": loss,
                    "train/throughput_samples_per_sec": speed
                })
    return records

def load_eval_json(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def main():
    parser = argparse.ArgumentParser(description="Upload completed Phase 1 training & evaluation metrics to Weights & Biases")
    parser.add_argument("--project", type=str, default="qwen-decision-head", help="W&B project name")
    parser.add_argument("--run_name", type=str, default="phase-1-frozen-backbone", help="W&B run name")
    parser.add_argument("--entity", type=str, default=None, help="W&B entity / team username")
    parser.add_argument("--api_key", type=str, default=None, help="W&B API Key")
    args = parser.parse_args()

    if args.api_key:
        wandb.login(key=args.api_key)
    else:
        # Check if already logged in or let wandb handle it
        wandb.login()

    run = wandb.init(
        project=args.project,
        name=args.run_name,
        entity=args.entity,
        notes="Phase 1: Frozen Qwen2.5-1.5B backbone with trained 2-layer Decision Head on SargeDev/jev-distill-corpus-v3 (655,806 samples).",
        tags=["phase-1", "frozen-backbone", "qwen2.5-1.5b", "jev-distill"],
        config={
            "phase": 1,
            "backbone": "models/Qwen2.5-1.5B-Instruct",
            "dataset": "SargeDev/jev-distill-corpus-v3",
            "total_samples": 655806,
            "batch_size": 32,
            "learning_rate": 1e-3,
            "optimizer": "AdamW",
            "loss_function": "KL-Divergence (Soft Teacher Target)",
            "architecture": {
                "head": "LayerNorm(1536) -> Linear(1536, 512) -> GELU -> Dropout(0.1) -> Linear(512, 10)",
                "backbone_status": "Frozen"
            },
            "hardware": "NVIDIA GeForce RTX 5060 Laptop GPU (8GB VRAM, sm_120)"
        }
    )

    # 1. Parse all training logs in order
    log_dir = r"C:\Users\omen\.gemini\antigravity\brain\bfd9b302-8dfc-4802-9c71-f46dfb39cd21\.system_generated\tasks"
    training_tasks = [
        "task-349.log", # 0 -> 100k
        "task-356.log", # 100k -> 200k
        "task-389.log", # 200k -> 300k
        "task-414.log", # 300k -> 400k
        "task-436.log", # 400k -> 500k
        "task-458.log", # 500k -> 600k
        "task-482.log", # 600k -> 655.8k
    ]

    all_train_steps = []
    seen_samples = set()
    for task_file in training_tasks:
        full_path = os.path.join(log_dir, task_file)
        steps = parse_task_log(full_path)
        for s in steps:
            # de-duplicate if any overlap
            if s["samples_seen"] not in seen_samples:
                seen_samples.add(s["samples_seen"])
                all_train_steps.append(s)

    all_train_steps.sort(key=lambda x: x["samples_seen"])
    print(f"Extracted {len(all_train_steps)} training steps from logs.")

    # 2. Evaluation results mapped by sample count
    eval_checkpoints = [
        {"samples": 100000, "tag": "100k", "file": "eval_results.json"},
        {"samples": 200000, "tag": "200k", "file": "eval_results_200k.json"},
        {"samples": 300000, "tag": "300k", "file": "eval_results_300k.json"},
        {"samples": 400000, "tag": "400k", "file": "eval_results_400k.json"},
        {"samples": 500000, "tag": "500k", "file": "eval_results_500k.json"},
        {"samples": 600000, "tag": "600k", "file": "eval_results_600k.json"},
        {"samples": 655806, "tag": "Final (655.8k)", "file": "eval_results_final.json"},
    ]

    table_data = []
    eval_lookup = {}

    for ec in eval_checkpoints:
        data = load_eval_json(ec["file"])
        if data:
            test_info = data.get("test_split", {})
            ood_info = data.get("ood_split", {})
            eval_metrics = {
                "eval/loss": test_info.get("mean_kl_divergence"),
                "eval/test_loss": test_info.get("mean_kl_divergence"),
                "eval/ood_loss": ood_info.get("mean_kl_divergence"),
                "eval/accuracy": test_info.get("top1_accuracy"),
                "eval/test_accuracy": test_info.get("top1_accuracy"),
                "eval/test_kl_div": test_info.get("mean_kl_divergence"),
                "eval/test_tvd": test_info.get("mean_tvd"),
                "eval/test_latency_ms": test_info.get("latency_ms_per_sample"),
                "eval/ood_accuracy": ood_info.get("top1_accuracy"),
                "eval/ood_kl_div": ood_info.get("mean_kl_divergence"),
                "eval/ood_tvd": ood_info.get("mean_tvd"),
                "eval/ood_latency_ms": ood_info.get("latency_ms_per_sample"),
            }
            eval_lookup[ec["samples"]] = eval_metrics
            table_data.append([
                ec["tag"],
                ec["samples"],
                test_info.get("top1_accuracy"),
                test_info.get("mean_kl_divergence"),
                test_info.get("mean_tvd"),
                ood_info.get("top1_accuracy"),
                ood_info.get("mean_kl_divergence"),
                ood_info.get("mean_tvd"),
                test_info.get("latency_ms_per_sample")
            ])

    # 3. Stream all steps into W&B
    print("Uploading training trajectory to W&B...")
    current_eval_idx = 0
    eval_sorted_keys = sorted(eval_lookup.keys())

    for step_data in all_train_steps:
        samples = step_data["samples_seen"]
        log_payload = {
            "samples_seen": samples,
            "train/loss": step_data["train/loss"],
            "train/throughput": step_data["train/throughput_samples_per_sec"],
        }
        # Check if an eval checkpoint occurred at or before this sample
        while current_eval_idx < len(eval_sorted_keys) and samples >= eval_sorted_keys[current_eval_idx]:
            eval_key = eval_sorted_keys[current_eval_idx]
            log_payload.update(eval_lookup[eval_key])
            current_eval_idx += 1

        wandb.log(log_payload, step=samples)

    # Log any remaining eval checkpoints (e.g. final)
    while current_eval_idx < len(eval_sorted_keys):
        eval_key = eval_sorted_keys[current_eval_idx]
        wandb.log({
            "samples_seen": eval_key,
            **eval_lookup[eval_key]
        }, step=eval_key)
        current_eval_idx += 1

    # 4. Upload summary table and evaluation artifacts
    columns = [
        "Checkpoint", "Samples Trained",
        "Test Acc (%)", "Test KL Div", "Test TVD",
        "OOD Acc (%)", "OOD KL Div", "OOD TVD", "Latency (ms)"
    ]
    summary_table = wandb.Table(columns=columns, data=table_data)
    wandb.log({"Phase 1 Checkpoint Comparison": summary_table})

    # Upload all evaluation JSON result files as W&B Artifact
    eval_artifact = wandb.Artifact(name="phase-1-evaluation-reports", type="evaluation-metrics")
    for ec in eval_checkpoints:
        if os.path.exists(ec["file"]):
            eval_artifact.add_file(ec["file"])
    run.log_artifact(eval_artifact)

    # Summary metrics
    final_data = load_eval_json("eval_results_final.json")
    if final_data:
        wandb.summary["final/eval_loss"] = final_data["test_split"]["mean_kl_divergence"]
        wandb.summary["final/test_loss"] = final_data["test_split"]["mean_kl_divergence"]
        wandb.summary["final/ood_loss"] = final_data["ood_split"]["mean_kl_divergence"]
        wandb.summary["final/accuracy"] = final_data["test_split"]["top1_accuracy"]
        wandb.summary["final/test_accuracy"] = final_data["test_split"]["top1_accuracy"]
        wandb.summary["final/test_kl_div"] = final_data["test_split"]["mean_kl_divergence"]
        wandb.summary["final/test_tvd"] = final_data["test_split"]["mean_tvd"]
        wandb.summary["final/ood_accuracy"] = final_data["ood_split"]["top1_accuracy"]
        wandb.summary["final/ood_kl_div"] = final_data["ood_split"]["mean_kl_divergence"]
        wandb.summary["final/ood_tvd"] = final_data["ood_split"]["mean_tvd"]
        wandb.summary["final/latency_ms"] = final_data["test_split"]["latency_ms_per_sample"]

    print(f"\nSuccessfully logged all metrics to Weights & Biases!")
    print(f"View Run Dashboard: {run.url}")
    wandb.finish()

if __name__ == "__main__":
    main()

