import os
import sys
import argparse
import json
import functools
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_from_disk, load_dataset

print = functools.partial(print, flush=True)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import QwenDecisionModel
from dataset import DecisionBatchCollate
from evaluate import evaluate_decision_model

def run_eval(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Running Comprehensive Decision Head Evaluation ===")
    print(f"Model: {args.model_name}")
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = QwenDecisionModel(
        model_name_or_path=args.model_name,
        max_choices=args.max_choices,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    )

    if os.path.exists(args.checkpoint_path):
        print(f"Loading checkpoint weights from: {args.checkpoint_path}")
        ckpt = torch.load(args.checkpoint_path, map_location="cpu")
        state_dict = ckpt["head_state_dict"] if "head_state_dict" in ckpt else ckpt
        model.head.load_state_dict(state_dict)
    else:
        print(f"Warning: Checkpoint {args.checkpoint_path} not found! Testing with randomly initialized head.")

    model.to(device)
    model.eval()

    collate_fn = DecisionBatchCollate(tokenizer, max_length=args.max_seq_len, max_choices=args.max_choices)
    results = {}

    if os.path.exists(args.data_path):
        print(f"Loading dataset from disk: {args.data_path}")
        dataset_dict = load_from_disk(args.data_path)
        test_data = dataset_dict["test"]
        ood_data = dataset_dict["ood"]
    else:
        test_data = load_dataset(args.dataset_name, split="test")
        ood_data = load_dataset(args.dataset_name, split="ood")

    # 1. In-distribution Test Split
    print("\n[1/2] Evaluating In-Distribution 'test' Split...")
    test_slice = test_data.select(range(min(len(test_data), args.eval_samples)))
    test_loader = DataLoader(test_slice, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False)
    test_metrics = evaluate_decision_model(model, test_loader, device)
    print(f"--> Test Split ({test_metrics['samples']} samples):")
    print(f"    Top-1 Accuracy: {test_metrics['top1_accuracy']}%")
    print(f"    Mean KL Divergence: {test_metrics['mean_kl_divergence']}")
    print(f"    Total Variation Distance (TVD): {test_metrics['mean_tvd']}")
    print(f"    Latency: {test_metrics['latency_ms_per_sample']} ms / decision")
    results["test_split"] = test_metrics

    # 2. Out-of-Distribution OOD Split
    print("\n[2/2] Evaluating Out-of-Distribution 'ood' Split...")
    ood_slice = ood_data.select(range(min(len(ood_data), args.eval_samples)))
    ood_loader = DataLoader(ood_slice, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False)
    ood_metrics = evaluate_decision_model(model, ood_loader, device)
    print(f"--> OOD Split ({ood_metrics['samples']} samples):")
    print(f"    Top-1 Accuracy: {ood_metrics['top1_accuracy']}%")
    print(f"    Mean KL Divergence: {ood_metrics['mean_kl_divergence']}")
    print(f"    Total Variation Distance (TVD): {ood_metrics['mean_tvd']}")
    print(f"    Latency: {ood_metrics['latency_ms_per_sample']} ms / decision")
    results["ood_split"] = ood_metrics

    # Save to json
    out_file = args.output_file
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved evaluation results to: {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="models/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--dataset_name", type=str, default="SargeDev/jev-distill-corpus-v3")
    parser.add_argument("--data_path", type=str, default="data/jev-distill-corpus-v3")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/decision_head_final.pt")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_seq_len", type=int, default=384)
    parser.add_argument("--max_choices", type=int, default=10)
    parser.add_argument("--eval_samples", type=int, default=1500)
    parser.add_argument("--output_file", type=str, default="eval_results.json")
    
    args = parser.parse_args()
    run_eval(args)


