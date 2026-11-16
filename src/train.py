import os
import sys
import time
import argparse
import json
import functools
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_from_disk, load_dataset

# Ensure unbuffered printing
print = functools.partial(print, flush=True)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import QwenDecisionModel
from dataset import DecisionBatchCollate
from evaluate import evaluate_decision_model

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Qwen Decision Head Training ===")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Backbone: {args.model_name}")
    print(f"Checkpoint Interval: Every {args.checkpoint_interval:,} samples")
    print(f"Batch Size: {args.batch_size}")
    
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n[1/4] Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("\n[2/4] Initializing Qwen Decision Model (Frozen Backbone)...")
    model = QwenDecisionModel(
        model_name_or_path=args.model_name,
        max_choices=args.max_choices,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    )
    model.to(device)

    # Train only the head
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=args.learning_rate, weight_decay=1e-4)

    start_sample = 0
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming training from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.head.load_state_dict(ckpt["head_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_sample = ckpt.get("samples_seen", 0)
        print(f"Resuming from sample count: {start_sample:,}")

    collate_fn = DecisionBatchCollate(tokenizer, max_length=args.max_seq_len, max_choices=args.max_choices)

    print("\n[3/4] Loading Full Dataset from Disk...")
    if os.path.exists(args.data_path):
        print(f"Loading cached dataset from disk: {args.data_path}")
        dataset_dict = load_from_disk(args.data_path)
    else:
        print(f"Loading dataset directly via HuggingFace: {args.dataset_name}")
        dataset_dict = load_dataset(args.dataset_name)

    train_data = dataset_dict["train"]
    val_data = dataset_dict.get("validation", None)
    
    if args.max_samples is not None and args.max_samples < len(train_data):
        train_data = train_data.select(range(args.max_samples))
        print(f"Selected slice of {len(train_data):,} samples for training.")

    total_train_samples = len(train_data)
    print(f"Total training samples for this run: {total_train_samples:,}")

    val_loader = None
    if val_data is not None:
        val_subset = val_data.select(range(min(len(val_data), args.val_samples)))
        val_loader = DataLoader(
            val_subset,
            batch_size=args.batch_size,
            collate_fn=collate_fn,
            shuffle=False
        )
        print(f"Prepared validation set ({len(val_subset)} samples).")

    print(f"\n[4/4] Starting Training...")
    if start_sample > 0 and start_sample < total_train_samples:
        train_data = train_data.select(range(start_sample, total_train_samples))

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        num_workers=0
    )

    model.head.train()
    running_loss = 0.0
    samples_seen = start_sample
    batch_idx = 0
    start_time = time.time()
    last_log_time = start_time

    next_checkpoint_at = ((samples_seen // args.checkpoint_interval) + 1) * args.checkpoint_interval
    log_interval = 20 # log every 20 batches (~640 samples)

    try:
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["targets"].to(device)
            num_choices = batch["num_choices"].to(device)
            current_batch_size = input_ids.size(0)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask, num_choices=num_choices)
            log_probs = F.log_softmax(logits, dim=-1)

            # Soft targets KL divergence loss
            target_clamped = torch.clamp(targets, min=1e-8)
            loss = F.kl_div(log_probs, target_clamped, reduction="batchmean", log_target=False)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.head.parameters(), max_norm=1.0)
            optimizer.step()

            samples_seen += current_batch_size
            running_loss += loss.item() * current_batch_size
            batch_idx += 1

            # Log status
            if batch_idx % log_interval == 0:
                elapsed = time.time() - last_log_time
                samples_in_interval = log_interval * args.batch_size
                throughput = samples_in_interval / elapsed if elapsed > 0 else 0
                avg_running_loss = running_loss / samples_in_interval
                pct = (samples_seen / total_train_samples) * 100
                print(f"Step {batch_idx:6d} | Progress: {samples_seen:7,d}/{total_train_samples:,} ({pct:5.1f}%) | Loss: {avg_running_loss:.4f} | Speed: {throughput:.1f} samples/s")
                running_loss = 0.0
                last_log_time = time.time()

            # Checkpointing every 100k samples
            if samples_seen >= next_checkpoint_at:
                ckpt_tag = f"{next_checkpoint_at // 1000}k"
                ckpt_path = os.path.join(args.output_dir, f"decision_head_{ckpt_tag}.pt")
                latest_path = os.path.join(args.output_dir, "decision_head_latest.pt")
                
                print(f"\n>>> Reached {samples_seen:,} samples! Saving checkpoint: {ckpt_path}")
                ckpt_payload = {
                    "samples_seen": samples_seen,
                    "head_state_dict": model.head.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "args": vars(args)
                }
                torch.save(ckpt_payload, ckpt_path)
                torch.save(ckpt_payload, latest_path)

                if val_loader is not None:
                    print("Running Validation Check...")
                    val_metrics = evaluate_decision_model(model, val_loader, device)
                    print(f"Validation @ {ckpt_tag}: Top-1 Acc = {val_metrics['top1_accuracy']}%, Mean KL = {val_metrics['mean_kl_divergence']}, TVD = {val_metrics['mean_tvd']}")
                    model.head.train()

                if args.max_samples is not None and samples_seen >= args.max_samples:
                    print(f"\n>>> Target reached: {samples_seen:,} samples (Checkpoint {ckpt_tag} complete)! Stopping as planned.")
                    break

                next_checkpoint_at += args.checkpoint_interval
                last_log_time = time.time()

    except KeyboardInterrupt:
        print("\n[!] Training paused by user. Saving emergency checkpoint...")
        emergency_path = os.path.join(args.output_dir, "decision_head_paused.pt")
        torch.save({
            "samples_seen": samples_seen,
            "head_state_dict": model.head.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, emergency_path)
        print(f"Checkpoint saved to: {emergency_path}. You can resume anytime using --resume {emergency_path}")
        return

    # Final save
    final_path = os.path.join(args.output_dir, "decision_head_final.pt")
    torch.save({
        "samples_seen": samples_seen,
        "head_state_dict": model.head.state_dict(),
        "args": vars(args)
    }, final_path)
    total_time = time.time() - start_time
    print(f"\n==========================================")
    print(f"Training Complete! Total Samples: {samples_seen:,} in {total_time/60:.2f} mins")
    print(f"Saved final weights to: {final_path}")
    print(f"==========================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="models/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--dataset_name", type=str, default="SargeDev/jev-distill-corpus-v3")
    parser.add_argument("--data_path", type=str, default="data/jev-distill-corpus-v3")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_seq_len", type=int, default=384)
    parser.add_argument("--max_choices", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--checkpoint_interval", type=int, default=100000)
    parser.add_argument("--val_samples", type=int, default=1500)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, default=None)
    
    args = parser.parse_args()
    train(args)
