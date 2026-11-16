import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
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
from model import QwenLoRADecisionModel
from dataset import DecisionBatchCollate
from evaluate import evaluate_decision_model

def get_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    return trainable_params, all_param

def train_phase2(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    print(f"=== Qwen Decision Head Phase 2: QLoRA Joint Training ===")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Tensor Core TF32: Enabled")
    print(f"Max Sequence Length: {args.max_seq_len}")
    print(f"Backbone: {args.model_name}")
    print(f"LoRA Target Modules: ['q_proj', 'k_proj', 'v_proj', 'o_proj']")
    print(f"Checkpoint Interval: Every {args.checkpoint_interval:,} samples")
    print(f"Batch Size: {args.batch_size} (Grad Accum: {args.gradient_accumulation_steps} -> Effective: {args.batch_size * args.gradient_accumulation_steps})")
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.reports_dir, exist_ok=True)

    # Setup W&B if enabled
    wandb_run = None
    if args.use_wandb and args.wandb_mode != "disabled":
        try:
            import wandb
            mode = "offline" if args.offline else args.wandb_mode
            wandb_run = wandb.init(
                id=args.wandb_run_id,
                project=args.wandb_project,
                name=args.wandb_run_name,
                resume="allow",
                mode=mode,
                config=vars(args),
                tags=["phase-2", "lora", "joint-training", "qwen2.5-1.5b"]
            )
            if mode == "offline":
                print(f"Weights & Biases running in OFFLINE mode (saving metrics locally to ./wandb).")
                print(f"You can sync anytime later when connected to internet using: wandb sync ./wandb/latest-run")
            else:
                print(f"Weights & Biases initialized: {wandb_run.url}")
        except Exception as e:
            print(f"W&B init skipped: {e}")

    print("\n[1/4] Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("\n[2/4] Initializing Qwen LoRA Decision Model...")
    head_weights = args.warm_start_head if (args.warm_start_head and os.path.exists(args.warm_start_head)) else None
    
    model = QwenLoRADecisionModel(
        model_name_or_path=args.model_name,
        max_choices=args.max_choices,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        head_weights_path=head_weights
    )
    model.to(device)

    trainable_p, total_p = get_trainable_parameters(model)
    print(f"Trainable Parameters: {trainable_p:,} / {total_p:,} ({100 * trainable_p / total_p:.3f}%)")

    # Grouped optimizer parameters: different learning rates for LoRA vs Head
    lora_params = [p for n, p in model.peft_model.named_parameters() if p.requires_grad]
    head_params = [p for p in model.head.parameters() if p.requires_grad]

    optimizer_grouped_parameters = [
        {"params": lora_params, "lr": args.lr_lora, "weight_decay": args.weight_decay},
        {"params": head_params, "lr": args.lr_head, "weight_decay": 1e-4}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters)

    start_sample = 0
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming training from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.head.load_state_dict(ckpt["head_state_dict"])
        if "lora_state_dict" in ckpt:
            model.peft_model.load_state_dict(ckpt["lora_state_dict"])
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
    test_data = dataset_dict.get("test", None)
    ood_data = dataset_dict.get("ood", None)

    if args.max_samples is not None and args.max_samples < len(train_data):
        train_data = train_data.select(range(args.max_samples))
        print(f"Selected slice of {len(train_data):,} samples for training.")

    total_train_samples = len(train_data)
    print(f"Total training samples for this run: {total_train_samples:,}")

    # Build evaluation loaders
    test_loader = None
    ood_loader = None
    if test_data is not None:
        test_subset = test_data.select(range(min(len(test_data), args.eval_samples)))
        test_loader = DataLoader(test_subset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False, pin_memory=True)
    if ood_data is not None:
        ood_subset = ood_data.select(range(min(len(ood_data), args.eval_samples)))
        ood_loader = DataLoader(ood_subset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False, pin_memory=True)

    print(f"\n[4/4] Starting Phase 2 Training...")
    if start_sample > 0 and start_sample < total_train_samples:
        train_data = train_data.select(range(start_sample, total_train_samples))

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    model.train()
    samples_seen = start_sample
    running_loss = 0.0
    accum_loss = 0.0
    log_interval = 20
    next_checkpoint_at = ((start_sample // args.checkpoint_interval) + 1) * args.checkpoint_interval

    start_time = time.time()
    last_log_time = time.time()
    batch_idx = 0

    optimizer.zero_grad()

    try:
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["targets"].to(device)
            num_choices = batch["num_choices"].to(device)
            current_batch_size = input_ids.size(0)

            logits = model(input_ids, attention_mask, num_choices=num_choices)
            log_probs = F.log_softmax(logits, dim=-1)

            # Soft targets KL divergence loss
            target_clamped = torch.clamp(targets, min=1e-8)
            loss = F.kl_div(log_probs, target_clamped, reduction="batchmean", log_target=False)
            
            # Normalize for gradient accumulation
            loss_scaled = loss / args.gradient_accumulation_steps
            loss_scaled.backward()

            samples_seen += current_batch_size
            running_loss += loss.item() * current_batch_size
            accum_loss += loss.item()
            batch_idx += 1

            # Optimizer step on gradient accumulation boundary
            if batch_idx % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            # Status logging
            if batch_idx % log_interval == 0:
                elapsed = time.time() - last_log_time
                samples_in_interval = log_interval * args.batch_size
                throughput = samples_in_interval / elapsed if elapsed > 0 else 0
                avg_running_loss = running_loss / samples_in_interval
                pct = (samples_seen / total_train_samples) * 100
                vram_gb = torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
                print(f"Step {batch_idx:6d} | Progress: {samples_seen:7,d}/{total_train_samples:,} ({pct:5.1f}%) | Loss: {avg_running_loss:.4f} | VRAM: {vram_gb:.2f}GB | Speed: {throughput:.1f} samples/s")
                
                if wandb_run:
                    # Skip re-logging steps that were already recorded before an interruption
                    curr_max_step = getattr(wandb_run, "step", 0)
                    if samples_seen >= curr_max_step:
                        wandb.log({
                            "samples_seen": samples_seen,
                            "train/loss": avg_running_loss,
                            "train/throughput": throughput,
                            "train/vram_gb": vram_gb
                        }, step=samples_seen)

                running_loss = 0.0
                last_log_time = time.time()

            # Checkpointing
            if samples_seen >= next_checkpoint_at:
                ckpt_tag = f"{next_checkpoint_at // 1000}k"
                ckpt_path = os.path.join(args.output_dir, f"decision_head_lora_{ckpt_tag}.pt")
                latest_path = os.path.join(args.output_dir, "decision_head_lora_latest.pt")
                
                print(f"\n>>> Reached {samples_seen:,} samples! Saving checkpoint: {ckpt_path}")
                ckpt_payload = {
                    "samples_seen": samples_seen,
                    "lora_state_dict": model.peft_model.state_dict(),
                    "head_state_dict": model.head.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "args": vars(args)
                }
                torch.save(ckpt_payload, ckpt_path)
                torch.save(ckpt_payload, latest_path)

                # Run evaluation on Test and OOD splits
                eval_metrics_dict = {}
                if test_loader is not None or ood_loader is not None:
                    print("Running Phase 2 Checkpoint Evaluation...")
                    model.eval()
                    if test_loader is not None:
                        test_metrics = evaluate_decision_model(model, test_loader, device)
                        print(f"--> [Test Split @ {ckpt_tag}]: Top-1 Acc = {test_metrics['top1_accuracy']}%, Loss = {test_metrics['eval_loss']}, TVD = {test_metrics['mean_tvd']}")
                        eval_metrics_dict["test_split"] = test_metrics
                        if wandb_run:
                            wandb.log({
                                "samples_seen": samples_seen,
                                "eval/loss": test_metrics["eval_loss"],
                                "eval/test_loss": test_metrics["eval_loss"],
                                "eval/accuracy": test_metrics["top1_accuracy"],
                                "eval/test_accuracy": test_metrics["top1_accuracy"],
                                "eval/test_tvd": test_metrics["mean_tvd"],
                                "eval/test_latency_ms": test_metrics["latency_ms_per_sample"],
                            }, step=samples_seen)

                    if ood_loader is not None:
                        ood_metrics = evaluate_decision_model(model, ood_loader, device)
                        print(f"--> [OOD Split  @ {ckpt_tag}]: Top-1 Acc = {ood_metrics['top1_accuracy']}%, Loss = {ood_metrics['eval_loss']}, TVD = {ood_metrics['mean_tvd']}")
                        eval_metrics_dict["ood_split"] = ood_metrics
                        if wandb_run:
                            wandb.log({
                                "samples_seen": samples_seen,
                                "eval/ood_loss": ood_metrics["eval_loss"],
                                "eval/ood_accuracy": ood_metrics["top1_accuracy"],
                                "eval/ood_tvd": ood_metrics["mean_tvd"],
                                "eval/ood_latency_ms": ood_metrics["latency_ms_per_sample"],
                            }, step=samples_seen)

                    # Save evaluation report to reports/phase_2/
                    report_file = os.path.join(args.reports_dir, f"eval_results_{ckpt_tag}.json")
                    with open(report_file, "w", encoding="utf-8") as f:
                        json.dump(eval_metrics_dict, f, indent=2)
                    print(f"Saved evaluation report to: {report_file}")

                    model.train()

                if args.max_samples is not None and samples_seen >= args.max_samples:
                    print(f"\n>>> Target reached: {samples_seen:,} samples (Checkpoint {ckpt_tag} complete)! Stopping as planned.")
                    break

                next_checkpoint_at += args.checkpoint_interval
                last_log_time = time.time()

    except KeyboardInterrupt:
        print("\n[!] Training paused by user. Saving emergency checkpoint...")
        emergency_path = os.path.join(args.output_dir, "decision_head_lora_paused.pt")
        torch.save({
            "samples_seen": samples_seen,
            "lora_state_dict": model.peft_model.state_dict(),
            "head_state_dict": model.head.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, emergency_path)
        print(f"Checkpoint saved to: {emergency_path}")
        return

    # Final save
    final_path = os.path.join(args.output_dir, "decision_head_lora_final.pt")
    torch.save({
        "samples_seen": samples_seen,
        "lora_state_dict": model.peft_model.state_dict(),
        "head_state_dict": model.head.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args)
    }, final_path)
    total_time = time.time() - start_time
    print(f"\n==========================================")
    print(f"Phase 2 Training Chunk Complete! Total Samples: {samples_seen:,} in {total_time/60:.2f} mins")
    print(f"Saved final weights to: {final_path}")
    print(f"==========================================")

    # Final comprehensive evaluation if reached end of dataset
    if test_loader is not None or ood_loader is not None:
        print("\nRunning Final Phase 2 Checkpoint Evaluation...")
        model.eval()
        eval_metrics_dict = {}
        if test_loader is not None:
            test_metrics = evaluate_decision_model(model, test_loader, device)
            print(f"--> [Final Test Split @ {samples_seen:,}]: Top-1 Acc = {test_metrics['top1_accuracy']}%, Loss = {test_metrics['eval_loss']}, TVD = {test_metrics['mean_tvd']}")
            eval_metrics_dict["test_split"] = test_metrics
            if wandb_run:
                wandb.log({
                    "samples_seen": samples_seen,
                    "eval/loss": test_metrics["eval_loss"],
                    "eval/test_loss": test_metrics["eval_loss"],
                    "eval/accuracy": test_metrics["top1_accuracy"],
                    "eval/test_accuracy": test_metrics["top1_accuracy"],
                    "eval/test_tvd": test_metrics["mean_tvd"],
                    "eval/test_latency_ms": test_metrics["latency_ms_per_sample"],
                }, step=samples_seen)

        if ood_loader is not None:
            ood_metrics = evaluate_decision_model(model, ood_loader, device)
            print(f"--> [Final OOD Split @ {samples_seen:,}]: Top-1 Acc = {ood_metrics['top1_accuracy']}%, Loss = {ood_metrics['eval_loss']}, TVD = {ood_metrics['mean_tvd']}")
            eval_metrics_dict["ood_split"] = ood_metrics
            if wandb_run:
                wandb.log({
                    "samples_seen": samples_seen,
                    "eval/ood_loss": ood_metrics["eval_loss"],
                    "eval/ood_accuracy": ood_metrics["top1_accuracy"],
                    "eval/ood_tvd": ood_metrics["mean_tvd"],
                    "eval/ood_latency_ms": ood_metrics["latency_ms_per_sample"],
                }, step=samples_seen)

        final_report_file = os.path.join(args.reports_dir, f"eval_results_final.json")
        with open(final_report_file, "w", encoding="utf-8") as f:
            json.dump(eval_metrics_dict, f, indent=2)
        print(f"Saved final evaluation report to: {final_report_file}")
    
    if wandb_run:
        wandb_run.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="models/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--dataset_name", type=str, default="SargeDev/jev-distill-corpus-v3")
    parser.add_argument("--data_path", type=str, default="data/jev-distill-corpus-v3")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--max_choices", type=int, default=10)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lr_lora", type=float, default=2e-4)
    parser.add_argument("--lr_head", type=float, default=4e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--checkpoint_interval", type=int, default=50000)
    parser.add_argument("--eval_samples", type=int, default=1500)
    parser.add_argument("--max_samples", type=int, default=50000)
    parser.add_argument("--output_dir", type=str, default="checkpoints/phase_2")
    parser.add_argument("--reports_dir", type=str, default="reports/phase_2")
    parser.add_argument("--warm_start_head", type=str, default="checkpoints/phase_1/decision_head_final.pt")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--use_wandb", action="store_true", default=True)
    parser.add_argument("--offline", action="store_true", default=False, help="Run without internet; saves W&B logs locally to sync later")
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb_project", type=str, default="qwen-decision-head")
    parser.add_argument("--wandb_run_name", type=str, default="phase-2-lora-joint-training")
    parser.add_argument("--wandb_run_id", type=str, default="phase-2-lora-joint-run")
    
    args = parser.parse_args()
    train_phase2(args)

