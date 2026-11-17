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
from datasets import load_dataset

# Ensure unbuffered printing
print = functools.partial(print, flush=True)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import QwenLoRADecisionModel
from dataset import DecisionBatchCollate

def train_phase3(
    model_name: str = "export/Qwen2.5-1.5B-Decision-Head-Merged",
    data_file: str = "data/phase3_consistency/consistency_train.jsonl",
    output_dir: str = "checkpoints/phase3_consistency",
    batch_size: int = 8,
    grad_accum: int = 2,
    lr_lora: float = 5e-5,
    lr_head: float = 1e-4,
    epochs: int = 1,
    max_seq_len: int = 384,
    max_choices: int = 10
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    print("=" * 70)
    print("PHASE 3: AXIOMATIC CONSISTENCY TRAINING (JEV-CALIBRATION)")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Base Model: {model_name}")
    print(f"Dataset: {data_file}")
    print(f"Batch Size: {batch_size} (Grad Accum: {grad_accum} -> Effective: {batch_size * grad_accum})")
    print("=" * 70)

    os.makedirs(output_dir, exist_ok=True)

    print("\n[1/4] Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("\n[2/4] Initializing Qwen Decision Model for Consistency Training...")
    head_weights = os.path.join(model_name, "decision_head.safetensors")
    if not os.path.exists(head_weights):
        head_weights = os.path.join(model_name, "decision_head.pt")
        
    model = QwenLoRADecisionModel(
        model_name_or_path=model_name,
        max_choices=max_choices,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        head_weights_path=head_weights if os.path.exists(head_weights) else None
    )
    model.to(device)

    # Grouped optimizer
    lora_params = [p for n, p in model.peft_model.named_parameters() if p.requires_grad]
    head_params = [p for p in model.head.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": lora_params, "lr": lr_lora, "weight_decay": 0.01},
        {"params": head_params, "lr": lr_head, "weight_decay": 1e-4}
    ])

    print("\n[3/4] Loading Consistency Dataset...")
    ds = load_dataset("json", data_files=data_file, split="train")
    print(f"Total consistency training examples: {len(ds):,}")
    
    collate_fn = DecisionBatchCollate(tokenizer, max_length=max_seq_len, max_choices=max_choices)
    train_loader = DataLoader(
        ds,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    print("\n[4/4] Starting Consistency Training Loop...")
    model.train()
    total_steps = len(train_loader) * epochs
    global_step = 0
    start_time = time.time()
    last_log_time = time.time()
    running_loss = 0.0
    log_interval = 25

    optimizer.zero_grad()
    
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch + 1}/{epochs} ---")
        for batch_idx, batch in enumerate(train_loader, 1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["targets"].to(device)
            num_choices = batch["num_choices"].to(device)
            
            logits = model(input_ids, attention_mask, num_choices=num_choices)
            log_probs = F.log_softmax(logits, dim=-1)

            # Soft targets KL divergence loss
            target_clamped = torch.clamp(targets, min=1e-8)
            loss = F.kl_div(log_probs, target_clamped, reduction="batchmean", log_target=False)
            
            (loss / grad_accum).backward()
            running_loss += loss.item()

            if batch_idx % grad_accum == 0 or batch_idx == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            if batch_idx % log_interval == 0:
                elapsed = time.time() - last_log_time
                samples_in_interval = log_interval * batch_size
                throughput = samples_in_interval / elapsed if elapsed > 0 else 0
                avg_loss = running_loss / log_interval
                running_loss = 0.0
                last_log_time = time.time()
                pct = (batch_idx / len(train_loader)) * 100
                vram_gb = torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
                print(f"Step {global_step:5d}/{total_steps} | Epoch {epoch+1} ({pct:5.1f}%) | Loss: {avg_loss:.4f} | Speed: {throughput:.1f} s/s | VRAM: {vram_gb:.2f}GB")

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.1f}s ({total_time / 60:.1f} min)")

    # Save final model
    print(f"Saving fine-tuned consistency checkpoint to: {output_dir}")
    save_path = os.path.join(output_dir, "phase3_consistency_final.pt")
    torch.save({
        "head_state_dict": model.head.state_dict(),
        "lora_state_dict": model.peft_model.state_dict(),
        "total_samples": len(ds)
    }, save_path)
    
    # Save standalone head weights
    from safetensors.torch import save_file
    save_file(model.head.state_dict(), os.path.join(output_dir, "decision_head.safetensors"))
    
    # Save LoRA adapter
    lora_dir = os.path.join(output_dir, "lora_adapter")
    model.peft_model.save_pretrained(lora_dir)
    print(f"Successfully saved Phase 3 model & LoRA adapter!")
    return output_dir

if __name__ == "__main__":
    train_phase3()
