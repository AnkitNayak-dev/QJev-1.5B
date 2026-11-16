import torch
import torch.nn.functional as F
import numpy as np
import time
from typing import Dict, Any, List

def evaluate_decision_model(model, dataloader, device) -> Dict[str, float]:
    """
    Evaluates the Qwen Decision Head on a given dataloader.
    Returns Top-1 Accuracy, Average KL Divergence, TVD (Total Variation Distance), and Latency.
    """
    model.eval()
    total_kl = 0.0
    total_top1_match = 0
    total_samples = 0
    total_tvd = 0.0
    
    start_time = time.time()
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["targets"].to(device)
            num_choices = batch["num_choices"].to(device)
            batch_size = input_ids.size(0)

            logits = model(input_ids, attention_mask, num_choices=num_choices)
            # Logits are masked with -1e4 for invalid choices
            log_probs = F.log_softmax(logits, dim=-1)
            pred_probs = torch.softmax(logits, dim=-1)

            # KL divergence: KL(Target || Pred) or KL(Pred || Target)
            # Soft targets KL: sum(targets * (log(targets + 1e-10) - log_probs))
            target_clamped = torch.clamp(targets, min=1e-8)
            kl = F.kl_div(log_probs, target_clamped, reduction="batchmean", log_target=False)
            total_kl += kl.item() * batch_size

            # Top 1 argmax match
            pred_top1 = torch.argmax(logits, dim=-1)
            target_top1 = torch.argmax(targets, dim=-1)
            matches = (pred_top1 == target_top1).sum().item()
            total_top1_match += matches

            # Total Variation Distance: 0.5 * sum(|pred - target|)
            tvd = 0.5 * torch.sum(torch.abs(pred_probs - targets), dim=-1).sum().item()
            total_tvd += tvd

            total_samples += batch_size

    elapsed = time.time() - start_time
    latency_ms_per_sample = (elapsed / total_samples) * 1000 if total_samples > 0 else 0

    return {
        "samples": total_samples,
        "eval_loss": round(total_kl / total_samples, 4) if total_samples > 0 else 0.0,
        "top1_accuracy": round((total_top1_match / total_samples) * 100, 2) if total_samples > 0 else 0.0,
        "mean_kl_divergence": round(total_kl / total_samples, 4) if total_samples > 0 else 0.0,
        "mean_tvd": round(total_tvd / total_samples, 4) if total_samples > 0 else 0.0,
        "latency_ms_per_sample": round(latency_ms_per_sample, 2),
        "total_eval_time_sec": round(elapsed, 2)
    }

