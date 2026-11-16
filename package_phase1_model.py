import os
import json
import shutil
import torch
from safetensors.torch import save_file

def main():
    print("=== Packaging Phase 1 Decision Head for Release & Backup ===")
    
    ckpt_path = "checkpoints/decision_head_final.pt"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")
    
    ckpt = torch.load(ckpt_path, map_location="cpu")
    head_state_dict = ckpt["head_state_dict"]
    
    # 1. Create permanent backup in checkpoints/phase1_backup/
    backup_dir = "checkpoints/phase1_backup"
    os.makedirs(backup_dir, exist_ok=True)
    shutil.copy2(ckpt_path, os.path.join(backup_dir, "decision_head_phase1_final.pt"))
    print(f"Backed up checkpoint to: {backup_dir}/decision_head_phase1_final.pt")

    # 2. Create standalone release package in export/Qwen2.5-1.5B-Decision-Head/
    export_dir = "export/Qwen2.5-1.5B-Decision-Head"
    os.makedirs(export_dir, exist_ok=True)

    # Save safetensors
    safetensors_path = os.path.join(export_dir, "decision_head.safetensors")
    save_file(head_state_dict, safetensors_path)
    print(f"Exported safetensors weights: {safetensors_path}")

    # Save pytorch .pt
    pt_path = os.path.join(export_dir, "decision_head.pt")
    torch.save(head_state_dict, pt_path)
    print(f"Exported PyTorch weights: {pt_path}")

    # Save config.json
    config = {
        "model_type": "qwen_decision_head",
        "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
        "hidden_size": 1536,
        "head_intermediate_size": 512,
        "max_choices": 10,
        "dropout": 0.1,
        "activation": "gelu",
        "output_type": "jev_probabilities",
        "training_corpus": "SargeDev/jev-distill-corpus-v3",
        "training_samples": 655806,
        "phase": 1,
        "metrics": {
            "test_accuracy_pct": 75.67,
            "test_kl_divergence": 0.1757,
            "test_tvd": 0.1664,
            "ood_accuracy_pct": 57.53,
            "ood_kl_divergence": 0.7884,
            "latency_ms_per_sample": 42.94
        }
    }
    with open(os.path.join(export_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"Exported config.json")

    # Copy clean modeling code
    modeling_code = '''import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

class DecisionHead(nn.Module):
    def __init__(self, hidden_size: int = 1536, intermediate_size: int = 512, max_choices: int = 10, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.dense = nn.Linear(hidden_size, intermediate_size)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(intermediate_size, max_choices)
        self.score_proj = nn.Linear(intermediate_size, 1)

    def forward(self, x: torch.Tensor, num_choices: torch.Tensor = None) -> torch.Tensor:
        h = self.norm(x)
        h = self.act(self.dense(h))
        h = self.dropout(h)
        logits = self.out_proj(h)

        if num_choices is not None:
            batch_size = logits.size(0)
            seq_range = torch.arange(logits.size(-1), device=logits.device).unsqueeze(0).expand(batch_size, -1)
            mask = seq_range >= num_choices.unsqueeze(1)
            logits = logits.masked_fill(mask, -1e4)

        return logits

class QwenDecisionModel(nn.Module):
    def __init__(self, base_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct", head_weights_path: str = None, device: str = "cuda"):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.backbone = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            dtype=torch.bfloat16,
            device_map=self.device
        )
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.head = DecisionHead(
            hidden_size=self.backbone.config.hidden_size,
            intermediate_size=512,
            max_choices=10
        ).to(self.device).to(torch.bfloat16)

        if head_weights_path:
            if head_weights_path.endswith(".safetensors"):
                from safetensors.torch import load_file
                state = load_file(head_weights_path)
            else:
                raw = torch.load(head_weights_path, map_location=self.device)
                state = raw.get("head_state_dict", raw)
            self.head.load_state_dict(state)

    @torch.no_grad()
    def predict(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, num_choices: torch.Tensor = None) -> torch.Tensor:
        outputs = self.backbone.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state
        last_token_indices = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(input_ids.size(0), device=input_ids.device)
        pooled = last_hidden[batch_indices, last_token_indices]
        logits = self.head(pooled, num_choices=num_choices)
        return torch.softmax(logits, dim=-1)
'''
    with open(os.path.join(export_dir, "modeling_decision_head.py"), "w", encoding="utf-8") as f:
        f.write(modeling_code)
    print(f"Exported modeling_decision_head.py")

    # Create quickstart inference script
    predict_code = '''import torch
from transformers import AutoTokenizer
from modeling_decision_head import QwenDecisionModel

def main():
    print("Loading Qwen Decision Head Model...")
    model = QwenDecisionModel(
        base_model_name="Qwen/Qwen2.5-1.5B-Instruct",
        head_weights_path="decision_head.safetensors"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

    system_prompt = "You are a fast decision-making AI. Select the best option from the choices below."
    question = "Which planet is known as the Red Planet?"
    choices = ["Venus", "Mars", "Jupiter", "Saturn"]

    formatted_text = f"<|im_start|>system\\n{system_prompt}<|im_end|>\\n<|im_start|>user\\nQuestion: {question}\\n"
    for i, choice in enumerate(choices):
        formatted_text += f"{chr(65+i)}. {choice}\\n"
    formatted_text += "Select the best choice.<|im_end|>\\n<|im_start|>assistant\\nChoice:"

    inputs = tokenizer(formatted_text, return_tensors="pt").to(model.device)
    num_choices = torch.tensor([len(choices)], device=model.device)

    probs = model.predict(inputs["input_ids"], inputs["attention_mask"], num_choices=num_choices)[0]

    print(f"\\nQuestion: {question}")
    for i, choice in enumerate(choices):
        p = probs[i].item() * 100
        print(f"  [{chr(65+i)}] {choice:<15} : {p:5.2f}%")

    top_idx = torch.argmax(probs[:len(choices)]).item()
    print(f"\\nPredicted Best Choice: [{chr(65+top_idx)}] {choices[top_idx]}")

if __name__ == "__main__":
    main()
'''
    with open(os.path.join(export_dir, "predict.py"), "w", encoding="utf-8") as f:
        f.write(predict_code)
    print(f"Exported predict.py")

    # Create README.md (Model Card)
    readme = """# Qwen2.5-1.5B JEV Decision Head (Phase 1)

This repository contains the **JEV-style Decision Head** trained on top of `Qwen/Qwen2.5-1.5B-Instruct` using `SargeDev/jev-distill-corpus-v3` (~656,000 samples).

Unlike traditional autoregressive language models that generate text token-by-token (taking hundreds of milliseconds), this model outputs a **pure probability distribution across choices in a single forward pass (~43 ms latency)**.

---

## Model Highlights
- **Backbone:** `Qwen/Qwen2.5-1.5B-Instruct` (Frozen)
- **Architecture:** 2-layer MLP Decision Head (`LayerNorm(1536) -> Linear(1536, 512) -> GELU -> Dropout(0.1) -> Linear(512, 10)`)
- **Loss:** Soft-target KL Divergence against JEV distilled probability distributions
- **Throughput / Latency:** **~43 ms per decision** on RTX 5060 Laptop (0 token generation overhead)
- **VRAM Footprint:** ~3.2 GB total VRAM in bfloat16

---

## Benchmark Results (Phase 1 Baseline)

| Split | Samples | Top-1 Accuracy | Mean KL Divergence | Total Variation Distance (TVD) | Latency |
|---|---|---|---|---|---|
| **In-Distribution (`test`)** | 1,500 | **75.67%** | **0.1757** | **0.1664** | **43.05 ms** |
| **Out-of-Distribution (`ood`)** | 1,500 | **57.53%** | **0.7884** | **0.4607** | **42.94 ms** |

---

## Quickstart

```python
import torch
from transformers import AutoTokenizer
from modeling_decision_head import QwenDecisionModel

# 1. Load model with decision head weights
model = QwenDecisionModel(
    base_model_name="Qwen/Qwen2.5-1.5B-Instruct",
    head_weights_path="decision_head.safetensors"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

# 2. Format query
question = "Which planet is known as the Red Planet?"
choices = ["Venus", "Mars", "Jupiter", "Saturn"]

prompt = f"<|im_start|>user\\nQuestion: {question}\\nA. Venus\\nB. Mars\\nC. Jupiter\\nD. Saturn<|im_end|>\\n<|im_start|>assistant\\nChoice:"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
num_choices = torch.tensor([len(choices)], device=model.device)

# 3. Single forward-pass prediction (43ms)
probs = model.predict(inputs["input_ids"], inputs["attention_mask"], num_choices=num_choices)[0]

top_idx = torch.argmax(probs[:len(choices)]).item()
print(f"Decision: {choices[top_idx]} ({probs[top_idx]*100:.1f}%)")
```
"""
    with open(os.path.join(export_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme)
    print(f"Exported README.md (Model Card)")

    print(f"\n[SUCCESS] Model packaged and ready to post at: {export_dir}")

if __name__ == "__main__":
    main()

