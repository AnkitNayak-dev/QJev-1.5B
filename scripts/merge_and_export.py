import os
import json
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from safetensors.torch import save_file, load_file

def main():
    print("=== Creating Full Merged Standalone Model (Phase 2) ===")
    
    base_model_path = "models/Qwen2.5-1.5B-Instruct" if os.path.exists("models/Qwen2.5-1.5B-Instruct") else "Qwen/Qwen2.5-1.5B-Instruct"
    lora_adapter_dir = "export/Qwen2.5-1.5B-Decision-Head-LoRA/lora_adapter"
    head_weights_path = "export/Qwen2.5-1.5B-Decision-Head-LoRA/decision_head.safetensors"
    
    export_dir = "export/Qwen2.5-1.5B-Decision-Head-Merged"
    os.makedirs(export_dir, exist_ok=True)
    
    # 1. Load Base Model and apply LoRA
    print(f"[1/4] Loading base model from {base_model_path}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    
    print(f"[2/4] Attaching LoRA adapter from {lora_adapter_dir}...")
    peft_model = PeftModel.from_pretrained(base_model, lora_adapter_dir)
    
    print(f"[3/4] Fusing (merge_and_unload) LoRA weights permanently into Qwen backbone...")
    merged_model = peft_model.merge_and_unload()
    
    print(f"[4/4] Saving merged standalone model and tokenizer to {export_dir}...")
    merged_model.save_pretrained(export_dir, safe_serialization=True)
    tokenizer.save_pretrained(export_dir)
    
    # Copy Decision Head weights
    shutil.copy2(head_weights_path, os.path.join(export_dir, "decision_head.safetensors"))
    shutil.copy2("export/Qwen2.5-1.5B-Decision-Head-LoRA/decision_head.pt", os.path.join(export_dir, "decision_head.pt"))
    print(f"Copied decision_head weights to {export_dir}")
    
    # Copy evaluation report metrics to config
    eval_final_path = "reports/phase_2/eval_results_final.json"
    if os.path.exists(eval_final_path):
        with open(eval_final_path, "r", encoding="utf-8") as f:
            final_metrics = json.load(f)
    else:
        final_metrics = {}
        
    merged_config = {
        "model_type": "qwen_decision_merged",
        "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
        "is_lora_merged": True,
        "hidden_size": 1536,
        "head_intermediate_size": 512,
        "max_choices": 10,
        "dropout": 0.1,
        "activation": "gelu",
        "output_type": "jev_probabilities",
        "training_corpus": "SargeDev/jev-distill-corpus-v3",
        "training_samples": 655806,
        "phase": 2,
        "metrics": final_metrics
    }
    with open(os.path.join(export_dir, "decision_config.json"), "w", encoding="utf-8") as f:
        json.dump(merged_config, f, indent=2)

    # Standalone inference script for merged model
    merged_inference_code = '''import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import load_file

class DecisionHead(nn.Module):
    def __init__(self, hidden_size: int = 1536, intermediate_size: int = 512, max_choices: int = 10, dropout: float = 0.1):
        super().__init__()
        self.max_choices = max_choices
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
            seq_range = torch.arange(self.max_choices, device=logits.device).unsqueeze(0).expand(batch_size, -1)
            mask = seq_range >= num_choices.unsqueeze(1)
            logits = logits.masked_fill(mask, -1e4)
        return logits

class QwenDecisionMergedModel(nn.Module):
    """
    Fully merged standalone Qwen2.5-1.5B JEV Decision Model.
    No PEFT or LoRA loading required.
    """
    def __init__(self, model_dir: str = ".", device: str = "cuda"):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.backbone = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.bfloat16,
            device_map=self.device
        )
        self.head = DecisionHead().to(device=self.device, dtype=torch.bfloat16)
        head_path = os.path.join(model_dir, "decision_head.safetensors")
        if os.path.exists(head_path):
            state = load_file(head_path)
            self.head.load_state_dict(state)
        self.eval()

    @torch.no_grad()
    def predict(self, prompt: str, tokenizer, choices: list[str]) -> dict:
        num_c = len(choices)
        formatted_prompt = f"{prompt}\\n\\nChoices:\\n" + "\\n".join([f"{i+1}. {c}" for i, c in enumerate(choices)])
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
        
        # Access internal transformer
        transformer = getattr(self.backbone, "model", self.backbone)
        out = transformer(inputs["input_ids"], attention_mask=inputs["attention_mask"])
        last_hidden = out.last_hidden_state[:, -1, :]
        
        logits = self.head(last_hidden, num_choices=torch.tensor([num_c], device=self.device))
        probs = F.softmax(logits[0, :num_c], dim=-1).cpu().tolist()
        return {choices[i]: round(probs[i], 4) for i in range(num_c)}

if __name__ == "__main__":
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    tokenizer = AutoTokenizer.from_pretrained(pkg_dir)
    model = QwenDecisionMergedModel(model_dir=pkg_dir)
    res = model.predict(
        "Should we invest in renewable energy infrastructure?",
        tokenizer,
        ["Strongly Agree", "Somewhat Agree", "Neutral", "Disagree"]
    )
    print("Decision Probabilities:", res)
'''
    with open(os.path.join(export_dir, "modeling_qwen_decision_merged.py"), "w", encoding="utf-8") as f:
        f.write(merged_inference_code)
    print("Exported modeling_qwen_decision_merged.py")

    # README for merged repo
    merged_readme = f'''---
license: apache-2.0
base_model: Qwen/Qwen2.5-1.5B-Instruct
tags:
- decision-making
- jev
- joint-expected-value
- qwen
- reasoning
datasets:
- SargeDev/jev-distill-corpus-v3
metrics:
- accuracy
- tvd
- kl_divergence
pipeline_tag: text-classification
---

# Qwen2.5-1.5B JEV Decision Model (Full Merged Weights)

This repository contains the **fully merged standalone weights** for the Qwen2.5-1.5B JEV Decision Model. The Phase 2 QLoRA adapters have been permanently fused into the base transformer weights using `merge_and_unload()`.

You do **not** need `peft` to use this model. It loads directly via `transformers` and `torch`.

## Key Performance Metrics
- **Out-of-Distribution (OOD) Accuracy:** **80.07%** (+22.54% over frozen backbone baseline)
- **Out-of-Distribution (OOD) TVD:** **0.2389** (48.1% error reduction)
- **In-Distribution (`test`) TVD:** **0.0921** (sub-9% error margin)
- **Inference Latency:** **31.49 ms** per sample

## Usage
```python
import torch
from transformers import AutoTokenizer
from modeling_qwen_decision_merged import QwenDecisionMergedModel

tokenizer = AutoTokenizer.from_pretrained("<your-username>/Qwen2.5-1.5B-Decision-Head-Merged")
model = QwenDecisionMergedModel(model_dir=".")

prompt = "We found a critical memory leak in the production payment service. What is the immediate priority?"
choices = [
    "Roll back to previous stable release immediately",
    "Deploy a hotfix directly into production without QA",
    "Wait until normal business hours to notify the team",
    "Ignore the alert until users complain"
]

results = model.predict(prompt, tokenizer, choices)
for c, p in results.items():
    print(f"{{c}}: {{p * 100:.2f}}%")
```
'''
    with open(os.path.join(export_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(merged_readme)
    print("Exported README.md")
    print(f"\n[DONE] Full merged standalone package ready in: {export_dir}")

if __name__ == "__main__":
    main()
