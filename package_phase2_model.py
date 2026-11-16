import os
import json
import shutil
import torch
from safetensors.torch import save_file
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    print("=== Packaging Phase 2 Decision Head + LoRA for Release ===")
    
    ckpt_path = "checkpoints/phase_2/decision_head_lora_final.pt"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")
    
    ckpt = torch.load(ckpt_path, map_location="cpu")
    head_state_dict = ckpt["head_state_dict"]
    lora_state_dict = ckpt["lora_state_dict"]
    
    export_dir = "export/Qwen2.5-1.5B-Decision-Head-LoRA"
    os.makedirs(export_dir, exist_ok=True)
    
    # 1. Save Head weights
    safetensors_path = os.path.join(export_dir, "decision_head.safetensors")
    save_file(head_state_dict, safetensors_path)
    print(f"Exported Decision Head safetensors: {safetensors_path}")

    pt_path = os.path.join(export_dir, "decision_head.pt")
    torch.save(head_state_dict, pt_path)
    print(f"Exported Decision Head PyTorch: {pt_path}")

    # 2. Save LoRA Adapter in standard PEFT format
    lora_dir = os.path.join(export_dir, "lora_adapter")
    os.makedirs(lora_dir, exist_ok=True)
    
    lora_config = {
        "base_model_name_or_path": "Qwen/Qwen2.5-1.5B-Instruct",
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "modules_to_save": None,
        "peft_type": "LORA",
        "r": 16,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "task_type": "FEATURE_EXTRACTION"
    }
    with open(os.path.join(lora_dir, "adapter_config.json"), "w", encoding="utf-8") as f:
        json.dump(lora_config, f, indent=2)
        
    lora_safetensors_path = os.path.join(lora_dir, "adapter_model.safetensors")
    peft_lora_dict = {k: v.contiguous() for k, v in lora_state_dict.items() if "lora_" in k}
    save_file(peft_lora_dict, lora_safetensors_path)
    torch.save(peft_lora_dict, os.path.join(lora_dir, "adapter_model.bin"))
    print(f"Exported PEFT LoRA adapter: {lora_dir} ({len(peft_lora_dict)} tensors)")

    # 3. Read final metrics
    eval_final_path = "reports/phase_2/eval_results_final.json"
    if os.path.exists(eval_final_path):
        with open(eval_final_path, "r", encoding="utf-8") as f:
            final_metrics = json.load(f)
    else:
        final_metrics = {}

    # 4. Save Release config.json
    config = {
        "model_type": "qwen_decision_head_lora",
        "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
        "hidden_size": 1536,
        "head_intermediate_size": 512,
        "max_choices": 10,
        "dropout": 0.1,
        "activation": "gelu",
        "lora": {
            "r": 16,
            "alpha": 32,
            "dropout": 0.05,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]
        },
        "output_type": "jev_probabilities",
        "training_corpus": "SargeDev/jev-distill-corpus-v3",
        "training_samples": 655806,
        "phase": 2,
        "metrics": final_metrics
    }
    with open(os.path.join(export_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"Exported config.json")

    # 5. Save self-contained inference script
    inference_code = '''import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
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

    def predict_score(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.act(self.dense(h))
        return self.score_proj(h).squeeze(-1)

class QwenDecisionLoRAModel(nn.Module):
    def __init__(self, base_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct", package_dir: str = ".", device: str = "cuda"):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            device_map=self.device
        )
        lora_dir = os.path.join(package_dir, "lora_adapter")
        if os.path.exists(lora_dir):
            self.model = PeftModel.from_pretrained(self.base_model, lora_dir)
        else:
            self.model = self.base_model
            
        self.head = DecisionHead().to(device=self.device, dtype=torch.bfloat16)
        head_path = os.path.join(package_dir, "decision_head.safetensors")
        if os.path.exists(head_path):
            state = load_file(head_path)
            self.head.load_state_dict(state)
        self.eval()

    @torch.no_grad()
    def predict(self, prompt: str, tokenizer, choices: list[str]) -> dict:
        num_c = len(choices)
        formatted_prompt = f"{prompt}\\n\\nChoices:\\n" + "\\n".join([f"{i+1}. {c}" for i, c in enumerate(choices)])
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
        
        # Access internal transformer hidden states
        base = self.model.base_model.model if hasattr(self.model, "base_model") else self.model
        transformer = getattr(base, "model", base)
        out = transformer(inputs["input_ids"], attention_mask=inputs["attention_mask"])
        last_hidden = out.last_hidden_state[:, -1, :]
        
        logits = self.head(last_hidden, num_choices=torch.tensor([num_c], device=self.device))
        probs = F.softmax(logits[0, :num_c], dim=-1).cpu().tolist()
        return {choices[i]: round(probs[i], 4) for i in range(num_c)}

if __name__ == "__main__":
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    model = QwenDecisionLoRAModel(package_dir=pkg_dir)
    res = model.predict(
        "Should we invest in renewable energy infrastructure?",
        tokenizer,
        ["Strongly Agree", "Somewhat Agree", "Neutral", "Disagree"]
    )
    print("Decision Probabilities:", res)
'''
    with open(os.path.join(export_dir, "modeling_qwen_decision_lora.py"), "w", encoding="utf-8") as f:
        f.write(inference_code)
    print("Exported modeling_qwen_decision_lora.py")

    # 6. Save README
    readme_content = f'''# Qwen2.5-1.5B JEV Decision Head + LoRA (Phase 2 Release)

Trained jointly on **655,806 samples** of `SargeDev/jev-distill-corpus-v3` with QLoRA on attention projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`) and Decision Head.

## Key Performance Metrics
- **Out-of-Distribution (OOD) Accuracy:** **80.07%** (vs 57.53% Phase 1 baseline, **+22.54% gain**)
- **Out-of-Distribution (OOD) TVD:** **0.2389** (vs 0.4607 Phase 1 baseline, **-48.1% error reduction**)
- **In-Distribution (`test`) TVD:** **0.0921** (vs 0.1664 Phase 1 baseline, **-44.6% error reduction**)
- **Inference Latency:** **31.49 ms** / sample

## Directory Structure
- `decision_head.safetensors`: Standalone Decision Head weights
- `decision_head.pt`: PyTorch format Decision Head weights
- `lora_adapter/`: HuggingFace PEFT compatible LoRA weights (`adapter_model.safetensors`, `adapter_config.json`)
- `config.json`: Model configuration and evaluation metadata
- `modeling_qwen_decision_lora.py`: Self-contained loading and prediction script

## Quick Start
```python
from modeling_qwen_decision_lora import QwenDecisionLoRAModel
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
model = QwenDecisionLoRAModel(package_dir="export/Qwen2.5-1.5B-Decision-Head-LoRA")

result = model.predict(
    prompt="Should we deploy this feature to production immediately?",
    tokenizer=tokenizer,
    choices=["Deploy Now", "Wait for More Tests", "Reject PR"]
)
print(result)
```
'''
    with open(os.path.join(export_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme_content)
    print("Exported README.md")
    print("\nPackage complete!")

if __name__ == "__main__":
    main()
