"""
QJev-1.5B: System 1 Decision Model Inference Demo
Fast, calibrated probability distribution across structured choices in ~31 ms.
"""

import os
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

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

class QJevModel(nn.Module):
    def __init__(self, model_id_or_path: str = "devankit7873/QJev-1.5B", device: str = None):
        super().__init__()
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        print(f"Loading QJev-1.5B from '{model_id_or_path}' on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id_or_path, trust_remote_code=True)
        self.backbone = AutoModelForCausalLM.from_pretrained(
            model_id_or_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            device_map=self.device
        )
        self.head = DecisionHead().to(device=self.device, dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
        
        # Load decision head weights from local folder or Hugging Face Hub
        if os.path.isdir(model_id_or_path) and os.path.exists(os.path.join(model_id_or_path, "decision_head.safetensors")):
            head_weights_path = os.path.join(model_id_or_path, "decision_head.safetensors")
        else:
            head_weights_path = hf_hub_download(repo_id=model_id_or_path, filename="decision_head.safetensors")
            
        state = load_file(head_weights_path)
        self.head.load_state_dict(state)
        self.eval()
        print("Model loaded successfully!")

    def format_prompt(self, context: str, choices: list[str]) -> str:
        choices_str = "\n".join([f"[{i+1}] {c}" for i, c in enumerate(choices)])
        return (
            "<|im_start|>system\n"
            "You are an expert decision engine. Evaluate the context and select the optimal decision from the given choices.<|im_end|>\n"
            f"<|im_start|>user\nContext:\n{context.strip()}\n\nQuestion:\nSelect the most appropriate option:\n\nOptions:\n{choices_str}<|im_end|>\n"
            "<|im_start|>assistant\nDecision:"
        )

    @torch.no_grad()
    def predict(self, context: str, choices: list[str]) -> dict:
        num_c = len(choices)
        prompt = self.format_prompt(context, choices)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        transformer = getattr(self.backbone, "model", self.backbone)
        out = transformer(inputs["input_ids"], attention_mask=inputs["attention_mask"])
        last_hidden = out.last_hidden_state[:, -1, :]
        
        logits = self.head(last_hidden, num_choices=torch.tensor([num_c], device=self.device))
        probs = F.softmax(logits[0, :num_c], dim=-1).cpu().tolist()
        
        best_idx = int(torch.argmax(logits[0, :num_c]).item())
        return {
            "recommended": choices[best_idx],
            "confidence": f"{probs[best_idx] * 100:.2f}%",
            "probabilities": {choice: f"{prob * 100:.2f}%" for choice, prob in zip(choices, probs)}
        }

def main():
    parser = argparse.ArgumentParser(description="QJev-1.5B Single-Pass Decision Inference")
    parser.add_argument("--model", type=str, default="devankit7873/QJev-1.5B", help="HF repo or local directory")
    args = parser.parse_args()

    # Prefer local export folder if available locally for speed, else HF Hub
    model_path = args.model
    if model_path == "devankit7873/QJev-1.5B" and os.path.exists("export/Qwen2.5-1.5B-Decision-Head-Merged"):
        model_path = "export/Qwen2.5-1.5B-Decision-Head-Merged"

    model = QJevModel(model_id_or_path=model_path)

    # Demo 1: Systems Engineering Incident
    context_sys = "Production database primary replica CPU spiked to 100% and connection pool is exhausted with 5,000 waiting queries."
    choices_sys = [
        "Trigger immediate automated failover to standby replica",
        "Restart the primary database daemon immediately",
        "Double connection pool limits without scaling compute",
        "Silence alerts and monitor for next 15 minutes"
    ]
    
    print("\n" + "=" * 60)
    print("DEMO: Incident Triage Decision")
    print(f"Context: {context_sys}")
    result_sys = model.predict(context_sys, choices_sys)
    print("Result:")
    print(json.dumps(result_sys, indent=2))
    print("=" * 60)

if __name__ == "__main__":
    main()
