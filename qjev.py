"""
QJev: High-Speed System 1 Decision Model Wrapper.
Supports Full Fused, 4-Bit NF4 Quantized, and Modular LoRA Adapter editions.

Usage:
    from qjev import QJev

    # Load any edition in 1 line:
    model = QJev.from_pretrained("devankit7873/QJev-1.5B")
    # or model = QJev.from_pretrained("devankit7873/QJev-1.5B-4bit")
    # or model = QJev.from_pretrained("devankit7873/QJev-1.5B-LoRA")

    result = model.decide(
        context="Production database CPU spiked to 100% and connection pool is exhausted.",
        choices=[
            "Trigger immediate automated failover to standby replica",
            "Restart primary database daemon",
            "Double connection pool limits",
            "Silence alerts and monitor"
        ]
    )
    print(result["best_choice"], result["confidence"])
"""

import os
import argparse
import json
from typing import List, Dict, Any, Optional, Union
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

    def forward(self, x: torch.Tensor, num_choices: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.dropout(self.act(self.dense(self.norm(x))))
        logits = self.out_proj(h)
        if num_choices is not None:
            batch_size = logits.size(0)
            seq_range = torch.arange(self.max_choices, device=logits.device).unsqueeze(0).expand(batch_size, -1)
            mask = seq_range >= num_choices.unsqueeze(1)
            logits = logits.masked_fill(mask, -1e4)
        return logits


class QJev:
    """
    Universal user-friendly client for QJev-1.5B decision models.
    """
    def __init__(self, model_id_or_path: str = "devankit7873/QJev-1.5B", device: Optional[str] = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        is_cuda = self.device.type == "cuda"
        dtype = torch.bfloat16 if is_cuda and torch.cuda.is_bf16_supported() else (torch.float16 if is_cuda else torch.float32)

        print(f"Loading QJev from '{model_id_or_path}' on {self.device}...")

        # Detect LoRA adapter repo vs Full/4-bit repo
        is_lora = "lora" in model_id_or_path.lower() or os.path.exists(os.path.join(model_id_or_path, "adapter_config.json"))

        if is_lora:
            from peft import PeftModel
            base_id = "Qwen/Qwen2.5-1.5B-Instruct"
            if os.path.exists("models/Qwen2.5-1.5B-Instruct"):
                base_id = "models/Qwen2.5-1.5B-Instruct"

            self.tokenizer = AutoTokenizer.from_pretrained(base_id, trust_remote_code=True)
            base_model = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=dtype, device_map=self.device)
            self.model = PeftModel.from_pretrained(base_model, model_id_or_path)
            self.model.eval()
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id_or_path, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id_or_path,
                torch_dtype=dtype if not "4bit" in model_id_or_path.lower() else None,
                device_map=self.device
            )
            self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load Decision Head
        self.head = DecisionHead().to(device=self.device, dtype=dtype)
        if os.path.isdir(model_id_or_path) and os.path.exists(os.path.join(model_id_or_path, "decision_head.safetensors")):
            head_path = os.path.join(model_id_or_path, "decision_head.safetensors")
        elif os.path.isdir(model_id_or_path) and os.path.exists(os.path.join(model_id_or_path, "decision_head.pt")):
            head_path = os.path.join(model_id_or_path, "decision_head.pt")
        else:
            head_path = hf_hub_download(repo_id=model_id_or_path, filename="decision_head.safetensors")

        if head_path.endswith(".safetensors"):
            state = load_file(head_path)
        else:
            ckpt = torch.load(head_path, map_location="cpu")
            state = ckpt.get("head_state_dict", ckpt)
            
        self.head.load_state_dict(state)
        self.head.eval()
        print("Ready for real-time decisions (<35 ms)!")

    @classmethod
    def from_pretrained(cls, model_id_or_path: str = "devankit7873/QJev-1.5B", device: Optional[str] = None) -> "QJev":
        return cls(model_id_or_path=model_id_or_path, device=device)

    def _format_prompt(self, context: str, choices: List[str], question: Optional[str] = None) -> str:
        choices_str = "\n".join([f"[{i+1}] {c}" for i, c in enumerate(choices)])
        q_str = f"\n\nQuestion:\n{question.strip()}" if question else "\n\nQuestion:\nSelect the most appropriate option:"
        return (
            "<|im_start|>system\n"
            "You are an expert decision judge. Evaluate the situation and determine the likelihood for each option.<|im_end|>\n"
            f"<|im_start|>user\nContext:\n{context.strip()}{q_str}\n\nOptions:\n{choices_str}<|im_end|>\n"
            "<|im_start|>assistant\nDecision:"
        )

    @torch.no_grad()
    def decide(self, context: str, choices: List[str], question: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate context against up to 10 choices in a single forward pass.
        Returns:
            {
                "best_choice": str,
                "confidence": float (percentage, e.g. 86.72),
                "probabilities": {choice: percentage, ...}
            }
        """
        if not choices:
            raise ValueError("choices list cannot be empty")
        if len(choices) > 10:
            raise ValueError(f"QJev supports up to 10 choices, got {len(choices)}")

        num_c = len(choices)
        prompt = self._format_prompt(context, choices, question)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Access internal transformer
        if hasattr(self.model, "peft_config"):
            inner = getattr(self.model.base_model.model, "model", self.model.base_model.model)
        else:
            inner = getattr(self.model, "model", self.model)

        out = inner(inputs["input_ids"], attention_mask=inputs["attention_mask"])
        last_hidden = out.last_hidden_state[:, -1, :]

        logits = self.head(last_hidden, num_choices=torch.tensor([num_c], device=self.device))
        probs = F.softmax(logits[0, :num_c], dim=-1).cpu().tolist()

        best_idx = int(torch.argmax(logits[0, :num_c]).item())

        return {
            "best_choice": choices[best_idx],
            "confidence": round(probs[best_idx] * 100, 2),
            "probabilities": {c: round(p * 100, 2) for c, p in zip(choices, probs)}
        }

    # Alias for flexibility
    predict = decide


def main():
    parser = argparse.ArgumentParser(description="QJev: Instant Single-Pass Decision Model")
    parser.add_argument("--model", type=str, default="devankit7873/QJev-1.5B", help="Model repo or local path")
    parser.add_argument("--context", type=str, default=None, help="Decision situation/context")
    parser.add_argument("--choices", nargs="+", default=None, help="List of candidate choices (up to 10)")
    args = parser.parse_args()

    # Prefer local export folder if available locally for speed
    model_id = args.model
    if model_id == "devankit7873/QJev-1.5B" and os.path.exists("export/Qwen2.5-1.5B-Decision-Head-Merged"):
        model_id = "export/Qwen2.5-1.5B-Decision-Head-Merged"

    client = QJev.from_pretrained(model_id)

    if args.context and args.choices:
        res = client.decide(args.context, args.choices)
    else:
        print("\n--- Example: Incident Triage Decision ---")
        context = "Production database primary replica CPU spiked to 100% and connection pool is exhausted."
        choices = [
            "Trigger immediate automated failover to standby replica",
            "Restart primary database daemon",
            "Double connection pool limits without scaling compute",
            "Silence alerts and monitor"
        ]
        res = client.decide(context, choices)

    print("\nDecision Result:")
    print(f"👉 Recommended: {res['best_choice']} ({res['confidence']}%)")
    print("\nProbabilities:")
    for choice, prob in res["probabilities"].items():
        print(f"  [{prob:>5.2f}%]  {choice}")


if __name__ == "__main__":
    main()
