import os
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

    def predict_score(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.act(self.dense(h))
        return self.score_proj(h).squeeze(-1)

class QJevModel(nn.Module):
    """
    QJev-1.5B: High-Speed System 1 Decision Model.
    Single-pass calibrated probability distributions over structured choices.
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

    @classmethod
    def from_pretrained(cls, model_id_or_path: str, device: str = "cuda"):
        return cls(model_dir=model_id_or_path, device=device)

    def format_prompt(self, prompt: str, choices: list[str]) -> str:
        opts_str = "\n".join([f"[{i+1}] {c}" for i, c in enumerate(choices)])
        return (
            "<|im_start|>system\n"
            "You are an expert decision judge. Evaluate the situation and determine the likelihood for each option.<|im_end|>\n"
            f"<|im_start|>user\nContext:\n{prompt.strip()}\n\nQuestion:\nSelect the most appropriate option:\n\nOptions:\n{opts_str}<|im_end|>\n"
            "<|im_start|>assistant\nDecision:"
        )

    @torch.no_grad()
    def predict(self, prompt: str, tokenizer, choices: list[str]) -> dict:
        num_c = len(choices)
        formatted_prompt = self.format_prompt(prompt, choices)
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
        
        transformer = getattr(self.backbone, "model", self.backbone)
        out = transformer(inputs["input_ids"], attention_mask=inputs["attention_mask"])
        last_hidden = out.last_hidden_state[:, -1, :]
        
        logits = self.head(last_hidden, num_choices=torch.tensor([num_c], device=self.device))
        probs = F.softmax(logits[0, :num_c], dim=-1).cpu().tolist()
        return {choices[i]: round(probs[i], 4) for i in range(num_c)}

if __name__ == "__main__":
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    tokenizer = AutoTokenizer.from_pretrained(pkg_dir)
    model = QJevModel.from_pretrained(pkg_dir)
    res = model.predict(
        "Production database CPU is spiking at 99%. What is the immediate priority?",
        tokenizer,
        ["Failover to read replica immediately", "Restart the primary database", "Increase connection pool", "Ignore the alert"]
    )
    print("QJev Decisions:", res)
