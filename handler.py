import os
import re
from typing import Dict, List, Any, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
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


class EndpointHandler:
    """
    Hugging Face Inference Endpoints Custom Handler for QJev-1.5B.
    Accepts:
      {
        "inputs": "Situation or context description",
        "parameters": {
          "choices": ["Choice 1", "Choice 2", ...]
        }
      }
      OR:
      {
        "context": "...",
        "question": "...",
        "choices": [...]
      }
    Returns standard text-classification scores:
      [
        {"label": "Choice 1", "score": 0.8672},
        {"label": "Choice 2", "score": 0.0703},
        ...
      ]
    """
    def __init__(self, path: str = ""):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

        # 1. Load Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 2. Load Base Transformer Backbone
        self.backbone = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=dtype,
            device_map=self.device
        )
        self.backbone.eval()

        # 3. Load MLP Decision Head
        self.head = DecisionHead().to(device=self.device, dtype=dtype)
        head_path = os.path.join(path, "decision_head.safetensors")
        if not os.path.exists(head_path):
            head_path = os.path.join(path, "decision_head.pt")
            
        if os.path.exists(head_path):
            if head_path.endswith(".safetensors"):
                state = load_file(head_path)
            else:
                ckpt = torch.load(head_path, map_location="cpu")
                state = ckpt.get("head_state_dict", ckpt)
            self.head.load_state_dict(state)
        self.head.eval()

    def _extract_choices_from_text(self, text: str) -> List[str]:
        # Try extracting patterns like "[1] ...", "1. ...", "Option A: ..."
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        choices = []
        for line in lines:
            m = re.match(r"^(?:\[?\d+\]?[\.\)]?|[A-E][\.\)]|Option\s+[A-E]:?)\s*(.+)$", line, re.IGNORECASE)
            if m:
                choices.append(m.group(1).strip())
        if len(choices) >= 2:
            return choices[:10]
        return ["Approve / Yes", "Reject / No"]

    def _format_prompt(self, context: str, question: str, choices: List[str]) -> str:
        choices_str = "\n".join([f"[{i+1}] {c}" for i, c in enumerate(choices)])
        q_str = f"\n\nQuestion:\n{question.strip()}" if question else "\n\nQuestion:\nSelect the most appropriate option:"
        return (
            "<|im_start|>system\n"
            "You are an expert decision judge. Evaluate the situation and determine the likelihood for each option.<|im_end|>\n"
            f"<|im_start|>user\nContext:\n{context.strip()}{q_str}\n\nOptions:\n{choices_str}<|im_end|>\n"
            "<|im_start|>assistant\nDecision:"
        )

    def _process_single(self, item: Dict[str, Any]) -> List[Dict[str, float]]:
        # Parse inputs
        context = ""
        question = ""
        choices = []

        if isinstance(item, str):
            context = item
            choices = self._extract_choices_from_text(item)
        elif isinstance(item, dict):
            # Check standard Hugging Face inputs/parameters format
            if "inputs" in item:
                context = str(item.get("inputs", ""))
                params = item.get("parameters", {}) or {}
                if isinstance(params, dict):
                    choices = params.get("choices", [])
                    question = params.get("question", "")

            # Check alternative keys
            if not context and "context" in item:
                context = str(item.get("context", ""))
            if not question and "question" in item:
                question = str(item.get("question", ""))
            if not choices and "choices" in item:
                choices = item.get("choices", [])
            if not choices and "options" in item:
                choices = item.get("options", [])

            # Fallback if no choices provided
            if not choices:
                choices = self._extract_choices_from_text(context)

        # Enforce max 10 choices (architecture limit)
        choices = [str(c) for c in choices][:10]
        if len(choices) < 2:
            choices = ["Approve / Yes", "Reject / No"]

        num_c = len(choices)
        prompt = self._format_prompt(context, question, choices)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            transformer = getattr(self.backbone, "model", self.backbone)
            out = transformer(inputs["input_ids"], attention_mask=inputs["attention_mask"])
            last_hidden = out.last_hidden_state[:, -1, :]
            logits = self.head(last_hidden, num_choices=torch.tensor([num_c], device=self.device))
            probs = F.softmax(logits[0, :num_c], dim=-1).cpu().tolist()

        return [{"label": choice, "score": round(prob, 4)} for choice, prob in zip(choices, probs)]

    def __call__(self, data: Union[Dict[str, Any], List[Dict[str, Any]]]) -> Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]]:
        """
        Receives request payload, returns predictions.
        """
        if isinstance(data, list):
            return [self._process_single(item) for item in data]
        elif isinstance(data, dict):
            # If payload is {"inputs": ["str1", "str2"], ...}
            if isinstance(data.get("inputs"), list):
                results = []
                for inp in data["inputs"]:
                    single_data = dict(data)
                    single_data["inputs"] = inp
                    results.append(self._process_single(single_data))
                return results
            return self._process_single(data)
        elif isinstance(data, str):
            return self._process_single(data)
        else:
            raise ValueError(f"Unsupported payload type: {type(data)}")
