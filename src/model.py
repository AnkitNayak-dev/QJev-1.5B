import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Any, Optional

class DecisionHead(nn.Module):
    """
    MLP Decision Head attached to frozen Qwen hidden states.
    Supports predicting calibrated distributions up to max_choices (default 10).
    """
    def __init__(self, hidden_size: int = 1536, intermediate_size: int = 512, max_choices: int = 10, dropout: float = 0.1):
        super().__init__()
        self.max_choices = max_choices
        self.norm = nn.LayerNorm(hidden_size)
        self.dense = nn.Linear(hidden_size, intermediate_size)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(intermediate_size, max_choices)
        
        # Binary / single score projection head
        self.score_proj = nn.Linear(intermediate_size, 1)

    def forward(self, hidden_state: torch.Tensor, num_choices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        hidden_state: (batch_size, hidden_size)
        num_choices: (batch_size,) optional integer tensor indicating valid options count
        """
        x = self.norm(hidden_state)
        x = self.act(self.dense(x))
        x = self.dropout(x)
        logits = self.out_proj(x) # (batch_size, max_choices)

        if num_choices is not None:
            # Mask out invalid choice indices with -inf
            mask = torch.arange(self.max_choices, device=logits.device).unsqueeze(0) >= num_choices.unsqueeze(1)
            logits = logits.masked_fill(mask, -1e4)

        return logits

    def predict_score(self, hidden_state: torch.Tensor) -> torch.Tensor:
        x = self.norm(hidden_state)
        x = self.act(self.dense(x))
        return self.score_proj(x).squeeze(-1)


class QwenDecisionModel(nn.Module):
    """
    Qwen2.5-1.5B backbone with frozen transformer weights and trainable DecisionHead.
    """
    def __init__(self, model_name_or_path: str = "Qwen/Qwen2.5-1.5B-Instruct", max_choices: int = 10, torch_dtype=torch.bfloat16):
        super().__init__()
        self.backbone = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            dtype=torch_dtype,
            device_map=None,
            low_cpu_mem_usage=True
        )
        # Freeze all backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

        hidden_size = self.backbone.config.hidden_size
        # The decision head is trained in float32 for maximum gradient stability
        self.head = DecisionHead(hidden_size=hidden_size, max_choices=max_choices).float()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, num_choices: Optional[torch.Tensor] = None):
        with torch.no_grad():
            outputs = self.backbone.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            # outputs.last_hidden_state has shape: (batch_size, seq_len, hidden_size)
            last_hidden_state = outputs.last_hidden_state
            
            # Extract the last valid non-padded token embedding for each sample in batch
            seq_lens = attention_mask.sum(dim=1) - 1 # 0-indexed last token
            batch_indices = torch.arange(input_ids.size(0), device=input_ids.device)
            pooled_rep = last_hidden_state[batch_indices, seq_lens] # (batch_size, hidden_size)

        logits = self.head(pooled_rep.float(), num_choices=num_choices)
        return logits


class QwenLoRADecisionModel(nn.Module):
    """
    Qwen2.5-1.5B with LoRA adapters + trainable DecisionHead for Phase 2 joint training.
    Base transformer weights are frozen; LoRA parameters and DecisionHead are trained jointly.
    """
    def __init__(
        self,
        model_name_or_path: str = "models/Qwen2.5-1.5B-Instruct",
        max_choices: int = 10,
        torch_dtype=torch.bfloat16,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        target_modules: Optional[list] = None,
        head_weights_path: Optional[str] = None
    ):
        super().__init__()
        import os
        from peft import LoraConfig, get_peft_model

        if target_modules is None:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

        base_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            dtype=torch_dtype,
            device_map=None,
            low_cpu_mem_usage=True
        )
        base_model.gradient_checkpointing_enable()

        peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none"
        )

        self.peft_model = get_peft_model(base_model, peft_config)
        # Directly target inner Qwen2Model to skip lm_head projection
        self.backbone_inner = self.peft_model.base_model.model.model

        hidden_size = base_model.config.hidden_size
        self.head = DecisionHead(hidden_size=hidden_size, max_choices=max_choices).float()

        if head_weights_path and os.path.exists(head_weights_path):
            raw = torch.load(head_weights_path, map_location="cpu")
            state = raw.get("head_state_dict", raw)
            self.head.load_state_dict(state)
            print(f"[Phase 2] Decision Head initialized from Phase 1 weights: {head_weights_path}")

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, num_choices: Optional[torch.Tensor] = None):
        outputs = self.backbone_inner(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        last_hidden_state = outputs.last_hidden_state
        seq_lens = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(input_ids.size(0), device=input_ids.device)
        pooled_rep = last_hidden_state[batch_indices, seq_lens]
        logits = self.head(pooled_rep.float(), num_choices=num_choices)
        return logits


