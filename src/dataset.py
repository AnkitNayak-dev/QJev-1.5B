import torch
from torch.utils.data import Dataset
from typing import Dict, Any, List, Optional

def format_prompt(tokenizer, state: str, question: str, options: Optional[List[str]] = None) -> str:
    """
    Fast direct formatting using Qwen's ChatML format (<|im_start|> ... <|im_end|>).
    Avoids slow Jinja2 template rendering in tight loops.
    """
    sys_str = "<|im_start|>system\nYou are an expert decision judge. Evaluate the situation and determine the likelihood for each option.<|im_end|>\n"
    
    user_body = f"Context:\n{state.strip()}\n\nQuestion:\n{question.strip()}"
    if options and len(options) > 0:
        opts_str = "\n".join([f"[{i+1}] {opt}" for i, opt in enumerate(options)])
        user_body += f"\n\nOptions:\n{opts_str}"
        
    user_str = f"<|im_start|>user\n{user_body}<|im_end|>\n"
    asst_str = "<|im_start|>assistant\nDecision:"
    
    return sys_str + user_str + asst_str

class DecisionBatchCollate:
    def __init__(self, tokenizer, max_length: int = 384, max_choices: int = 10):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_choices = max_choices

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        prompts = []
        targets = []
        num_choices_list = []

        for item in batch:
            state = item.get("state", "") or ""
            question = item.get("question", "") or ""
            options = item.get("options", [])
            target = item.get("target", [])
            kind = item.get("kind", "choice")

            if options is None:
                options = []
            
            # Normalize noul / boolean options if not explicitly listed
            if kind == "noul" and len(options) == 0:
                options = ["False", "True"]

            prompt = format_prompt(self.tokenizer, state, question, options)
            prompts.append(prompt)

            # Pad or truncate target distribution to max_choices
            if isinstance(target, (int, float)):
                target_dist = [float(target)]
            elif isinstance(target, list):
                target_dist = [float(x) for x in target]
            else:
                target_dist = [1.0]

            n_choices = min(len(options) if options else len(target_dist), self.max_choices)
            if n_choices < 1:
                n_choices = 1

            # Pad target vector to max_choices
            padded_target = target_dist[:self.max_choices] + [0.0] * max(0, self.max_choices - len(target_dist))
            targets.append(padded_target[:self.max_choices])
            num_choices_list.append(n_choices)

        tokenized = self.tokenizer(
            prompts,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "targets": torch.tensor(targets, dtype=torch.float32),
            "num_choices": torch.tensor(num_choices_list, dtype=torch.long)
        }
