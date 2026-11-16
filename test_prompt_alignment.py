import os
import sys
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
import jevbench as jb

sys.path.append(os.path.abspath("export/Qwen2.5-1.5B-Decision-Head-Merged"))
from modeling_qwen_decision_merged import QwenDecisionMergedModel

def format_trained_prompt(state: str, question: str, options: list[str]) -> str:
    sys_str = "<|im_start|>system\nYou are an expert decision judge. Evaluate the situation and determine the likelihood for each option.<|im_end|>\n"
    user_body = f"Context:\n{state.strip()}\n\nQuestion:\n{question.strip()}"
    if options and len(options) > 0:
        opts_str = "\n".join([f"[{i+1}] {opt}" for i, opt in enumerate(options)])
        user_body += f"\n\nOptions:\n{opts_str}"
    user_str = f"<|im_start|>user\n{user_body}<|im_end|>\n"
    asst_str = "<|im_start|>assistant\nDecision:"
    return sys_str + user_str + asst_str

class AlignedJevBackend:
    name = "qwen2.5-1.5b-aligned"
    concurrent = False
    
    def __init__(self):
        model_dir = "export/Qwen2.5-1.5B-Decision-Head-Merged"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = QwenDecisionMergedModel(model_dir=model_dir, device="cuda")

    @torch.no_grad()
    def _predict_dist(self, state: str, question: str, options: list[str]) -> list[float]:
        num_c = len(options)
        prompt = format_trained_prompt(state, question, options)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        transformer = getattr(self.model.backbone, "model", self.model.backbone)
        out = transformer(inputs["input_ids"], attention_mask=inputs["attention_mask"])
        last_hidden = out.last_hidden_state[:, -1, :]
        logits = self.model.head(last_hidden, num_choices=torch.tensor([num_c], device=self.device))
        probs = F.softmax(logits[0, :num_c], dim=-1).cpu().tolist()
        return probs

    def ask(self, state: str, questions: dict) -> dict:
        out = {}
        for key, q in questions.items():
            q_type = q.get("type", "choice")
            instructions = q.get("instructions", "")
            
            if q_type == "noul":
                # Matches dataset.py line 45: ["False", "True"] -> index 1 is True!
                options = ["False", "True"]
                probs = self._predict_dist(state, instructions, options)
                p_true = probs[1]
                out[key] = {"type": "noul", "noul": float(p_true)}
                
            elif q_type == "choice":
                criteria = q.get("criteria", {})
                opt_keys = list(criteria.keys())
                options = [f"{k}: {criteria[k]}" for k in opt_keys]
                probs = self._predict_dist(state, instructions, options)
                prob_dict = {k: float(p) for k, p in zip(opt_keys, probs)}
                total = sum(prob_dict.values())
                if total > 0:
                    prob_dict = {k: v / total for k, v in prob_dict.items()}
                out[key] = {"type": "choice", "probabilities": prob_dict}
                
            elif q_type == "score":
                criteria = q.get("criteria", [])
                options = [str(c) for c in criteria]
                probs = self._predict_dist(state, instructions, options)
                prob_dict = {str(i): float(p) for i, p in enumerate(probs)}
                total = sum(prob_dict.values())
                if total > 0:
                    prob_dict = {k: v / total for k, v in prob_dict.items()}
                out[key] = {"type": "score", "probabilities": prob_dict}
        return out

if __name__ == "__main__":
    backend = AlignedJevBackend()
    print("Testing aligned prompt format on examples suite...")
    report = jb.evaluate(backend, suite="examples", cache=None)
    print("\nAligned Result on Examples:")
    print(report.table("group"))
