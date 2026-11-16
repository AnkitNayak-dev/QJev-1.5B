import os
import sys
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
import jevbench as jb

sys.path.append(os.path.abspath("export/Qwen2.5-1.5B-Decision-Head-Merged"))
from modeling_qwen_decision_merged import QwenDecisionMergedModel

def create_jevbench_model():
    model_dir = "export/Qwen2.5-1.5B-Decision-Head-Merged"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    decision_model = QwenDecisionMergedModel(model_dir=model_dir, device="cuda")
    
    class QwenJevBenchBackend:
        name = "qwen2.5-1.5b-decision-head"
        concurrent = False
        
        def ask(self, state: str, questions: dict) -> dict:
            out = {}
            for key, q in questions.items():
                q_type = q.get("type", "choice")
                instructions = q.get("instructions", "")
                
                if q_type == "noul":
                    # Binary yes/no question
                    prompt = f"{state}\n\nQuestion: {instructions}"
                    choices = ["Yes", "No"]
                    res = decision_model.predict(prompt, tokenizer, choices)
                    p_yes = res.get("Yes", 0.5)
                    out[key] = {"type": "noul", "noul": float(p_yes)}
                    
                elif q_type == "choice":
                    # Multi-choice question with criteria dict {opt_id: desc}
                    criteria = q.get("criteria", {})
                    opt_keys = list(criteria.keys())
                    # Format options
                    choices = [f"{k}: {criteria[k]}" for k in opt_keys]
                    prompt = f"{state}\n\nQuestion: {instructions}"
                    res = decision_model.predict(prompt, tokenizer, choices)
                    
                    # Map probabilities back to option keys
                    prob_dict = {}
                    for k, choice_str in zip(opt_keys, choices):
                        prob_dict[k] = float(res.get(choice_str, 0.0))
                    # Normalize to strictly sum to 1.0
                    total = sum(prob_dict.values())
                    if total > 0:
                        prob_dict = {k: v / total for k, v in prob_dict.items()}
                    out[key] = {"type": "choice", "probabilities": prob_dict}
                    
                elif q_type == "score":
                    # Ordinal score with criteria list [level1, level2, ...]
                    criteria = q.get("criteria", [])
                    choices = [str(c) for c in criteria]
                    prompt = f"{state}\n\nQuestion: {instructions}"
                    res = decision_model.predict(prompt, tokenizer, choices)
                    
                    prob_dict = {}
                    for i, choice_str in enumerate(choices):
                        prob_dict[str(i)] = float(res.get(choice_str, 0.0))
                    total = sum(prob_dict.values())
                    if total > 0:
                        prob_dict = {k: v / total for k, v in prob_dict.items()}
                    out[key] = {"type": "score", "probabilities": prob_dict}
            return out
            
    return QwenJevBenchBackend()

if __name__ == "__main__":
    print("Initializing Qwen JevBench Backend...")
    backend = create_jevbench_model()
    
    print("Testing on examples suite (1 case)...")
    report = jb.evaluate(backend, suite="examples", progress=True)
    print("\n=== Examples Suite Result ===")
    print(report.table("group"))
