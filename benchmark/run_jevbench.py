import os
import sys
import gc
import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import jevbench as jb

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(repo_root, "src"))
from model import DecisionHead

# -------------------------------------------------------------
# 1. Phase 1 Backend: Frozen Backbone + Decision Head
# -------------------------------------------------------------
class Phase1Backend:
    name = "qwen2.5-1.5b-phase1-frozen-base"
    concurrent = False

    def __init__(self):
        base_path = "models/Qwen2.5-1.5B-Instruct" if os.path.exists("models/Qwen2.5-1.5B-Instruct") else "Qwen/Qwen2.5-1.5B-Instruct"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Phase 1] Loading base model: {base_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(base_path)
        self.backbone = AutoModelForCausalLM.from_pretrained(
            base_path,
            torch_dtype=torch.bfloat16,
            device_map=self.device
        )
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.head = DecisionHead().to(self.device, dtype=torch.bfloat16)
        head_path = "export/Qwen2.5-1.5B-Decision-Head/decision_head.safetensors"
        from safetensors.torch import load_file
        state = load_file(head_path)
        self.head.load_state_dict(state)
        self.head.eval()

    @torch.no_grad()
    def _predict(self, prompt: str, choices: list[str]) -> dict:
        num_c = len(choices)
        formatted_prompt = f"{prompt}\n\nChoices:\n" + "\n".join([f"{i+1}. {c}" for i, c in enumerate(choices)])
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
        
        transformer = getattr(self.backbone, "model", self.backbone)
        out = transformer(inputs["input_ids"], attention_mask=inputs["attention_mask"])
        last_hidden = out.last_hidden_state[:, -1, :]
        logits = self.head(last_hidden, num_choices=torch.tensor([num_c], device=self.device))
        probs = F.softmax(logits[0, :num_c], dim=-1).cpu().tolist()
        return {choices[i]: round(probs[i], 4) for i in range(num_c)}

    def ask(self, state: str, questions: dict) -> dict:
        out = {}
        for key, q in questions.items():
            q_type = q.get("type", "choice")
            instructions = q.get("instructions", "")
            
            if q_type == "noul":
                prompt = f"{state}\n\nQuestion: {instructions}"
                choices = ["Yes", "No"]
                res = self._predict(prompt, choices)
                p_yes = res.get("Yes", 0.5)
                out[key] = {"type": "noul", "noul": float(p_yes)}
                
            elif q_type == "choice":
                criteria = q.get("criteria", {})
                opt_keys = list(criteria.keys())
                choices = [f"{k}: {criteria[k]}" for k in opt_keys]
                prompt = f"{state}\n\nQuestion: {instructions}"
                res = self._predict(prompt, choices)
                prob_dict = {k: float(res.get(c, 0.0)) for k, c in zip(opt_keys, choices)}
                total = sum(prob_dict.values())
                if total > 0:
                    prob_dict = {k: v / total for k, v in prob_dict.items()}
                out[key] = {"type": "choice", "probabilities": prob_dict}
                
            elif q_type == "score":
                criteria = q.get("criteria", [])
                choices = [str(c) for c in criteria]
                prompt = f"{state}\n\nQuestion: {instructions}"
                res = self._predict(prompt, choices)
                prob_dict = {str(i): float(res.get(c, 0.0)) for i, c in enumerate(choices)}
                total = sum(prob_dict.values())
                if total > 0:
                    prob_dict = {k: v / total for k, v in prob_dict.items()}
                out[key] = {"type": "score", "probabilities": prob_dict}
        return out


# -------------------------------------------------------------
# 2. Phase 2 Backend: Merged Qwen + LoRA + Decision Head
# -------------------------------------------------------------
class Phase2Backend:
    name = "qwen2.5-1.5b-phase2-full-model"
    concurrent = False

    def __init__(self):
        model_dir = "export/Qwen2.5-1.5B-Decision-Head-Merged"
        print(f"[Phase 2] Loading merged model from: {model_dir}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = QwenDecisionMergedModel(model_dir=model_dir, device="cuda")

    def ask(self, state: str, questions: dict) -> dict:
        out = {}
        for key, q in questions.items():
            q_type = q.get("type", "choice")
            instructions = q.get("instructions", "")
            
            if q_type == "noul":
                prompt = f"{state}\n\nQuestion: {instructions}"
                choices = ["Yes", "No"]
                res = self.model.predict(prompt, self.tokenizer, choices)
                p_yes = res.get("Yes", 0.5)
                out[key] = {"type": "noul", "noul": float(p_yes)}
                
            elif q_type == "choice":
                criteria = q.get("criteria", {})
                opt_keys = list(criteria.keys())
                choices = [f"{k}: {criteria[k]}" for k in opt_keys]
                prompt = f"{state}\n\nQuestion: {instructions}"
                res = self.model.predict(prompt, self.tokenizer, choices)
                prob_dict = {k: float(res.get(c, 0.0)) for k, c in zip(opt_keys, choices)}
                total = sum(prob_dict.values())
                if total > 0:
                    prob_dict = {k: v / total for k, v in prob_dict.items()}
                out[key] = {"type": "choice", "probabilities": prob_dict}
                
            elif q_type == "score":
                criteria = q.get("criteria", [])
                choices = [str(c) for c in criteria]
                prompt = f"{state}\n\nQuestion: {instructions}"
                res = self.model.predict(prompt, self.tokenizer, choices)
                prob_dict = {str(i): float(res.get(c, 0.0)) for i, c in enumerate(choices)}
                total = sum(prob_dict.values())
                if total > 0:
                    prob_dict = {k: v / total for k, v in prob_dict.items()}
                out[key] = {"type": "score", "probabilities": prob_dict}
        return out


def main():
    os.makedirs("reports/jevbench", exist_ok=True)
    
    # --- RUN PHASE 1 ---
    print("\n" + "="*60)
    print("STEP 1: Evaluating Phase 1 (Frozen Backbone + Head)")
    print("="*60)
    backend_p1 = Phase1Backend()
    report_p1 = jb.evaluate(backend_p1, suite="jevbench-mini", progress=True)
    p1_path = "reports/jevbench/jevbench_phase1_report.json"
    report_p1.save(p1_path)
    print(f"Saved Phase 1 JevBench report to: {p1_path}")
    print("\nPhase 1 Results by Group:")
    print(report_p1.table("group"))
    
    # Cleanup memory
    del backend_p1
    gc.collect()
    torch.cuda.empty_cache()
    
    # --- RUN PHASE 2 ---
    print("\n" + "="*60)
    print("STEP 2: Evaluating Phase 2 (Full Merged Model)")
    print("="*60)
    backend_p2 = Phase2Backend()
    report_p2 = jb.evaluate(backend_p2, suite="jevbench-mini", progress=True)
    p2_path = "reports/jevbench/jevbench_phase2_report.json"
    report_p2.save(p2_path)
    print(f"Saved Phase 2 JevBench report to: {p2_path}")
    print("\nPhase 2 Results by Group:")
    print(report_p2.table("group"))
    
    # --- COMPARISON ---
    print("\n" + "="*60)
    print("FINAL JEVBENCH COMPARISON (Phase 1 vs. Phase 2)")
    print("="*60)
    print(f"{'Dimension':<25} | {'Phase 1 (Frozen)':<18} | {'Phase 2 (Full Model)':<20} | {'Delta':<10}")
    print("-" * 80)
    
    p1_scores = report_p1.groups
    p2_scores = report_p2.groups
    all_keys = sorted(set(list(p1_scores.keys()) + list(p2_scores.keys())))
    
    for k in all_keys:
        s1 = p1_scores.get(k, 0.0)
        s2 = p2_scores.get(k, 0.0)
        delta = s2 - s1
        sign = "+" if delta >= 0 else ""
        print(f"{k:<25} | {s1:>16.1f}% | {s2:>18.1f}% | {sign}{delta:>7.1f}%")
        
    print("-" * 80)
    delta_overall = report_p2.overall - report_p1.overall
    sign_ov = "+" if delta_overall >= 0 else ""
    print(f"{'OVERALL SCORE':<25} | {report_p1.overall:>16.1f}% | {report_p2.overall:>18.1f}% | {sign_ov}{delta_overall:>7.1f}%")
    print("="*60)

if __name__ == "__main__":
    main()
