import os
import sys
import gc
import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import jevbench as jb

sys.path.append(os.path.abspath("export/Qwen2.5-1.5B-Decision-Head-Merged"))
from modeling_qwen_decision_merged import QwenDecisionMergedModel, DecisionHead

def format_trained_chatml(state: str, question: str, options: list[str]) -> str:
    """Exact prompt format from src/dataset.py used across all 655k training samples."""
    sys_str = "<|im_start|>system\nYou are an expert decision judge. Evaluate the situation and determine the likelihood for each option.<|im_end|>\n"
    user_body = f"Context:\n{state.strip()}\n\nQuestion:\n{question.strip()}"
    if options and len(options) > 0:
        opts_str = "\n".join([f"[{i+1}] {opt}" for i, opt in enumerate(options)])
        user_body += f"\n\nOptions:\n{opts_str}"
    user_str = f"<|im_start|>user\n{user_body}<|im_end|>\n"
    asst_str = "<|im_start|>assistant\nDecision:"
    return sys_str + user_str + asst_str

# -------------------------------------------------------------
# 1. Aligned Phase 1 Backend (Frozen Backbone + Head)
# -------------------------------------------------------------
class AlignedPhase1Backend:
    name = "qwen2.5-1.5b-phase1-aligned"
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
        from safetensors.torch import load_file
        head_path = "export/Qwen2.5-1.5B-Decision-Head/decision_head.safetensors"
        self.head.load_state_dict(load_file(head_path))
        self.head.eval()

    @torch.no_grad()
    def _predict_dist(self, state: str, question: str, options: list[str]) -> list[float]:
        num_c = len(options)
        prompt = format_trained_chatml(state, question, options)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        transformer = getattr(self.backbone, "model", self.backbone)
        out = transformer(inputs["input_ids"], attention_mask=inputs["attention_mask"])
        last_hidden = out.last_hidden_state[:, -1, :]
        logits = self.head(last_hidden, num_choices=torch.tensor([num_c], device=self.device))
        probs = F.softmax(logits[0, :num_c], dim=-1).cpu().tolist()
        return probs

    def ask(self, state: str, questions: dict) -> dict:
        out = {}
        for key, q in questions.items():
            q_type = q.get("type", "choice")
            instructions = q.get("instructions", "")
            
            if q_type == "noul":
                options = ["False", "True"]
                probs = self._predict_dist(state, instructions, options)
                out[key] = {"type": "noul", "noul": float(probs[1])}
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

# -------------------------------------------------------------
# 2. Aligned Phase 2 Backend (Full Merged Model)
# -------------------------------------------------------------
class AlignedPhase2Backend:
    name = "qwen2.5-1.5b-phase2-aligned"
    concurrent = False

    def __init__(self):
        model_dir = "export/Qwen2.5-1.5B-Decision-Head-Merged"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Phase 2] Loading merged model from: {model_dir}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = QwenDecisionMergedModel(model_dir=model_dir, device="cuda")

    @torch.no_grad()
    def _predict_dist(self, state: str, question: str, options: list[str]) -> list[float]:
        num_c = len(options)
        prompt = format_trained_chatml(state, question, options)
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
                options = ["False", "True"]
                probs = self._predict_dist(state, instructions, options)
                out[key] = {"type": "noul", "noul": float(probs[1])}
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


def main():
    os.makedirs("reports/jevbench_aligned", exist_ok=True)
    
    # ---------------- PHASE 1 ----------------
    print("\n" + "="*60)
    print("STEP 1: Evaluating Phase 1 (Frozen Backbone + Head)")
    print("="*60)
    p1 = AlignedPhase1Backend()
    print("Running Phase 1 @ strict tolerance 0.05...")
    r1_strict = jb.evaluate(p1, suite="jevbench-mini", tolerance=0.05, progress=True)
    r1_strict.save("reports/jevbench_aligned/p1_strict_005.json")
    print(r1_strict.table("group"))
    
    print("\nRunning Phase 1 @ realistic tolerance 0.10...")
    r1_loose = jb.evaluate(p1, suite="jevbench-mini", tolerance=0.10, progress=True)
    r1_loose.save("reports/jevbench_aligned/p1_loose_010.json")
    print(r1_loose.table("group"))
    
    del p1
    gc.collect()
    torch.cuda.empty_cache()

    # ---------------- PHASE 2 ----------------
    print("\n" + "="*60)
    print("STEP 2: Evaluating Phase 2 (Full Merged Model)")
    print("="*60)
    p2 = AlignedPhase2Backend()
    print("Running Phase 2 @ strict tolerance 0.05...")
    r2_strict = jb.evaluate(p2, suite="jevbench-mini", tolerance=0.05, progress=True)
    r2_strict.save("reports/jevbench_aligned/p2_strict_005.json")
    print(r2_strict.table("group"))

    print("\nRunning Phase 2 @ realistic tolerance 0.10...")
    r2_loose = jb.evaluate(p2, suite="jevbench-mini", tolerance=0.10, progress=True)
    r2_loose.save("reports/jevbench_aligned/p2_loose_010.json")
    print(r2_loose.table("group"))

    print("\n" + "="*60)
    print("SUMMARY OF ALL RUNS")
    print("="*60)
    print(f"Phase 1 Strict (0.05):  {r1_strict.overall:.1f}%")
    print(f"Phase 1 Realistic (0.10): {r1_loose.overall:.1f}%")
    print(f"Phase 2 Strict (0.05):  {r2_strict.overall:.1f}%")
    print(f"Phase 2 Realistic (0.10): {r2_loose.overall:.1f}%")
    print("="*60)

if __name__ == "__main__":
    main()
