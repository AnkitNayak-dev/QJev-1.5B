import os
import sys
import gc
import json
import time
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import jevbench as jb

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(repo_root)
from qjev import QJev

def run_self_test():
    model_dir = os.path.join(repo_root, "export", "Qwen2.5-1.5B-Decision-Head-Merged")
    print("=" * 70)
    print("STARTING BENCHMARKHEAVEN (JEVBENCH) SELF-TEST")
    print(f"Loading local flagship model from: {model_dir}")
    print("=" * 70)
    
    start_load = time.time()
    qjev = QJev(model_id_or_path=model_dir, device="cuda")
    print(f"Model loaded in {time.time() - start_load:.2f}s on {qjev.device}")
    
    # Define wire format callable for JevBench
    def model_fn(state, questions):
        out = {}
        for key, q in questions.items():
            q_type = q.get("type", "choice")
            instructions = q.get("instructions", "")
            
            if q_type == "noul":
                res = qjev.decide(context=str(state), question=instructions, choices=["false", "true"])
                p_true = float(res["probabilities"]["true"]) / 100.0
                out[key] = {"type": "noul", "noul": p_true}
                
            elif q_type == "choice":
                criteria = q.get("criteria", {})
                opt_keys = list(criteria.keys())
                choices = [f"{k}: {criteria[k]}" for k in opt_keys]
                res = qjev.decide(context=str(state), question=instructions, choices=choices)
                prob_dict = {k: float(res["probabilities"][c]) / 100.0 for k, c in zip(opt_keys, choices)}
                total = sum(prob_dict.values())
                if total > 0:
                    prob_dict = {k: v / total for k, v in prob_dict.items()}
                out[key] = {"type": "choice", "probabilities": prob_dict}
                
            elif q_type == "score":
                criteria = q.get("criteria", [])
                choices = [str(c) for c in criteria]
                res = qjev.decide(context=str(state), question=instructions, choices=choices)
                prob_dict = {str(i): float(res["probabilities"][c]) / 100.0 for i, c in enumerate(choices)}
                total = sum(prob_dict.values())
                if total > 0:
                    prob_dict = {k: v / total for k, v in prob_dict.items()}
                out[key] = {"type": "score", "probabilities": prob_dict}
        return out

    backend = jb.from_callable(model_fn, name="QJev-1.5B")
    
    print("\nRunning official reported suite: jevbench-mini (1,248 trials across 240 cases)...")
    start_eval = time.time()
    # Official standard tolerance is 0.05
    report = jb.evaluate(backend, suite="jevbench-mini", tolerance=0.05, progress=True)
    eval_duration = time.time() - start_eval
    
    os.makedirs(os.path.join(repo_root, "reports", "jevbench"), exist_ok=True)
    report_file = os.path.join(repo_root, "reports", "jevbench", "self_test_reported_v1.json")
    report.save(report_file)
    print(f"\nEvaluation finished in {eval_duration:.2f}s ({1248/eval_duration:.1f} decisions/sec)")
    print(f"Report saved to: {report_file}")
    
    print("\n" + "=" * 70)
    print("RESULTS BY DOMAIN (Official standard tol=0.05):")
    print("=" * 70)
    try:
        print(report.table("domain"))
    except Exception as e:
        print(f"Domain table note: {e}")
        
    print("\n" + "=" * 70)
    print("RESULTS BY AXIOM GROUP (Official standard tol=0.05):")
    print("=" * 70)
    print(report.table("group"))
    
    print("\n" + "=" * 70)
    print(f"OVERALL JEVBENCH OFFICIAL SCORE: {report.overall:.2f}%")
    print("=" * 70)

if __name__ == "__main__":
    run_self_test()
