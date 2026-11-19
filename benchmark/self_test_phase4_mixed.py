import os
import sys
import json
import time
import jevbench as jb

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(repo_root)
from qjev import QJev

def run_phase4_mixed_evaluation(tolerance: float = 0.05):
    model_dir = os.path.join(repo_root, "export", "QJev-1.5B-Phase4-Mixed")
        
    print("=" * 70)
    print("RUNNING OFFICIAL JEVBENCH EVALUATION ON BALANCED PHASE 4 MODEL")
    print(f"Model Directory: {model_dir}")
    print(f"Tolerance: fixed {tolerance} (Official Benchmark Standard)")
    print("=" * 70)
    
    start_load = time.time()
    qjev = QJev(model_id_or_path=model_dir, device="cuda")
    print(f"Model loaded in {time.time() - start_load:.2f}s on {qjev.device}")
    
    def model_fn(state, questions):
        out = {}
        for key, q in questions.items():
            q_type = q.get("type", "choice")
            instructions = q.get("instructions", "")
            
            if q_type == "noul":
                res = qjev.decide(context=str(state), question=instructions, choices=["false", "true"])
                p_true = float(res["probabilities"].get("true", 50.0)) / 100.0
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

    backend = jb.from_callable(model_fn, name="QJev-1.5B-Phase4-Mixed")
    
    print(f"\nRunning official reported suite: jevbench-mini (tolerance={tolerance})...")
    start_eval = time.time()
    report = jb.evaluate(backend, suite="jevbench-mini", tolerance=tolerance, progress=True)
    eval_duration = time.time() - start_eval
    
    os.makedirs(os.path.join(repo_root, "reports", "jevbench"), exist_ok=True)
    report_file = os.path.join(repo_root, "reports", "jevbench", "self_test_phase4_mixed.json")
    report.save(report_file)
    print(f"\nEvaluation finished in {eval_duration:.2f}s")
    print(f"Report saved to: {report_file}")
    
    print("\n" + "=" * 70)
    print("BALANCED PHASE 4 RESULTS BY AXIOM GROUP (tolerance=0.05):")
    print("=" * 70)
    print(report.table("group"))
    
    print("\n" + "=" * 70)
    print(f"NEW OFFICIAL OVERALL SCORE: {report.overall * 100:.2f}%")
    print("=" * 70)
    
    p3_file = os.path.join(repo_root, "reports", "jevbench", "self_test_phase3_consistency.json")
    if os.path.exists(p3_file):
        with open(p3_file) as f:
            p3_data = json.load(f)
        p3_overall = p3_data["scores"]["overall"] * 100
        delta = (report.overall * 100) - p3_overall
        sign = "+" if delta >= 0 else ""
        print(f"Phase 3 Baseline: {p3_overall:.2f}%")
        print(f"Phase 4 Mixed:    {report.overall * 100:.2f}% (Delta: {sign}{delta:.2f}%)")
        print("=" * 70)
        
    return report.overall

if __name__ == "__main__":
    run_phase4_mixed_evaluation()
