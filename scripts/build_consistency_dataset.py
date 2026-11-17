import os
import random
from datasets import load_from_disk, Dataset
import json

random.seed(42)

def generate_consistency_dataset(num_samples: int = 15000):
    ds_path = "data/jev-distill-corpus-v3"
    print(f"Loading base dataset from {ds_path}...")
    ds = load_from_disk(ds_path)
    
    # We use calibration and train splits to sample multi-choice items
    pool = []
    for item in ds["calibration"]:
        if item["kind"] == "choice" and len(item["options"]) >= 2 and isinstance(item["target"], list):
            pool.append(item)
            
    print(f"Found {len(pool)} multi-choice items in calibration split.")
    if len(pool) < num_samples:
        for item in ds["train"]:
            if item["kind"] == "choice" and len(item["options"]) >= 2 and isinstance(item["target"], list):
                pool.append(item)
                if len(pool) >= num_samples:
                    break
                    
    print(f"Total candidate multi-choice pool: {len(pool)}")
    selected = pool[:num_samples]
    
    pairs = []
    for idx, item in enumerate(selected):
        state = item["state"]
        q_text = item["question"]
        opts = item["options"]
        targets = item["target"]
        
        # 1. Base Multi-Choice item
        pairs.append({
            "id": f"p3_{idx}_base",
            "state": state,
            "question": q_text,
            "options": opts,
            "target": targets,
            "kind": "choice"
        })
        
        # Pick the top option and an alternate option
        top_idx = targets.index(max(targets))
        top_opt = opts[top_idx]
        top_prob = targets[top_idx]
        
        # 2. Choice Indicator Probe: "Is the correct answer the following option? Option: {top_opt}"
        # Expected noul probability for true is top_prob
        ind_q = f"Question: {q_text}\nIs the correct answer the following option?\nOption: {top_opt}"
        pairs.append({
            "id": f"p3_{idx}_indicator",
            "state": state,
            "question": ind_q,
            "options": ["false", "true"],
            "target": [round(1.0 - top_prob, 4), round(top_prob, 4)],
            "kind": "noul"
        })
        
        # 3. Choice Complement Probe: "Is the correct answer anything other than the following option? Option: {top_opt}"
        # Expected noul probability for true is 1.0 - top_prob
        comp_q = f"Question: {q_text}\nIs the correct answer anything other than the following option?\nOption: {top_opt}"
        pairs.append({
            "id": f"p3_{idx}_complement",
            "state": state,
            "question": comp_q,
            "options": ["false", "true"],
            "target": [round(top_prob, 4), round(1.0 - top_prob, 4)],
            "kind": "noul"
        })
        
        # 4. Choice Permutation: shuffle options and targets together
        indices = list(range(len(opts)))
        random.shuffle(indices)
        shuffled_opts = [opts[i] for i in indices]
        shuffled_targets = [targets[i] for i in indices]
        pairs.append({
            "id": f"p3_{idx}_perm",
            "state": state,
            "question": q_text,
            "options": shuffled_opts,
            "target": shuffled_targets,
            "kind": "choice"
        })

    print(f"Generated {len(pairs)} total consistency training items ({len(selected)} base scenarios x 4 forms).")
    
    out_dir = "data/phase3_consistency"
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "consistency_train.jsonl")
    with open(out_file, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
            
    print(f"Saved consistency dataset to: {out_file}")
    return out_file

if __name__ == "__main__":
    generate_consistency_dataset(num_samples=5000)
