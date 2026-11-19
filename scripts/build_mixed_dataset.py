import os
import random
import json
from datasets import load_from_disk

random.seed(42)

OTHER_THANS = (
    "Question: {question}\nIs the correct answer anything other than the following option?\nOption: {description}",
    "Question: {question}\nIs the following option the wrong answer?\nOption: {description}",
    "{question}\nWould it be incorrect to choose this option: {description}?",
)

IS_OPTIONS = (
    "Question: {question}\nIs the correct answer the following option?\nOption: {description}",
    "{question}\nIs this the best answer: {description}?",
)

NEGATIONS = (
    "Answer yes if the correct answer to the following question is no, and answer no if it is yes. Question: {question}",
    "Is it false that the answer to the following question is yes? Question: {question}",
    "Consider this question: {question}\nWould the correct answer be no?",
)

def build_mixed_dataset(num_original: int = 30000, num_consistency: int = 10000):
    ds_path = "data/jev-distill-corpus-v3"
    print(f"Loading base dataset from {ds_path}...")
    ds = load_from_disk(ds_path)
    
    # 1. Sample Original High-Quality Items (75%)
    print(f"Sampling {num_original:,} original items from distillation corpus...")
    original_items = []
    
    # Gather from calibration and train
    for split in ["calibration", "train"]:
        for item in ds[split]:
            k = item["kind"]
            opts = item["options"]
            tgt = item["target"]
            if isinstance(tgt, list) and len(tgt) >= 2 and sum(tgt) > 0:
                s = sum(tgt)
                norm_tgt = [round(t / s, 4) for t in tgt]
                original_items.append({
                    "id": f"orig_{len(original_items)}",
                    "state": item["state"],
                    "question": item["question"],
                    "options": opts,
                    "target": norm_tgt,
                    "kind": k
                })
            if len(original_items) >= num_original:
                break
        if len(original_items) >= num_original:
            break
            
    print(f"Collected {len(original_items):,} original items.")
    
    # 2. Build Targeted Consistency Pairs (25%)
    print(f"Generating {num_consistency:,} targeted consistency items...")
    consistency_items = []
    
    # We take candidate choice items from calibration
    calib_choices = [it for it in ds["calibration"] if it["kind"] == "choice" and len(it["options"]) >= 2]
    random.shuffle(calib_choices)
    
    for idx, item in enumerate(calib_choices):
        state = item["state"]
        q_text = item["question"]
        opts = item["options"]
        s = sum(item["target"])
        targets = [round(t / s, 4) for t in item["target"]]
        n_opts = len(opts)
        
        top_idx = targets.index(max(targets))
        top_opt = opts[top_idx]
        top_prob = targets[top_idx]
        
        # Choice Complement: P(true) = 1.0 - prob
        tmpl_comp = random.choice(OTHER_THANS)
        comp_q = tmpl_comp.format(question=q_text, description=top_opt)
        consistency_items.append({
            "id": f"mix_comp_{idx}",
            "state": state,
            "question": comp_q,
            "options": ["false", "true"],
            "target": [round(top_prob, 4), round(1.0 - top_prob, 4)],
            "kind": "noul"
        })
        
        # Choice Indicator: P(true) = prob
        tmpl_is = random.choice(IS_OPTIONS)
        ind_q = tmpl_is.format(question=q_text, description=top_opt)
        consistency_items.append({
            "id": f"mix_ind_{idx}",
            "state": state,
            "question": ind_q,
            "options": ["false", "true"],
            "target": [round(1.0 - top_prob, 4), round(top_prob, 4)],
            "kind": "noul"
        })
        
        # Choice Permutation
        p_idx = list(range(n_opts))
        random.shuffle(p_idx)
        consistency_items.append({
            "id": f"mix_perm_{idx}",
            "state": state,
            "question": q_text,
            "options": [opts[i] for i in p_idx],
            "target": [targets[i] for i in p_idx],
            "kind": "choice"
        })
        
        # Option Merge (if >= 3 options)
        if n_opts >= 3:
            i, j = random.sample(range(n_opts), 2)
            merged_opt = f"Either: {opts[i]}; or: {opts[j]}"
            merged_prob = round(targets[i] + targets[j], 4)
            new_opts = [merged_opt] + [opts[k] for k in range(n_opts) if k not in (i, j)]
            new_targets = [merged_prob] + [targets[k] for k in range(n_opts) if k not in (i, j)]
            new_targets = [round(t / sum(new_targets), 4) for t in new_targets]
            consistency_items.append({
                "id": f"mix_merge_{idx}",
                "state": state,
                "question": q_text,
                "options": new_opts,
                "target": new_targets,
                "kind": "choice"
            })
            
        if len(consistency_items) >= num_consistency:
            break
            
    consistency_items = consistency_items[:num_consistency]
    print(f"Collected {len(consistency_items):,} consistency items.")
    
    # 3. Combine and Shuffle
    all_mixed = original_items + consistency_items
    random.shuffle(all_mixed)
    print(f"\nTotal Balanced Dataset: {len(all_mixed):,} items (75% Original / 25% Consistency).")
    
    out_dir = "data/phase4_mixed"
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "mixed_train.jsonl")
    with open(out_file, "w", encoding="utf-8") as f:
        for it in all_mixed:
            f.write(json.dumps(it) + "\n")
            
    print(f"Saved balanced dataset to: {out_file}")
    return out_file

if __name__ == "__main__":
    build_mixed_dataset()
