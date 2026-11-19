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

SPLIT_PREDICATES = (
    ("and the text mentions a specific number", "and the text mentions no specific number"),
    ("and the text mentions a date or time", "and the text mentions no date or time"),
    ("and the text names a person, team or organization", "and the text names no person, team or organization"),
    ("and the text contains a question", "and the text contains no question"),
)

def build_phase4_dataset(num_base_scenarios: int = 6000):
    ds_path = "data/jev-distill-corpus-v3"
    print(f"Loading base dataset from {ds_path}...")
    ds = load_from_disk(ds_path)
    
    # Collect candidate choice items and noul items
    choice_pool = []
    noul_pool = []
    
    for split_name in ["calibration", "train"]:
        for item in ds[split_name]:
            if item["kind"] == "choice" and len(item["options"]) >= 2 and isinstance(item["target"], list):
                # Ensure targets sum close to 1
                s = sum(item["target"])
                if 0.95 <= s <= 1.05:
                    choice_pool.append(item)
            elif item["kind"] == "noul" and len(item["options"]) == 2 and isinstance(item["target"], list):
                s = sum(item["target"])
                if 0.95 <= s <= 1.05:
                    noul_pool.append(item)
            if len(choice_pool) >= num_base_scenarios and len(noul_pool) >= num_base_scenarios // 2:
                break
        if len(choice_pool) >= num_base_scenarios and len(noul_pool) >= num_base_scenarios // 2:
            break
            
    print(f"Collected {len(choice_pool)} choice scenarios and {len(noul_pool)} noul scenarios.")
    
    items = []
    
    # 1. Process Choice Scenarios into Multi-Form Metamorphic Families
    for idx, item in enumerate(choice_pool[:num_base_scenarios]):
        state = item["state"]
        q_text = item["question"]
        opts = item["options"]
        targets = [round(t / sum(item["target"]), 4) for t in item["target"]]
        n_opts = len(opts)
        
        # Base item
        items.append({
            "id": f"p4_c_{idx}_base",
            "state": state,
            "question": q_text,
            "options": opts,
            "target": targets,
            "kind": "choice"
        })
        
        # Pick top option and a random non-top option
        top_idx = targets.index(max(targets))
        other_indices = [i for i in range(n_opts) if i != top_idx]
        sample_indices = [top_idx]
        if other_indices:
            sample_indices.append(random.choice(other_indices))
            
        for opt_idx in sample_indices:
            opt = opts[opt_idx]
            prob = targets[opt_idx]
            
            # Choice Indicator (P(true) = prob)
            tmpl_is = random.choice(IS_OPTIONS)
            ind_q = tmpl_is.format(question=q_text, description=opt)
            items.append({
                "id": f"p4_c_{idx}_ind_{opt_idx}",
                "state": state,
                "question": ind_q,
                "options": ["false", "true"],
                "target": [round(1.0 - prob, 4), round(prob, 4)],
                "kind": "noul"
            })
            
            # Choice Complement (P(true) = 1.0 - prob)
            tmpl_other = random.choice(OTHER_THANS)
            comp_q = tmpl_other.format(question=q_text, description=opt)
            items.append({
                "id": f"p4_c_{idx}_comp_{opt_idx}",
                "state": state,
                "question": comp_q,
                "options": ["false", "true"],
                "target": [round(prob, 4), round(1.0 - prob, 4)],
                "kind": "noul"
            })
            
        # Option Permutation
        perm_idx = list(range(n_opts))
        random.shuffle(perm_idx)
        items.append({
            "id": f"p4_c_{idx}_perm",
            "state": state,
            "question": q_text,
            "options": [opts[i] for i in perm_idx],
            "target": [targets[i] for i in perm_idx],
            "kind": "choice"
        })
        
        # Option Merge (if >= 3 options)
        if n_opts >= 3:
            i, j = random.sample(range(n_opts), 2)
            merged_opt = f"Either: {opts[i]}; or: {opts[j]}"
            merged_prob = round(targets[i] + targets[j], 4)
            new_opts = [merged_opt] + [opts[k] for k in range(n_opts) if k not in (i, j)]
            new_targets = [merged_prob] + [targets[k] for k in range(n_opts) if k not in (i, j)]
            # normalize sum
            new_targets = [round(t / sum(new_targets), 4) for t in new_targets]
            items.append({
                "id": f"p4_c_{idx}_merge",
                "state": state,
                "question": q_text,
                "options": new_opts,
                "target": new_targets,
                "kind": "choice"
            })
            
        # Option Split (if <= 8 options so it stays within max_choices=10)
        if n_opts <= 8:
            split_idx = top_idx
            o_desc = opts[split_idx]
            o_prob = targets[split_idx]
            pred_yes, pred_no = random.choice(SPLIT_PREDICATES)
            part_a = f"{o_desc}, {pred_yes}"
            part_b = f"{o_desc}, {pred_no}"
            half_prob = round(o_prob / 2.0, 4)
            split_opts = [part_a, part_b] + [opts[k] for k in range(n_opts) if k != split_idx]
            split_targets = [half_prob, round(o_prob - half_prob, 4)] + [targets[k] for k in range(n_opts) if k != split_idx]
            split_targets = [round(t / sum(split_targets), 4) for t in split_targets]
            items.append({
                "id": f"p4_c_{idx}_split",
                "state": state,
                "question": q_text,
                "options": split_opts,
                "target": split_targets,
                "kind": "choice"
            })
            
    # 2. Process Noul Scenarios into Negation Pairs
    for idx, item in enumerate(noul_pool[:num_base_scenarios // 2]):
        state = item["state"]
        q_text = item["question"]
        targets = [round(t / sum(item["target"]), 4) for t in item["target"]]
        
        # Base Noul
        items.append({
            "id": f"p4_n_{idx}_base",
            "state": state,
            "question": q_text,
            "options": ["false", "true"],
            "target": targets,
            "kind": "noul"
        })
        
        # Negation Wrapper
        neg_tmpl = random.choice(NEGATIONS)
        neg_q = neg_tmpl.format(question=q_text)
        items.append({
            "id": f"p4_n_{idx}_neg",
            "state": state,
            "question": neg_q,
            "options": ["false", "true"],
            "target": [targets[1], targets[0]], # flipped
            "kind": "noul"
        })

    random.shuffle(items)
    print(f"\nTotal Phase 4 targeted consistency examples: {len(items):,}")
    
    out_dir = "data/phase4_strict"
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "phase4_strict_train.jsonl")
    with open(out_file, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
            
    print(f"Successfully generated and saved Phase 4 dataset to: {out_file}")
    return out_file

if __name__ == "__main__":
    build_phase4_dataset()
