import os
import sys

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(repo_root)

print("=" * 70, flush=True)
print("STARTING BALANCED DATA MIXING PIPELINE: TRAIN -> MERGE -> EVALUATE", flush=True)
print("=" * 70, flush=True)

# 1. Train
print("\n>>> STEP 1: Running Balanced Phase 4 Training (2,500 steps)...", flush=True)
from src.train_phase4_mixed import train_phase4_mixed
train_phase4_mixed(max_steps=2500)

# 2. Merge
print("\n>>> STEP 2: Merging Adapter into Standalone Model...", flush=True)
from scripts.merge_phase4_mixed import merge_phase4_mixed
merge_phase4_mixed()

# 3. Benchmark
print("\n>>> STEP 3: Evaluating Official JevBench (tolerance=0.05)...", flush=True)
from benchmark.self_test_phase4_mixed import run_phase4_mixed_evaluation
run_phase4_mixed_evaluation()

print("\n" + "=" * 70, flush=True)
print("BALANCED DATA MIXING PIPELINE COMPLETED SUCCESSFULLY!", flush=True)
print("=" * 70, flush=True)
