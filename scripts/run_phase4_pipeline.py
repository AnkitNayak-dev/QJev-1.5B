import os
import sys
import time

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(repo_root)

print("=" * 70, flush=True)
print("STARTING FULL PHASE 4 PIPELINE: TRAIN -> MERGE -> EVALUATE", flush=True)
print("=" * 70, flush=True)

# 1. Train
print("\n>>> STEP 1: Running Phase 4 Strict Margin Training (2000 steps)...", flush=True)
from src.train_phase4_strict import train_phase4
train_phase4(max_steps=2000)

# 2. Merge
print("\n>>> STEP 2: Merging Phase 4 LoRA Adapter into Standalone Model...", flush=True)
from scripts.merge_phase4 import merge_phase4
merge_phase4()

# 3. Benchmark
print("\n>>> STEP 3: Evaluating Official JevBench (tolerance=0.05)...", flush=True)
from benchmark.self_test_phase4 import run_phase4_evaluation
run_phase4_evaluation()

print("\n" + "=" * 70, flush=True)
print("PHASE 4 PIPELINE COMPLETED SUCCESSFULLY!", flush=True)
print("=" * 70, flush=True)
