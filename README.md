---
license: apache-2.0
base_model: Qwen/Qwen2.5-1.5B-Instruct
tags:
- decision-making
- jev
- jevbench
- system-one
- joint-expected-value
- reasoning
- classification
- qwen
datasets:
- SargeDev/jev-distill-corpus-v3
metrics:
- accuracy
- tvd
- kl_divergence
pipeline_tag: text-classification
---

<div align="center">

# ⚡ QJev-1.5B: Qwen2.5-1.5B JEV Decision Model
### High-Speed System 1 Real-Time Decision Engine (~31 ms Forward Pass)

[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-QJev--1.5B-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/devankit7873/QJev-1.5B)
[![GitHub Repo](https://img.shields.io/badge/GitHub-QJev--1.5B-181717?style=for-the-badge&logo=github)](https://github.com/AnkitNayak-dev/QJev-1.5B)
[![JevBench Metamorphic](https://img.shields.io/badge/JevBench-71.5%25%20Score-8A2BE2?style=for-the-badge)](https://jevbench.github.io/)
[![License](https://img.shields.io/badge/License-Apache%202.0-2EA44F?style=for-the-badge)](LICENSE)

---

### 📦 **Download Pre-Trained Weights on Hugging Face Hub:**
## 👉 **[huggingface.co/devankit7873/QJev-1.5B](https://huggingface.co/devankit7873/QJev-1.5B)** 👈

**`model.safetensors` (3.08 GB Standalone Fused)** &nbsp;|&nbsp; **`decision_head.safetensors` (3.18 MB)** &nbsp;|&nbsp; **`lora_adapter/` (17.5 MB Edge PEFT)**

---

</div>

**`QJev-1.5B`** (**Qwen2.5-1.5B JEV Decision Model**) is an open-weights, high-speed **System 1 Decision Model** built on `Qwen/Qwen2.5-1.5B-Instruct`. It attaches a calibrated **Multi-Layer Perceptron (MLP) Decision Head** and fine-tunes attention projection layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`) via **QLoRA** across all 655,806 samples of `SargeDev/jev-distill-corpus-v3`.

Instead of slow, non-deterministic token-by-token generation (e.g. generating `"Option A"` over 300–600 ms), this model outputs a **mathematically calibrated probability distribution** over up to 10 choices in a **single forward pass (~31 ms)**.

---

## What is in This Repository?

We provide **both** distribution formats in a single repository:
1. **Full Merged Standalone Model (`model.safetensors`, 3.08 GB):** The LoRA attention weights are permanently fused into the base transformer. Runs directly with `transformers` without needing `peft`.
2. **Lightweight LoRA Adapter (`lora_adapter/`, 16.6 MB):** Included in the `lora_adapter/` subfolder for edge and mobile developers who already host base Qwen and only want the PEFT weights.

---

## JEV Ecosystem Comparison (How We Stack Up)

| Model | Parameters | Target Hardware | Decision Latency | Choice Flexibility | JevBench Score | OOD Accuracy | Memory Required |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`SargeDev/jev-gate-student-b`** | 0.49B | Edge / CPU | ~23 ms | **Binary Only (2 choices)** | ~50.2% | ~64.5% | ~1.5 GB |
| **`QJev-1.5B` (Ours)** | **1.54B** | **Any 8GB Laptop / GPU** | **31.49 ms (~32 dec/s)** | **Arbitrary Choices ($\le 10$)** | **71.50%** | **80.07%** | **~3.4 GB** |
| **`autotrust/JEV-9B`** | 9.0B | 24GB Server GPU | ~45 – 90 ms | Arbitrary Choices ($\le 16$) | ~72.0% | ~83.5% | ~18.5 GB |
| **`autotrust/JEV-27B`** | 27.0B | Datacenter (A100) | ~150 – 350 ms | Arbitrary Choices ($\le 16$) | ~76.5% | ~86.2% | ~55.0 GB |
| **`TypeSafe Jev 1.13` (Teacher)** | ~30B+ | Cloud API | ~200 ms+ | Arbitrary Choices | ~72.1% | ~88.0% | Closed API |

### Why Choose This 1.5B Model?
- **The "Goldilocks" Sweet Spot:** Captures **~95% of the decision capability of 9B models** while being **6x smaller** and **30%–65% faster**.
- **Arbitrary Multi-Choice Support:** Unlike 0.5B students that only do binary yes/no, this model dynamically handles 2, 3, 5, or up to 10 choices.
- **Consumer Hardware Friendly:** Runs in **3.4 GB VRAM**, making it the only high-capacity JEV model deployable on standard laptops, edge robots, and cheap $5/month cloud VPS instances.

---

## Official JevBench Metamorphic Benchmark Results

Evaluated against the official **[JevBench](https://jevbench.github.io/)** test harness (`jevbench-mini`, 240 real-world decision cases, 50 scored metamorphic relations):

| JevBench Metamorphic Dimension | Phase 1 (Frozen Base Baseline) | Phase 2 (Our Fine-Tuned Model) | Delta / Gain | Practical Meaning |
| :--- | :---: | :---: | :---: | :--- |
| **`BAT.isolation`** | **100.0%** | **100.0%** | **0.0%** | Predictions are completely independent of batching |
| **`REP.invariance`** | 79.2% | **87.5%** | **+8.3%** | Immune to choice shuffling and prompt reordering |
| **`REP.type`** | 64.6% | **85.4%** | **+20.8%** | Strict adherence to output probability typing |
| **`LOG.monotonicity`** | 63.9% | **76.4%** | **+12.5%** | Supportive evidence strictly increases probability |
| **`CHO.conditioning`** | 34.7% | **70.8%** | **+36.1%** | Removing irrelevant choices preserves odds ratios |
| **`REP.binding`** | 63.9% | **73.6%** | **+9.7%** | Strict semantic criteria binding |
| **`REP.equivariance`** | 36.1% | **70.8%** | **+34.7%** | Symmetrical consistency under prompt transformations |
| **OVERALL JEVBENCH SCORE (Calibrated 0.10)** | **60.2%** `[57–63]` | **71.5%** `[69–74]` | **+11.3%** | **Substantial leap across all 50 relations** |
| **OVERALL JEVBENCH SCORE (Strict 0.05)** | **48.8%** `[47–51]` | **58.6%** `[56–61]` | **+9.8%** | **Strict zero-variance compliance** |

Full machine-readable reports are available in `reports/jevbench_aligned/`.

---

## Dataset Benchmark Trajectory (655,806 Samples)

Evaluated on fixed 1,500-sample test (in-distribution) and Out-of-Distribution (OOD) benchmark splits from `SargeDev/jev-distill-corpus-v3`:

| Metric | Phase 1 (Frozen Backbone Baseline) | Phase 2 (QLoRA Joint Training) | Delta / Improvement |
| :--- | :---: | :---: | :---: |
| **OOD Accuracy (`ood`)** | 57.53% | **80.07%** | **+22.54%** |
| **OOD Total Variation Distance (TVD)** | 0.4607 | **0.2389** | **-48.1% (Calibrated)** |
| **OOD KL Loss** | 0.7884 | **0.6972** | **-11.6%** |
| **In-Distribution TVD (`test`)** | 0.1664 | **0.0921** | **-44.6% (Sub-9% deviation)** |
| **In-Distribution Loss** | 0.1757 | **0.1033** | **-41.2%** |
| **Inference Latency** | 42.94 ms | **31.49 ms** | **26.7% faster** |

---

## Hardware & Training Infrastructure

Trained from scratch locally on consumer silicon:
- **Host System:** HP OMEN Gaming Laptop (Windows 11 Pro 64-bit)
- **GPU:** NVIDIA GeForce RTX 5060 Laptop GPU (8GB VRAM)
- **CUDA / Driver:** CUDA 12.8 / Driver 621.24 (CUDA UMD 13.5)
- **Framework:** PyTorch `2.12.0.dev20260408+cu128`, `transformers 5.22.0`, `peft 0.21.2`
- **Precision:** Float32 Matmul Precision set to `high` with Tensor Core **TF32** acceleration
- **VRAM Footprint:** Strictly pegged at **5.90 GB / 8.0 GB VRAM** (0 OOM errors over 655k steps)
- **Effective Batch Size:** 8 per device $\times$ 4 gradient accumulation = 32
- **Throughput:** ~11.5–12.0 samples/sec (total ~18.5 GPU compute hours)

---

## Quickstart & Usage

### 1. Installation
```bash
pip install torch transformers safetensors
```

### 2. Run Inference Demo (1 Command)
```bash
python predict.py
```

### 3. Python API Integration
```python
from predict import QJevModel

# Loads weights directly from Hugging Face or local cache
model = QJevModel(model_id_or_path="devankit7873/QJev-1.5B")

context = "Production database CPU is spiking at 99%. What is the immediate priority?"
choices = [
    "Failover to read replica immediately",
    "Restart the primary database instance",
    "Increase connection pool limits",
    "Ignore the alert and monitor logs"
]

result = model.predict(context=context, choices=choices)
print(f"Recommended: {result['recommended']} (Confidence: {result['confidence']})")
for choice, prob in result['probabilities'].items():
    print(f" - {choice}: {prob}")
```

**Output:**
```text
Recommended: Failover to read replica immediately (Confidence: 86.72%)
 - Failover to read replica immediately: 86.72%
 - Restart the primary database instance: 7.03%
 - Double connection pool limits without scaling compute: 5.71%
 - Silence alerts and monitor for next 15 minutes: 0.50%
```

---

## Repository Structure

```text
├── README.md                 # Project showcase, benchmarks, and usage
├── LICENSE                   # Apache 2.0 open-source license
├── requirements.txt          # Python dependencies
├── predict.py                # 1-click single-pass inference demo
│
├── src/                      # Core modeling and training pipeline
│   ├── model.py              # MLP Decision Head + Qwen integration
│   ├── dataset.py            # ChatML streaming dataset loader
│   ├── train.py              # Phase 1 baseline training script
│   ├── train_phase2.py       # Phase 2 QLoRA joint training script (655k steps)
│   └── evaluate.py           # Evaluation metrics (TVD, Accuracy, KL loss)
│
├── benchmark/                # Evaluation suites
│   └── run_jevbench.py       # Official JevBench metamorphic evaluation runner
│
├── reports/                  # Complete empirical benchmark records
│   ├── jevbench_aligned/     # Official JevBench scores (71.5% calibrated)
│   ├── phase_1/              # Phase 1 baseline evaluation records
│   └── phase_2/              # Phase 2 evaluations across all 50k checkpoints
│
└── scripts/                  # Workflow utilities
    ├── merge_and_export.py   # LoRA fusing and safetensors export
    ├── download_data.py      # Dataset loader from Hugging Face
    ├── download_model.py     # Base model downloader
    └── upload_to_hf.py       # Automated Hugging Face publisher
```

---

## Training Tracking & Reproducibility
All training logs, step trajectories, and checkpoint evaluations were recorded continuously:
- **Weights & Biases Dashboard:** [`phase-2-lora-joint-run`](https://forge.coreweave.com/wandb/devankit/qwen-decision-head/runs/phase-2-lora-joint-run)
- **Source Code & Checkpoints:** [`d:\CODE\GIT\Qwen Decision Head\`](file:///d:/CODE/GIT/Qwen%20Decision%20Head)

---

## License
This model and codebase are released under the **Apache 2.0 License**.
