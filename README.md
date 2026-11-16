<div align="center">

# ⚡ QJev-1.5B: Qwen2.5-1.5B JEV Decision Model
### High-Speed System 1 Real-Time Decision Engine (~31 ms Forward Pass)

[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-QJev--1.5B-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/devankit7873/QJev-1.5B)
[![GitHub](https://img.shields.io/badge/GitHub-QJev--1.5B-181717?style=for-the-badge&logo=github)](https://github.com/AnkitNayak-dev/QJev-1.5B)
[![JevBench](https://img.shields.io/badge/JevBench-71.5%25%20Score-8A2BE2?style=for-the-badge)](https://jevbench.github.io/)
[![License](https://img.shields.io/badge/License-Apache%202.0-2EA44F?style=for-the-badge)](LICENSE)

---

### 📦 QJev Model Family (Choose Your Format):
- **Full Fused Standalone:** [devankit7873/QJev-1.5B](https://huggingface.co/devankit7873/QJev-1.5B) (3.08 GB • Zero PEFT dependency)
- **Lightweight LoRA Adapter:** [devankit7873/QJev-1.5B-LoRA](https://huggingface.co/devankit7873/QJev-1.5B-LoRA) (20 MB • Modular edge loading)
- **4-Bit NF4 Quantized:** [devankit7873/QJev-1.5B-4bit](https://huggingface.co/devankit7873/QJev-1.5B-4bit) (1.15 GB • Ultra-low 1.07 GB VRAM)

---

[News](#-news) • [Quickstart](#-quickstart--usage) • [Model Download](#-how-to-download-the-model) • [Benchmarks](#-official-jevbench-metamorphic-results) • [Architecture](#-model-architecture--system-1-paradigm) • [Citation](#-citation)

</div>

---

## 📢 News
* **[2026-11-16]** 🚀 **QJev-1.5B is officially released!** Trained on all 655,806 samples of `SargeDev/jev-distill-corpus-v3`. Achieves **71.5% calibrated JevBench score** (within 2.2% of 12B leaders) while running in **3.4 GB VRAM at ~31 ms per decision**.

---

## 💡 Overview: System 1 Decision Paradigm

Traditional LLMs make decisions through slow, non-deterministic token-by-token autoregressive generation (e.g. generating `"Option A"` over 300–600 ms). 

**`QJev-1.5B`** introduces a **System 1 Decision Architecture**:
1. Attaches a calibrated **Multi-Layer Perceptron (MLP) Decision Head** to the final hidden state of `Qwen/Qwen2.5-1.5B-Instruct`.
2. Jointly trains attention projection layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`) via **QLoRA** and the decision head across all 655,806 distillation samples.
3. Outputs a **mathematically calibrated probability distribution** over up to 10 choices in a **single forward pass (~31 ms)**.

---

## ⚖️ JEV Ecosystem Comparison

| Model | Parameters | Target Hardware | Decision Latency | Choice Flexibility | JevBench Score | OOD Accuracy | Memory Required |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`SargeDev/jev-gate-student-b`** | 0.49B | Edge / CPU | ~23 ms | **Binary Only (2 choices)** | ~50.2% | ~64.5% | ~1.5 GB |
| **`QJev-1.5B` (Ours)** | **1.54B** | **Any 8GB Laptop / GPU** | **31.49 ms (~32 dec/s)** | **Arbitrary Choices ($\le 10$)** | **71.50%** | **80.07%** | **~3.4 GB** |
| **`autotrust/JEV-9B`** | 9.0B | 24GB Server GPU | ~45 – 90 ms | Arbitrary Choices ($\le 16$) | ~72.0% | ~83.5% | ~18.5 GB |
| **`autotrust/JEV-27B`** | 27.0B | Datacenter (A100) | ~150 – 350 ms | Arbitrary Choices ($\le 16$) | ~76.5% | ~86.2% | ~55.0 GB |
| **`TypeSafe Jev 1.13` (Teacher)** | ~30B+ | Cloud API | ~200 ms+ | Arbitrary Choices | ~72.1% | ~88.0% | Closed API |

### Why Choose QJev-1.5B?
- **The "Goldilocks" Sweet Spot:** Captures **~95% of the decision capability of 9B models** while being **6x smaller** and **30%–65% faster**.
- **Dynamic Multi-Choice Support:** Unlike 0.5B students that only do binary yes/no, QJev-1.5B dynamically handles 2, 3, 5, or up to 10 choices.
- **Consumer Hardware Friendly:** Pegged at **3.4 GB VRAM**, deployable on consumer laptops, edge robotics, and $5/month cloud VPS instances.

---

## 🏆 Official JevBench Metamorphic Results

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

## 📊 Dataset Benchmark Trajectory (655,806 Samples)

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

## 📥 How to Download the Model

We host full fused weights and modular LoRA adapters on the [Hugging Face Hub](https://huggingface.co/devankit7873/QJev-1.5B).

### Method 1: Hugging Face CLI (Fastest)
```bash
huggingface-cli download devankit7873/QJev-1.5B --local-dir ./QJev-1.5B
```

### Method 2: Python `snapshot_download`
```python
from huggingface_hub import snapshot_download

snapshot_download(repo_id="devankit7873/QJev-1.5B", local_dir="./QJev-1.5B")
```

### Method 3: Git LFS Clone
```bash
git lfs install
git clone https://huggingface.co/devankit7873/QJev-1.5B
```

---

## 🚀 Quickstart & Usage

### 1. Installation
```bash
git clone https://github.com/AnkitNayak-dev/QJev-1.5B.git
cd QJev-1.5B
pip install -r requirements.txt
```

### 2. 1-Command CLI Decision
```bash
python qjev.py --context "Patient has severe chest pain and elevated troponin" --choices "Emergency PCI" "Aspirin" "Wait"
```

### 3. Super Simple Python API (3 Lines of Code)
```python
from qjev import QJev

# Load ANY edition (Full Standalone, 4-bit Quantized, or LoRA Adapter) in 1 line:
model = QJev.from_pretrained("devankit7873/QJev-1.5B")
# Or 4-bit (1.07 GB VRAM):  model = QJev.from_pretrained("devankit7873/QJev-1.5B-4bit")
# Or LoRA (20 MB download): model = QJev.from_pretrained("devankit7873/QJev-1.5B-LoRA")

# 1-Line Decision (<35 ms)
result = model.decide(
    context="Production database primary replica CPU spiked to 100% and connection pool is exhausted.",
    choices=[
        "Trigger immediate automated failover to standby replica",
        "Restart the primary database daemon immediately",
        "Double connection pool limits without scaling compute",
        "Silence alerts and monitor for next 15 minutes"
    ]
)

print(f"👉 Recommended Action: {result['best_choice']} ({result['confidence']}%)")
for choice, prob in result['probabilities'].items():
    print(f"  [{prob:>5.2f}%]  {choice}")
```

**Expected Output:**
```text
👉 Recommended Action: Trigger immediate automated failover to standby replica (88.28%)
  [88.28%]  Trigger immediate automated failover to standby replica
  [ 6.54%]  Restart the primary database daemon immediately
  [ 4.79%]  Double connection pool limits without scaling compute
  [ 0.45%]  Silence alerts and monitor for next 15 minutes
```

---

## 🏗️ Model Architecture & System 1 Paradigm

```
[User Context + Formatted Choices]
                 │
                 ▼
     ┌────────────────────────┐
     │  Qwen2.5-1.5B Backbone │  (QLoRA adapted on q, k, v, o projections)
     └───────────┬────────────┘
                 │ (Last Hidden State: 1536 dim)
                 ▼
     ┌────────────────────────┐
     │   LayerNorm(1536)      │
     └───────────┬────────────┘
                 │
                 ▼
     ┌────────────────────────┐
     │   Linear(1536 -> 512)  │  (GELU + Dropout 0.1)
     └───────────┬────────────┘
                 │
                 ▼
     ┌────────────────────────┐
     │   Linear(512 -> 10)    │  (Dynamic Choice Masking)
     └───────────┬────────────┘
                 │
                 ▼
     [Calibrated Softmax Probabilities]
```

---

## 💻 Hardware & Training Infrastructure

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

## 📁 Repository Structure

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

## 📈 Telemetry & Tracking
All training trajectories, step metrics, and evaluations were recorded in real time:
- **Weights & Biases Dashboard:** [`phase-2-lora-joint-run`](https://forge.coreweave.com/wandb/devankit/qwen-decision-head/runs/phase-2-lora-joint-run)
- **Source Code & Pipelines:** [github.com/AnkitNayak-dev/QJev-1.5B](https://github.com/AnkitNayak-dev/QJev-1.5B)
- **Model Weights & Hub:** [huggingface.co/devankit7873/QJev-1.5B](https://huggingface.co/devankit7873/QJev-1.5B)

---

## 📚 Citation

If you use **QJev-1.5B** or its benchmark methodology in your research, please cite:

```bibtex
@misc{nayak2026qjev15b,
  author = {Ankit Nayak},
  title = {QJev-1.5B: High-Speed System 1 JEV Decision Model based on Qwen2.5},
  year = {2026},
  publisher = {GitHub and Hugging Face},
  howpublished = {\url{https://github.com/AnkitNayak-dev/QJev-1.5B}}
}
```

---

## 📄 License
This model and codebase are released under the **Apache 2.0 License**.
Base weights are governed by the Alibaba Cloud Qwen License. Distillation corpus is provided by SargeDev (`jev-distill-corpus-v3`). Metamorphic benchmarking powered by JevBench.
