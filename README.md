---
license: apache-2.0
base_model: Qwen/Qwen2.5-1.5B-Instruct
library_name: transformers
tags:
- transformers
- pytorch
- causal-lm
- qwen
- qwen2
- decision-making
- system-one
- reasoning
- jev
- jevbench
- joint-expected-value
- classification
- safetensors
- qlora
- agent
- decision-model
- text-classification
- fast-inference
datasets:
- SargeDev/jev-distill-corpus-v3
metrics:
- accuracy
- tvd
- kl_divergence
pipeline_tag: text-classification
---

<div align="center">

# ⚡ QJev-1.5B: High-Speed System 1 JEV Decision Model
### Single-Pass Real-Time Decision Engine (~31 ms Forward Pass • Ranked #1 Sub-2B Model)

[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-devankit7873%2FQJev--1.5B-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/devankit7873/QJev-1.5B)
[![GitHub](https://img.shields.io/badge/GitHub-QJev--1.5B-181717?style=for-the-badge&logo=github)](https://github.com/AnkitNayak-dev/QJev-1.5B)
[![JevBench](https://img.shields.io/badge/JevBench%20Official-62.53%25%20(tol=0.05)-8A2BE2?style=for-the-badge)](https://benchmarkheaven.com/jev-models)
[![JevBench Relaxed](https://img.shields.io/badge/JevBench%20Relaxed-71.50%25%20(tol=0.10)-blue?style=for-the-badge)](https://benchmarkheaven.com/jev-models)
[![License](https://img.shields.io/badge/License-Apache%202.0-2EA44F?style=for-the-badge)](LICENSE)

---

### 🌳 Hugging Face Model Tree & Flavors:
```
Qwen/Qwen2.5-1.5B-Instruct  (Root Base)
 └── ⚡ devankit7873/QJev-1.5B  (Flagship Fused 16-bit • 3.08 GB • Zero PEFT dependency)
      ├── 🔌 devankit7873/QJev-1.5B-LoRA  (PEFT Adapter • 20 MB • Modular edge loading)
      └── 🗜️ devankit7873/QJev-1.5B-4bit  (NF4 Quantized • 1.15 GB • 1.07 GB VRAM)
```

[News](#-news) • [Overview](#-overview-system-1-decision-paradigm) • [Leaderboard & Benchmarks](#-official-benchmarkheaven--jevbench-results) • [Quickstart](#-quickstart--usage) • [Multi-Engine Serving](#-multi-engine-inference-deployment) • [Architecture](#-model-architecture) • [Repo Structure](#-repository-structure) • [Citation](#-citation)

</div>

---

## 📢 News
* **[2026-11-17] 🎯 Phase 3 Consistency Release & Multi-Engine Serving:**
  - Trained with cross-format symmetric consistency loss, boosting official **JevBench strict tolerance (0.05) score from 58.67% to 62.53%** (+3.86% overall, +15.2% on choice conditioning).
  - Officially ranks as the **#1 decision model in the sub-2B tier**, outperforming multiple 4B baselines on [BenchmarkHeaven](https://benchmarkheaven.com/jev-models).
  - Shipped native deployment support for **vLLM**, **Ollama (`Modelfile`)**, **llama.cpp (GGUF)**, and **production FastAPI (`serve.py`)**.
* **[2026-11-16] 🚀 Initial Release:**
  - Full QLoRA joint training across all 655,806 distillation samples of `SargeDev/jev-distill-corpus-v3`.
  - Achieves **71.50% relaxed metamorphic score** in **3.4 GB VRAM at ~31 ms per decision**.

---

## 💡 Overview: System 1 Decision Paradigm

Traditional LLMs make decisions through slow, non-deterministic token-by-token autoregressive generation (e.g. generating `"Option A"` over 300–600 ms). 

**`QJev-1.5B`** introduces a **System 1 Decision Architecture**:
1. Attaches a calibrated **Multi-Layer Perceptron (MLP) Decision Head** to the final hidden state of `Qwen/Qwen2.5-1.5B-Instruct`.
2. Jointly trains attention projection layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`) via **QLoRA** and the decision head across all 655,806 distillation samples.
3. In Phase 3, fine-tunes with soft target KL-divergence consistency loss across multiple-choice, boolean probes, and permutation groups.
4. Outputs a **mathematically calibrated probability distribution** over up to 10 choices in a **single forward pass (~31 ms)**.

---

## 🏆 Official BenchmarkHeaven & JevBench Results

Evaluated against the official **JevBench** test harness (`jevbench-mini`, 240 cases, 50 scored metamorphic relations, 1,468 trials):

### 1. Leaderboard Placement (BenchmarkHeaven JEV Leaderboard)

Under the official BenchmarkHeaven standard (**strict tolerance = 0.05**):

| Model | Parameters | Benchmark Score (tol=0.05) | Rank / Category | Latency | VRAM Footprint |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `EXAONE-1.2B` | 1.2B | 47.7% | Sub-2B | ~25 ms | ~2.8 GB |
| `SargeDev/jev-gate-student-b` | 0.49B | 50.2% | Sub-2B (Binary Only) | ~23 ms | ~1.5 GB |
| `decision-2b` | 2.0B | 54.8% | 2B Tier | ~40 ms | ~4.5 GB |
| `pjev-2b` | 2.0B | 56.5% | 2B Tier | ~38 ms | ~4.4 GB |
| **`QJev-1.5B` (Ours, Phase 3)** | **1.54B** | **62.53%** | **👑 #1 Sub-2B (Beats 4B)** | **~31 ms** | **~3.4 GB (1.07 GB in 4-bit)** |
| `jiwo-4b` | 4.0B | 60.2% | 4B Tier | ~65 ms | ~8.8 GB |
| `decision-4b-v12` | 4.0B | 60.4% | 4B Tier | ~62 ms | ~8.5 GB |
| `autotrust/JEV-9B` | 9.0B | 72.0% | 9B Tier | ~85 ms | ~18.5 GB |
| `autotrust/JEV-27B` | 27.0B | 76.5% | Server Tier | ~220 ms | ~55.0 GB |

> **Note on Evaluation Tolerances:**
> - **Official Strict (`tol = 0.05`):** **62.53%** (Official score reported on BenchmarkHeaven public leaderboard).
> - **Relaxed Exploration (`tol = 0.10`):** **71.50%** (Lenient consistency threshold).

### 2. Breakdown Across Metamorphic Dimensions

| Dimension | Baseline (Pre-Consistency) | QJev-1.5B (Phase 3 Consistency) | Gain | Practical Meaning |
| :--- | :---: | :---: | :---: | :--- |
| **`BAT.isolation`** | **100.0%** | **100.0%** | **0.0%** | Batching independence (Zero cross-talk between batch items) |
| **`REP.type`** | 77.1% | **83.3%** | **+6.2%** | Format coherence across multi-choice, boolean, and ordinal prompts |
| **`REP.invariance`** | 73.2% | **74.4%** | **+1.2%** | Immune to choice reordering and prompt syntax perturbations |
| **`LOG.bounds`** | 62.5% | **67.9%** | **+5.4%** | Strict adherence to probabilistic upper and lower bounds |
| **`CHO.conditioning`** | 41.7% | **56.9%** | **+15.2%** | Removing irrelevant choices preserves odds ratios (Independence of Irrelevant Alternatives) |
| **`REP.equivariance`** | 48.6% | **55.6%** | **+7.0%** | Symmetrical probability shifts under negation transformations |
| **`MEA.partition`** | 30.7% | **41.1%** | **+10.4%** | Probability mass preservation across option clustering |
| **OVERALL (Strict 0.05)** | **58.67%** | **62.53%** | **+3.86%** | **Official Benchmark Standard** |

Machine-readable evaluation reports are archived in `reports/jevbench/`.

---

## 🚀 Quickstart & Usage

### 1. Installation
```bash
git clone https://github.com/AnkitNayak-dev/QJev-1.5B.git
cd QJev-1.5B
pip install -r requirements.txt
```

### 2. Fast Python API (3 Lines of Code)
```python
from qjev import QJev

# Load ANY edition (Full Standalone, 4-bit Quantized, or LoRA Adapter) in 1 line:
model = QJev.from_pretrained("devankit7873/QJev-1.5B")
# Or 4-bit (1.07 GB VRAM):  model = QJev.from_pretrained("devankit7873/QJev-1.5B-4bit")
# Or LoRA (20 MB download): model = QJev.from_pretrained("devankit7873/QJev-1.5B-LoRA")

# 1-Line Decision (~31 ms)
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

### 3. CLI Command-Line Decision
```bash
python qjev.py --context "Patient has severe chest pain and elevated troponin" --choices "Emergency PCI" "Aspirin" "Wait"
```

---

## 🌐 Multi-Engine Inference Deployment

QJev-1.5B natively supports **all standard LLM serving runtimes**:

### 1. High-Performance Microservice Server (`serve.py`)
Run the unified FastAPI server exposing REST, System One, and OpenAI compatible endpoints:
```bash
python serve.py --model devankit7873/QJev-1.5B --port 8000
```
Endpoints provided:
* **REST Decision API:** `POST /v1/decide`
* **Official Jev Protocol (BenchmarkHeaven):** `POST /v1/systemone`
* **OpenAI Drop-In Compatible:** `POST /v1/chat/completions`

Query via cURL:
```bash
curl -X POST http://localhost:8000/v1/decide \
  -H "Content-Type: application/json" \
  -d '{
    "context": "Customer requests immediate refund after 45 days.",
    "options": ["Approve refund", "Offer store credit", "Deny request"]
  }'
```

### 2. Ollama (Local Mac / PC / Linux)
Package and run locally with Ollama using the included `Modelfile`:
```bash
ollama create qjev-1.5b -f Modelfile
ollama run qjev-1.5b
```
Or query from Python via `QJev`:
```python
from qjev import QJev
model = QJev.from_ollama("http://localhost:11434", model="qjev-1.5b")
result = model.decide("Route packet to cluster A or B?", ["Cluster A", "Cluster B"])
```

### 3. llama.cpp / GGUF (LM Studio & Edge CPU)
Export to GGUF format for edge devices using the export tool:
```bash
python scripts/export_gguf.py --model-dir export/QJev-1.5B-Phase3 --out-file qjev-1.5b-f16.gguf
./llama-quantize qjev-1.5b-f16.gguf qjev-1.5b-q4_k_m.gguf Q4_K_M
```

### 4. vLLM (Production Continuous Batching)
```bash
vllm serve devankit7873/QJev-1.5B --task classify --port 8000
```

---

## 🏗️ Model Architecture

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

## 📁 Repository Structure

```text
├── README.md                           # Project showcase, benchmarks, and documentation
├── LICENSE                             # Apache 2.0 open-source license
├── requirements.txt                    # Python dependencies
├── Modelfile                           # Ollama build definition
├── qjev.py                             # Universal Python SDK & CLI client
├── serve.py                            # Production FastAPI server (REST, Jev, OpenAI)
├── predict.py                          # Single-pass forward evaluation demo
├── handler.py                          # Hugging Face Inference Endpoint handler
│
├── benchmark/                          # JevBench self-test and evaluation suites
│   ├── run_jevbench.py                 # JevBench metamorphic test harness
│   ├── self_test_benchmarkheaven.py    # Official BenchmarkHeaven standard runner
│   ├── self_test_phase3.py             # Phase 3 consistency verification runner
│   └── self_test_step1.py              # Calibrated temperature sweep suite
│
├── reports/                            # Empirical evaluation records
│   └── jevbench/                       # JSON benchmark results across phases
│       ├── self_test_phase3_consistency.json # Official 62.53% report
│       ├── jevbench_phase1_report.json
│       └── jevbench_phase2_report.json
│
├── src/                                # Core modeling and training code
│   ├── model.py                        # MLP Decision Head + Qwen architecture
│   ├── dataset.py                      # Streaming ChatML dataset builder
│   ├── train.py                        # Phase 1 baseline training
│   ├── train_phase2.py                 # Phase 2 QLoRA joint training (655k samples)
│   ├── train_phase3_consistency.py     # Phase 3 KL consistency alignment
│   └── evaluate.py                     # Out-of-Distribution evaluation metrics
│
└── scripts/                            # Engineering and deployment utilities
    ├── build_consistency_dataset.py    # 20k paired consistency dataset generator
    ├── merge_phase3.py                 # Phase 3 LoRA fusing & packaging
    ├── merge_and_export.py             # Base export utility
    ├── export_gguf.py                  # GGUF converter for llama.cpp / LM Studio
    ├── publish_all_hf.py               # Hugging Face Hub publisher
    ├── download_data.py                # Dataset downloader
    └── download_model.py               # Base model downloader
```

---

## 💻 Hardware & Training Infrastructure

- **Hardware:** HP OMEN Laptop (NVIDIA GeForce RTX 5060 Laptop GPU, 8GB VRAM)
- **Host OS:** Windows 11 Pro 64-bit
- **CUDA / Driver:** CUDA 12.8 / Driver 621.24
- **Precision:** Float32 Matmul Precision (`high`) with Tensor Core TF32 acceleration
- **Training Trajectory:**
  - **Phase 1:** MLP Head alignment on frozen backbone (1.5 hrs).
  - **Phase 2:** Joint QLoRA + MLP training across 655,806 samples (18.5 hrs, strictly 5.9 GB VRAM).
  - **Phase 3:** Paired consistency KL alignment (27 mins, 1,250 steps, batch size 16).

---

## 📚 Citation

If you use **QJev-1.5B** or its benchmark methodology, please cite:

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
Base weights are governed by Alibaba Cloud Qwen License. Distillation corpus is provided by SargeDev (`jev-distill-corpus-v3`).
