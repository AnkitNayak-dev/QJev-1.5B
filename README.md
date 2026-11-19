<div align="center">

# QJev-1.5B: High-Speed System 1 JEV Decision Model
**Single-Pass Joint Expected Value Decision Engine (~31 ms Forward Pass)**

[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-AnkitNayak-dev%2FQJev--1.5B-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co/AnkitNayak-dev/QJev-1.5B)
[![GitHub](https://img.shields.io/badge/GitHub-QJev--1.5B-181717?style=flat-square&logo=github)](https://github.com/AnkitNayak-dev/QJev-1.5B)
[![JevBench Strict](https://img.shields.io/badge/JevBench%20Strict-62.53%25%20(tol=0.05)-8A2BE2?style=flat-square)](https://benchmarkheaven.com/jev-models)
[![JevBench Relaxed](https://img.shields.io/badge/JevBench%20Relaxed-71.50%25%20(tol=0.10)-blue?style=flat-square)](https://benchmarkheaven.com/jev-models)
[![License](https://img.shields.io/badge/License-Apache%202.0-2EA44F?style=flat-square)](LICENSE)

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#benchmark-results">Benchmark Results</a> •
  <a href="#model-variants">Model Variants</a> •
  <a href="#quickstart">Quickstart</a> •
  <a href="#serving-and-deployment">Serving and Deployment</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#repository-structure">Repository Structure</a> •
  <a href="#citation">Citation</a>
</p>

</div>

---

### Model Hierarchy

```text
Qwen/Qwen2.5-1.5B-Instruct (Base Model)
 └── AnkitNayak-dev/QJev-1.5B (Flagship Fused 16-bit, 3.08 GB)
      ├── AnkitNayak-dev/QJev-1.5B-LoRA (PEFT Adapter, 20 MB)
      └── AnkitNayak-dev/QJev-1.5B-4bit (NF4 Quantized, 1.15 GB)
```

---

## Overview

Traditional large language models perform decision-making through sequential, autoregressive token generation. This process incurs substantial latency (often 300 to 600 ms) and introduces non-deterministic sampling variance when parsing formatted outputs.

**QJev-1.5B** implements a direct System 1 decision mechanism by coupling a calibrated Multi-Layer Perceptron (MLP) decision head to the final hidden representation of `Qwen/Qwen2.5-1.5B-Instruct`. Rather than generating text tokens, the model computes a calibrated probability distribution across up to 10 candidate actions in a single forward pass (~31 ms on consumer hardware).

### Key Characteristics

- **Single Forward Pass Execution:** Decision latency of ~31 ms (~32 decisions/second per device).
- **Calibrated Probability Distribution:** Direct outputs represent normalized decision probabilities rather than heuristic sequence log-probabilities.
- **Dynamic Choice Support:** Accommodates arbitrary choice cardinalities from 2 up to 10 options with dynamic logit masking.
- **Consistency Regularization:** Aligned via Phase 3 soft-target symmetric cross-format consistency loss to ensure stability across option permutations, boolean probes, and coarsened subsets.

---

## Benchmark Results

QJev-1.5B is evaluated against the official [JevBench](https://jevbench.github.io/) metamorphic test harness (`jevbench-mini`, 240 evaluation cases, 50 scored relations, 1,468 trials) following standard BenchmarkHeaven submission criteria.

### Leaderboard Comparison

All models evaluated under official strict calibration criteria (**tolerance = 0.05**):

| Model | Parameters | Official Score (tol=0.05) | Category | Inference Latency | Memory Footprint |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `EXAONE-1.2B` | 1.2B | 47.7% | Sub-2B | ~25 ms | ~2.8 GB |
| `SargeDev/jev-gate-student-b` | 0.49B | 50.2% | Sub-2B (Binary Only) | ~23 ms | ~1.5 GB |
| `decision-2b` | 2.0B | 54.8% | 2B Tier | ~40 ms | ~4.5 GB |
| `pjev-2b` | 2.0B | 56.5% | 2B Tier | ~38 ms | ~4.4 GB |
| **`QJev-1.5B` (Ours, Phase 3)** | **1.54B** | **62.53%** | **Rank 1 Sub-2B** | **~31 ms** | **3.4 GB (1.07 GB in 4-bit)** |
| `jiwo-4b` | 4.0B | 60.2% | 4B Tier | ~65 ms | ~8.8 GB |
| `decision-4b-v12` | 4.0B | 60.4% | 4B Tier | ~62 ms | ~8.5 GB |
| `autotrust/JEV-9B` | 9.0B | 72.0% | 9B Tier | ~85 ms | ~18.5 GB |
| `autotrust/JEV-27B` | 27.0B | 76.5% | Datacenter Tier | ~220 ms | ~55.0 GB |

*Tolerance specification: The official BenchmarkHeaven evaluation suite strictly uses `tolerance = 0.05` for headline Capability scores. Under exploratory relaxed evaluation (`tolerance = 0.10`), QJev-1.5B achieves 71.50%.*

### Metamorphic Consistency Breakdown

| Dimension | Baseline (Pre-Consistency) | QJev-1.5B (Phase 3 Consistency) | Delta | Formal Property Evaluated |
| :--- | :---: | :---: | :---: | :--- |
| `BAT.isolation` | 100.0% | **100.0%** | 0.0% | Invariance to batch composition and sample ordering |
| `REP.type` | 77.1% | **83.3%** | +6.2% | Invariance across prompt format representations (MCQ, boolean, ordinal) |
| `REP.invariance` | 73.2% | **74.4%** | +1.2% | Stability under option key renaming and syntax perturbations |
| `LOG.bounds` | 62.5% | **67.9%** | +5.4% | Adherence to probabilistic upper and lower bounds |
| `CHO.conditioning` | 41.7% | **56.9%** | +15.2% | Preservation of relative odds upon elimination of irrelevant options |
| `REP.equivariance` | 48.6% | **55.6%** | +7.0% | Consistent probability shifts under negation transformations |
| `MEA.partition` | 30.7% | **41.1%** | +10.4% | Probability mass conservation under option clustering |
| **Overall (Official Strict 0.05)** | **58.67%** | **62.53%** | **+3.86%** | **Standard JevBench Capability Benchmark** |

Machine-readable evaluation artifacts are archived under `reports/jevbench/`.

---

## Model Variants

QJev-1.5B is available in three deployment formats:

| Variant | Hugging Face Repository | Disk Size | VRAM Footprint | Recommended Use Case |
| :--- | :--- | :---: | :---: | :--- |
| **Full Fused (FP16)** | [AnkitNayak-dev/QJev-1.5B](https://huggingface.co/AnkitNayak-dev/QJev-1.5B) | 3.08 GB | 3.4 GB | Production servers, vLLM, standalone microservices |
| **PEFT Adapter** | [AnkitNayak-dev/QJev-1.5B-LoRA](https://huggingface.co/AnkitNayak-dev/QJev-1.5B-LoRA) | 20.7 MB | Shared | Multi-tenant deployments caching `Qwen2.5-1.5B-Instruct` |
| **NF4 Quantized** | [AnkitNayak-dev/QJev-1.5B-4bit](https://huggingface.co/AnkitNayak-dev/QJev-1.5B-4bit) | 1.15 GB | 1.07 GB | Resource-constrained edge instances, local workstations |

---

## Quickstart

### Installation

```bash
git clone https://github.com/AnkitNayak-dev/QJev-1.5B.git
cd QJev-1.5B
pip install -r requirements.txt
```

### Python API

```python
from qjev import QJev

# Load any model variant directly from Hugging Face Hub
model = QJev.from_pretrained("AnkitNayak-dev/QJev-1.5B")
# Or 4-bit quantized: model = QJev.from_pretrained("AnkitNayak-dev/QJev-1.5B-4bit")
# Or PEFT adapter:    model = QJev.from_pretrained("AnkitNayak-dev/QJev-1.5B-LoRA")

# Execute decision inference
result = model.decide(
    context="Production database primary replica CPU utilization reached 100% and connection pool is exhausted.",
    choices=[
        "Trigger immediate automated failover to standby replica",
        "Restart the primary database daemon immediately",
        "Double connection pool limits without scaling compute",
        "Silence alerts and monitor for next 15 minutes"
    ]
)

print(f"Recommended Action: {result['best_choice']} ({result['confidence']}%)")
for choice, prob in result['probabilities'].items():
    print(f"  [{prob:>5.2f}%]  {choice}")
```

#### Output

```text
Recommended Action: Trigger immediate automated failover to standby replica (88.28%)
  [88.28%]  Trigger immediate automated failover to standby replica
  [ 6.54%]  Restart the primary database daemon immediately
  [ 4.79%]  Double connection pool limits without scaling compute
  [ 0.45%]  Silence alerts and monitor for next 15 minutes
```

### Command-Line Interface

```bash
python qjev.py --context "Patient presents with acute chest pain and elevated troponin levels." --choices "Emergency PCI" "Aspirin" "Observation"
```

---

## Serving and Deployment

### 1. Unified Microservice Server (`serve.py`)

A production FastAPI service providing REST, Jev protocol, and OpenAI-compatible endpoints:

```bash
python serve.py --model AnkitNayak-dev/QJev-1.5B --port 8000
```

#### Endpoints
- `POST /v1/decide` - Direct decision endpoint returning structured probabilities.
- `POST /v1/systemone` - Official BenchmarkHeaven JEV evaluation protocol.
- `POST /v1/chat/completions` - Drop-in OpenAI-compatible wrapper.

#### Query via cURL

```bash
curl -X POST http://localhost:8000/v1/decide \
  -H "Content-Type: application/json" \
  -d '{
    "context": "Customer requests immediate refund after policy window has expired.",
    "options": ["Approve refund", "Issue store credit", "Deny request"]
  }'
```

### 2. Ollama Integration

A preconfigured `Modelfile` is provided in the repository root:

```bash
ollama create qjev-1.5b -f Modelfile
ollama run qjev-1.5b
```

Querying via the Python SDK:

```python
from qjev import QJev

client = QJev.from_ollama("http://localhost:11434", model="qjev-1.5b")
result = client.decide("Route network traffic to cluster Alpha or Beta?", ["Alpha", "Beta"])
```

### 3. llama.cpp and GGUF Export

Convert weights to GGUF format for edge CPU and local deployment:

```bash
python scripts/export_gguf.py --model-dir export/QJev-1.5B-Phase3 --out-file qjev-1.5b-f16.gguf
./llama-quantize qjev-1.5b-f16.gguf qjev-1.5b-q4_k_m.gguf Q4_K_M
```

### 4. vLLM Serving

```bash
vllm serve AnkitNayak-dev/QJev-1.5B --task classify --port 8000
```

---

## Architecture

```text
[Prompt Context + Formatted Candidate Choices]
                       │
                       ▼
           ┌────────────────────────┐
           │ Qwen2.5-1.5B Backbone  │  (QLoRA adapted: q, k, v, o projections)
           └───────────┬────────────┘
                       │ (Final Hidden State: 1536 dim)
                       ▼
           ┌────────────────────────┐
           │    LayerNorm(1536)     │
           └───────────┬────────────┘
                       │
                       ▼
           ┌────────────────────────┐
           │  Linear(1536 -> 512)   │  (GELU Activation, Dropout 0.1)
           └───────────┬────────────┘
                       │
                       ▼
           ┌────────────────────────┐
           │   Linear(512 -> 10)    │  (Dynamic Logit Masking for N Choices)
           └───────────┬────────────┘
                       │
                       ▼
         [Calibrated Softmax Probabilities]
```

---

## Training and Implementation Details

- **Base Architecture:** `Qwen/Qwen2.5-1.5B-Instruct`
- **Training Corpus:** `SargeDev/jev-distill-corpus-v3` (655,806 distillation examples)
- **Training Stages:**
  - **Phase 1 (MLP Head Initialization):** 50,000 steps with frozen backbone weights.
  - **Phase 2 (QLoRA Joint Optimization):** Full-corpus training across 655,806 samples with rank-16 LoRA adapters targeting all attention projections.
  - **Phase 3 (Cross-Format Consistency Alignment):** 1,250 steps of symmetric KL-divergence minimization across paired multiple-choice, boolean, and subset queries.
- **Hardware Configuration:** NVIDIA GeForce RTX 5060 Laptop GPU (8GB VRAM), CUDA 12.8, PyTorch 2.12 with TF32 precision.

---

## Repository Structure

```text
├── README.md                           # Project documentation and benchmarks
├── LICENSE                             # Apache 2.0 open-source license
├── requirements.txt                    # Python dependencies
├── Modelfile                           # Ollama configuration
├── qjev.py                             # Unified Python SDK and CLI client
├── serve.py                            # Production FastAPI service
├── predict.py                          # Forward pass evaluation script
├── handler.py                          # Hugging Face inference handler
│
├── benchmark/                          # Benchmark evaluation suites
│   ├── run_jevbench.py                 # JevBench test harness runner
│   ├── self_test_benchmarkheaven.py    # Official BenchmarkHeaven evaluation runner
│   ├── self_test_phase3.py             # Phase 3 verification runner
│   └── self_test_step1.py              # Calibration sweep script
│
├── reports/                            # Empirical evaluation logs
│   └── jevbench/                       # Benchmark records
│       ├── self_test_phase3_consistency.json # Official 62.53% report
│       ├── jevbench_phase1_report.json
│       └── jevbench_phase2_report.json
│
├── src/                                # Model definitions and training routines
│   ├── model.py                        # MLP Decision Head and Qwen integration
│   ├── dataset.py                      # Streaming ChatML dataset loader
│   ├── train.py                        # Phase 1 initialization
│   ├── train_phase2.py                 # Phase 2 QLoRA training
│   ├── train_phase3_consistency.py     # Phase 3 consistency alignment
│   └── evaluate.py                     # Metric evaluation routines
│
└── scripts/                            # Utilities and deployment tools
    ├── build_consistency_dataset.py    # Consistency dataset generator
    ├── merge_phase3.py                 # Weight fusing and export
    ├── merge_and_export.py             # Model export routine
    ├── export_gguf.py                  # GGUF conversion utility
    ├── publish_all_hf.py               # Hugging Face hub publishing script
    ├── download_data.py                # Dataset download helper
    └── download_model.py               # Base model download helper
```

---

## Citation

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

## License

This project is licensed under the [Apache 2.0 License](LICENSE).
Base model weights are governed by the Alibaba Cloud Qwen License. Training data is provided by SargeDev (`jev-distill-corpus-v3`).
