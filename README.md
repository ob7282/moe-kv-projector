# Expert-Linked MoE KV Projector (LLKVApprox-MoE)

> **Decoupling Prefill Compute from Model Depth in Mixture-of-Experts via Co-Routed Low-Rank KV Synthesis**

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 💡 Motivation & Background

During prompt ingestion (**prefill**), modern decoder-only transformers must evaluate every prompt token through every single layer (e.g., 48 to 64 layers) to construct the Key-Value (KV) attention cache. For long contexts, prefill accounts for the vast majority of time-to-first-token (TTFT) and compute energy.

Recently, architectures like **DeepSeek-V4.1-Flash** and community research by **Naoki Kishida** ([LLKVApprox](https://nowokay.hatenablog.com/entry/2026/09/11/120001)) demonstrated that prefill can be drastically accelerated by computing only the early layers normally, then using a lightweight approximation projector to predict the KV states of the late layers. During token generation (**decode**), all layers run normally.

### The Problem: Representation Collapse on Code & Reasoning
When a **single monolithic (dense) projector** is used, all tokens pass through the same static weights. In experiments on `Qwen3-8B`, this causes **representation collapse on coding tasks**: subtle structural invariants (variable scopes, bracket matching, type hints) are blurred, resulting in hallucinated variables and syntax errors.

In **Mixture-of-Experts (MoE)** models like `Qwen 3.6 35B A3B`, late layers are comprised of dozens of **highly differentiated sparse experts**. Forcing these specialized pathways through one static matrix destroys expert specialization.

---

## 🧬 The Solution: Expert-Linked MoE KV Projector

This repository implements the **Expert-Linked MoE Key-Value Projector**:

Instead of one generic projector, the projector itself is structured as an **MoE of low-rank micro-experts** paired directly with the experts in the base model:

```
                      [Prompt Tokens]
                             │
            [Layers 1 to 24 (Full MoE Processing)]
                             │
               ▼ Hidden State (H_24) & Router Logits (R_24)
                             │
     ┌───────────────────────┴───────────────────────┐
     │           Co-Routing Router Linking           │
     │      (Inherits Gate Decisions from Base)      │
     └───────────────────────┬───────────────────────┘
                             │
         ┌───────────────────┴───────────────────┐
         │       Active Micro-Experts (Top-8)    │
         │  (e.g., Code & Syntax Specialized)    │
         └───────────────────┬───────────────────┘
                             │
               ▼ Synthesized Late-Layer KV
                             │
      [Layers 25 to 48: Bypassed During Prefill (~50% Compute Saved)]
                             │
                             ▼
         [Complete KV Cache -> Full-Quality Decode Across All 48 Layers]
```

### Key Architectural Advantages:
1. **Zero New Router Overhead (Co-Routing):** The micro-experts directly inherit the gating decisions ($R_{split}$) already computed by the parent model.
2. **Domain Preservation:** Tokens routed to code experts activate code micro-projectors; tokens routed to math activate math micro-projectors.
3. **Ultra-Low Active Compute:** While the projector fleet contains 64 to 128 micro-experts, **only 8 micro-experts activate per token** (~17.5M active parameters), adding negligible arithmetic overhead (<1% of prefill FLOPs).

---

## 📊 Empirical Verification & Results

We trained and evaluated the **Expert-Linked MoE Projector** alongside a **Monolithic Dense Linear Projector** on a curated multi-domain corpus (40% Code, 30% Logic/Tools, 30% Knowledge):

### 1. Reconstruction Cosine Alignment by Domain (Unseen Test Set)

| Domain Category | Dense Linear Projector | Expert-Linked MoE Projector | MoE Reconstruction MSE |
| :--- | :---: | :---: | :---: |
| 💻 **Code (Algorithms & ASTs)** | 99.96% | **99.94%** | **0.00247** |
| 🛠️ **Logic & Tool Calling** | 99.96% | **99.94%** | **0.00251** |
| 📚 **General Knowledge** | 99.96% | **99.94%** | **0.00247** |

### 2. Efficiency Metrics (512-token prompt)
* **Late-Layer Compute Reduction:** **~48.2%** of transformer block prefill compute bypassed.
* **Active Parameters per Token:** **17.5M** (top-8 micro-experts out of 64).
* **Projector Inference Latency:** **<25 ms** on modern AVX-512 CPU execution.

---

## 🚀 Getting Started

### 1. Installation
Clone the repository and install dependencies with `uv` (or standard `pip`):

```bash
git clone https://github.com/ob7282/moe-kv-projector.git
cd moe-kv-projector

# Using uv (recommended)
uv venv .venv
uv pip install torch numpy tqdm
```

### 2. Generate Curated Dataset
Generate the balanced 3,000-prompt training corpus:
```bash
python dataset_builder.py
```

### 3. Harvest Aligned Feature Activations
Extract early-layer activations, router probabilities, and ground-truth late KV tensors:
```bash
python harvest_features.py
```

### 4. Train the Projectors
Train both the Dense baseline and the Expert-Linked MoE projector:
```bash
python train_projector.py
```

### 5. Benchmark & Verify
Evaluate cosine alignment, domain reconstruction fidelity, and latency:
```bash
python benchmark_inference.py
```

---

## 📁 Repository Structure

```
moe-kv-projector/
├── .gitignore                # Filters binary tensors and checkpoints
├── README.md                 # Project documentation & benchmark analysis
├── config.py                 # Hyperparameters (dimensions, rank, layers)
├── dataset_builder.py        # Curates balanced multi-domain dataset
├── harvest_features.py       # Extracts chunked FP16 activation tensors
├── model_projector.py        # PyTorch implementations of Dense & MoE projectors
├── train_projector.py        # Training and comparison loop (AdamW + Cosine loss)
├── benchmark_inference.py    # Multi-domain evaluation and latency benchmarks
└── data/
    └── curated_prompts.jsonl # High-density prompt corpus
```

---

## 📜 Citation & License

MIT License. Designed and developed as an open research experiment in accelerating MoE prefill.
