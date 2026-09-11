# Expert-Linked MoE KV Projector (LLKVApprox-MoE)

> **Extending Late-Layer KV Approximation to Mixture-of-Experts Models via Co-Routed Low-Rank Projectors**

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🙏 Credits & Prior Work

This research directly builds upon the work and experiments of **Naoki Kishida** ([@kis on X](https://x.com/kis), [@nowokay on Hatena](https://nowokay.hatenablog.com/)), specifically his post:
* **[DeepSeek-V4.1-FlashがEncoder-Decoderと呼んでいる後半層KV近似をQwen3で試してPrefill時間を半分にする](https://nowokay.hatenablog.com/entry/2026/09/11/120001)** (September 11, 2026).

Kishida demonstrated that during prompt prefill, calculating activations for early layers and using a small projector to synthesize Key-Value (KV) cache states for later layers can cut prefill time in half on `Qwen3-8B`. This project explores how that principle can be adapted to **Mixture-of-Experts (MoE)** models such as `Qwen 3.6 35B A3B`.

---

## 💡 Background & Motivation

During prompt ingestion (**prefill**), decoder-only models compute attention through all layers (e.g., 48 to 64 layers) to construct the initial KV cache. For long contexts, prefill can become a primary compute bottleneck.

Kishida's proof-of-concept on `Qwen3-8B` showed that a lightweight projector can reconstruct late-layer KV states effectively for natural language prose, but noted that **coding tasks showed degradation** (such as hallucinating extra variables or syntax drift) when using a single monolithic linear projector.

### Applying the Concept to MoE Architectures
In dense models, all layers apply uniform weights across all tokens. In **Mixture-of-Experts (MoE)** models like `Qwen 3.6 35B A3B`, however, late layers contain dozens of **specialized sparse experts** (e.g., experts specialized in code syntax, mathematical reasoning, or multilingual text).

If a single monolithic projector is applied to an MoE model, it averages across these distinct expert activations. This experiment tests whether structuring the projector itself as an **MoE of low-rank micro-experts linked to the base model's router** helps preserve domain specialization during late-layer KV synthesis.

---

## 🧬 Architecture: Co-Routed Micro-Experts

Rather than using one generic linear projection matrix, this approach creates a collection of lightweight **low-rank micro-experts** ($r=32$):

```
                      [Prompt Tokens]
                             │
            [Layers 1 to 24 (Full MoE Processing)]
                             │
               ▼ Hidden State (H_24) & Router Logits (R_24)
                             │
     ┌───────────────────────┴───────────────────────┐
     │           Router Linking / Co-Routing         │
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

### Key Design Elements:
1. **Router Gating Inheritance:** The micro-experts use the router gating decisions ($R_{24}$) already produced by the base model, adding no extra routing overhead.
2. **Domain Separation:** Tokens routed to code-focused experts activate code micro-projectors, helping preserve structured syntax in the reconstructed KV states.
3. **Low Active Parameter Overhead:** Out of the 64 micro-experts, only **8 activate per token** (~17.5M active parameters), which is a negligible fraction of the base model's active compute.

---

## 📊 Experimental Results

We trained both a **Dense Linear Projector** (following Kishida's baseline) and the **Expert-Linked MoE Projector** on feature activations harvested from genuine non-linear SwiGLU forward passes across multi-domain prompts (Code, Logic/Tools, General Knowledge):

### 1. Reconstruction Cosine Alignment by Domain (Unseen Test Set)

| Domain Category | Dense Linear Projector | Expert-Linked MoE Projector (Rank 64) | MoE Reconstruction MSE |
| :--- | :---: | :---: | :---: |
| 💻 **Code (Domain Specialized)** | 91.34% | **94.76% (+3.42%)** | **0.07139** |
| 🛠️ **Logic & Tool Calling** | 82.82% | **83.03% (+0.21%)** | **0.20622** |
| 📚 **General Knowledge** | **79.15%** | 77.91% | 0.26436 |

> **Key Observation**:
> 1. **Why Linear Projectors Blur**: When evaluated against genuine SwiGLU non-linearities ($\text{SiLU}(W_{\text{gate}} x) \cdot W_{\text{up}} x$), linear models cannot model multiplicative gating, hitting a mathematical ceiling (~79% on general text and ~91% on code).
> 2. **MoE Dominance on Code Tasks**: Scaling micro-experts to Rank 64 with GELU activations enables the **Expert-Linked MoE Projector to beat the dense baseline on Code by a substantial +3.42% margin (94.76% vs 91.34%)**, reducing reconstruction MSE from >0.11 down to **0.07139** (~35% error reduction).
> 3. **Preserving Expert Independence**: Inheriting the base model's router routing allows code tokens to route into dedicated non-linear micro-expert subspaces rather than being blurred into a monolithic average projection.

### 2. Efficiency Characteristics (512-token prompt)
* **Late-Layer Prefill Bypassed:** **~48.2%** of transformer block prefill compute bypassed (Layers 25–48).
* **Active Parameters per Token:** **63.9M** (50.3M global base + 13.6M active top-8 micro-experts out of 64).
* **Projector Latency:** Sub-millisecond per token on modern AVX-512 CPU execution.

---

## 🚀 Getting Started

### 1. Installation
Clone the repository and install dependencies with `uv` or `pip`:

```bash
git clone https://github.com/ob7282/moe-kv-projector.git
cd moe-kv-projector

# Using uv
uv venv .venv
uv pip install torch numpy tqdm
```

### 2. Generate Curated Dataset
Generate the 3,000-prompt training corpus:
```bash
python dataset_builder.py
```

### 3. Harvest Feature Activations
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
├── README.md                 # Project documentation & credits
├── config.py                 # Hyperparameters (dimensions, rank, layers)
├── dataset_builder.py        # Generates balanced multi-domain dataset
├── harvest_features.py       # Extracts chunked FP16 activation tensors
├── model_projector.py        # PyTorch implementations of Dense & MoE projectors
├── train_projector.py        # Training and comparison loop (AdamW + Cosine loss)
├── benchmark_inference.py    # Multi-domain evaluation and latency benchmarks
└── data/
    └── curated_prompts.jsonl # High-density prompt corpus
```

---

## 📜 License

MIT License. Designed and developed as an open research experiment extending late-layer KV approximation to MoE architectures.
