"""
Configuration and Hyperparameters for Expert-Linked MoE KV Projector
Target Model: Qwen 3.6 35B A3B
"""

import os
from pathlib import Path

# Base Paths (dynamically resolved to repository root)
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
FEATURES_DIR = DATA_DIR / "features"
MODELS_DIR = BASE_DIR / "models"
PROMPTS_FILE = DATA_DIR / "curated_prompts.jsonl"
PROJECTOR_CHECKPOINT = MODELS_DIR / "moe_kv_projector.pt"

# Local GGUF Model Path
GGUF_MODEL_PATH = Path(r"C:\LocalAI\models\Qwen3.6-35B-A3B-MTP-UD-Q4_K_M.gguf")

# Architecture Constants (Qwen 3.6 35B A3B)
D_MODEL = 2048              # Base token representation dimension
NUM_TOTAL_LAYERS = 48       # Total transformer depth
SPLIT_LAYER = 24            # Prefill cutoff layer (early layers compute full MoE)
TARGET_LAYERS = 24          # Number of late layers to approximate (Layers 25-48)

# Attention & KV Dimensions
NUM_KV_HEADS = 4            # Grouped Query Attention (GQA) KV heads
HEAD_DIM = 128              # Head dimension
D_KV_PER_LAYER = NUM_KV_HEADS * HEAD_DIM  # 512 per layer
TOTAL_TARGET_KV_DIM = TARGET_LAYERS * D_KV_PER_LAYER  # 12,288 total late KV features

# MoE Router & Micro-Expert Hyperparameters
NUM_EXPERTS = 64            # Number of routed experts
TOP_K_EXPERTS = 8           # Top-K active experts per token
PROJECTOR_RANK = 32         # Low-rank bottleneck dimension for each micro-expert

# Training Hyperparameters
BATCH_SIZE = 16
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01
EPOCHS = 15
COSINE_LOSS_WEIGHT = 0.1
MAX_SEQ_LEN = 512
DEVICE = "cpu"              # Optimized for Zen 4 AVX-512 VNNI (8 physical cores)
NUM_WORKER_THREADS = 8
