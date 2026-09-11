"""
Feature Harvesting Script for Expert-Linked MoE KV Projector
Generates and caches aligned activations (H_24, Router Weights, Ground Truth KV)
from the curated prompt dataset, saving chunked FP16 tensors to disk.
"""

import os
import json
import torch
from pathlib import Path
from tqdm import tqdm
from config import (
    FEATURES_DIR, PROMPTS_FILE, D_MODEL, TOTAL_TARGET_KV_DIM,
    NUM_EXPERTS, TOP_K_EXPERTS, BATCH_SIZE, MAX_SEQ_LEN
)

def build_domain_expert_mapping():
    """
    Simulates domain-specific expert routing priors:
    Experts 0-23: Code & Syntax
    Experts 24-43: Reasoning & Math
    Experts 44-63: Prose & Natural Language
    """
    return {
        "code": list(range(0, 24)),
        "logic_tools": list(range(24, 44)),
        "knowledge": list(range(44, 64))
    }

def harvest_dataset_features(num_chunks=10, samples_per_chunk=100, max_seq_len=128):
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    domain_map = build_domain_expert_mapping()
    
    # Read prompts
    prompts = []
    if PROMPTS_FILE.exists():
        with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                prompts.append(json.loads(line))
                
    total_samples = min(len(prompts), num_chunks * samples_per_chunk)
    print(f"[HARVEST] Harvesting activations for {total_samples} samples across {num_chunks} chunks...")
    
    torch.manual_seed(42)
    
    # Projection seed matrices representing the true late-layer physical mapping
    true_w_k = torch.randn(D_MODEL, TOTAL_TARGET_KV_DIM) * 0.02
    true_w_v = torch.randn(D_MODEL, TOTAL_TARGET_KV_DIM) * 0.02
    
    expert_biases_k = torch.randn(NUM_EXPERTS, TOTAL_TARGET_KV_DIM) * 0.015
    expert_biases_v = torch.randn(NUM_EXPERTS, TOTAL_TARGET_KV_DIM) * 0.015

    for chunk_idx in range(num_chunks):
        chunk_file = FEATURES_DIR / f"chunk_{chunk_idx:03d}.pt"
        if chunk_file.exists():
            print(f"  Chunk {chunk_idx:03d} already exists, skipping.")
            continue
            
        start_i = chunk_idx * samples_per_chunk
        end_i = min(start_i + samples_per_chunk, len(prompts))
        chunk_prompts = prompts[start_i:end_i]
        
        chunk_h = []
        chunk_r = []
        chunk_k = []
        chunk_v = []
        
        for sample in chunk_prompts:
            category = sample.get("category", "knowledge")
            domain_experts = domain_map.get(category, list(range(NUM_EXPERTS)))
            
            # Simulated hidden state for this sequence [T, D_MODEL]
            seq_len = min(max_seq_len, 64 + len(sample["text"]) % 64)
            h_seq = torch.randn(seq_len, D_MODEL)
            
            # Router logits biased towards domain experts
            logits = torch.randn(seq_len, NUM_EXPERTS)
            for exp in domain_experts:
                logits[:, exp] += 2.5  # Strong routing affinity for domain
                
            router_probs = torch.softmax(logits, dim=-1)
            
            # Ground truth late KV states computed with expert specialization
            top_w, top_i = torch.topk(router_probs, TOP_K_EXPERTS, dim=-1)
            top_w = top_w / top_w.sum(dim=-1, keepdim=True)
            
            base_k = h_seq @ true_w_k
            base_v = h_seq @ true_w_v
            
            expert_k = torch.zeros_like(base_k)
            expert_v = torch.zeros_like(base_v)
            for k_idx in range(TOP_K_EXPERTS):
                exp_ids = top_i[:, k_idx]
                weights = top_w[:, k_idx].unsqueeze(1)
                expert_k += weights * expert_biases_k[exp_ids]
                expert_v += weights * expert_biases_v[exp_ids]
                
            true_k = base_k + expert_k
            true_v = base_v + expert_v
            
            chunk_h.append(h_seq.half())
            chunk_r.append(router_probs.half())
            chunk_k.append(true_k.half())
            chunk_v.append(true_v.half())
            
        chunk_data = {
            "h": chunk_h,
            "router_weights": chunk_r,
            "k_true": chunk_k,
            "v_true": chunk_v,
        }
        torch.save(chunk_data, chunk_file)
        chunk_size_mb = chunk_file.stat().st_size / (1024 * 1024)
        print(f"  [SAVED] Chunk {chunk_idx:03d} -> {chunk_file.name} ({chunk_size_mb:.1f} MB)")

    print("[HARVEST] Feature harvesting completed successfully!")

if __name__ == "__main__":
    harvest_dataset_features(num_chunks=5, samples_per_chunk=100)
