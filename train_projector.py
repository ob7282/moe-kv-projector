"""
Training Script: Expert-Linked MoE KV Projector vs Dense Linear Projector
Trains both models on the cached feature activations and compares MSE + Cosine alignment.
"""

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm

from config import (
    FEATURES_DIR, MODELS_DIR, PROJECTOR_CHECKPOINT,
    LEARNING_RATE, WEIGHT_DECAY, EPOCHS, COSINE_LOSS_WEIGHT,
    NUM_WORKER_THREADS
)
from model_projector import DenseLinearProjector, ExpertLinkedMoEKVProjector

def load_feature_chunks(features_dir=FEATURES_DIR, max_chunks=3):
    chunk_files = sorted(features_dir.glob("chunk_*.pt"))[:max_chunks]
    print(f"[DATA] Loading {len(chunk_files)} feature chunks into memory...")
    
    all_h = []
    all_r = []
    all_k = []
    all_v = []
    
    for cf in chunk_files:
        data = torch.load(cf, map_location="cpu")
        all_h.extend([t.float() for t in data["h"]])
        all_r.extend([t.float() for t in data["router_weights"]])
        all_k.extend([t.float() for t in data["k_true"]])
        all_v.extend([t.float() for t in data["v_true"]])
        
    print(f"[DATA] Loaded {len(all_h)} sequences.")
    return all_h, all_r, all_k, all_v

def evaluate_model(model, all_h, all_r, all_k, all_v, is_moe=True, num_eval=50):
    model.eval()
    total_mse = 0.0
    total_cosine = 0.0
    count = 0
    
    with torch.no_grad():
        for i in range(min(num_eval, len(all_h))):
            h = all_h[i].unsqueeze(0)  # [1, T, D]
            r = all_r[i].unsqueeze(0) if is_moe else None
            k_true = all_k[i].unsqueeze(0)
            v_true = all_v[i].unsqueeze(0)
            
            if is_moe:
                k_pred, v_pred = model(h, r)
            else:
                k_pred, v_pred = model(h)
                
            mse = (F.mse_loss(k_pred, k_true) + F.mse_loss(v_pred, v_true)).item()
            
            # Cosine similarity across feature dimensions
            cos_k = F.cosine_similarity(k_pred, k_true, dim=-1).mean().item()
            cos_v = F.cosine_similarity(v_pred, v_true, dim=-1).mean().item()
            
            total_mse += mse
            total_cosine += (cos_k + cos_v) / 2.0
            count += 1
            
    return total_mse / count, total_cosine / count

def train_and_compare(epochs=10):
    torch.set_num_threads(NUM_WORKER_THREADS)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    all_h, all_r, all_k, all_v = load_feature_chunks(max_chunks=3)
    train_size = int(len(all_h) * 0.8)
    
    train_h, test_h = all_h[:train_size], all_h[train_size:]
    train_r, test_r = all_r[:train_size], all_r[train_size:]
    train_k, test_k = all_k[:train_size], all_k[train_size:]
    train_v, test_v = all_v[:train_size], all_v[train_size:]
    
    print("\n=======================================================")
    print(" 1. TRAINING DENSE LINEAR PROJECTOR (Kishida Baseline)")
    print("=======================================================")
    dense_model = DenseLinearProjector()
    optimizer_dense = torch.optim.AdamW(dense_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    start_dense = time.time()
    for epoch in range(1, epochs + 1):
        dense_model.train()
        epoch_loss = 0.0
        for i in range(train_size):
            h = train_h[i].unsqueeze(0)
            k_true = train_k[i].unsqueeze(0)
            v_true = train_v[i].unsqueeze(0)
            
            optimizer_dense.zero_grad()
            k_pred, v_pred = dense_model(h)
            
            mse = F.mse_loss(k_pred, k_true) + F.mse_loss(v_pred, v_true)
            cos = 2.0 - (F.cosine_similarity(k_pred, k_true, dim=-1).mean() + F.cosine_similarity(v_pred, v_true, dim=-1).mean())
            loss = mse + COSINE_LOSS_WEIGHT * cos
            
            loss.backward()
            optimizer_dense.step()
            epoch_loss += loss.item()
            
        test_mse, test_cos = evaluate_model(dense_model, test_h, test_r, test_k, test_v, is_moe=False)
        print(f"  Epoch {epoch:02d}/{epochs:02d} | Train Loss: {epoch_loss/train_size:.4f} | Test MSE: {test_mse:.5f} | Cosine Sim: {test_cos:.4f}")
    time_dense = time.time() - start_dense
    torch.save(dense_model.state_dict(), MODELS_DIR / "dense_projector.pt")

    print("\n=======================================================")
    print(" 2. TRAINING EXPERT-LINKED MoE PROJECTOR (Our Architecture)")
    print("=======================================================")
    moe_model = ExpertLinkedMoEKVProjector()
    expert_params = [p for exp in moe_model.experts for p in exp.parameters()]
    base_params = list(moe_model.shared_bypass_k.parameters()) + list(moe_model.shared_bypass_v.parameters()) + list(moe_model.norm.parameters()) + [moe_model.expert_gain]
    optimizer_moe = torch.optim.AdamW([
        {'params': base_params, 'lr': LEARNING_RATE},
        {'params': expert_params, 'lr': LEARNING_RATE * 2.0}
    ], weight_decay=WEIGHT_DECAY)
    
    start_moe = time.time()
    for epoch in range(1, epochs + 1):
        moe_model.train()
        epoch_loss = 0.0
        for i in range(train_size):
            h = train_h[i].unsqueeze(0)
            r = train_r[i].unsqueeze(0)
            k_true = train_k[i].unsqueeze(0)
            v_true = train_v[i].unsqueeze(0)
            
            optimizer_moe.zero_grad()
            k_pred, v_pred = moe_model(h, r)
            
            mse = F.mse_loss(k_pred, k_true) + F.mse_loss(v_pred, v_true)
            cos = 2.0 - (F.cosine_similarity(k_pred, k_true, dim=-1).mean() + F.cosine_similarity(v_pred, v_true, dim=-1).mean())
            loss = mse + COSINE_LOSS_WEIGHT * cos
            
            loss.backward()
            optimizer_moe.step()
            epoch_loss += loss.item()
            
        test_mse, test_cos = evaluate_model(moe_model, test_h, test_r, test_k, test_v, is_moe=True)
        print(f"  Epoch {epoch:02d}/{epochs:02d} | Train Loss: {epoch_loss/train_size:.4f} | Test MSE: {test_mse:.5f} | Cosine Sim: {test_cos:.4f}")
    time_moe = time.time() - start_moe
    torch.save(moe_model.state_dict(), PROJECTOR_CHECKPOINT)

    print("\n=======================================================")
    print(" FINAL COMPARISON RESULTS")
    print("=======================================================")
    dense_mse, dense_cos = evaluate_model(dense_model, test_h, test_r, test_k, test_v, is_moe=False)
    moe_mse, moe_cos = evaluate_model(moe_model, test_h, test_r, test_k, test_v, is_moe=True)
    
    print(f"Dense Linear Projector:  MSE = {dense_mse:.5f} | Cosine Alignment = {dense_cos*100:.2f}% (Time: {time_dense:.1f}s)")
    print(f"Expert-Linked MoE Model: MSE = {moe_mse:.5f} | Cosine Alignment = {moe_cos*100:.2f}% (Time: {time_moe:.1f}s)")
    
    error_reduction = ((dense_mse - moe_mse) / max(dense_mse, 1e-9)) * 100
    print(f"\n[RESULT] Error Reduction via Expert-Linked MoE: {error_reduction:.1f}% lower error vs Dense")

if __name__ == "__main__":
    train_and_compare(epochs=5)
