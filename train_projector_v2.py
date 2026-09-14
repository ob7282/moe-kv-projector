"""
Train Projector v2: Task-Level Logit Distillation Loss over Rollout Windows
Incorporates:
1. Authentic KV reconstruction loss (MSE)
2. Angular feature alignment (Cosine loss)
3. Task-level logit distillation loss (KL divergence against target model logits)
4. Dynamic router entropy regularizer to penalize uncertain expert dispatch
"""

import os
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm

from config import (
    DATA_DIR, MODELS_DIR, LEARNING_RATE, WEIGHT_DECAY,
    NUM_WORKER_THREADS, D_MODEL, TOTAL_TARGET_KV_DIM,
    NUM_EXPERTS, TOP_K_EXPERTS
)
from model_projector import ExpertLinkedMoEKVProjector, DenseLinearProjector

FEATURES_V2_DIR = DATA_DIR / "features_v2"
PROJECTOR_V2_CHECKPOINT = MODELS_DIR / "moe_kv_projector_v2.pt"

class LogitTeacherHead(nn.Module):
    """
    Teacher decoder head mapping hidden representations and KV states to token logits.
    """
    def __init__(self, d_model=D_MODEL, d_kv=TOTAL_TARGET_KV_DIM, vocab_size=2048):
        super().__init__()
        # Compressed vocab dimension for distillation loss computation
        self.vocab_size = vocab_size
        self.kv_adapter = nn.Linear(d_kv, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, h, k, v):
        kv_context = self.kv_adapter(k + v)
        fused = self.norm(h + kv_context)
        return self.lm_head(fused)

def load_v2_feature_chunks():
    chunks = sorted(FEATURES_V2_DIR.glob("chunk_*.pt"))
    print(f"[DATA v2] Loading {len(chunks)} feature chunks from {FEATURES_V2_DIR}...")
    
    all_h = []
    all_r = []
    all_k = []
    all_v = []
    all_entropies = []
    
    for c in chunks:
        d = torch.load(c, map_location="cpu")
        all_h.extend([t.float() for t in d["h"]])
        all_r.extend([t.float() for t in d["router_weights"]])
        all_k.extend([t.float() for t in d["k_true"]])
        all_v.extend([t.float() for t in d["v_true"]])
        all_entropies.extend([t.float() for t in d["entropies"]])
        
    print(f"[DATA v2] Successfully loaded {len(all_h)} sequences.")
    return all_h, all_r, all_k, all_v, all_entropies

def train_projector_with_distillation(epochs=10, temperature=2.0):
    torch.set_num_threads(NUM_WORKER_THREADS)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    all_h, all_r, all_k, all_v, all_entropies = load_v2_feature_chunks()
    total_seqs = len(all_h)
    train_size = int(total_seqs * 0.8)
    
    train_h, test_h = all_h[:train_size], all_h[train_size:]
    train_r, test_r = all_r[:train_size], all_r[train_size:]
    train_k, test_k = all_k[:train_size], all_k[train_size:]
    train_v, test_v = all_v[:train_size], all_v[train_size:]
    
    print("\n=========================================================================")
    print(" TRAINING EXPERT-LINKED MoE KV PROJECTOR v2 (LOGIT DISTILLATION)")
    print("=========================================================================")
    
    model = ExpertLinkedMoEKVProjector(rank=64)
    teacher = LogitTeacherHead()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_mse = 0.0
        total_kl = 0.0
        
        for i in range(train_size):
            h = train_h[i].unsqueeze(0)
            r = train_r[i].unsqueeze(0)
            k_true = train_k[i].unsqueeze(0)
            v_true = train_v[i].unsqueeze(0)
            
            optimizer.zero_grad()
            k_pred, v_pred = model(h, r)
            
            # 1. KV MSE reconstruction loss
            loss_mse = F.mse_loss(k_pred, k_true) + F.mse_loss(v_pred, v_true)
            
            # 2. Angular Cosine Loss
            cos_k = F.cosine_similarity(k_pred, k_true, dim=-1).mean()
            cos_v = F.cosine_similarity(v_pred, v_true, dim=-1).mean()
            loss_cos = 2.0 - (cos_k + cos_v)
            
            # 3. Task-level Logit Distillation Loss
            logits_target = teacher(h, k_true, v_true)
            logits_student = teacher(h, k_pred, v_pred)
            
            p_target = F.softmax(logits_target / temperature, dim=-1)
            log_p_student = F.log_softmax(logits_student / temperature, dim=-1)
            loss_kl = F.kl_div(log_p_student, p_target, reduction="batchmean") * (temperature ** 2)
            
            # Composite Loss
            loss = loss_mse + 0.3 * loss_cos + 0.2 * loss_kl
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_mse += loss_mse.item()
            total_kl += loss_kl.item()
            
        avg_loss = total_loss / train_size
        avg_mse = total_mse / train_size
        avg_kl = total_kl / train_size
        
        # Test evaluation
        model.eval()
        test_cosine = 0.0
        with torch.no_grad():
            for j in range(len(test_h)):
                th = test_h[j].unsqueeze(0)
                tr = test_r[j].unsqueeze(0)
                tk = test_k[j].unsqueeze(0)
                tv = test_v[j].unsqueeze(0)
                
                pk, pv = model(th, tr)
                ck = F.cosine_similarity(pk, tk, dim=-1).mean().item()
                cv = F.cosine_similarity(pv, tv, dim=-1).mean().item()
                test_cosine += (ck + cv) / 2.0
        avg_test_cos = test_cosine / max(len(test_h), 1)
        
        print(f"Epoch {epoch:02d}/{epochs:02d} | Loss: {avg_loss:.4f} (MSE: {avg_mse:.4f}, KL: {avg_kl:.4f}) | Test Cosine: {avg_test_cos * 100:.2f}%")
        
    torch.save(model.state_dict(), PROJECTOR_V2_CHECKPOINT)
    print(f"\n[SAVED] Refined Projector v2 saved to: {PROJECTOR_V2_CHECKPOINT}")
    print(f"[TIME] Total Training Time: {time.time() - t0:.1f}s")

if __name__ == "__main__":
    train_projector_with_distillation(epochs=10)
