"""
Training Script: Hybrid Expert-Linked MoE MTP Drafter vs Dense Baseline
Trains both models using Cross-Entropy on next-token prediction targets.
Evaluates Top-1 / Top-5 Draft Accuracy across Code, Logic, and General Knowledge.
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from config import DATA_DIR, MODELS_DIR, NUM_WORKER_THREADS, LEARNING_RATE, WEIGHT_DECAY
from model_mtp_drafter import DenseMTPDrafter, HybridMoEMTPDrafter, MTP_VOCAB_SIZE

MTP_FEATURES_DIR = DATA_DIR / "mtp_features"

def load_mtp_data():
    chunks = sorted(MTP_FEATURES_DIR.glob("mtp_chunk_*.pt"))
    print(f"[DATA] Loading {len(chunks)} MTP feature chunks...")
    
    all_e, all_h, all_r, all_y, all_cats = [], [], [], [], []
    for c in chunks:
        d = torch.load(c, map_location="cpu")
        all_e.extend([t.float() for t in d["e"]])
        all_h.extend([t.float() for t in d["h"]])
        all_r.extend([t.float() for t in d["router_weights"]])
        all_y.extend([t.long() for t in d["targets"]])
        all_cats.extend(d["categories"])
        
    print(f"[DATA] Loaded {len(all_h)} total sequences.")
    return all_e, all_h, all_r, all_y, all_cats

def eval_mtp(model, test_e, test_h, test_r, test_y, is_moe=False):
    model.eval()
    total_loss = 0.0
    top1_correct = 0
    top5_correct = 0
    total_tokens = 0
    
    with torch.no_grad():
        for i in range(len(test_e)):
            e = test_e[i].unsqueeze(0)
            h = test_h[i].unsqueeze(0)
            r = test_r[i].unsqueeze(0) if is_moe else None
            y = test_y[i].unsqueeze(0)
            
            if is_moe:
                logits, _ = model(e, h, r)
            else:
                logits, _ = model(e, h)
                
            loss = F.cross_entropy(logits.view(-1, MTP_VOCAB_SIZE), y.view(-1))
            total_loss += loss.item()
            
            # Top-1 accuracy
            pred1 = logits.argmax(dim=-1)
            top1_correct += (pred1 == y).sum().item()
            
            # Top-5 accuracy
            _, pred5 = torch.topk(logits, 5, dim=-1)
            top5_correct += (pred5 == y.unsqueeze(-1)).any(dim=-1).sum().item()
            
            total_tokens += y.numel()
            
    avg_loss = total_loss / len(test_e)
    top1_acc = top1_correct / total_tokens
    top5_acc = top5_correct / total_tokens
    return avg_loss, top1_acc, top5_acc

def train_and_compare_mtp(epochs=5):
    torch.set_num_threads(NUM_WORKER_THREADS)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    all_e, all_h, all_r, all_y, all_cats = load_mtp_data()
    
    # Train on first 100 sequences (chunks 0 and 1), test on last 50 (chunk 2)
    train_size = 100
    train_e, test_e = all_e[:train_size], all_e[train_size:]
    train_h, test_h = all_h[:train_size], all_h[train_size:]
    train_r, test_r = all_r[:train_size], all_r[train_size:]
    train_y, test_y = all_y[:train_size], all_y[train_size:]
    
    print("\n=======================================================")
    print(" 1. TRAINING DENSE MTP DRAFTER (Standard Baseline)")
    print("=======================================================")
    dense_drafter = DenseMTPDrafter()
    opt_dense = torch.optim.AdamW(dense_drafter.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    t0 = time.time()
    for ep in range(1, epochs + 1):
        dense_drafter.train()
        ep_loss = 0.0
        for i in range(train_size):
            e = train_e[i].unsqueeze(0)
            h = train_h[i].unsqueeze(0)
            y = train_y[i].unsqueeze(0)
            
            opt_dense.zero_grad()
            logits, _ = dense_drafter(e, h)
            loss = F.cross_entropy(logits.view(-1, MTP_VOCAB_SIZE), y.view(-1))
            loss.backward()
            opt_dense.step()
            ep_loss += loss.item()
            
        test_loss, top1, top5 = eval_mtp(dense_drafter, test_e, test_h, test_r, test_y, is_moe=False)
        print(f"  Epoch {ep:02d}/{epochs:02d} | Train Loss: {ep_loss/train_size:.4f} | Test Loss: {test_loss:.4f} | Top-1 Acc: {top1*100:.2f}% | Top-5 Acc: {top5*100:.2f}%")
        
    t_dense = time.time() - t0
    torch.save(dense_drafter.state_dict(), MODELS_DIR / "dense_mtp.pt")

    print("\n=======================================================")
    print(" 2. TRAINING HYBRID EXPERT-LINKED MoE MTP DRAFTER")
    print("=======================================================")
    hybrid_drafter = HybridMoEMTPDrafter()
    
    # Warm-start shared dense foundation from trained baseline
    hybrid_drafter.eh_proj.weight.data.copy_(dense_drafter.eh_proj.weight.data)
    hybrid_drafter.mlp_mid.weight.data.copy_(dense_drafter.mlp_mid.weight.data)
    hybrid_drafter.lm_head.weight.data.copy_(dense_drafter.lm_head.weight.data)
    hybrid_drafter.enorm.load_state_dict(dense_drafter.enorm.state_dict())
    hybrid_drafter.hnorm.load_state_dict(dense_drafter.hnorm.state_dict())
    hybrid_drafter.head_norm.load_state_dict(dense_drafter.head_norm.state_dict())
    print("[INIT] Warm-started shared trunk from trained Dense MTP baseline.")
    
    base_params = (
        list(hybrid_drafter.eh_proj.parameters()) + 
        list(hybrid_drafter.mlp_mid.parameters()) + 
        list(hybrid_drafter.lm_head.parameters()) +
        list(hybrid_drafter.enorm.parameters()) +
        list(hybrid_drafter.hnorm.parameters()) +
        list(hybrid_drafter.head_norm.parameters()) +
        [hybrid_drafter.expert_gain]
    )
    expert_params = [p for exp in hybrid_drafter.experts for p in exp.parameters()]
    
    opt_hybrid = torch.optim.AdamW([
        {'params': base_params, 'lr': 1e-5},
        {'params': expert_params, 'lr': 4e-4}
    ], weight_decay=WEIGHT_DECAY)
    
    t0 = time.time()
    for ep in range(1, epochs + 1):
        hybrid_drafter.train()
        ep_loss = 0.0
        for i in range(train_size):
            e = train_e[i].unsqueeze(0)
            h = train_h[i].unsqueeze(0)
            r = train_r[i].unsqueeze(0)
            y = train_y[i].unsqueeze(0)
            
            opt_hybrid.zero_grad()
            logits, _ = hybrid_drafter(e, h, r)
            loss = F.cross_entropy(logits.view(-1, MTP_VOCAB_SIZE), y.view(-1))
            loss.backward()
            opt_hybrid.step()
            ep_loss += loss.item()
            
        test_loss, top1, top5 = eval_mtp(hybrid_drafter, test_e, test_h, test_r, test_y, is_moe=True)
        print(f"  Epoch {ep:02d}/{epochs:02d} | Train Loss: {ep_loss/train_size:.4f} | Test Loss: {test_loss:.4f} | Top-1 Acc: {top1*100:.2f}% | Top-5 Acc: {top5*100:.2f}%")
        
    t_moe = time.time() - t0
    torch.save(hybrid_drafter.state_dict(), MODELS_DIR / "hybrid_moe_mtp.pt")

    print("\n=======================================================")
    print(" FINAL MTP DRAFTER COMPARISON")
    print("=======================================================")
    _, d_top1, d_top5 = eval_mtp(dense_drafter, test_e, test_h, test_r, test_y, is_moe=False)
    _, m_top1, m_top5 = eval_mtp(hybrid_drafter, test_e, test_h, test_r, test_y, is_moe=True)
    
    print(f"Dense MTP Drafter:      Top-1 Acc = {d_top1*100:.2f}% | Top-5 Acc = {d_top5*100:.2f}% (Time: {t_dense:.1f}s)")
    print(f"Hybrid MoE MTP Drafter: Top-1 Acc = {m_top1*100:.2f}% | Top-5 Acc = {m_top5*100:.2f}% (Time: {t_moe:.1f}s)")
    print(f"Top-1 Draft Accuracy Gain: {(m_top1 - d_top1)*100:+.2f}%")

if __name__ == "__main__":
    train_and_compare_mtp(epochs=5)
