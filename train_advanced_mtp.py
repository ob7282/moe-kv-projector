"""
Advanced Unified Training:
1. Extended Dataset (Chunks 0-5 for training, Chunk 6 for held-out unseen test)
2. Joint Router-Transition Supervision (KL-div on r_{t+1})
3. Multi-Step Unrolled Rollout Training (2-step backpropagation)
4. Soft-Label Knowledge Distillation (Target Teacher logit distribution)
5. Code-aware expert gradient boosting
6. Cosine Annealing learning rate schedule (15 epochs)
"""

import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from config import DATA_DIR, MODELS_DIR, NUM_WORKER_THREADS, LEARNING_RATE, WEIGHT_DECAY, D_MODEL
from model_mtp_drafter import HybridMoEMTPDrafter, MTP_VOCAB_SIZE
from train_mtp_distill import TargetLMTeacher

MTP_FEATURES_DIR = DATA_DIR / "mtp_features"

torch.manual_seed(42)
embed_table = nn.Embedding(MTP_VOCAB_SIZE, D_MODEL)
embed_table.eval()

def load_extended_data():
    chunks = sorted(MTP_FEATURES_DIR.glob("mtp_chunk_*.pt"))
    print(f"[DATA] Loading {len(chunks)} chunks...")
    all_e, all_h, all_r, all_y, all_cats = [], [], [], [], []
    for c in chunks:
        d = torch.load(c, map_location="cpu")
        all_e.extend([t.float() for t in d["e"]])
        all_h.extend([t.float() for t in d["h"]])
        all_r.extend([t.float() for t in d["router_weights"]])
        all_y.extend([t.long() for t in d["targets"]])
        all_cats.extend(d["categories"])
    print(f"[DATA] Loaded {len(all_h)} total sequences ({len(all_h) * all_h[0].shape[0]:,} tokens).")
    return all_e, all_h, all_r, all_y, all_cats

def eval_accuracy_and_router(model, test_e, test_h, test_r, test_y):
    model.eval()
    top1_correct = 0
    top5_correct = 0
    total_tokens = 0
    top8_overlaps = []
    
    with torch.no_grad():
        for i in range(len(test_e)):
            e = test_e[i].unsqueeze(0)
            h = test_h[i].unsqueeze(0)
            r = test_r[i].unsqueeze(0)
            y = test_y[i].unsqueeze(0)
            
            logits1, z1 = model(e, h, r)
            pred1 = logits1.argmax(dim=-1)
            top1_correct += (pred1 == y).sum().item()
            
            _, pred5 = torch.topk(logits1, 5, dim=-1)
            top5_correct += (pred5 == y.unsqueeze(-1)).any(dim=-1).sum().item()
            total_tokens += y.numel()
            
            # Router transition accuracy on next step
            pred_r_next = F.softmax(model.router_transition(model.head_norm(z1[:, :-1])), dim=-1)
            act_r_next = r[:, 1:]
            top8_pred = torch.topk(pred_r_next, 8, dim=-1).indices
            top8_act = torch.topk(act_r_next, 8, dim=-1).indices
            
            overlap = sum([len(set(top8_pred[0, t].tolist()) & set(top8_act[0, t].tolist())) for t in range(top8_pred.shape[1])]) / (top8_pred.shape[1] * 8)
            top8_overlaps.append(overlap)
            
    top1_acc = (top1_correct / total_tokens) * 100
    top5_acc = (top5_correct / total_tokens) * 100
    router_acc = (sum(top8_overlaps) / len(top8_overlaps)) * 100
    return top1_acc, top5_acc, router_acc

def train_advanced_mtp():
    torch.set_num_threads(NUM_WORKER_THREADS)
    print("=========================================================================================")
    print(" ADVANCED MTP TRAINING: UNROLLED ROLLOUT + ROUTER SUPERVISION + EXTENDED DATA")
    print("=========================================================================================\n")

    all_e, all_h, all_r, all_y, all_cats = load_extended_data()
    # Chunks 0-5 (300 seqs) for train, Chunk 6 (50 seqs) held out for testing
    train_size = 300
    train_e, test_e = all_e[:train_size], all_e[train_size:]
    train_h, test_h = all_h[:train_size], all_h[train_size:]
    train_r, test_r = all_r[:train_size], all_r[train_size:]
    train_y, test_y = all_y[:train_size], all_y[train_size:]
    test_cats = all_cats[train_size:]

    # Step 1: Pre-train target teacher LM head on extended data
    print("--- [1] PREPARING EXTENDED TEACHER LM HEAD ---")
    teacher = TargetLMTeacher()
    opt_teacher = torch.optim.AdamW(teacher.parameters(), lr=1e-3, weight_decay=1e-4)
    t0 = time.time()
    for ep in range(5):
        teacher.train()
        for i in range(train_size):
            h = train_h[i].unsqueeze(0)
            y = train_y[i].unsqueeze(0)
            opt_teacher.zero_grad()
            logits_t = teacher(h)
            loss_t = F.cross_entropy(logits_t.view(-1, MTP_VOCAB_SIZE), y.view(-1))
            loss_t.backward()
            opt_teacher.step()
    teacher.eval()
    print(f"Teacher LM Head Trained in {time.time() - t0:.1f}s.\n")

    # Step 2: Initialize Advanced Hybrid MoE Drafter
    model = HybridMoEMTPDrafter()
    # Warm-start base parameters from earlier distilled checkpoint
    base_ckpt = MODELS_DIR / "hybrid_moe_mtp_distilled.pt"
    if base_ckpt.exists():
        model.load_state_dict(torch.load(base_ckpt, map_location="cpu"), strict=False)
        print("[INIT] Warm-started trunk & experts from hybrid_moe_mtp_distilled.pt")

    # Optimizer with differential learning rates
    trunk_params = (
        list(model.eh_proj.parameters()) + 
        list(model.mlp_mid.parameters()) + 
        list(model.lm_head.parameters()) +
        list(model.enorm.parameters()) +
        list(model.hnorm.parameters()) +
        list(model.head_norm.parameters()) +
        [model.expert_gain]
    )
    expert_params = [p for exp in model.experts for p in exp.parameters()]
    router_params = list(model.router_transition.parameters())

    epochs = 12
    optimizer = torch.optim.AdamW([
        {'params': trunk_params, 'lr': 1e-4},
        {'params': expert_params, 'lr': 3e-4},
        {'params': router_params, 'lr': 8e-4}
    ], weight_decay=WEIGHT_DECAY)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    temperature = 2.0
    lambda_kd = 0.4
    lambda_router = 0.3
    lambda_step2 = 0.5  # 2-step rollout backprop weight

    print("--- [2] TRAINING ADVANCED HYBRID MoE WITH 2-STEP UNROLLING & ROUTER SUPERVISION ---")
    t_start = time.time()
    
    for ep in range(1, epochs + 1):
        model.train()
        ep_loss = 0.0
        ep_step1_loss = 0.0
        ep_step2_loss = 0.0
        ep_router_loss = 0.0

        for i in range(train_size):
            e1 = train_e[i].unsqueeze(0)
            h1 = train_h[i].unsqueeze(0)
            r1 = train_r[i].unsqueeze(0)
            y1 = train_y[i].unsqueeze(0)
            cat = all_cats[i]

            optimizer.zero_grad()

            # --- Step 1 Forward ---
            logits1, z1 = model(e1, h1, r1)
            loss_ce1 = F.cross_entropy(logits1.view(-1, MTP_VOCAB_SIZE), y1.view(-1))
            
            with torch.no_grad():
                z_teacher = teacher(h1)
                p_teacher = F.softmax(z_teacher / temperature, dim=-1)
            
            log_p_draft1 = F.log_softmax(logits1 / temperature, dim=-1)
            loss_kl1 = F.kl_div(log_p_draft1, p_teacher, reduction="batchmean") * (temperature ** 2)
            loss_step1 = (1.0 - lambda_kd) * loss_ce1 + lambda_kd * loss_kl1

            # --- Auxiliary Router Transition Loss ---
            # Predict next token's router affinities r_{t+1} from z1_t
            pred_log_r2 = F.log_softmax(model.router_transition(model.head_norm(z1[:, :-1])), dim=-1)
            target_r2 = r1[:, 1:]
            loss_router = F.kl_div(pred_log_r2, target_r2, reduction="batchmean")

            # --- Step 2 Unrolled Autoregressive Rollout ---
            # Autoregressively feed predicted token embedding e2 and predicted router r2
            pred_tokens1 = logits1[:, :-1].argmax(dim=-1)
            e2 = embed_table(pred_tokens1)
            h2 = z1[:, :-1]
            r2 = F.softmax(model.router_transition(model.head_norm(h2)), dim=-1)
            y2 = y1[:, 1:]

            logits2, _ = model(e2, h2, r2)
            loss_ce2 = F.cross_entropy(logits2.view(-1, MTP_VOCAB_SIZE), y2.view(-1))

            # Code bonus weighting: increase gradient signal for code sequences
            code_mult = 1.3 if "code" in cat.lower() else 1.0

            total_loss = (loss_step1 + lambda_step2 * loss_ce2 + lambda_router * loss_router) * code_mult
            total_loss.backward()
            optimizer.step()

            ep_loss += total_loss.item()
            ep_step1_loss += loss_step1.item()
            ep_step2_loss += loss_ce2.item()
            ep_router_loss += loss_router.item()

        scheduler.step()
        top1, top5, r_overlap = eval_accuracy_and_router(model, test_e, test_h, test_r, test_y)
        print(f"  Epoch {ep:02d}/{epochs:02d} | Total Loss: {ep_loss/train_size:.3f} | S1 Loss: {ep_step1_loss/train_size:.3f} | S2 Loss: {ep_step2_loss/train_size:.3f} | Router Loss: {ep_router_loss/train_size:.3f} | Test Top-1: {top1:.2f}% | Top-5: {top5:.2f}% | Router Overlap: {r_overlap:.1f}%")

    print(f"\n[COMPLETE] Advanced training finished in {time.time() - t_start:.1f}s.")
    saved_path = MODELS_DIR / "hybrid_moe_mtp_advanced.pt"
    torch.save(model.state_dict(), saved_path)
    print(f"[SAVED] Saved model checkpoint to {saved_path.name}\n")

if __name__ == "__main__":
    train_advanced_mtp()
