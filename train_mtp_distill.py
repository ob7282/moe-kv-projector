"""
Phase 3: Soft-Label Knowledge Distillation for Hybrid Micro-MoE MTP Drafter
Trains the Hybrid MoE drafter using a combined loss:
L = (1 - lambda) * L_CE + lambda * (tau^2) * KL(P_target || P_draft)
Measures accuracy gains and speculative acceptance on the unseen test set.
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from config import DATA_DIR, MODELS_DIR, NUM_WORKER_THREADS, LEARNING_RATE, WEIGHT_DECAY, D_MODEL
from model_mtp_drafter import DenseMTPDrafter, HybridMoEMTPDrafter, MTP_VOCAB_SIZE

MTP_FEATURES_DIR = DATA_DIR / "mtp_features"

class TargetLMTeacher(nn.Module):
    """Teacher LM Head mapping target hidden states to target token distributions."""
    def __init__(self, d_model=D_MODEL, vocab_size=MTP_VOCAB_SIZE):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, vocab_size, bias=False)
        )
    def forward(self, h):
        return self.net(h)

def load_mtp_data():
    chunks = sorted(MTP_FEATURES_DIR.glob("mtp_chunk_*.pt"))
    all_e, all_h, all_r, all_y, all_cats = [], [], [], [], []
    for c in chunks:
        d = torch.load(c, map_location="cpu")
        all_e.extend([t.float() for t in d["e"]])
        all_h.extend([t.float() for t in d["h"]])
        all_r.extend([t.float() for t in d["router_weights"]])
        all_y.extend([t.long() for t in d["targets"]])
        all_cats.extend(d["categories"])
    return all_e, all_h, all_r, all_y, all_cats

def eval_accuracy(model, test_e, test_h, test_r, test_y, is_moe=True):
    model.eval()
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
                
            pred1 = logits.argmax(dim=-1)
            top1_correct += (pred1 == y).sum().item()
            
            _, pred5 = torch.topk(logits, 5, dim=-1)
            top5_correct += (pred5 == y.unsqueeze(-1)).any(dim=-1).sum().item()
            total_tokens += y.numel()
            
    return (top1_correct / total_tokens) * 100, (top5_correct / total_tokens) * 100

def eval_speculative_acceptance(drafter, test_e, test_h, test_r, test_y, test_cats):
    drafter.eval()
    domains = ["Code (AST & Syntax)", "Logic & Tool Calling", "General Knowledge"]
    stats = {
        dom: {"correct": 0, "total": 0} for dom in domains
    }
    stats["total"] = {"correct": 0, "total": 0}
    
    with torch.no_grad():
        for i in range(len(test_e)):
            e = test_e[i].unsqueeze(0)
            h = test_h[i].unsqueeze(0)
            r = test_r[i].unsqueeze(0)
            y = test_y[i].unsqueeze(0)
            cat_raw = test_cats[i]
            
            if "code" in cat_raw.lower(): dom = "Code (AST & Syntax)"
            elif "logic" in cat_raw.lower() or "tool" in cat_raw.lower(): dom = "Logic & Tool Calling"
            else: dom = "General Knowledge"
            
            logits, _ = drafter(e, h, r)
            pred = logits.argmax(dim=-1)
            
            corr = (pred == y).sum().item()
            tot = y.numel()
            
            stats[dom]["correct"] += corr
            stats[dom]["total"] += tot
            stats["total"]["correct"] += corr
            stats["total"]["total"] += tot
            
    res = {}
    for k in domains + ["total"]:
        res[k] = (stats[k]["correct"] / max(stats[k]["total"], 1)) * 100
    return res

def run_distillation_experiment():
    torch.set_num_threads(NUM_WORKER_THREADS)
    print("=========================================================================================")
    print(" PHASE 3: SOFT-LABEL KNOWLEDGE DISTILLATION (KL-DIV) FOR HYBRID MoE DRAFTER")
    print("=========================================================================================\n")

    all_e, all_h, all_r, all_y, all_cats = load_mtp_data()
    train_size = 100
    train_e, test_e = all_e[:train_size], all_e[train_size:]
    train_h, test_h = all_h[:train_size], all_h[train_size:]
    train_r, test_r = all_r[:train_size], all_r[train_size:]
    train_y, test_y = all_y[:train_size], all_y[train_size:]
    test_cats = all_cats[train_size:]

    # Step 1: Train/Prepare Target Teacher Head
    print("--- [1] PREPARING TARGET MODEL TEACHER DISTRIBUTION ---")
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
    
    # Evaluate teacher accuracy
    t_corr = 0
    t_tot = 0
    with torch.no_grad():
        for i in range(len(test_e)):
            h = test_h[i].unsqueeze(0)
            y = test_y[i].unsqueeze(0)
            pred = teacher(h).argmax(dim=-1)
            t_corr += (pred == y).sum().item()
            t_tot += y.numel()
    print(f"Teacher LM Head Trained in {time.time() - t0:.1f}s. Test Accuracy: {(t_corr/t_tot)*100:.2f}%\n")

    # Step 2: Load trained baseline Hybrid MoE to measure baseline performance
    baseline_moe = HybridMoEMTPDrafter()
    baseline_moe.load_state_dict(torch.load(MODELS_DIR / "hybrid_moe_mtp.pt", map_location="cpu"), strict=False)
    baseline_top1, baseline_top5 = eval_accuracy(baseline_moe, test_e, test_h, test_r, test_y, is_moe=True)
    baseline_rates = eval_speculative_acceptance(baseline_moe, test_e, test_h, test_r, test_y, test_cats)

    # Step 3: Train Hybrid MoE with Soft-Label Knowledge Distillation
    print("--- [2] TRAINING HYBRID MoE DRAFTER WITH DISTILLATION LOSS ---")
    distill_moe = HybridMoEMTPDrafter()
    # Initialize from the baseline checkpoint to fine-tune with soft supervision
    distill_moe.load_state_dict(torch.load(MODELS_DIR / "hybrid_moe_mtp.pt", map_location="cpu"), strict=False)

    opt_distill = torch.optim.AdamW(distill_moe.parameters(), lr=1e-4, weight_decay=WEIGHT_DECAY)
    
    temperature = 2.0
    lambda_kd = 0.5  # 50% hard ground truth, 50% soft target distribution
    epochs = 5

    t0 = time.time()
    for ep in range(1, epochs + 1):
        distill_moe.train()
        ep_ce = 0.0
        ep_kl = 0.0
        for i in range(train_size):
            e = train_e[i].unsqueeze(0)
            h = train_h[i].unsqueeze(0)
            r = train_r[i].unsqueeze(0)
            y = train_y[i].unsqueeze(0)

            with torch.no_grad():
                z_teacher = teacher(h)
                p_teacher = F.softmax(z_teacher / temperature, dim=-1)

            opt_distill.zero_grad()
            logits_draft, _ = distill_moe(e, h, r)
            
            # 1. Hard Cross-Entropy Loss
            loss_ce = F.cross_entropy(logits_draft.view(-1, MTP_VOCAB_SIZE), y.view(-1))
            
            # 2. Soft KL-Divergence Loss
            log_p_draft = F.log_softmax(logits_draft / temperature, dim=-1)
            loss_kl = F.kl_div(log_p_draft, p_teacher, reduction="batchmean") * (temperature ** 2)
            
            total_loss = (1 - lambda_kd) * loss_ce + lambda_kd * loss_kl
            total_loss.backward()
            opt_distill.step()

            ep_ce += loss_ce.item()
            ep_kl += loss_kl.item()

        top1, top5 = eval_accuracy(distill_moe, test_e, test_h, test_r, test_y, is_moe=True)
        print(f"  Epoch {ep:02d}/{epochs:02d} | CE Loss: {ep_ce/train_size:.4f} | KL Loss: {ep_kl/train_size:.4f} | Test Top-1: {top1:.2f}% | Top-5: {top5:.2f}%")

    t_distill = time.time() - t0
    torch.save(distill_moe.state_dict(), MODELS_DIR / "hybrid_moe_mtp_distilled.pt")
    print(f"\n[SAVED] Distilled model saved to models/hybrid_moe_mtp_distilled.pt\n")

    distill_top1, distill_top5 = eval_accuracy(distill_moe, test_e, test_h, test_r, test_y, is_moe=True)
    distill_rates = eval_speculative_acceptance(distill_moe, test_e, test_h, test_r, test_y, test_cats)

    # Step 4: Comparative Reporting
    print("--- [3] EVALUATION SUMMARY: HARD CE vs SOFT DISTILLATION ---")
    print(f"{'Metric':<30} | {'Hard CE MoE (Baseline)':<24} | {'Soft Distilled MoE':<22} | {'Net Improvement':<18}")
    print("-" * 100)
    print(f"{'Top-1 Test Accuracy':<30} | {baseline_top1:>22.2f}% | {distill_top1:>20.2f}% | {distill_top1 - baseline_top1:>+16.2f}%")
    print(f"{'Top-5 Test Accuracy':<30} | {baseline_top5:>22.2f}% | {distill_top5:>20.2f}% | {distill_top5 - baseline_top5:>+16.2f}%")
    print("-" * 100)
    print(f"{'Overall Acceptance (alpha)':<30} | {baseline_rates['total']:>22.2f}% | {distill_rates['total']:>20.2f}% | {distill_rates['total'] - baseline_rates['total']:>+16.2f}%")
    print(f"{'Code (AST & Syntax)':<30} | {baseline_rates['Code (AST & Syntax)']:>22.2f}% | {distill_rates['Code (AST & Syntax)']:>20.2f}% | {distill_rates['Code (AST & Syntax)'] - baseline_rates['Code (AST & Syntax)']:>+16.2f}%")
    print(f"{'Logic & Tool Calling':<30} | {baseline_rates['Logic & Tool Calling']:>22.2f}% | {distill_rates['Logic & Tool Calling']:>20.2f}% | {distill_rates['Logic & Tool Calling'] - baseline_rates['Logic & Tool Calling']:>+16.2f}%")
    print(f"{'General Knowledge':<30} | {baseline_rates['General Knowledge']:>22.2f}% | {distill_rates['General Knowledge']:>20.2f}% | {distill_rates['General Knowledge'] - baseline_rates['General Knowledge']:>+16.2f}%")

if __name__ == "__main__":
    run_distillation_experiment()
