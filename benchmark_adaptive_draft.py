"""
Phase 4 Benchmark: Confidence-Gated Adaptive Draft Length (N*)
Evaluates adaptive halting based on top-1 draft confidence tau in [0.35, 0.65] vs fixed horizons.
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from config import DATA_DIR, MODELS_DIR, NUM_WORKER_THREADS, D_MODEL
from model_mtp_drafter import HybridMoEMTPDrafter, MTP_VOCAB_SIZE

MTP_FEATURES_DIR = DATA_DIR / "mtp_features"

torch.manual_seed(42)
embed_table = nn.Embedding(MTP_VOCAB_SIZE, D_MODEL)
embed_table.eval()

def rollout_adaptive(model, curr_e, curr_h, curr_r, max_n, tau_threshold=0.0):
    """
    Rolls out up to max_n tokens. Halts early if top-1 probability < tau_threshold.
    """
    drafts = []
    e = curr_e
    h = curr_h
    r = curr_r
    
    for _ in range(max_n):
        logits, h = model(e, h, r)
        probs = F.softmax(logits[0, 0], dim=-1)
        conf, pred = torch.max(probs, dim=-1)
        
        # Check early exit condition
        if tau_threshold > 0.0 and conf.item() < tau_threshold and len(drafts) > 0:
            break
            
        pred_id = pred.item()
        drafts.append(pred_id)
        e = embed_table(torch.tensor([[pred_id]]))
        r = model.transition_router(h, r, beta=0.5)
        
    return drafts

def run_adaptive_draft_benchmark():
    torch.set_num_threads(NUM_WORKER_THREADS)
    print("=========================================================================================")
    print(" PHASE 4 BENCHMARK: CONFIDENCE-GATED ADAPTIVE DRAFT LENGTH (N*)")
    print("=========================================================================================\n")

    moe_ckpt = MODELS_DIR / "hybrid_moe_mtp_distilled.pt"
    if not moe_ckpt.exists():
        moe_ckpt = MODELS_DIR / "hybrid_moe_mtp.pt"
        
    moe_model = HybridMoEMTPDrafter()
    moe_model.load_state_dict(torch.load(moe_ckpt, map_location="cpu"), strict=False)
    moe_model.eval()

    d = torch.load(MTP_FEATURES_DIR / "mtp_chunk_002.pt", map_location="cpu")
    test_e = [t.float() for t in d["e"]]
    test_h = [t.float() for t in d["h"]]
    test_r = [t.float() for t in d["router_weights"]]
    test_y = [t.long() for t in d["targets"]]
    test_cats = d["categories"]

    configs = [
        {"name": "Fixed N = 1", "max_n": 1, "tau": 0.0},
        {"name": "Fixed N = 2", "max_n": 2, "tau": 0.0},
        {"name": "Fixed N = 3", "max_n": 3, "tau": 0.0},
        {"name": "Fixed N = 5", "max_n": 5, "tau": 0.0},
        {"name": "Fixed N = 8", "max_n": 8, "tau": 0.0},
        {"name": "Adaptive N* <= 5 (tau = 0.35)", "max_n": 5, "tau": 0.35},
        {"name": "Adaptive N* <= 5 (tau = 0.45)", "max_n": 5, "tau": 0.45},
        {"name": "Adaptive N* <= 5 (tau = 0.55)", "max_n": 5, "tau": 0.55},
        {"name": "Adaptive N* <= 5 (tau = 0.65)", "max_n": 5, "tau": 0.65},
        {"name": "Adaptive N* <= 8 (tau = 0.45)", "max_n": 8, "tau": 0.45},
    ]

    domains = ["Code (AST & Syntax)", "Logic & Tool Calling", "General Knowledge"]

    stats = {
        cfg["name"]: {
            "drafted_count": 0,
            "accepted_count": 0,
            "yield_sum": 0.0,
            "evals": 0,
            "by_dom_yield": {dom: 0.0 for dom in domains},
            "by_dom_evals": {dom: 0 for dom in domains},
        } for cfg in configs
    }

    t0 = time.time()
    with torch.no_grad():
        for seq_idx in range(len(test_e)):
            e_seq = test_e[seq_idx]
            h_seq = test_h[seq_idx]
            r_seq = test_r[seq_idx]
            y_seq = test_y[seq_idx]
            cat_raw = test_cats[seq_idx]

            if "code" in cat_raw.lower(): dom = "Code (AST & Syntax)"
            elif "logic" in cat_raw.lower() or "tool" in cat_raw.lower(): dom = "Logic & Tool Calling"
            else: dom = "General Knowledge"

            seq_len = y_seq.shape[0]

            for t in range(0, seq_len - 8, 4):
                gt_8 = y_seq[t: t + 8].tolist()
                curr_e = e_seq[t].unsqueeze(0).unsqueeze(0)
                curr_h = h_seq[t].unsqueeze(0).unsqueeze(0)
                curr_r = r_seq[t].unsqueeze(0).unsqueeze(0)

                for cfg in configs:
                    name = cfg["name"]
                    drafts = rollout_adaptive(moe_model, curr_e, curr_h, curr_r, cfg["max_n"], cfg["tau"])
                    
                    # Verification against ground truth
                    accepted = 0
                    for k in range(len(drafts)):
                        if drafts[k] == gt_8[k]:
                            accepted += 1
                        else:
                            break
                            
                    stats[name]["drafted_count"] += len(drafts)
                    stats[name]["accepted_count"] += accepted
                    stats[name]["yield_sum"] += (1 + accepted)
                    stats[name]["evals"] += 1
                    stats[name]["by_dom_yield"][dom] += (1 + accepted)
                    stats[name]["by_dom_evals"][dom] += 1

    print(f"[EVAL] Completed adaptive draft benchmark in {time.time() - t0:.1f}s.\n")

    print("--- [1] SPECULATIVE EFFICIENCY & DRAFT PRECISION TABLE ---")
    print(f"{'Strategy':<30} | {'Avg Draft (N*)':<16} | {'Tokens/Pass':<14} | {'Draft Precision':<18} | {'Drafting ROI':<14}")
    print("-" * 102)
    for cfg in configs:
        name = cfg["name"]
        ev = stats[name]["evals"]
        avg_drafted = stats[name]["drafted_count"] / ev
        avg_yield = stats[name]["yield_sum"] / ev
        precision = (stats[name]["accepted_count"] / max(stats[name]["drafted_count"], 1)) * 100
        # ROI: Accepted tokens generated per draft token spent
        roi = stats[name]["accepted_count"] / max(stats[name]["drafted_count"], 1)
        print(f"{name:<30} | {avg_drafted:>14.2f}  | {avg_yield:>12.2f}x | {precision:>16.1f}% | {roi:>12.2f}")

    print("\n--- [2] DOMAIN BREAKDOWN (YIELD PER PASS) ---")
    print(f"{'Strategy':<30} | {'Code (AST)':<14} | {'Logic & Tools':<15} | {'General Knowledge':<18}")
    print("-" * 84)
    for cfg in configs:
        name = cfg["name"]
        c_ev = stats[name]["by_dom_evals"]["Code (AST & Syntax)"]
        l_ev = stats[name]["by_dom_evals"]["Logic & Tool Calling"]
        k_ev = stats[name]["by_dom_evals"]["General Knowledge"]
        c_y = stats[name]["by_dom_yield"]["Code (AST & Syntax)"] / c_ev
        l_y = stats[name]["by_dom_yield"]["Logic & Tool Calling"] / l_ev
        k_y = stats[name]["by_dom_yield"]["General Knowledge"] / k_ev
        print(f"{name:<30} | {c_y:>12.2f}x | {l_y:>13.2f}x | {k_y:>16.2f}x")

if __name__ == "__main__":
    run_adaptive_draft_benchmark()
