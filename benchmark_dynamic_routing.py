"""
Phase 1 Benchmark: Dynamic Micro-Routing vs Static Routing vs Dense Baseline
Evaluates multi-step speculative rollouts (N=1 to N=8) with dynamic router transitions.
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from config import DATA_DIR, MODELS_DIR, NUM_WORKER_THREADS, D_MODEL, NUM_EXPERTS
from model_mtp_drafter import DenseMTPDrafter, HybridMoEMTPDrafter, MTP_VOCAB_SIZE

MTP_FEATURES_DIR = DATA_DIR / "mtp_features"

torch.manual_seed(42)
embed_table = nn.Embedding(MTP_VOCAB_SIZE, D_MODEL)
embed_table.eval()

def run_dynamic_routing_benchmark(max_n=8):
    torch.set_num_threads(NUM_WORKER_THREADS)
    print("=========================================================================================")
    print(" PHASE 1 BENCHMARK: DYNAMIC MICRO-ROUTING vs STATIC ROUTING vs DENSE BASELINE")
    print("=========================================================================================\n")

    dense_ckpt = MODELS_DIR / "dense_mtp.pt"
    moe_ckpt = MODELS_DIR / "hybrid_moe_mtp.pt"

    dense_model = DenseMTPDrafter()
    dense_model.load_state_dict(torch.load(dense_ckpt, map_location="cpu"))
    dense_model.eval()

    moe_model = HybridMoEMTPDrafter()
    # Allow strict=False to accommodate the newly added router_transition weights
    moe_model.load_state_dict(torch.load(moe_ckpt, map_location="cpu"), strict=False)
    moe_model.eval()

    # Load unseen test chunk 2
    d = torch.load(MTP_FEATURES_DIR / "mtp_chunk_002.pt", map_location="cpu")
    test_e = [t.float() for t in d["e"]]
    test_h = [t.float() for t in d["h"]]
    test_r = [t.float() for t in d["router_weights"]]
    test_y = [t.long() for t in d["targets"]]
    test_cats = d["categories"]

    horizons = [1, 2, 4, 6, 8]
    domains = ["Code (AST & Syntax)", "Logic & Tool Calling", "General Knowledge"]

    # Tracking metrics
    stats = {
        n: {
            "dense": {"total": 0, "by_dom": {dom: 0 for dom in domains}},
            "static_moe": {"total": 0, "by_dom": {dom: 0 for dom in domains}},
            "dynamic_moe": {"total": 0, "by_dom": {dom: 0 for dom in domains}},
            "evals": 0,
            "evals_by_dom": {dom: 0 for dom in domains}
        } for n in horizons
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

            for t in range(0, seq_len - max_n, 4):
                gt = y_seq[t: t + max_n].tolist()

                for n in horizons:
                    gt_n = gt[:n]

                    # 1. Dense Baseline Rollout
                    dense_drafts = []
                    curr_e = e_seq[t].unsqueeze(0).unsqueeze(0)
                    curr_h = h_seq[t].unsqueeze(0).unsqueeze(0)
                    for _ in range(n):
                        logits, z = dense_model(curr_e, curr_h)
                        pred = logits.argmax(dim=-1).item()
                        dense_drafts.append(pred)
                        curr_e = embed_table(torch.tensor([[pred]]))
                        curr_h = z
                    d_acc = 0
                    for k in range(n):
                        if dense_drafts[k] == gt_n[k]: d_acc += 1
                        else: break

                    # 2. Static-Routing MoE Rollout
                    static_drafts = []
                    curr_e = e_seq[t].unsqueeze(0).unsqueeze(0)
                    curr_h = h_seq[t].unsqueeze(0).unsqueeze(0)
                    curr_r = r_seq[t].unsqueeze(0).unsqueeze(0)
                    for _ in range(n):
                        logits, z = moe_model(curr_e, curr_h, curr_r)
                        pred = logits.argmax(dim=-1).item()
                        static_drafts.append(pred)
                        curr_e = embed_table(torch.tensor([[pred]]))
                        curr_h = z
                    s_acc = 0
                    for k in range(n):
                        if static_drafts[k] == gt_n[k]: s_acc += 1
                        else: break

                    # 3. Dynamic-Routing MoE Rollout
                    dyn_drafts = []
                    curr_e = e_seq[t].unsqueeze(0).unsqueeze(0)
                    curr_h = h_seq[t].unsqueeze(0).unsqueeze(0)
                    curr_r = r_seq[t].unsqueeze(0).unsqueeze(0)
                    for _ in range(n):
                        logits, z = moe_model(curr_e, curr_h, curr_r)
                        pred = logits.argmax(dim=-1).item()
                        dyn_drafts.append(pred)
                        curr_e = embed_table(torch.tensor([[pred]]))
                        curr_h = z
                        # Autoregressive micro-router transition
                        curr_r = moe_model.transition_router(z, curr_r, beta=0.5)
                    dyn_acc = 0
                    for k in range(n):
                        if dyn_drafts[k] == gt_n[k]: dyn_acc += 1
                        else: break

                    stats[n]["dense"]["total"] += (1 + d_acc)
                    stats[n]["static_moe"]["total"] += (1 + s_acc)
                    stats[n]["dynamic_moe"]["total"] += (1 + dyn_acc)
                    stats[n]["evals"] += 1

                    stats[n]["dense"]["by_dom"][dom] += (1 + d_acc)
                    stats[n]["static_moe"]["by_dom"][dom] += (1 + s_acc)
                    stats[n]["dynamic_moe"]["by_dom"][dom] += (1 + dyn_acc)
                    stats[n]["evals_by_dom"][dom] += 1

    print(f"[EVAL] Completed in {time.time() - t0:.1f}s.\n")

    print("--- [1] OVERALL SPECULATIVE YIELD COMPARISON ---")
    print(f"{'Horizon':<10} | {'Dense Baseline':<16} | {'Static MoE':<16} | {'Dynamic MoE (New)':<18} | {'Dynamic vs Dense Gain':<20}")
    print("-" * 88)
    for n in horizons:
        ev = stats[n]["evals"]
        d_y = stats[n]["dense"]["total"] / ev
        s_y = stats[n]["static_moe"]["total"] / ev
        dyn_y = stats[n]["dynamic_moe"]["total"] / ev
        gain = ((dyn_y - d_y) / d_y) * 100
        print(f"N = {n:<6} | {d_y:>14.2f}x | {s_y:>14.2f}x | {dyn_y:>16.2f}x | {gain:>+18.2f}%")

    print("\n--- [2] CODE (AST & SYNTAX) YIELD ACROSS HORIZONS ---")
    print(f"{'Horizon':<10} | {'Dense Code Yield':<18} | {'Static MoE':<14} | {'Dynamic MoE (New)':<18} | {'Net Dynamic Gain':<18}")
    print("-" * 85)
    for n in horizons:
        ev = stats[n]["evals_by_dom"]["Code (AST & Syntax)"]
        d_y = stats[n]["dense"]["by_dom"]["Code (AST & Syntax)"] / ev
        s_y = stats[n]["static_moe"]["by_dom"]["Code (AST & Syntax)"] / ev
        dyn_y = stats[n]["dynamic_moe"]["by_dom"]["Code (AST & Syntax)"] / ev
        gain = ((dyn_y - d_y) / d_y) * 100
        print(f"N = {n:<6} | {d_y:>16.2f}x | {s_y:>12.2f}x | {dyn_y:>16.2f}x | {gain:>+16.2f}%")

    print("\n--- [3] LOGIC & TOOL CALLING YIELD ACROSS HORIZONS ---")
    print(f"{'Horizon':<10} | {'Dense Logic Yield':<18} | {'Static MoE':<14} | {'Dynamic MoE (New)':<18} | {'Net Dynamic Gain':<18}")
    print("-" * 85)
    for n in horizons:
        ev = stats[n]["evals_by_dom"]["Logic & Tool Calling"]
        d_y = stats[n]["dense"]["by_dom"]["Logic & Tool Calling"] / ev
        s_y = stats[n]["static_moe"]["by_dom"]["Logic & Tool Calling"] / ev
        dyn_y = stats[n]["dynamic_moe"]["by_dom"]["Logic & Tool Calling"] / ev
        gain = ((dyn_y - d_y) / d_y) * 100
        print(f"N = {n:<6} | {d_y:>16.2f}x | {s_y:>12.2f}x | {dyn_y:>16.2f}x | {gain:>+16.2f}%")

if __name__ == "__main__":
    run_dynamic_routing_benchmark()
