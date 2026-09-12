"""
Comprehensive Unified Speculative Engine Benchmark
Compares:
1. Dense Baseline (Linear-3)
2. Dense Baseline (Linear-5)
3. First Hybrid MoE (Linear-3)
4. First Hybrid MoE (Linear-5)
5. First Hybrid MoE (Tree-7b Fixed)
6. Unified Engine - Balanced (Dynamic Routing + Distillation + Adaptive Tree)
7. Unified Engine - High Yield (Dynamic Routing + Distillation + Expressive Adaptive Tree)
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from config import DATA_DIR, MODELS_DIR, NUM_WORKER_THREADS, D_MODEL
from model_mtp_drafter import DenseMTPDrafter, HybridMoEMTPDrafter, MTP_VOCAB_SIZE

MTP_FEATURES_DIR = DATA_DIR / "mtp_features"

torch.manual_seed(42)
embed_table = nn.Embedding(MTP_VOCAB_SIZE, D_MODEL)
embed_table.eval()

def generate_linear(model, is_moe, curr_e, curr_h, curr_r, n, dynamic=False):
    chain = []
    e = curr_e
    h = curr_h
    r = curr_r
    for _ in range(n):
        if is_moe:
            logits, h = model(e, h, r)
            if dynamic:
                r = model.transition_router(h, r, beta=0.5)
        else:
            logits, h = model(e, h)
        pred = logits.argmax(dim=-1).item()
        chain.append(pred)
        e = embed_table(torch.tensor([[pred]]))
    return [chain], n

def generate_tree_2_3_2_fixed(model, curr_e, curr_h, curr_r):
    logits1, h1 = model(curr_e, curr_h, curr_r)
    r1 = model.transition_router(h1, curr_r, beta=0.5)
    top2_c1 = torch.topk(logits1[0, 0], k=2).indices.tolist()
    
    paths = []
    c1_best = top2_c1[0]
    e2 = embed_table(torch.tensor([[c1_best]]))
    logits2, h2 = model(e2, h1, r1)
    r2 = model.transition_router(h2, r1, beta=0.5)
    top2_c2 = torch.topk(logits2[0, 0], k=2).indices.tolist()
    for c2 in top2_c2:
        e3 = embed_table(torch.tensor([[c2]]))
        logits3, _ = model(e3, h2, r2)
        c3 = logits3.argmax(dim=-1).item()
        paths.append([c1_best, c2, c3])
        
    c1_sec = top2_c1[1]
    e2_sec = embed_table(torch.tensor([[c1_sec]]))
    logits2_sec, _ = model(e2_sec, h1, r1)
    c2_sec = logits2_sec.argmax(dim=-1).item()
    paths.append([c1_sec, c2_sec])
    return paths, 7

def generate_unified_adaptive(model, curr_e, curr_h, curr_r, mode="balanced"):
    draft_count = 1
    paths = []
    
    # Depth 1
    logits1, h1 = model(curr_e, curr_h, curr_r)
    probs1 = F.softmax(logits1[0, 0], dim=-1)
    top2_p1, top2_idx1 = torch.topk(probs1, k=2)
    p1_1, p1_2 = top2_p1[0].item(), top2_p1[1].item()
    c1_1, c1_2 = top2_idx1[0].item(), top2_idx1[1].item()
    
    r1 = model.transition_router(h1, curr_r, beta=0.5)
    
    # Depth 2 primary
    e2_1 = embed_table(torch.tensor([[c1_1]]))
    logits2_1, h2_1 = model(e2_1, h1, r1)
    draft_count += 1
    probs2_1 = F.softmax(logits2_1[0, 0], dim=-1)
    top2_p2, top2_idx2 = torch.topk(probs2_1, k=2)
    p2_1, p2_2 = top2_p2[0].item(), top2_p2[1].item()
    c2_1, c2_2 = top2_idx2[0].item(), top2_idx2[1].item()
    
    r2_1 = model.transition_router(h2_1, r1, beta=0.5)
    
    # Depth 3 on primary
    e3_1 = embed_table(torch.tensor([[c2_1]]))
    logits3_1, _ = model(e3_1, h2_1, r2_1)
    draft_count += 1
    c3_1 = logits3_1.argmax(dim=-1).item()
    paths.append([c1_1, c2_1, c3_1])
    
    if mode == "high_yield":
        # Expand second candidate at Depth 2 if competitive
        if p2_2 >= 0.15:
            e3_2 = embed_table(torch.tensor([[c2_2]]))
            logits3_2, _ = model(e3_2, h2_1, r2_1)
            draft_count += 1
            c3_2 = logits3_2.argmax(dim=-1).item()
            paths.append([c1_1, c2_2, c3_2])
            
        # Branch from root if ambiguous
        if p1_1 < 0.70 and (p1_2 / max(p1_1, 1e-6)) >= 0.25:
            e2_2 = embed_table(torch.tensor([[c1_2]]))
            logits2_2, _ = model(e2_2, h1, r1)
            draft_count += 1
            c2_sec = logits2_2.argmax(dim=-1).item()
            paths.append([c1_2, c2_sec])
    else:
        # Balanced mode: tighter thresholds
        if p2_1 < 0.65 and (p2_2 / max(p2_1, 1e-6)) >= 0.35 and p2_2 >= 0.25:
            paths.append([c1_1, c2_2])
            draft_count += 1
            
        if p1_1 < 0.65 and (p1_2 / max(p1_1, 1e-6)) >= 0.35:
            e2_sec = embed_table(torch.tensor([[c1_2]]))
            logits2_sec, _ = model(e2_sec, h1, r1)
            draft_count += 1
            c2_sec = logits2_sec.argmax(dim=-1).item()
            paths.append([c1_2, c2_sec])
            
    return paths, draft_count

def verify_tree_paths(candidate_paths, ground_truth):
    max_acc = 0
    gt_len = len(ground_truth)
    for path in candidate_paths:
        acc = 0
        for k in range(min(len(path), gt_len)):
            if path[k] == ground_truth[k]:
                acc += 1
            else:
                break
        if acc > max_acc:
            max_acc = acc
    return max_acc

def run_comprehensive_benchmark():
    torch.set_num_threads(NUM_WORKER_THREADS)
    print("=========================================================================================")
    print(" UNIFIED SPECULATIVE ENGINE BENCHMARK: COMBINED vs FIRST HYBRID MoE vs DENSE")
    print("=========================================================================================\n")

    dense_ckpt = MODELS_DIR / "dense_mtp.pt"
    first_moe_ckpt = MODELS_DIR / "hybrid_moe_mtp.pt"
    distill_moe_ckpt = MODELS_DIR / "hybrid_moe_mtp_distilled.pt"

    dense_model = DenseMTPDrafter()
    dense_model.load_state_dict(torch.load(dense_ckpt, map_location="cpu"))
    dense_model.eval()

    first_moe_model = HybridMoEMTPDrafter()
    first_moe_model.load_state_dict(torch.load(first_moe_ckpt, map_location="cpu"), strict=False)
    first_moe_model.eval()

    unified_model = HybridMoEMTPDrafter()
    unified_model.load_state_dict(torch.load(distill_moe_ckpt, map_location="cpu"), strict=False)
    unified_model.eval()

    d = torch.load(MTP_FEATURES_DIR / "mtp_chunk_002.pt", map_location="cpu")
    test_e = [t.float() for t in d["e"]]
    test_h = [t.float() for t in d["h"]]
    test_r = [t.float() for t in d["router_weights"]]
    test_y = [t.long() for t in d["targets"]]
    test_cats = d["categories"]

    domains = ["Code (AST & Syntax)", "Logic & Tool Calling", "General Knowledge"]
    competitors = [
        "1. Dense Baseline (Linear-3)",
        "2. Dense Baseline (Linear-5)",
        "3. First Hybrid MoE (Linear-3)",
        "4. First Hybrid MoE (Linear-5)",
        "5. First Hybrid MoE (Tree-7b)",
        "6. Unified Engine (Balanced Tree)",
        "7. Unified Engine (High-Yield Tree)",
    ]

    stats = {
        c: {
            "drafted_tokens": 0,
            "accepted_tokens": 0,
            "yield_sum": 0.0,
            "evals": 0,
            "by_dom_yield": {dom: 0.0 for dom in domains},
            "by_dom_evals": {dom: 0 for dom in domains},
        } for c in competitors
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

            for t in range(0, seq_len - 6, 4):
                gt = y_seq[t: t + 6].tolist()
                curr_e = e_seq[t].unsqueeze(0).unsqueeze(0)
                curr_h = h_seq[t].unsqueeze(0).unsqueeze(0)
                curr_r = r_seq[t].unsqueeze(0).unsqueeze(0)

                # 1. Dense Linear-3
                p1, d1 = generate_linear(dense_model, False, curr_e, curr_h, None, 3)
                acc1 = verify_tree_paths(p1, gt)
                stats["1. Dense Baseline (Linear-3)"]["drafted_tokens"] += d1
                stats["1. Dense Baseline (Linear-3)"]["accepted_tokens"] += acc1
                stats["1. Dense Baseline (Linear-3)"]["yield_sum"] += (1 + acc1)
                stats["1. Dense Baseline (Linear-3)"]["by_dom_yield"][dom] += (1 + acc1)
                stats["1. Dense Baseline (Linear-3)"]["by_dom_evals"][dom] += 1
                stats["1. Dense Baseline (Linear-3)"]["evals"] += 1

                # 2. Dense Linear-5
                p2, d2 = generate_linear(dense_model, False, curr_e, curr_h, None, 5)
                acc2 = verify_tree_paths(p2, gt)
                stats["2. Dense Baseline (Linear-5)"]["drafted_tokens"] += d2
                stats["2. Dense Baseline (Linear-5)"]["accepted_tokens"] += acc2
                stats["2. Dense Baseline (Linear-5)"]["yield_sum"] += (1 + acc2)
                stats["2. Dense Baseline (Linear-5)"]["by_dom_yield"][dom] += (1 + acc2)
                stats["2. Dense Baseline (Linear-5)"]["by_dom_evals"][dom] += 1
                stats["2. Dense Baseline (Linear-5)"]["evals"] += 1

                # 3. First Hybrid MoE Linear-3
                p3, d3 = generate_linear(first_moe_model, True, curr_e, curr_h, curr_r, 3, dynamic=False)
                acc3 = verify_tree_paths(p3, gt)
                stats["3. First Hybrid MoE (Linear-3)"]["drafted_tokens"] += d3
                stats["3. First Hybrid MoE (Linear-3)"]["accepted_tokens"] += acc3
                stats["3. First Hybrid MoE (Linear-3)"]["yield_sum"] += (1 + acc3)
                stats["3. First Hybrid MoE (Linear-3)"]["by_dom_yield"][dom] += (1 + acc3)
                stats["3. First Hybrid MoE (Linear-3)"]["by_dom_evals"][dom] += 1
                stats["3. First Hybrid MoE (Linear-3)"]["evals"] += 1

                # 4. First Hybrid MoE Linear-5
                p4, d4 = generate_linear(first_moe_model, True, curr_e, curr_h, curr_r, 5, dynamic=False)
                acc4 = verify_tree_paths(p4, gt)
                stats["4. First Hybrid MoE (Linear-5)"]["drafted_tokens"] += d4
                stats["4. First Hybrid MoE (Linear-5)"]["accepted_tokens"] += acc4
                stats["4. First Hybrid MoE (Linear-5)"]["yield_sum"] += (1 + acc4)
                stats["4. First Hybrid MoE (Linear-5)"]["by_dom_yield"][dom] += (1 + acc4)
                stats["4. First Hybrid MoE (Linear-5)"]["by_dom_evals"][dom] += 1
                stats["4. First Hybrid MoE (Linear-5)"]["evals"] += 1

                # 5. First Hybrid MoE Tree-7b (Fixed Tree)
                p5, d5 = generate_tree_2_3_2_fixed(first_moe_model, curr_e, curr_h, curr_r)
                acc5 = verify_tree_paths(p5, gt)
                stats["5. First Hybrid MoE (Tree-7b)"]["drafted_tokens"] += d5
                stats["5. First Hybrid MoE (Tree-7b)"]["accepted_tokens"] += acc5
                stats["5. First Hybrid MoE (Tree-7b)"]["yield_sum"] += (1 + acc5)
                stats["5. First Hybrid MoE (Tree-7b)"]["by_dom_yield"][dom] += (1 + acc5)
                stats["5. First Hybrid MoE (Tree-7b)"]["by_dom_evals"][dom] += 1
                stats["5. First Hybrid MoE (Tree-7b)"]["evals"] += 1

                # 6. Unified Engine (Balanced Tree)
                p6, d6 = generate_unified_adaptive(unified_model, curr_e, curr_h, curr_r, mode="balanced")
                acc6 = verify_tree_paths(p6, gt)
                stats["6. Unified Engine (Balanced Tree)"]["drafted_tokens"] += d6
                stats["6. Unified Engine (Balanced Tree)"]["accepted_tokens"] += acc6
                stats["6. Unified Engine (Balanced Tree)"]["yield_sum"] += (1 + acc6)
                stats["6. Unified Engine (Balanced Tree)"]["by_dom_yield"][dom] += (1 + acc6)
                stats["6. Unified Engine (Balanced Tree)"]["by_dom_evals"][dom] += 1
                stats["6. Unified Engine (Balanced Tree)"]["evals"] += 1

                # 7. Unified Engine (High-Yield Tree)
                p7, d7 = generate_unified_adaptive(unified_model, curr_e, curr_h, curr_r, mode="high_yield")
                acc7 = verify_tree_paths(p7, gt)
                stats["7. Unified Engine (High-Yield Tree)"]["drafted_tokens"] += d7
                stats["7. Unified Engine (High-Yield Tree)"]["accepted_tokens"] += acc7
                stats["7. Unified Engine (High-Yield Tree)"]["yield_sum"] += (1 + acc7)
                stats["7. Unified Engine (High-Yield Tree)"]["by_dom_yield"][dom] += (1 + acc7)
                stats["7. Unified Engine (High-Yield Tree)"]["by_dom_evals"][dom] += 1
                stats["7. Unified Engine (High-Yield Tree)"]["evals"] += 1

    print(f"[EVAL] Comprehensive benchmark completed in {time.time() - t0:.1f}s.\n")

    baseline_yield = stats["1. Dense Baseline (Linear-3)"]["yield_sum"] / stats["1. Dense Baseline (Linear-3)"]["evals"]
    first_moe_yield = stats["3. First Hybrid MoE (Linear-3)"]["yield_sum"] / stats["3. First Hybrid MoE (Linear-3)"]["evals"]

    print("--- [1] MASTER SPECULATIVE PERFORMANCE MATRIX ---")
    print(f"{'Engine Architecture':<38} | {'Avg Draft':<11} | {'Yield (Tok/Pass)':<18} | {'Draft Precision':<16} | {'vs Dense L-3':<14} | {'vs First MoE L-3':<16}")
    print("-" * 125)
    for c in competitors:
        ev = stats[c]["evals"]
        avg_d = stats[c]["drafted_tokens"] / ev
        y = stats[c]["yield_sum"] / ev
        prec = (stats[c]["accepted_tokens"] / max(stats[c]["drafted_tokens"], 1)) * 100
        vs_dense = ((y - baseline_yield) / baseline_yield) * 100
        vs_first = ((y - first_moe_yield) / first_moe_yield) * 100
        print(f"{c:<38} | {avg_d:>9.2f}   | {y:>16.2f}x | {prec:>14.1f}% | {vs_dense:>+12.2f}% | {vs_first:>+14.2f}%")

    print("\n--- [2] DOMAIN YIELD BREAKDOWN (Tokens Produced per Verification Pass) ---")
    print(f"{'Engine Architecture':<38} | {'Code (AST)':<14} | {'Logic & Tools':<15} | {'General Knowledge':<18}")
    print("-" * 92)
    for c in competitors:
        c_ev = stats[c]["by_dom_evals"]["Code (AST & Syntax)"]
        l_ev = stats[c]["by_dom_evals"]["Logic & Tool Calling"]
        k_ev = stats[c]["by_dom_evals"]["General Knowledge"]
        c_y = stats[c]["by_dom_yield"]["Code (AST & Syntax)"] / c_ev
        l_y = stats[c]["by_dom_yield"]["Logic & Tool Calling"] / l_ev
        k_y = stats[c]["by_dom_yield"]["General Knowledge"] / k_ev
        print(f"{c:<38} | {c_y:>12.2f}x | {l_y:>13.2f}x | {k_y:>16.2f}x")

if __name__ == "__main__":
    run_comprehensive_benchmark()
