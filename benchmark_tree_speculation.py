"""
Phase 2 Benchmark: Tree-Based Speculative Drafting vs Linear Draft Chains
Evaluates speculative acceptance across candidate tree topologies vs linear chains.
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

def generate_linear_chain(model, is_moe, curr_e, curr_h, curr_r, n):
    """Generates a linear draft chain of length n."""
    chain = []
    e = curr_e
    h = curr_h
    r = curr_r
    for _ in range(n):
        if is_moe:
            logits, h = model(e, h, r)
            r = model.transition_router(h, r, beta=0.5)
        else:
            logits, h = model(e, h)
        pred = logits.argmax(dim=-1).item()
        chain.append(pred)
        e = embed_table(torch.tensor([[pred]]))
    return [chain]

def generate_tree_1_2_2(model, curr_e, curr_h, curr_r):
    logits1, h1 = model(curr_e, curr_h, curr_r)
    r1 = model.transition_router(h1, curr_r, beta=0.5)
    c1 = logits1.argmax(dim=-1).item()
    
    e2 = embed_table(torch.tensor([[c1]]))
    logits2, h2 = model(e2, h1, r1)
    r2 = model.transition_router(h2, r1, beta=0.5)
    top2_c2 = torch.topk(logits2[0, 0], k=2).indices.tolist()
    
    paths = []
    for c2 in top2_c2:
        e3 = embed_table(torch.tensor([[c2]]))
        logits3, _ = model(e3, h2, r2)
        c3 = logits3.argmax(dim=-1).item()
        paths.append([c1, c2, c3])
    return paths

def generate_tree_2_2_2(model, curr_e, curr_h, curr_r):
    logits1, h1 = model(curr_e, curr_h, curr_r)
    r1 = model.transition_router(h1, curr_r, beta=0.5)
    top2_c1 = torch.topk(logits1[0, 0], k=2).indices.tolist()
    
    paths = []
    for c1 in top2_c1:
        e2 = embed_table(torch.tensor([[c1]]))
        logits2, _ = model(e2, h1, r1)
        top2_c2 = torch.topk(logits2[0, 0], k=2).indices.tolist()
        for c2 in top2_c2:
            paths.append([c1, c2])
    return paths

def generate_tree_1_3_3(model, curr_e, curr_h, curr_r):
    logits1, h1 = model(curr_e, curr_h, curr_r)
    r1 = model.transition_router(h1, curr_r, beta=0.5)
    c1 = logits1.argmax(dim=-1).item()
    
    e2 = embed_table(torch.tensor([[c1]]))
    logits2, h2 = model(e2, h1, r1)
    r2 = model.transition_router(h2, r1, beta=0.5)
    top3_c2 = torch.topk(logits2[0, 0], k=3).indices.tolist()
    
    paths = []
    for c2 in top3_c2:
        e3 = embed_table(torch.tensor([[c2]]))
        logits3, _ = model(e3, h2, r2)
        c3 = logits3.argmax(dim=-1).item()
        paths.append([c1, c2, c3])
    return paths

def generate_tree_2_3_2(model, curr_e, curr_h, curr_r):
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
    return paths

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

def run_tree_speculation_benchmark():
    torch.set_num_threads(NUM_WORKER_THREADS)
    print("=========================================================================================")
    print(" PHASE 2 BENCHMARK: TREE-BASED SPECULATIVE DRAFTING vs LINEAR DRAFT CHAINS")
    print("=========================================================================================\n")

    dense_ckpt = MODELS_DIR / "dense_mtp.pt"
    moe_ckpt = MODELS_DIR / "hybrid_moe_mtp.pt"

    dense_model = DenseMTPDrafter()
    dense_model.load_state_dict(torch.load(dense_ckpt, map_location="cpu"))
    dense_model.eval()

    moe_model = HybridMoEMTPDrafter()
    moe_model.load_state_dict(torch.load(moe_ckpt, map_location="cpu"), strict=False)
    moe_model.eval()

    d = torch.load(MTP_FEATURES_DIR / "mtp_chunk_002.pt", map_location="cpu")
    test_e = [t.float() for t in d["e"]]
    test_h = [t.float() for t in d["h"]]
    test_r = [t.float() for t in d["router_weights"]]
    test_y = [t.long() for t in d["targets"]]
    test_cats = d["categories"]

    domains = ["Code (AST & Syntax)", "Logic & Tool Calling", "General Knowledge"]
    schemes = [
        "Dense Linear-3 (3 tokens)",
        "Dense Linear-5 (5 tokens)",
        "MoE Linear-3 (3 tokens)",
        "MoE Linear-5 (5 tokens)",
        "MoE Tree-5 [1-2-2] (5 tokens)",
        "MoE Tree-6 [2-2] (6 tokens)",
        "MoE Tree-7 [1-3-3] (7 tokens)",
        "MoE Tree-7b [2-2+1] (7 tokens)",
    ]

    stats = {
        s: {
            "yield": 0.0,
            "accepted_tokens": 0,
            "total_drafted": 0,
            "by_dom": {dom: 0.0 for dom in domains},
            "evals_by_dom": {dom: 0 for dom in domains},
            "evals": 0,
        } for s in schemes
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
                p_dl3 = generate_linear_chain(dense_model, False, curr_e, curr_h, None, 3)
                acc_dl3 = verify_tree_paths(p_dl3, gt)
                stats["Dense Linear-3 (3 tokens)"]["accepted_tokens"] += acc_dl3
                stats["Dense Linear-3 (3 tokens)"]["total_drafted"] += 3
                stats["Dense Linear-3 (3 tokens)"]["yield"] += (1 + acc_dl3)
                stats["Dense Linear-3 (3 tokens)"]["by_dom"][dom] += (1 + acc_dl3)
                stats["Dense Linear-3 (3 tokens)"]["evals_by_dom"][dom] += 1
                stats["Dense Linear-3 (3 tokens)"]["evals"] += 1

                # 2. Dense Linear-5
                p_dl5 = generate_linear_chain(dense_model, False, curr_e, curr_h, None, 5)
                acc_dl5 = verify_tree_paths(p_dl5, gt)
                stats["Dense Linear-5 (5 tokens)"]["accepted_tokens"] += acc_dl5
                stats["Dense Linear-5 (5 tokens)"]["total_drafted"] += 5
                stats["Dense Linear-5 (5 tokens)"]["yield"] += (1 + acc_dl5)
                stats["Dense Linear-5 (5 tokens)"]["by_dom"][dom] += (1 + acc_dl5)
                stats["Dense Linear-5 (5 tokens)"]["evals_by_dom"][dom] += 1
                stats["Dense Linear-5 (5 tokens)"]["evals"] += 1

                # 3. MoE Linear-3
                p_ml3 = generate_linear_chain(moe_model, True, curr_e, curr_h, curr_r, 3)
                acc_ml3 = verify_tree_paths(p_ml3, gt)
                stats["MoE Linear-3 (3 tokens)"]["accepted_tokens"] += acc_ml3
                stats["MoE Linear-3 (3 tokens)"]["total_drafted"] += 3
                stats["MoE Linear-3 (3 tokens)"]["yield"] += (1 + acc_ml3)
                stats["MoE Linear-3 (3 tokens)"]["by_dom"][dom] += (1 + acc_ml3)
                stats["MoE Linear-3 (3 tokens)"]["evals_by_dom"][dom] += 1
                stats["MoE Linear-3 (3 tokens)"]["evals"] += 1

                # 4. MoE Linear-5
                p_ml5 = generate_linear_chain(moe_model, True, curr_e, curr_h, curr_r, 5)
                acc_ml5 = verify_tree_paths(p_ml5, gt)
                stats["MoE Linear-5 (5 tokens)"]["accepted_tokens"] += acc_ml5
                stats["MoE Linear-5 (5 tokens)"]["total_drafted"] += 5
                stats["MoE Linear-5 (5 tokens)"]["yield"] += (1 + acc_ml5)
                stats["MoE Linear-5 (5 tokens)"]["by_dom"][dom] += (1 + acc_ml5)
                stats["MoE Linear-5 (5 tokens)"]["evals_by_dom"][dom] += 1
                stats["MoE Linear-5 (5 tokens)"]["evals"] += 1

                # 5. MoE Tree-5 [1-2-2]
                p_t5 = generate_tree_1_2_2(moe_model, curr_e, curr_h, curr_r)
                acc_t5 = verify_tree_paths(p_t5, gt)
                stats["MoE Tree-5 [1-2-2] (5 tokens)"]["accepted_tokens"] += acc_t5
                stats["MoE Tree-5 [1-2-2] (5 tokens)"]["total_drafted"] += 5
                stats["MoE Tree-5 [1-2-2] (5 tokens)"]["yield"] += (1 + acc_t5)
                stats["MoE Tree-5 [1-2-2] (5 tokens)"]["by_dom"][dom] += (1 + acc_t5)
                stats["MoE Tree-5 [1-2-2] (5 tokens)"]["evals_by_dom"][dom] += 1
                stats["MoE Tree-5 [1-2-2] (5 tokens)"]["evals"] += 1

                # 6. MoE Tree-6 [2-2]
                p_t6 = generate_tree_2_2_2(moe_model, curr_e, curr_h, curr_r)
                acc_t6 = verify_tree_paths(p_t6, gt)
                stats["MoE Tree-6 [2-2] (6 tokens)"]["accepted_tokens"] += acc_t6
                stats["MoE Tree-6 [2-2] (6 tokens)"]["total_drafted"] += 6
                stats["MoE Tree-6 [2-2] (6 tokens)"]["yield"] += (1 + acc_t6)
                stats["MoE Tree-6 [2-2] (6 tokens)"]["by_dom"][dom] += (1 + acc_t6)
                stats["MoE Tree-6 [2-2] (6 tokens)"]["evals_by_dom"][dom] += 1
                stats["MoE Tree-6 [2-2] (6 tokens)"]["evals"] += 1

                # 7. MoE Tree-7 [1-3-3]
                p_t7 = generate_tree_1_3_3(moe_model, curr_e, curr_h, curr_r)
                acc_t7 = verify_tree_paths(p_t7, gt)
                stats["MoE Tree-7 [1-3-3] (7 tokens)"]["accepted_tokens"] += acc_t7
                stats["MoE Tree-7 [1-3-3] (7 tokens)"]["total_drafted"] += 7
                stats["MoE Tree-7 [1-3-3] (7 tokens)"]["yield"] += (1 + acc_t7)
                stats["MoE Tree-7 [1-3-3] (7 tokens)"]["by_dom"][dom] += (1 + acc_t7)
                stats["MoE Tree-7 [1-3-3] (7 tokens)"]["evals_by_dom"][dom] += 1
                stats["MoE Tree-7 [1-3-3] (7 tokens)"]["evals"] += 1

                # 8. MoE Tree-7b [2-2+1]
                p_t7b = generate_tree_2_3_2(moe_model, curr_e, curr_h, curr_r)
                acc_t7b = verify_tree_paths(p_t7b, gt)
                stats["MoE Tree-7b [2-2+1] (7 tokens)"]["accepted_tokens"] += acc_t7b
                stats["MoE Tree-7b [2-2+1] (7 tokens)"]["total_drafted"] += 7
                stats["MoE Tree-7b [2-2+1] (7 tokens)"]["yield"] += (1 + acc_t7b)
                stats["MoE Tree-7b [2-2+1] (7 tokens)"]["by_dom"][dom] += (1 + acc_t7b)
                stats["MoE Tree-7b [2-2+1] (7 tokens)"]["evals_by_dom"][dom] += 1
                stats["MoE Tree-7b [2-2+1] (7 tokens)"]["evals"] += 1

    print(f"[EVAL] Tree speculation benchmark completed in {time.time() - t0:.1f}s.\n")

    baseline_yield = stats["Dense Linear-3 (3 tokens)"]["yield"] / stats["Dense Linear-3 (3 tokens)"]["evals"]

    print("--- [1] OVERALL SPECULATIVE SPEEDUP & ACCEPTANCE COMPARISON ---")
    print(f"{'Drafting Scheme':<32} | {'Tokens Drafted':<15} | {'Tokens/Pass (Yield)':<20} | {'Draft Precision':<16} | {'vs Dense L-3 Gain':<18}")
    print("-" * 110)
    for s in schemes:
        ev = stats[s]["evals"]
        mean_yield = stats[s]["yield"] / ev
        prec = (stats[s]["accepted_tokens"] / stats[s]["total_drafted"]) * 100
        tokens_ver = stats[s]["total_drafted"] // ev
        gain = ((mean_yield - baseline_yield) / baseline_yield) * 100
        print(f"{s:<32} | {tokens_ver:>15} | {mean_yield:>18.2f}x | {prec:>15.1f}% | {gain:>+16.2f}%")

    print("\n--- [2] DOMAIN-SPECIFIC SPECULATIVE YIELD (Tokens Produced per Verification Pass) ---")
    print(f"{'Drafting Scheme':<32} | {'Code (AST)':<14} | {'Logic & Tools':<15} | {'General Knowledge':<18}")
    print("-" * 88)
    for s in schemes:
        c_ev = stats[s]["evals_by_dom"]["Code (AST & Syntax)"]
        l_ev = stats[s]["evals_by_dom"]["Logic & Tool Calling"]
        k_ev = stats[s]["evals_by_dom"]["General Knowledge"]
        c_y = stats[s]["by_dom"]["Code (AST & Syntax)"] / c_ev
        l_y = stats[s]["by_dom"]["Logic & Tool Calling"] / l_ev
        k_y = stats[s]["by_dom"]["General Knowledge"] / k_ev
        print(f"{s:<32} | {c_y:>12.2f}x | {l_y:>13.2f}x | {k_y:>16.2f}x")

if __name__ == "__main__":
    run_tree_speculation_benchmark()
