"""
Feature Harvesting for Multi-Token Prediction (MTP) / Speculative Drafting
Extracts:
1. Current token embeddings e(x_t)
2. Final-layer hidden states h_t
3. Final-layer router distributions r_t
4. Target next tokens y_t = x_{t+1}
across multi-domain prompts (Code, Logic/Tools, General Knowledge).
"""

import os
import re
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from config import DATA_DIR, PROMPTS_FILE, D_MODEL, NUM_EXPERTS, TOP_K_EXPERTS
from model_mtp_drafter import MTP_VOCAB_SIZE

MTP_FEATURES_DIR = DATA_DIR / "mtp_features"
MTP_FEATURES_DIR.mkdir(parents=True, exist_ok=True)
VOCAB_FILE = DATA_DIR / "mtp_vocab.json"
with open(VOCAB_FILE, "r", encoding="utf-8") as f:
    vocab = json.load(f)

class SwiGLUExpert(nn.Module):
    def __init__(self, d_model=D_MODEL, d_ffn=1024):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ffn, bias=False)
        self.w_up = nn.Linear(d_model, d_ffn, bias=False)
        self.w_down = nn.Linear(d_ffn, d_model, bias=False)
    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))

class NonLinearMoEBlock(nn.Module):
    def __init__(self, d_model=D_MODEL, num_experts=NUM_EXPERTS, top_k=TOP_K_EXPERTS, d_ffn=1024):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.norm = nn.LayerNorm(d_model)
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList([SwiGLUExpert(d_model, d_ffn) for _ in range(num_experts)])

    def forward(self, x):
        normed = self.norm(x)
        logits = self.gate(normed)
        weights, indices = torch.topk(torch.softmax(logits, dim=-1), self.top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        
        out = torch.zeros_like(x)
        for k in range(self.top_k):
            idx_k = indices[:, :, k]
            w_k = weights[:, :, k].unsqueeze(-1)
            for exp_id in range(self.num_experts):
                mask = (idx_k == exp_id)
                if mask.any():
                    tokens = normed[mask]
                    exp_out = self.experts[exp_id](tokens)
                    out[mask] += w_k[mask] * exp_out
        return x + out, torch.softmax(logits, dim=-1)

class TargetMoEBackbone(nn.Module):
    def __init__(self, vocab_size=MTP_VOCAB_SIZE, d_model=D_MODEL):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.block1 = NonLinearMoEBlock(d_model)
        self.block2 = NonLinearMoEBlock(d_model)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, token_ids):
        # token_ids: [B, T]
        e = self.embed(token_ids)
        x, _ = self.block1(e)
        h, r = self.block2(x)
        h = self.final_norm(h)
        return e, h, r

def harvest_mtp_dataset(num_samples=150, seq_len=128, chunk_size=50):
    torch.manual_seed(42)
    backbone = TargetMoEBackbone()
    backbone.eval()
    
    # Load curated prompts to assign domains
    with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
        prompts = [json.loads(line) for line in f]
        
    print(f"[HARVEST-MTP] Harvesting MTP features for {num_samples} sequences across domains...")
    
    chunks = []
    current_chunk = {"e": [], "h": [], "router_weights": [], "targets": [], "categories": []}
    
    with torch.no_grad():
        for i in range(num_samples):
            p = prompts[i % len(prompts)]
            category = p.get("category", "General Knowledge")
            
            # Tokenize prompt text using authentic vocabulary
            toks = re.findall(r"\w+|[^\w\s]", p["text"])
            token_ids = [vocab.get(t, vocab["<unk>"]) for t in toks]
            if len(token_ids) < seq_len:
                token_ids = (token_ids * (seq_len // max(len(token_ids), 1) + 1))[:seq_len]
            else:
                token_ids = token_ids[:seq_len]
            tokens = torch.tensor([token_ids], dtype=torch.long)
                
            e, h, r = backbone(tokens)
            
            # Input features at positions 0 to T-2
            e_in = e[:, :-1, :].squeeze(0).half()  # [T-1, D]
            h_in = h[:, :-1, :].squeeze(0).half()  # [T-1, D]
            r_in = r[:, :-1, :].squeeze(0).half()  # [T-1, NUM_EXPERTS]
            
            # Target tokens at positions 1 to T-1
            y_target = tokens[:, 1:].squeeze(0)    # [T-1]
            
            current_chunk["e"].append(e_in)
            current_chunk["h"].append(h_in)
            current_chunk["router_weights"].append(r_in)
            current_chunk["targets"].append(y_target)
            current_chunk["categories"].append(category)
            
            if len(current_chunk["h"]) >= chunk_size or i == num_samples - 1:
                chunk_id = len(chunks)
                chunk_file = MTP_FEATURES_DIR / f"mtp_chunk_{chunk_id:03d}.pt"
                torch.save(current_chunk, chunk_file)
                print(f"  [SAVED] {chunk_file.name} ({len(current_chunk['h'])} sequences)")
                chunks.append(chunk_file)
                current_chunk = {"e": [], "h": [], "router_weights": [], "targets": [], "categories": []}
                
    print("[HARVEST-MTP] Complete! MTP feature dataset generated.")

if __name__ == "__main__":
    harvest_mtp_dataset()
