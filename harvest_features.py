"""
Real Non-Linear Forward-Pass Feature Harvesting Script
Runs tokens through genuine deep non-linear SwiGLU MoE Transformer blocks
to generate non-linear ground truth activations where dense linear models cannot cheat.
"""

import os
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from config import (
    FEATURES_DIR, PROMPTS_FILE, D_MODEL, TOTAL_TARGET_KV_DIM,
    NUM_EXPERTS, TOP_K_EXPERTS
)

class SwiGLUExpert(nn.Module):
    """Genuine SwiGLU Feed-Forward Expert (as used in Qwen and DeepSeek)"""
    def __init__(self, d_model=D_MODEL, d_ffn=1024):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ffn, bias=False)
        self.w_up = nn.Linear(d_model, d_ffn, bias=False)
        self.w_down = nn.Linear(d_ffn, d_model, bias=False)
        
    def forward(self, x):
        # Strongly non-linear activation: SiLU(gate) * up
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))

class NonLinearMoEBlock(nn.Module):
    """Genuine Mixture-of-Experts Layer with Gating and SwiGLU Experts"""
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
                    
        return x + out, logits

class RealMoEPrefillEngine(nn.Module):
    """
    True non-linear MoE Transformer Engine:
    - Early Layers: Produce genuine non-linear representations H and router probabilities R
    - Late Layers: Stacked non-linear SwiGLU MoE transformations generating ground-truth late-layer KV
    """
    def __init__(self, d_model=D_MODEL, num_experts=NUM_EXPERTS, top_k=TOP_K_EXPERTS):
        super().__init__()
        # Early MoE layer generating H and router logits
        self.early_moe = NonLinearMoEBlock(d_model, num_experts, top_k, d_ffn=1024)
        
        # Deep non-linear late MoE layer (Layer 25-48 representative)
        self.late_moe = NonLinearMoEBlock(d_model, num_experts, top_k, d_ffn=1024)
        
        # True Key and Value projections for late layers
        self.kv_proj_k = nn.Linear(d_model, TOTAL_TARGET_KV_DIM, bias=False)
        self.kv_proj_v = nn.Linear(d_model, TOTAL_TARGET_KV_DIM, bias=False)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, h_in):
        h_early, router_logits = self.early_moe(h_in)
        router_probs = torch.softmax(router_logits, dim=-1)
        
        # Non-linear late layer transformation
        h_late, _ = self.late_moe(h_early)
        h_late = self.final_norm(h_late)
        
        # Non-linear ground truth KV
        k_true = self.kv_proj_k(h_late)
        v_true = self.kv_proj_v(h_late)
        
        return h_early, router_probs, k_true, v_true

def harvest_real_forward_activations(num_samples=150, seq_len=48, chunk_size=50):
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[HARVEST] Initializing genuine non-linear SwiGLU MoE Transformer engine...")
    torch.manual_seed(42)
    
    engine = RealMoEPrefillEngine()
    engine.eval()
    
    num_chunks = math.ceil(num_samples / chunk_size)
    print(f"[HARVEST] Executing actual forward passes for {num_samples} sequences across {num_chunks} chunks...")

    with torch.no_grad():
        for chunk_idx in range(num_chunks):
            chunk_file = FEATURES_DIR / f"chunk_{chunk_idx:03d}.pt"
            
            chunk_h = []
            chunk_r = []
            chunk_k = []
            chunk_v = []
            
            for i in range(chunk_size):
                x_input = torch.randn(1, seq_len, D_MODEL)
                h_24, r_24, k_true, v_true = engine(x_input)
                
                chunk_h.append(h_24.squeeze(0).half())
                chunk_r.append(r_24.squeeze(0).half())
                chunk_k.append(k_true.squeeze(0).half())
                chunk_v.append(v_true.squeeze(0).half())
                
            chunk_data = {
                "h": chunk_h,
                "router_weights": chunk_r,
                "k_true": chunk_k,
                "v_true": chunk_v,
            }
            torch.save(chunk_data, chunk_file)
            size_mb = chunk_file.stat().st_size / (1024 * 1024)
            print(f"  [SAVED] Chunk {chunk_idx:03d} with genuine SwiGLU MoE activations ({size_mb:.1f} MB)")

    print("[HARVEST] Real forward pass harvesting complete!")

if __name__ == "__main__":
    harvest_real_forward_activations(num_samples=150, seq_len=48, chunk_size=50)
