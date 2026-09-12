"""
PyTorch Architecture: Hybrid Expert-Linked MoE MTP Drafter vs Dense Baseline
Target Model: Qwen 3.6 35B A3B MTP
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import D_MODEL, NUM_EXPERTS, TOP_K_EXPERTS

MTP_VOCAB_SIZE = 2048
MTP_DRAFT_RANK = 64

class DenseMTPDrafter(nn.Module):
    """
    Standard monolithic Dense MTP Drafter (matching Qwen 3.6 / DeepSeek-V3 MTP design).
    Fuses token embedding and hidden state through a dense projection matrix.
    """
    def __init__(self, d_model=D_MODEL, vocab_size=MTP_VOCAB_SIZE):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        self.enorm = nn.LayerNorm(d_model)
        self.hnorm = nn.LayerNorm(d_model)
        
        # Matches blk.40.nextn.eh_proj: [4096, 2048]
        self.eh_proj = nn.Linear(d_model * 2, d_model, bias=False)
        self.mlp_mid = nn.Linear(d_model, d_model, bias=False)
        self.act = nn.GELU()
        
        self.head_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Weight initialization
        nn.init.normal_(self.eh_proj.weight, std=0.02)
        nn.init.normal_(self.mlp_mid.weight, std=0.02)
        nn.init.normal_(self.lm_head.weight, std=0.02)

    def forward(self, e, h):
        """
        e: [B, T, D_MODEL] - Token embedding at position t
        h: [B, T, D_MODEL] - Final-layer hidden state at position t
        """
        e_norm = self.enorm(e)
        h_norm = self.hnorm(h)
        eh = torch.cat([e_norm, h_norm], dim=-1)
        
        # Dense trunk projection with residual
        proj = self.eh_proj(eh)
        z = proj + self.mlp_mid(self.act(proj))
        
        logits = self.lm_head(self.head_norm(z))
        return logits, z


class MicroMTPExpert(nn.Module):
    """
    Individual non-linear draft micro-expert paired with a base model expert.
    Provides domain-specialized token transition predictions (AST, syntax, logic).
    """
    def __init__(self, in_dim=D_MODEL * 2, out_dim=D_MODEL, rank=MTP_DRAFT_RANK):
        super().__init__()
        self.down = nn.Linear(in_dim, rank, bias=False)
        self.mid = nn.Linear(rank, rank, bias=False)
        self.up = nn.Linear(rank, out_dim, bias=False)
        self.act1 = nn.GELU()
        self.act2 = nn.GELU()
        
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.mid.weight, a=math.sqrt(5))
        # Initialized near zero so experts start as additive residual boosters
        nn.init.normal_(self.up.weight, std=0.005)

    def forward(self, x):
        z1 = self.act1(self.down(x))
        z2 = self.act2(self.mid(z1)) + z1
        return self.up(z2)


class HybridMoEMTPDrafter(nn.Module):
    """
    Hybrid Shared-Base + Expert-Linked Micro-MoE MTP Drafter.
    Combines:
    1. Full-capacity Dense Foundation (guarantees general conversational accuracy)
    2. 64 Deep Micro-Experts inheriting router gating from the base model
       (specializes in code syntax, math symbols, and tool calling)
    """
    def __init__(
        self,
        d_model=D_MODEL,
        vocab_size=MTP_VOCAB_SIZE,
        num_experts=NUM_EXPERTS,
        top_k=TOP_K_EXPERTS,
        rank=MTP_DRAFT_RANK
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.num_experts = num_experts
        self.top_k = top_k
        
        self.enorm = nn.LayerNorm(d_model)
        self.hnorm = nn.LayerNorm(d_model)
        
        # Dense Foundation Trunk (1.0x scale)
        self.eh_proj = nn.Linear(d_model * 2, d_model, bias=False)
        self.mlp_mid = nn.Linear(d_model, d_model, bias=False)
        self.act = nn.GELU()
        
        nn.init.normal_(self.eh_proj.weight, std=0.02)
        nn.init.normal_(self.mlp_mid.weight, std=0.02)

        # Micro-Experts Fleet (64 specialized draft micro-blocks)
        self.experts = nn.ModuleList([
            MicroMTPExpert(in_dim=d_model * 2, out_dim=d_model, rank=rank)
            for _ in range(num_experts)
        ])
        
        # Learnable per-expert draft authority gain
        self.expert_gain = nn.Parameter(torch.ones(num_experts))
        
        self.head_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        nn.init.normal_(self.lm_head.weight, std=0.02)

    def forward(self, e, h, router_weights=None):
        """
        e: [B, T, D_MODEL] - Token embedding at position t
        h: [B, T, D_MODEL] - Final hidden state from base model at position t
        router_weights: [B, T, NUM_EXPERTS] - Softmax router probabilities from base model at position t
        """
        B, T, D = h.shape
        e_norm = self.enorm(e)
        h_norm = self.hnorm(h)
        eh = torch.cat([e_norm, h_norm], dim=-1)  # [B, T, 4096]
        
        # 1. Base Dense Trunk Output
        proj = self.eh_proj(eh)
        base_z = proj + self.mlp_mid(self.act(proj))

        if router_weights is None:
            logits = self.lm_head(self.head_norm(base_z))
            return logits, base_z

        # 2. Inherited Top-K Expert Dispatch
        top_weights, top_indices = torch.topk(router_weights, self.top_k, dim=-1)
        top_weights = top_weights / (top_weights.sum(dim=-1, keepdim=True) + 1e-9)

        expert_z = torch.zeros_like(base_z)

        eh_flat = eh.view(-1, D * 2)
        top_indices_flat = top_indices.view(-1, self.top_k)
        top_weights_flat = top_weights.view(-1, self.top_k)

        for k in range(self.top_k):
            indices_k = top_indices_flat[:, k]
            weights_k = top_weights_flat[:, k].unsqueeze(1)

            unique_experts = torch.unique(indices_k)
            for exp_idx in unique_experts:
                mask = (indices_k == exp_idx)
                if not mask.any():
                    continue
                sub_eh = eh_flat[mask]
                sub_z = self.experts[exp_idx.item()](sub_eh)
                w_sub = weights_k[mask] * self.expert_gain[exp_idx.item()]
                expert_z.view(-1, D)[mask] += w_sub * sub_z

        total_z = base_z + expert_z
        logits = self.lm_head(self.head_norm(total_z))
        return logits, total_z

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    dense_mtp = DenseMTPDrafter()
    hybrid_mtp = HybridMoEMTPDrafter()
    
    print(f"Dense MTP Drafter Parameters: {count_parameters(dense_mtp):,}")
    print(f"Hybrid MoE MTP Drafter Parameters: {count_parameters(hybrid_mtp):,}")
    
    # Active parameters per draft token
    active_params = count_parameters(dense_mtp) + 8 * count_parameters(hybrid_mtp.experts[0])
    print(f"Hybrid MoE Active Parameters per Token: {active_params:,}")
    
    # Smoke test forward
    dummy_e = torch.randn(2, 32, D_MODEL)
    dummy_h = torch.randn(2, 32, D_MODEL)
    dummy_r = F.softmax(torch.randn(2, 32, NUM_EXPERTS), dim=-1)
    
    out_dense, _ = dense_mtp(dummy_e, dummy_h)
    out_moe, _ = hybrid_mtp(dummy_e, dummy_h, dummy_r)
    print(f"Dense Logits Shape: {out_dense.shape}")
    print(f"Hybrid MoE Logits Shape: {out_moe.shape} [SUCCESS]")
