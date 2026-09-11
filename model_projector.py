"""
PyTorch Architecture: Expert-Linked MoE KV Projector (LLKVApprox-MoE)
Target Model: Qwen 3.6 35B A3B
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import D_MODEL, TOTAL_TARGET_KV_DIM, NUM_EXPERTS, TOP_K_EXPERTS, PROJECTOR_RANK

class DenseLinearProjector(nn.Module):
    """
    Baseline monolithic linear projector (Kishida / LLKVApprox style).
    Averages all tokens through single global projection matrices.
    """
    def __init__(self, d_model=D_MODEL, d_kv=TOTAL_TARGET_KV_DIM):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.proj_k = nn.Linear(d_model, d_kv, bias=False)
        self.proj_v = nn.Linear(d_model, d_kv, bias=False)
        
        nn.init.normal_(self.proj_k.weight, std=0.02)
        nn.init.normal_(self.proj_v.weight, std=0.02)

    def forward(self, h):
        # h: [B, T, D_MODEL]
        h_norm = self.norm(h)
        k_pred = self.proj_k(h_norm)
        v_pred = self.proj_v(h_norm)
        return k_pred, v_pred


class MicroExpert(nn.Module):
    """
    Individual non-linear micro-expert paired 1:1 with a base model expert.
    Features decoupled K and V latent bottlenecks to eliminate cross-subspace interference.
    """
    def __init__(self, d_model=D_MODEL, d_kv=TOTAL_TARGET_KV_DIM, rank=PROJECTOR_RANK):
        super().__init__()
        # Dedicated Key projection pipeline (retrieval address subspace)
        self.down_k = nn.Linear(d_model, rank, bias=False)
        self.up_k = nn.Linear(rank, d_kv, bias=False)
        self.act_k = nn.GELU()

        # Dedicated Value projection pipeline (information content subspace)
        self.down_v = nn.Linear(d_model, rank, bias=False)
        self.up_v = nn.Linear(rank, d_kv, bias=False)
        self.act_v = nn.GELU()

        nn.init.kaiming_uniform_(self.down_k.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.down_v.weight, a=math.sqrt(5))
        nn.init.normal_(self.up_k.weight, std=0.02)
        nn.init.normal_(self.up_v.weight, std=0.02)

    def forward(self, h):
        z_k = self.act_k(self.down_k(h))
        z_v = self.act_v(self.down_v(h))
        k = self.up_k(z_k)
        v = self.up_v(z_v)
        return k, v


class ExpertLinkedMoEKVProjector(nn.Module):
    """
    Expert-Linked MoE KV Projector.
    Routes token representations to the corresponding micro-experts based on
    the base model's router weights, preserving code and domain specialization.
    """
    def __init__(
        self,
        d_model=D_MODEL,
        d_kv=TOTAL_TARGET_KV_DIM,
        num_experts=NUM_EXPERTS,
        top_k=TOP_K_EXPERTS,
        rank=PROJECTOR_RANK,
        bypass_scale=0.5
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.rank = rank
        self.bypass_scale = bypass_scale
        self.norm = nn.LayerNorm(d_model)
        
        # Shared global base bypass
        self.shared_bypass_k = nn.Linear(d_model, d_kv, bias=False)
        self.shared_bypass_v = nn.Linear(d_model, d_kv, bias=False)
        nn.init.normal_(self.shared_bypass_k.weight, std=0.02)
        nn.init.normal_(self.shared_bypass_v.weight, std=0.02)

        # Scaled micro-experts fleet
        self.experts = nn.ModuleList([
            MicroExpert(d_model, d_kv, rank=rank) for _ in range(num_experts)
        ])

        # Learnable per-expert adaptive gain
        self.expert_gain = nn.Parameter(torch.ones(num_experts))

    def forward(self, h, router_weights=None):
        """
        h: [B, T, D_MODEL] - Hidden state from layer 24
        router_weights: [B, T, NUM_EXPERTS] - Softmax gating probabilities from layer 24
        """
        B, T, D = h.shape
        h_norm = self.norm(h)
        
        # Scaled base shared representation
        base_k = self.bypass_scale * self.shared_bypass_k(h_norm)
        base_v = self.bypass_scale * self.shared_bypass_v(h_norm)

        if router_weights is None:
            # Fallback to base representation if router weights omitted
            return base_k, base_v

        # Extract top-k active experts and normalized weights
        top_weights, top_indices = torch.topk(router_weights, self.top_k, dim=-1)
        top_weights = top_weights / (top_weights.sum(dim=-1, keepdim=True) + 1e-9)

        expert_k = torch.zeros_like(base_k)
        expert_v = torch.zeros_like(base_v)

        # Flatten batch and seq dimensions for sparse expert dispatch
        h_flat = h_norm.view(-1, D)
        top_indices_flat = top_indices.view(-1, self.top_k)
        top_weights_flat = top_weights.view(-1, self.top_k)

        # Dispatch to active micro-experts
        for k in range(self.top_k):
            indices_k = top_indices_flat[:, k]
            weights_k = top_weights_flat[:, k].unsqueeze(1)

            # Unique active experts in this rank slot
            unique_experts = torch.unique(indices_k)
            for exp_idx in unique_experts:
                mask = (indices_k == exp_idx)
                if not mask.any():
                    continue
                h_sub = h_flat[mask]
                k_sub, v_sub = self.experts[exp_idx.item()](h_sub)
                w_sub = weights_k[mask] * self.expert_gain[exp_idx.item()]
                
                expert_k.view(-1, base_k.shape[-1])[mask] += w_sub * k_sub
                expert_v.view(-1, base_v.shape[-1])[mask] += w_sub * v_sub

        total_k = base_k + expert_k
        total_v = base_v + expert_v
        return total_k, total_v

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    dense = DenseLinearProjector()
    moe = ExpertLinkedMoEKVProjector()
    print(f"Dense Linear Projector Parameters: {count_parameters(dense):,}")
    print(f"Expert-Linked MoE Projector Parameters: {count_parameters(moe):,}")
    
    # Smoke test forward pass
    dummy_h = torch.randn(2, 64, D_MODEL)
    dummy_r = F.softmax(torch.randn(2, 64, NUM_EXPERTS), dim=-1)
    
    k_moe, v_moe = moe(dummy_h, dummy_r)
    print(f"MoE Output Shapes: K={k_moe.shape}, V={v_moe.shape} [SUCCESS]")
