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


class DeepSpecialistExpert(nn.Module):
    """
    Deep 2-layer non-linear specialist expert with internal residual connections
    and decoupled K and V pathways. Operates as a precision residual corrector
    for complex AST, syntax, and domain structures.
    """
    def __init__(self, d_model=D_MODEL, d_kv=TOTAL_TARGET_KV_DIM, rank=PROJECTOR_RANK):
        super().__init__()
        # Key pathway: down -> mid (residual) -> up
        self.down_k = nn.Linear(d_model, rank, bias=False)
        self.mid_k = nn.Linear(rank, rank, bias=False)
        self.up_k = nn.Linear(rank, d_kv, bias=False)
        self.act_k1 = nn.GELU()
        self.act_k2 = nn.GELU()

        # Value pathway: down -> mid (residual) -> up
        self.down_v = nn.Linear(d_model, rank, bias=False)
        self.mid_v = nn.Linear(rank, rank, bias=False)
        self.up_v = nn.Linear(rank, d_kv, bias=False)
        self.act_v1 = nn.GELU()
        self.act_v2 = nn.GELU()

        nn.init.kaiming_uniform_(self.down_k.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.down_v.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.mid_k.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.mid_v.weight, a=math.sqrt(5))
        # Small initialization so specialists start as additive zero-centered residual correctors
        nn.init.normal_(self.up_k.weight, std=0.005)
        nn.init.normal_(self.up_v.weight, std=0.005)

    def forward(self, h):
        # Key transformation with residual skip
        z_k1 = self.act_k1(self.down_k(h))
        z_k = self.act_k2(self.mid_k(z_k1)) + z_k1
        k = self.up_k(z_k)

        # Value transformation with residual skip
        z_v1 = self.act_v1(self.down_v(h))
        z_v = self.act_v2(self.mid_v(z_v1)) + z_v1
        v = self.up_v(z_v)

        return k, v

MicroExpert = DeepSpecialistExpert  # Backward compatibility alias


class ExpertLinkedMoEKVProjector(nn.Module):
    """
    Hybrid Shared-Base + Deep-Specialist MoE Projector.
    Combines:
    1. Full-capacity 1.0x Dense Shared Base (covers broad language and general knowledge)
    2. Deep 2-layer Non-linear Specialist Fleet (provides precision AST/code residual corrections)
    """
    def __init__(
        self,
        d_model=D_MODEL,
        d_kv=TOTAL_TARGET_KV_DIM,
        num_experts=NUM_EXPERTS,
        top_k=TOP_K_EXPERTS,
        rank=PROJECTOR_RANK,
        bypass_scale=1.0
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.rank = rank
        self.bypass_scale = bypass_scale
        self.norm = nn.LayerNorm(d_model)
        
        # Full-capacity 1.0x shared base projector
        self.shared_bypass_k = nn.Linear(d_model, d_kv, bias=False)
        self.shared_bypass_v = nn.Linear(d_model, d_kv, bias=False)
        nn.init.normal_(self.shared_bypass_k.weight, std=0.02)
        nn.init.normal_(self.shared_bypass_v.weight, std=0.02)

        # Deep specialist fleet
        self.experts = nn.ModuleList([
            DeepSpecialistExpert(d_model, d_kv, rank=rank) for _ in range(num_experts)
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
        
        # Full-capacity 1.0x base shared representation
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
