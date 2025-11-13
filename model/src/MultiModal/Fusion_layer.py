import torch
import torch.nn as nn
import torch.nn.functional as F


class FairFusion(nn.Module):

    def __init__(self, text_dim: int, num_dim: int, fusion_dim: int = 512):
        super().__init__()
        self.t_proj = nn.Linear(text_dim, fusion_dim)
        self.n_proj = nn.Linear(num_dim, fusion_dim)
        self.b_proj = nn.Bilinear(fusion_dim, fusion_dim, fusion_dim)
        self.gate = nn.Sequential(nn.Linear(fusion_dim * 3, fusion_dim),
                                  nn.ReLU(), nn.Linear(fusion_dim, 3))

    def forward(self, text_emb: torch.Tensor,
                num_emb: torch.Tensor) -> torch.Tensor:
        t = self.t_proj(text_emb)  # [B, D]
        n = self.n_proj(num_emb)  # [B, D]
        b = self.b_proj(t, n)  # [B, D]
        gate_in = torch.cat([t, n, b], dim=-1)
        logits = self.gate(gate_in)  # [B, 3]
        weights = F.softmax(logits, dim=-1)  # [B, 3]
        w_t = weights[:, 0].unsqueeze(-1)
        w_n = weights[:, 1].unsqueeze(-1)
        w_b = weights[:, 2].unsqueeze(-1)
        fused = w_t * t + w_n * n + w_b * b
        return fused
