from Fusion_layer import FairFusion
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from typing import Optional, List, Dict, Tuple, Any
from Type_Token_manager import TypeTokenManager


# -------------------------
# Router: LayerNorm + noise + dropout -> logits
# -------------------------
class Router(nn.Module):

    def __init__(self,
                 fusion_dim: int,
                 n_experts: int,
                 noise_std: float = 1.0,
                 dropout: float = 0.1):
        super().__init__()
        self.noise_std = noise_std
        self.net = nn.Sequential(nn.LayerNorm(fusion_dim),
                                 nn.Linear(fusion_dim, fusion_dim // 2),
                                 nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(fusion_dim // 2, n_experts))

    def forward(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:
        logits = self.net(x)
        if training and self.noise_std > 0.0:
            noise = torch.randn_like(logits) * (self.noise_std /
                                                (logits.size(-1)**0.5))
            logits = logits + noise
        weights = F.softmax(logits, dim=-1)
        return weights, logits  # return logits for auxiliary balancing calculation


# -------------------------
# Sparse dispatch: Top-1 routing (runs only selected experts)
# -------------------------
def top1_dispatch_and_combine(experts: nn.ModuleList, fused: torch.Tensor,
                              top_idx: torch.LongTensor) -> torch.Tensor:
    """
    fused: [B, D]
    top_idx: [B] ints in [0, n_experts)
    Return: refined [B, D] computed by running only required expert slices.
    """
    device = fused.device
    B, D = fused.shape
    n_experts = len(experts)
    refined = torch.zeros_like(fused, device=device)

    # For each expert, run on the subset of samples that chose it
    for e in range(n_experts):
        mask = (top_idx == e)
        if not mask.any():
            continue
        idx = mask.nonzero(as_tuple=False).squeeze(1)
        inp = fused.index_select(0, idx)  # [k, D]
        out = experts[e](inp)  # [k, D]
        refined.index_copy_(0, idx, out)  # put back
    return refined


# -------------------------
# Load-balancing loss (auxiliary)
# -------------------------
def load_balance_loss(weights_logits: torch.Tensor) -> torch.Tensor:
    """
    weights_logits: the router logits BEFORE softmax or AFTER? 
    We'll accept logits returned alongside weights; but easier use softmax weights.
    Input expected: logits (not necessary) but we can recompute.
    Simpler: pass the softmax weights (B, n_experts).
    L_balance = n_experts * sum(mean(weights, dim=0) * log(mean(weights, dim=0))) (entropy-like)
    Another common is: encourage mean router load close to uniform -> MSE with 1/n
    """
    weights = weights_logits  # here we expect softmax weights [B, E]
    mean_usage = weights.mean(dim=0)  # [E]
    E = weights.size(1)
    ideal = torch.full_like(mean_usage, 1.0 / E)
    # MSE load penalty (simple, stable)
    loss = F.mse_loss(mean_usage, ideal)
    return loss


class PsychometricMoE(nn.Module):

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_traits: int = 5,
        num_char_types: int = 10,
        num_numeric_features: int = 50,
        fusion_dim: int = 512,
        numeric_hidden: int = 128,
        mod_dropout: float = 0.3,
        n_experts: int = 4,
        type_embed_dim: int = 64,
        router_noise: float = 1.0,
    ):
        super().__init__()
        self.mod_dropout = mod_dropout
        self.n_experts = n_experts

        # Encoders
        self.transformer = AutoModel.from_pretrained(model_name)
        self.text_dim = self.transformer.config.hidden_size
        self.numeric_processor = nn.Sequential(
            nn.Linear(num_numeric_features, numeric_hidden),
            nn.LayerNorm(numeric_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # Type tokens
        self.type_manager = TypeTokenManager(type_embed_dim)
        self.type_to_text = nn.Linear(type_embed_dim, self.text_dim)
        self.type_to_num = nn.Linear(type_embed_dim, numeric_hidden)

        # Fair fusion
        self.fusion = FairFusion(self.text_dim, numeric_hidden, fusion_dim)

        # Experts + router
        self.experts = nn.ModuleList(
            [Expert(fusion_dim) for _ in range(n_experts)])
        self.router = Router(fusion_dim,
                             n_experts,
                             noise_std=router_noise,
                             dropout=0.1)

        # Heads
        self.trait_head = nn.Sequential(nn.Linear(fusion_dim, 256), nn.ReLU(),
                                        nn.Linear(256, num_traits))
        self.irt_head = nn.Sequential(nn.Linear(fusion_dim, 256), nn.ReLU(),
                                      nn.Linear(256, 3))
        self.char_head = nn.Sequential(nn.Linear(fusion_dim, 256), nn.ReLU(),
                                       nn.Linear(256, num_char_types))

        # placeholders
        self.register_buffer("zero_text", torch.zeros(1, self.text_dim))
        self.register_buffer("zero_num", torch.zeros(1, numeric_hidden))

    def _infer_B(self, input_ids, numeric_features):
        if input_ids is not None:
            return input_ids.size(0)
        if numeric_features is not None:
            return numeric_features.size(0)
        raise ValueError("At least one modality must be provided.")

    def _get_text_emb(self, input_ids, attention_mask, B):
        if input_ids is None:
            return self.zero_text.expand(B, -1)
        out = self.transformer(input_ids=input_ids,
                               attention_mask=attention_mask)
        return out.last_hidden_state[:, 0, :]

    def _get_num_emb(self, numeric_features, B, device):
        if numeric_features is None:
            return self.zero_num.expand(B, -1).to(device)
        return self.numeric_processor(numeric_features)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        numeric_features: Optional[torch.FloatTensor] = None,
        type_strings: Optional[List[str]] = None,
        expert_type: Optional[List[int]] = None,
        use_hard_top1: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str,
                                                              torch.Tensor]]:
        """
        Returns: trait, irt, char, aux_losses dict
        aux_losses contains:
            - 'load_balance': scalar balancing loss
            - 'router_entropy': optional router entropy (for diagnostics)
        """

        device = next(self.parameters()).device
        B = self._infer_B(input_ids, numeric_features)

        # get modal embeddings
        text_emb = self._get_text_emb(input_ids, attention_mask, B).to(device)
        num_emb = self._get_num_emb(numeric_features, B, device).to(device)

        # type conditioning
        if type_strings is not None:
            type_emb = self.type_manager(type_strings).to(device)
            text_emb = text_emb + self.type_to_text(type_emb)
            num_emb = num_emb + self.type_to_num(type_emb)

        # modality dropout (sample-level)
        if self.training and self.mod_dropout > 0.0:
            keep = 1.0 - self.mod_dropout
            mask_t = torch.bernoulli(torch.full(
                (B, 1), keep, device=device)).to(text_emb.dtype)
            mask_n = torch.bernoulli(torch.full(
                (B, 1), keep, device=device)).to(num_emb.dtype)
            text_emb = text_emb * mask_t
            num_emb = num_emb * mask_n

        # fusion
        fused = self.fusion(text_emb, num_emb)  # [B, D]

        # router -> weights, logits
        weights, logits = self.router(
            fused, training=self.training)  # weights: [B, E]

        aux: Dict[str, torch.Tensor] = {}
        # compute load balance loss from soft weights (MSE-based)
        aux['load_balance'] = load_balance_loss(weights)

        # choose top-1 indexes
        top1 = weights.argmax(dim=-1)  # [B]

        # if user forced expert_type, respect that (allow list or tensor)
        if expert_type is not None:
            if not torch.is_tensor(expert_type):
                expert_type = torch.tensor(expert_type,
                                           dtype=torch.long,
                                           device=device)
            if expert_type.dim() == 0:
                expert_type = expert_type.unsqueeze(0)
            if expert_type.size(0) != B:
                if expert_type.size(0) == 1:
                    expert_type = expert_type.expand(B)
                else:
                    raise ValueError(
                        "expert_type length must be 1 or batch size")
            top1 = expert_type.to(device)

        # If use_hard_top1 False (soft), we can combine expert outputs weighted (costly).
        # For production we prefer hard top1 (sparse): run only the selected experts.
        if use_hard_top1:
            refined = top1_dispatch_and_combine(self.experts, fused, top1)
        else:
            # lightweight fallback: run all experts and weighted-sum (slower but simpler)
            expert_outs = torch.stack(
                [self.experts[i](fused) for i in range(self.n_experts)],
                dim=1)  # [B, E, D]
            refined = (expert_outs * weights.unsqueeze(-1)).sum(dim=1)

        # heads
        trait = self.trait_head(refined)
        irt = self.irt_head(refined)
        char = self.char_head(refined)

        # diagnostics
        aux['router_entropy'] = -(weights * (weights + 1e-12).log()).sum(
            dim=-1).mean()  # mean entropy

        return trait, irt, char, aux
