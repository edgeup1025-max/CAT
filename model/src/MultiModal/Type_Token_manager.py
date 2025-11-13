import torch
import torch.nn as nn
from typing import List


class TypeTokenManager(nn.Module):

    def __init__(self, embed_dim: int = 64):
        super().__init__()
        self.embed_dim = embed_dim
        self.type2id = {}
        self.embedding = nn.Embedding(1, embed_dim)

    def add_type(self, type_str: str):
        if type_str in self.type2id:
            return self.type2id[type_str]
        new_id = len(self.type2id)
        self.type2id[type_str] = new_id
        new_emb = nn.Embedding(new_id + 1, self.embed_dim)
        new_emb.weight.data[:-1] = self.embedding.weight.data.clone()
        new_emb.weight.data[-1].uniform_(-0.02, 0.02)
        self.embedding = new_emb
        return new_id

    def encode_types(self, type_list: List[str]) -> torch.LongTensor:
        ids = [self.add_type(t) for t in type_list]
        return torch.tensor(ids, dtype=torch.long)

    def forward(self, type_list: List[str]) -> torch.Tensor:
        ids = self.encode_types(type_list).to(
            next(self.parameters()).device if len(list(self.parameters())) >
            0 else "cpu")
        return self.embedding(ids)
