import math

import torch
from torch import nn


def reshape_by_heads(qkv: torch.Tensor, head_num: int) -> torch.Tensor:
    batch_size, n, _ = qkv.size()
    qkv_dim = qkv.size(-1) // head_num
    return qkv.reshape(batch_size, n, head_num, qkv_dim).transpose(1, 2)


def multi_head_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, rank3_ninf_mask: torch.Tensor | None = None) -> torch.Tensor:
    score = torch.matmul(q, k.transpose(2, 3))
    score_scaled = score / math.sqrt(q.size(-1))
    if rank3_ninf_mask is not None:
        score_scaled = score_scaled + rank3_ninf_mask[:, None, :, :]
    weights = torch.softmax(score_scaled, dim=3)
    out = torch.matmul(weights, v)
    return out.transpose(1, 2).reshape(q.size(0), q.size(2), -1)


class AddAndNormalizationModule(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.norm = nn.InstanceNorm1d(embedding_dim, affine=True)

    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        added = input1 + input2
        return self.norm(added.transpose(1, 2)).transpose(1, 2)


class FeedForwardModule(nn.Module):
    def __init__(self, embedding_dim: int, ff_hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SymNCOEncoderLayer(nn.Module):
    def __init__(self, embedding_dim: int, head_num: int, qkv_dim: int, ff_hidden_dim: int):
        super().__init__()
        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)
        self.add_norm_1 = AddAndNormalizationModule(embedding_dim)
        self.feed_forward = FeedForwardModule(embedding_dim, ff_hidden_dim)
        self.add_norm_2 = AddAndNormalizationModule(embedding_dim)
        self.head_num = head_num

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = reshape_by_heads(self.Wq(x), self.head_num)
        k = reshape_by_heads(self.Wk(x), self.head_num)
        v = reshape_by_heads(self.Wv(x), self.head_num)
        out_concat = multi_head_attention(q, k, v)
        out = self.multi_head_combine(out_concat)
        out = self.add_norm_1(x, out)
        return self.add_norm_2(out, self.feed_forward(out))


class SymNCOGraphEncoder(nn.Module):
    def __init__(self, embedding_dim: int, n_heads: int, n_layers: int, ff_hidden_dim: int = 512):
        super().__init__()
        self.init_embed = nn.Linear(2, embedding_dim)
        qkv_dim = embedding_dim // n_heads
        self.layers = nn.ModuleList(
            [SymNCOEncoderLayer(embedding_dim, n_heads, qkv_dim, ff_hidden_dim) for _ in range(n_layers)]
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        assert mask is None, "SymNCOGraphEncoder does not support masks"
        h = self.init_embed(x)
        for layer in self.layers:
            h = layer(h)
        return h, h.mean(dim=1)


__all__ = ["SymNCOGraphEncoder"]