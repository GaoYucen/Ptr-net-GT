"""Forest-aware 条件边策略共享的概率和候选摘要工具。"""

from __future__ import annotations

import math

import torch
from torch import Tensor


def categorical_entropy(log_probabilities: Tensor) -> Tensor:
    """计算分类熵，并避免产生 ``0 * -inf``。"""
    finite = torch.isfinite(log_probabilities)
    safe_log_probabilities = torch.where(
        finite, log_probabilities, torch.zeros_like(log_probabilities)
    )
    probabilities = safe_log_probabilities.exp() * finite.to(log_probabilities.dtype)
    return -(probabilities * safe_log_probabilities).sum(dim=-1)


def masked_conditional_log_probabilities(
    logits: Tensor,
    mask: Tensor,
    temperature: float,
) -> Tensor:
    """分别归一化每个 tail 行，不在不同 tail 的 logits 之间进行比较。"""
    if logits.shape != mask.shape or logits.ndim != 3 or mask.dtype != torch.bool:
        raise ValueError("logits and mask must have shape (batch, tails, heads)")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    valid_rows = ~mask.all(dim=-1, keepdim=True)
    safe_logits = logits.masked_fill(mask & valid_rows, -torch.inf)
    safe_logits = torch.where(valid_rows, safe_logits, torch.zeros_like(logits))
    log_p = torch.log_softmax(safe_logits / temperature, dim=-1)
    return log_p.masked_fill(mask | ~valid_rows, -torch.inf)


def native_head_summary(log_p: Tensor, distances: Tensor) -> Tensor:
    """返回便于分离尺度的原生 head 分布摘要。

    三个特征依次为归一化熵、归一化预期边代价和最大条件概率。
    完全被 mask 的 tail 行映射为零。
    """
    if log_p.shape != distances.shape or log_p.ndim != 3:
        raise ValueError("log_p and distances must have shape (batch, tails, heads)")
    finite = torch.isfinite(log_p)
    probabilities = torch.where(finite, log_p.exp(), torch.zeros_like(log_p))
    safe_log_p = torch.where(finite, log_p, torch.zeros_like(log_p))
    entropy_terms = probabilities * safe_log_p
    entropy = -entropy_terms.sum(dim=-1) / math.log(max(log_p.size(-1), 2))
    expected_cost = (probabilities * distances).sum(dim=-1) / math.sqrt(2.0)
    best_probability = probabilities.amax(dim=-1)
    summary = torch.stack((entropy, expected_cost, best_probability), dim=-1)
    valid_rows = finite.any(dim=-1, keepdim=True)
    return torch.where(valid_rows, summary, torch.zeros_like(summary))


def joint_action_entropy(tail_log_p: Tensor, head_log_p: Tensor) -> Tensor:
    """计算分解策略的 H(tail) + E_tail[H(head | tail)]。"""
    if tail_log_p.ndim != 2 or head_log_p.ndim != 3:
        raise ValueError("tail and head log probabilities have incompatible ranks")
    if head_log_p.shape[:2] != tail_log_p.shape:
        raise ValueError("tail and conditional head shapes are incompatible")
    tail_finite = torch.isfinite(tail_log_p)
    tail_probabilities = torch.where(tail_finite, tail_log_p.exp(), torch.zeros_like(tail_log_p))
    safe_tail_log_p = torch.where(tail_finite, tail_log_p, torch.zeros_like(tail_log_p))
    tail_terms = tail_probabilities * safe_tail_log_p
    head_finite = torch.isfinite(head_log_p)
    head_probabilities = torch.where(head_finite, head_log_p.exp(), torch.zeros_like(head_log_p))
    safe_head_log_p = torch.where(head_finite, head_log_p, torch.zeros_like(head_log_p))
    head_terms = head_probabilities * safe_head_log_p
    head_entropy = -head_terms.sum(dim=-1)
    return -tail_terms.sum(dim=-1) + (tail_probabilities * head_entropy).sum(dim=-1)
