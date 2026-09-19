"""构造模型适配器使用的标准策略梯度目标。"""

from __future__ import annotations

from torch import Tensor


def reinforce_loss(cost: Tensor, log_likelihood: Tensor) -> Tensor:
    """使用中心化批次 baseline，返回批次平均的 REINFORCE 目标。"""
    if cost.ndim != 1 or log_likelihood.shape != cost.shape:
        raise ValueError("cost and log_likelihood must be vectors with equal shape")
    advantage = (cost - cost.mean()).detach()
    return (advantage * log_likelihood).mean()
