"""Forest-aware 边策略的统一接口与张量化执行器。

该模块只规定成功范式的共同逻辑：先为每个合法 tail 生成条件 head
提案，再比较所有 tail，最后执行一条合法边。AM、PtrNet、GPN 只需要
实现各自风格的两个评分函数和一步上下文更新。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar, runtime_checkable

import torch
from torch import Generator, Tensor

from groupopt.framework.neural import BatchedConstructionProcess, ConstructionOutput

StateT = TypeVar("StateT")


@dataclass(frozen=True, slots=True)
class ForestHeadProposal:
    """所有 ``tail -> head`` 条件分布及其供 tail 评分使用的特征。"""

    log_probabilities: Tensor
    features: Tensor

    def validate(self, batch_size: int, node_count: int) -> None:
        if self.log_probabilities.shape != (batch_size, node_count, node_count):
            raise ValueError("head proposal must have shape (batch, tails, heads)")
        if self.features.ndim != 3 or self.features.shape[:2] != (
            batch_size,
            node_count,
        ):
            raise ValueError("head proposal features must start with (batch, tails)")


@runtime_checkable
class ForestAwareDecoder(Protocol[StateT]):
    """宿主模型接入完整 GroupOpt 时必须实现的三个步骤。"""

    def score_heads(self, state: StateT, pair_mask: Tensor) -> ForestHeadProposal:
        """为每个合法 tail 返回条件 head 分布和候选边特征。"""
        ...

    def score_tails(
        self,
        state: StateT,
        head_proposal: ForestHeadProposal,
        tail_mask: Tensor,
    ) -> Tensor:
        """比较所有开放路径端点，返回归一化的 tail 对数概率。"""
        ...

    def update_context(self, selected_tail: Tensor, selected_head: Tensor) -> None:
        """在边被接受后更新宿主 decoder 的递归上下文。"""
        ...


@dataclass(slots=True)
class CallableForestAwareDecoder(Generic[StateT]):
    """让模型用闭包接入接口，同时保持原参数名称和 checkpoint 兼容性。"""

    head_scorer: Callable[[StateT, Tensor], ForestHeadProposal]
    tail_scorer: Callable[[StateT, ForestHeadProposal, Tensor], Tensor]
    context_updater: Callable[[Tensor, Tensor], None]

    def score_heads(self, state: StateT, pair_mask: Tensor) -> ForestHeadProposal:
        return self.head_scorer(state, pair_mask)

    def score_tails(
        self,
        state: StateT,
        head_proposal: ForestHeadProposal,
        tail_mask: Tensor,
    ) -> Tensor:
        return self.tail_scorer(state, head_proposal, tail_mask)

    def update_context(self, selected_tail: Tensor, selected_head: Tensor) -> None:
        self.context_updater(selected_tail, selected_head)


def decode_forest_edges(
    coordinates: Tensor,
    process: BatchedConstructionProcess[Tensor, StateT],
    decoder: ForestAwareDecoder[StateT],
    decode_type: str,
    generator: Generator | None,
) -> ConstructionOutput:
    """执行 ``p(tail|Forest) p(head|tail,Forest)`` 的统一 rollout。"""
    state = process.initial_state(coordinates)
    batch_size, node_count, _ = coordinates.shape
    batch_index = torch.arange(batch_size, device=coordinates.device)
    tails: list[Tensor] = []
    heads: list[Tensor] = []
    selected_log_probabilities: list[Tensor] = []
    tail_entropies: list[Tensor] = []
    action_entropies: list[Tensor] = []

    while not process.is_terminal(state):
        pair_mask = process.action_mask(state)
        proposal = decoder.score_heads(state, pair_mask)
        proposal.validate(batch_size, node_count)
        tail_mask = process.base_mask(state)
        tail_log_p = decoder.score_tails(state, proposal, tail_mask)
        if tail_log_p.shape != (batch_size, node_count):
            raise ValueError("tail scorer must return shape (batch, tails)")
        if not torch.isneginf(tail_log_p.masked_select(tail_mask)).all():
            raise ValueError("tail scorer assigned probability to an illegal tail")

        selected_tail = _select(tail_log_p, decode_type, generator)
        selected_head_log_p = proposal.log_probabilities[batch_index, selected_tail]
        selected_head = _select(selected_head_log_p, decode_type, generator)
        if pair_mask[batch_index, selected_tail, selected_head].any():
            raise RuntimeError("Forest-aware decoder selected an illegal edge")

        selected_log_probabilities.append(
            tail_log_p.gather(1, selected_tail[:, None]).squeeze(1)
            + selected_head_log_p.gather(1, selected_head[:, None]).squeeze(1)
        )
        tail_entropies.append(_categorical_entropy(tail_log_p))
        action_entropies.append(_joint_action_entropy(tail_log_p, proposal.log_probabilities))
        tails.append(selected_tail)
        heads.append(selected_head)
        state = process.transition(state, selected_tail, selected_head)
        decoder.update_context(selected_tail, selected_head)

    tail_tensor = torch.stack(tails, dim=1)
    head_tensor = torch.stack(heads, dim=1)
    return ConstructionOutput(
        cost=process.objective(state, tail_tensor, head_tensor),
        log_likelihood=torch.stack(selected_log_probabilities, dim=1).sum(dim=1),
        tails=tail_tensor,
        heads=head_tensor,
        successor=process.solution(state),
        tail_entropy=torch.stack(tail_entropies, dim=1).mean(dim=1),
        action_entropy=torch.stack(action_entropies, dim=1).mean(dim=1),
    )


def _categorical_entropy(log_probabilities: Tensor) -> Tensor:
    finite = torch.isfinite(log_probabilities)
    safe_log_p = torch.where(finite, log_probabilities, torch.zeros_like(log_probabilities))
    probabilities = safe_log_p.exp() * finite.to(log_probabilities.dtype)
    return -(probabilities * safe_log_p).sum(dim=-1)


def _joint_action_entropy(tail_log_p: Tensor, head_log_p: Tensor) -> Tensor:
    tail_probabilities = torch.where(
        torch.isfinite(tail_log_p), tail_log_p.exp(), torch.zeros_like(tail_log_p)
    )
    return _categorical_entropy(tail_log_p) + (
        tail_probabilities * _categorical_entropy(head_log_p)
    ).sum(dim=-1)


def _select(
    log_probabilities: Tensor,
    decode_type: str,
    generator: Generator | None,
) -> Tensor:
    if decode_type == "greedy":
        return log_probabilities.argmax(dim=1)
    if decode_type == "sampling":
        return torch.multinomial(
            log_probabilities.exp(), num_samples=1, generator=generator
        ).squeeze(1)
    raise ValueError(f"unknown decode type: {decode_type}")
