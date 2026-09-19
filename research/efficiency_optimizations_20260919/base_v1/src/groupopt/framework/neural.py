"""连接核心范式与神经方法的可选 PyTorch 桥接层。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypeVar, runtime_checkable

from torch import Generator, Tensor

InstanceT_contra = TypeVar("InstanceT_contra", contravariant=True)
StateT = TypeVar("StateT")


@dataclass(frozen=True, slots=True)
class ConstructionOutput:
    """所有神经方法适配器统一返回的张量结果。"""

    cost: Tensor
    log_likelihood: Tensor
    tails: Tensor
    heads: Tensor
    successor: Tensor
    tail_entropy: Tensor
    action_entropy: Tensor


@runtime_checkable
class BatchedConstructionProcess(Protocol[InstanceT_contra, StateT]):
    """供神经模型适配器使用的张量化问题插件。

    布尔 mask 遵循 attention 约定：``True`` 表示非法。可行性、状态转移、
    固定 base 行为和目标函数均由问题插件负责，而不是由神经模型负责。
    """

    def initial_state(self, instance: InstanceT_contra) -> StateT: ...

    def is_terminal(self, state: StateT) -> bool: ...

    def base_mask(self, state: StateT) -> Tensor: ...

    def representative_mask(self, state: StateT, selected_base: Tensor) -> Tensor: ...

    def action_mask(self, state: StateT, fixed_base: Tensor | None = None) -> Tensor: ...

    def fixed_base(self, state: StateT, anchor: int = 0) -> Tensor: ...

    def transition(
        self, state: StateT, selected_base: Tensor, selected_representative: Tensor
    ) -> StateT: ...

    def objective(
        self, state: StateT, selected_bases: Tensor, selected_representatives: Tensor
    ) -> Tensor: ...

    def solution(self, state: StateT) -> Tensor: ...


@runtime_checkable
class ConstructionModel(Protocol):
    """所有模型族共享的稳定神经适配器接口。"""

    def forward(
        self,
        instance: Tensor,
        decode_type: str = "sampling",
        base_mode: str = "native_conditional_free",
        anchor: int = 0,
        temperature: float = 1.0,
        generator: Generator | None = None,
    ) -> ConstructionOutput:
        """为框架动作评分，并返回一个可行的构造结果。"""
        ...
