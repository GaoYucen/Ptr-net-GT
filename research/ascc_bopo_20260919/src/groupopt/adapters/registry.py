"""供不同模型族接入 GroupOpt 的统一扩展点。"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from torch import nn

from groupopt.models.am import AttentionModel
from groupopt.models.gpn import GraphPointerNetwork
from groupopt.models.pomo import POMOModel
from groupopt.models.ptrnet import PointerNetwork
from groupopt.problems import BatchedCVRPConstruction, BatchedMCycleCoverConstruction

ModelBuilder = Callable[[Mapping[str, Any]], nn.Module]


class ModelRegistry:
    """在不修改框架代码的情况下注册并构建神经方法。"""

    def __init__(self) -> None:
        self._builders: dict[str, ModelBuilder] = {}

    def register(self, name: str, builder: ModelBuilder) -> None:
        if not name or not name.isidentifier():
            raise ValueError("model name must be a non-empty Python identifier")
        if name in self._builders:
            raise ValueError(f"model is already registered: {name}")
        self._builders[name] = builder

    def build(self, config: Mapping[str, Any]) -> nn.Module:
        name = str(config.get("model", "am"))
        try:
            builder = self._builders[name]
        except KeyError as error:
            raise ValueError(
                f"unknown model {name!r}; available models: {', '.join(self.names())}"
            ) from error
        return builder(config)

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._builders))


def _am(config: Mapping[str, Any]) -> nn.Module:
    problem = str(config.get("problem", "tsp"))
    if problem == "tsp":
        input_dim = 2
        construction_process = None
    elif problem == "cvrp":
        input_dim = 5
        construction_process = BatchedCVRPConstruction()
    elif problem == "min_m_ccp":
        input_dim = 2
        construction_process = BatchedMCycleCoverConstruction(int(config["cycles"]))
    else:
        raise ValueError(f"AM host does not support problem: {problem}")
    return AttentionModel(
        input_dim=input_dim,
        embedding_dim=int(config["embedding_dim"]),
        n_heads=int(config["heads"]),
        n_encoder_layers=int(config["encoder_layers"]),
        feed_forward_dim=int(config["feed_forward_dim"]),
        normalization=str(config["normalization"]),
        construction_process=construction_process,
    )


def _ptrnet(config: Mapping[str, Any]) -> nn.Module:
    return PointerNetwork(
        embedding_dim=int(config["embedding_dim"]),
        n_encoder_layers=int(config["encoder_layers"]),
    )


def _gpn(config: Mapping[str, Any]) -> nn.Module:
    return GraphPointerNetwork(
        embedding_dim=int(config["embedding_dim"]),
        n_encoder_layers=int(config["encoder_layers"]),
    )


def _pomo(config: Mapping[str, Any]) -> nn.Module:
    return POMOModel(
        embedding_dim=int(config["embedding_dim"]),
        head_num=int(config["heads"]),
        qkv_dim=int(config.get("qkv_dim", 16)),
        encoder_layers=int(config["encoder_layers"]),
        feed_forward_dim=int(config["feed_forward_dim"]),
        pomo_size=int(config.get("pomo_size", 8)),
    )


model_registry = ModelRegistry()
model_registry.register("am", _am)
model_registry.register("ptrnet", _ptrnet)
model_registry.register("gpn", _gpn)
model_registry.register("pomo", _pomo)


def build_model(config: Mapping[str, Any]) -> nn.Module:
    """根据实验配置构建一个已注册的神经适配器。"""
    return model_registry.build(config)
