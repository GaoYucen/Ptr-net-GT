"""用于鲁棒性实验的确定性 Euclidean TSP 实例分布。"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor

TSPDistribution = Literal[
    "uniform",
    "clustered",
    "clustered_strong",
    "corner_biased",
    "narrow_strip",
    "clustered_outliers",
]

TSP_DISTRIBUTIONS: tuple[TSPDistribution, ...] = (
    "uniform",
    "clustered",
    "clustered_strong",
    "corner_biased",
    "narrow_strip",
    "clustered_outliers",
)


def generate_tsp_coordinates(
    sample_count: int,
    node_count: int,
    distribution: TSPDistribution,
    seed: int,
) -> Tensor:
    """生成位于 ``[0, 1]^2`` 的固定 CPU 张量，且不改变全局随机数状态。"""
    if sample_count < 1 or node_count < 2:
        raise ValueError("sample_count must be positive and node_count at least two")
    if distribution not in TSP_DISTRIBUTIONS:
        raise ValueError(f"unknown TSP distribution: {distribution}")

    generator = torch.Generator(device="cpu").manual_seed(seed)
    shape = (sample_count, node_count, 2)
    if distribution == "uniform":
        coordinates = torch.rand(shape, generator=generator)
    elif distribution == "clustered":
        coordinates = _clustered(shape, cluster_count=4, sigma=0.075, generator=generator)
    elif distribution == "clustered_strong":
        coordinates = _clustered(shape, cluster_count=4, sigma=0.025, generator=generator)
    elif distribution == "corner_biased":
        uniform = torch.rand(shape, generator=generator)
        sides = torch.randint(0, 2, shape, generator=generator).to(torch.float32)
        near_zero = uniform.pow(4)
        coordinates = torch.where(sides.bool(), 1.0 - near_zero, near_zero)
    elif distribution == "narrow_strip":
        longitudinal = torch.rand(sample_count, node_count, generator=generator) - 0.5
        transverse = 0.035 * torch.randn(
            sample_count, node_count, generator=generator
        )
        angles = torch.rand(sample_count, generator=generator) * torch.pi
        cosines = angles.cos()[:, None]
        sines = angles.sin()[:, None]
        x = 0.5 + longitudinal * cosines - transverse * sines
        y = 0.5 + longitudinal * sines + transverse * cosines
        coordinates = torch.stack((x, y), dim=-1).clamp(0.0, 1.0)
    else:
        clustered = _clustered(shape, cluster_count=2, sigma=0.035, generator=generator)
        outliers = torch.rand(shape, generator=generator)
        outlier_mask = torch.rand(
            sample_count, node_count, 1, generator=generator
        ) < 0.15
        coordinates = torch.where(outlier_mask, outliers, clustered)

    if coordinates.shape != shape or not torch.isfinite(coordinates).all():
        raise RuntimeError("generated TSP coordinates are invalid")
    return coordinates.to(torch.float32)


def generate_cvrp_instances(
    sample_count: int,
    customer_count: int,
    capacity: int,
    generator: torch.Generator,
    device: torch.device | str = "cpu",
) -> Tensor:
    """生成 ``[客户坐标, 需求, 仓库坐标]`` 格式的标准 uniform CVRP 实例。"""
    if sample_count < 1 or customer_count < 2:
        raise ValueError("sample_count must be positive and customer_count at least two")
    if capacity < 9:
        raise ValueError("capacity must accommodate the maximum demand 9")
    target_device = torch.device(device)
    depot = torch.rand(
        sample_count, 1, 2, device=target_device, generator=generator
    )
    customers = torch.rand(
        sample_count,
        customer_count,
        2,
        device=target_device,
        generator=generator,
    )
    integer_demand = torch.randint(
        1,
        10,
        (sample_count, customer_count, 1),
        device=target_device,
        generator=generator,
    )
    demand = integer_demand.to(customers.dtype) / float(capacity)
    repeated_depot = depot.expand(sample_count, customer_count, 2)
    return torch.cat((customers, demand, repeated_depot), dim=-1)


def _clustered(
    shape: tuple[int, int, int],
    cluster_count: int,
    sigma: float,
    generator: torch.Generator,
) -> Tensor:
    sample_count, node_count, _ = shape
    centers = 0.12 + 0.76 * torch.rand(
        sample_count, cluster_count, 2, generator=generator
    )
    assignments = torch.randint(
        0, cluster_count, (sample_count, node_count), generator=generator
    )
    selected_centers = centers.gather(
        1, assignments[..., None].expand(sample_count, node_count, 2)
    )
    noise = sigma * torch.randn(shape, generator=generator)
    return (selected_centers + noise).clamp(0.0, 1.0)
