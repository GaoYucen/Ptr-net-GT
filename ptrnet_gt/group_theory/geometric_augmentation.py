import math

import torch


def augment_xy_data_by_8_fold(problems: torch.Tensor) -> torch.Tensor:
    x = problems[:, :, [0]]
    y = problems[:, :, [1]]
    views = [
        torch.cat((x, y), dim=2),
        torch.cat((1 - x, y), dim=2),
        torch.cat((x, 1 - y), dim=2),
        torch.cat((1 - x, 1 - y), dim=2),
        torch.cat((y, x), dim=2),
        torch.cat((1 - y, x), dim=2),
        torch.cat((y, 1 - x), dim=2),
        torch.cat((1 - y, 1 - x), dim=2),
    ]
    return torch.cat(views, dim=0)


def _sr_transform(x: torch.Tensor, y: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    phi = torch.where(idx < 0.5, idx * 4 * math.pi, (idx - 0.5) * 4 * math.pi)
    x = x - 0.5
    y = y - 0.5
    x_prime = torch.cos(phi) * x - torch.sin(phi) * y
    y_prime = torch.sin(phi) * x + torch.cos(phi) * y
    return torch.where(idx < 0.5, torch.cat((x_prime + 0.5, y_prime + 0.5), dim=2), torch.cat((y_prime + 0.5, x_prime + 0.5), dim=2))


def augment_xy_data_by_n_fold(problems: torch.Tensor, n: int) -> torch.Tensor:
    if n <= 1:
        return problems
    if n == 8:
        return augment_xy_data_by_8_fold(problems)
    x = problems[:, :, [0]]
    y = problems[:, :, [1]]
    out = [problems]
    idx = torch.rand(n - 1, device=problems.device)
    for i in range(n - 1):
        out.append(_sr_transform(x, y, idx[i]))
    return torch.cat(out, dim=0)


__all__ = ["augment_xy_data_by_8_fold", "augment_xy_data_by_n_fold"]