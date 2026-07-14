import torch


def mask_long2bool(mask: torch.Tensor, n: int | None = None) -> torch.Tensor:
    if mask.dtype == torch.bool:
        return mask
    if mask.dtype != torch.int64:
        raise TypeError(f"Unsupported mask dtype: {mask.dtype}")

    width = 64 if n is None else min(64, n)
    bits = ((mask[..., None] >> torch.arange(width, device=mask.device, dtype=torch.int64)) & 1).to(torch.bool)
    if n is not None:
        bits = bits.reshape(*mask.shape[:-1], -1)[..., :n]
    return bits.reshape(*mask.shape, width) if n is None else bits


def mask_long_scatter(mask: torch.Tensor, values: torch.Tensor, check_unset: bool = True) -> torch.Tensor:
    if mask.dtype != torch.int64:
        raise TypeError(f"mask_long_scatter expects int64 mask, got {mask.dtype}")
    if values.dim() != mask.dim():
        raise ValueError("values must have the same rank as mask")

    result = mask.clone()
    one = torch.ones(1, dtype=torch.int64, device=mask.device)
    for bit in range(values.size(-1)):
        bit_value = (one << values[..., bit]).to(result.dtype)
        if check_unset and ((result[..., bit] & bit_value) != 0).any():
            raise ValueError("Attempted to set an already active bit")
        result[..., bit] |= bit_value
    return result


__all__ = ["mask_long2bool", "mask_long_scatter"]