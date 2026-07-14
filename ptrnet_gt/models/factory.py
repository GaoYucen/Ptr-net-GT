from __future__ import annotations

from ptrnet_gt.baselines import AttentionModel, PointerNetwork
from ptrnet_gt.models import ComponentMergeDecoder


def build_model(config: dict, problem):
    model_cfg = config["model"]
    model_name = model_cfg["name"]
    common_kwargs = dict(
        embedding_dim=model_cfg.get("embedding_dim", 128),
        hidden_dim=model_cfg.get("hidden_dim", 128),
        problem=problem,
        n_encode_layers=model_cfg.get("n_encode_layers", 2),
        normalization=model_cfg.get("normalization", "batch"),
        tanh_clipping=model_cfg.get("tanh_clipping", 10.0),
        checkpoint_encoder=model_cfg.get("checkpoint_encoder", False),
        shrink_size=model_cfg.get("shrink_size"),
    )

    if model_name == "component_merge":
        return ComponentMergeDecoder(
            context_mode=model_cfg.get("context_mode", "cross_step"),
            mask_inner=model_cfg.get("mask_inner", True),
            mask_logits=model_cfg.get("mask_logits", True),
            n_heads=model_cfg.get("n_heads", 8),
            **common_kwargs,
        )
    if model_name == "attention":
        return AttentionModel(mask_inner=True, mask_logits=True, **common_kwargs)
    if model_name == "pointer":
        return PointerNetwork(**common_kwargs)
    raise ValueError(f"Unknown model: {model_name}")
