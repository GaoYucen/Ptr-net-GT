from __future__ import annotations

from ptrnet_gt.models import AttentionModelDecoder, ComponentMergeDecoder, PointerNetworkDecoder


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
            action_mode=model_cfg.get("action_mode", "tail_head"),
            use_dynamic_role_features=model_cfg.get("use_dynamic_role_features", False),
            use_role_feature_projection=model_cfg.get("use_role_feature_projection", False),
            role_embedding_dim=model_cfg.get("role_embedding_dim"),
            use_distance_projection=model_cfg.get("use_distance_projection", False),
            distance_embedding_dim=model_cfg.get("distance_embedding_dim"),
            mask_inner=model_cfg.get("mask_inner", True),
            mask_logits=model_cfg.get("mask_logits", True),
            n_heads=model_cfg.get("n_heads", 8),
            **common_kwargs,
        )
    if model_name == "pointer_network":
        return PointerNetworkDecoder(
            n_heads=model_cfg.get("n_heads", 8),
            **common_kwargs,
        )
    if model_name == "attention_model":
        return AttentionModelDecoder(
            n_heads=model_cfg.get("n_heads", 8),
            **common_kwargs,
        )
    raise ValueError(f"Unsupported model for current TSP-only codebase: {model_name}")
