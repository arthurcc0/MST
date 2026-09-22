"""Shared DINOv2 / DINOv3 size and slice-fusion helpers."""

from __future__ import annotations

DINOV2_SIZES = {
    "s": {"hub": "dinov2_vits14", "embed": 384, "heads": 6},
    "b": {"hub": "dinov2_vitb14", "embed": 768, "heads": 12},
    "l": {"hub": "dinov2_vitl14", "embed": 1024, "heads": 16},
    "g": {"hub": "dinov2_vitg14", "embed": 1536, "heads": 24},
}

_SIZE_ALIASES = {
    "s": "s",
    "small": "s",
    "vits": "s",
    "b": "b",
    "base": "b",
    "vitb": "b",
    "l": "l",
    "large": "l",
    "vitl": "l",
    "g": "g",
    "giant": "g",
    "vitg": "g",
}


def normalize_dino_size(model_size: str) -> str:
    key = str(model_size).strip().lower()
    if key not in _SIZE_ALIASES:
        raise ValueError(
            f"Unsupported model_size {model_size!r}; use s/b/l/g "
            f"(or small/base/large/giant)."
        )
    return _SIZE_ALIASES[key]


def dinov2_hub_name(model_size: str, *, use_registers: bool = False) -> str:
    size = normalize_dino_size(model_size)
    name = DINOV2_SIZES[size]["hub"]
    return f"{name}_reg" if use_registers else name


def slice_fusion_nhead(emb_ch: int, encoder_heads: int, preferred: int = 12) -> int:
    """Pick an nhead that divides ``emb_ch``. ViT-L is 1024-d (16 heads), not 12."""
    for candidate in (preferred, encoder_heads, 16, 8, 6, 4, 1):
        if candidate and emb_ch % int(candidate) == 0:
            return int(candidate)
    return 1


def encoder_num_heads(encoder, fallback: int = 12) -> int:
    for attr in ("num_heads", "n_heads", "num_attention_heads"):
        value = getattr(encoder, attr, None)
        if value:
            return int(value)
    config = getattr(encoder, "config", None)
    if config is not None:
        value = getattr(config, "num_attention_heads", None)
        if value:
            return int(value)
    return int(fallback)


def load_compatible_state_dict(model, state_dict: dict) -> dict:
    """Load matching tensors; skip unexpected keys and shape mismatches.

    Needed when a small MST checkpoint is passed into a larger DINO backbone:
    ``load_state_dict(..., strict=False)`` still errors on same-name, different-shape.
    """
    current = model.state_dict()
    compatible = {}
    skipped_shape = []
    unexpected = []
    for key, value in state_dict.items():
        if key not in current:
            unexpected.append(key)
            continue
        if tuple(current[key].shape) != tuple(value.shape):
            skipped_shape.append(
                f"{key}: ckpt{tuple(value.shape)} != model{tuple(current[key].shape)}"
            )
            continue
        compatible[key] = value
    missing = [key for key in current if key not in compatible]
    model.load_state_dict(compatible, strict=False)
    return {
        "loaded": len(compatible),
        "missing": missing,
        "unexpected": unexpected,
        "skipped_shape": skipped_shape,
    }
