from mst.models.dino_common import (
    dinov2_hub_name,
    load_compatible_state_dict,
    normalize_dino_size,
    slice_fusion_nhead,
)


def test_normalize_dino_size_aliases():
    assert normalize_dino_size("base") == "b"
    assert normalize_dino_size("large") == "l"
    assert normalize_dino_size("L") == "l"
    assert normalize_dino_size("giant") == "g"


def test_dinov2_hub_names():
    assert dinov2_hub_name("b") == "dinov2_vitb14"
    assert dinov2_hub_name("l", use_registers=True) == "dinov2_vitl14_reg"
    assert dinov2_hub_name("g") == "dinov2_vitg14"


def test_slice_fusion_nhead_divides_embed():
    assert 384 % slice_fusion_nhead(384, 6) == 0
    assert 768 % slice_fusion_nhead(768, 12) == 0
    assert slice_fusion_nhead(1024, 16) == 16
    assert 1024 % slice_fusion_nhead(1024, 16) == 0
    assert 1536 % slice_fusion_nhead(1536, 24) == 0
    # bottleneck emb_ch // 4 for ViT-L
    assert slice_fusion_nhead(256, 16) == 16


def test_load_compatible_state_dict_skips_shape_mismatch():
    import torch
    import torch.nn as nn

    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Linear(8, 8)
            self.head = nn.Linear(8, 2)

    model = Toy()
    before = model.encoder.weight.detach().clone()
    ckpt = {
        "encoder.weight": torch.zeros(4, 4),
        "encoder.bias": torch.zeros(4),
        "head.weight": torch.ones_like(model.head.weight),
        "head.bias": torch.ones_like(model.head.bias),
        "other.weight": torch.zeros(1),
    }
    report = load_compatible_state_dict(model, ckpt)
    assert report["loaded"] == 2
    assert len(report["skipped_shape"]) == 2
    assert torch.equal(model.encoder.weight, before)
    assert torch.equal(model.head.weight, torch.ones_like(model.head.weight))
