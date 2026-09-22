"""DINOv3 slice classifier — separate from DinoClassifierSlice (DINOv2)."""

from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange
from transformers import AutoConfig, AutoModel

from .base_model import BasicClassifier
from .dino_common import encoder_num_heads, normalize_dino_size, slice_fusion_nhead
from .utils.transformer_blocks import TransformerEncoderLayer

DINOV3_HF_IDS = {
    "s": "facebook/dinov3-vits16-pretrain-lvd1689m",
    "b": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "l": "facebook/dinov3-vitl16-pretrain-lvd1689m",
}


def _build_dinov3_encoder(model_size: str, pretrained: bool) -> nn.Module:
    size = normalize_dino_size(model_size)
    if size not in DINOV3_HF_IDS:
        raise ValueError(f"Unsupported DINOv3 model_size {model_size!r}; choose from {list(DINOV3_HF_IDS)}")

    model_id = DINOV3_HF_IDS[size]
    if pretrained:
        encoder = AutoModel.from_pretrained(model_id, attn_implementation="eager")
    else:
        config = AutoConfig.from_pretrained(model_id)
        encoder = AutoModel.from_config(config, attn_implementation="eager")
    return encoder


class DinoClassifierSliceV3(BasicClassifier):
    """3D slice classifier with DINOv3 ViT backbone and MST slice fusion."""

    def __init__(
        self,
        in_ch,
        out_ch,
        spatial_dims=2,
        pretrained=True,
        save_attn=False,
        rotary_positional_encoding=None,
        optimizer_kwargs={"lr": 1e-6, "weight_decay": 1e-2},
        model_size="s",
        use_bottleneck=False,
        use_slice_pos_emb=False,
        enable_linear=True,
        enable_trans=True,
        slice_fusion="transformer",
        freeze=False,
        encoder_lr=None,
        **kwargs,
    ):
        kwargs.pop("paired_sampling", None)
        kwargs.pop("freeze_backbone", None)
        kwargs.pop("unfreeze_encoder_blocks", None)
        kwargs.pop("learning_rate", None)
        kwargs.pop("class_weight", None)
        kwargs.pop("model_name", None)
        kwargs.pop("model_version", None)
        kwargs.pop("use_registers", None)
        loss_kwargs = kwargs.pop("loss_kwargs", {})
        loss_cls = kwargs.pop("loss", torch.nn.CrossEntropyLoss)
        super().__init__(
            in_ch,
            out_ch,
            spatial_dims=spatial_dims,
            optimizer_kwargs=optimizer_kwargs,
            loss=loss_cls,
            loss_kwargs=loss_kwargs,
        )
        self.encoder_lr = encoder_lr
        self.save_attn = save_attn
        self.attention_maps = []
        self.attention_maps_slice = []
        self.slice_fusion_type = slice_fusion
        self.model_size = normalize_dino_size(model_size)

        self.encoder = _build_dinov3_encoder(model_size=self.model_size, pretrained=pretrained)
        self.patch_size = int(self.encoder.config.patch_size)
        self.num_register_tokens = int(self.encoder.config.num_register_tokens)

        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False

        emb_ch = int(self.encoder.config.hidden_size)
        if use_bottleneck:
            self.bottleneck = nn.Linear(emb_ch, emb_ch // 4)
            emb_ch = emb_ch // 4
        self.emb_ch = emb_ch

        if slice_fusion == "transformer":
            if use_slice_pos_emb:
                self.slice_pos_emb = nn.Embedding(256, emb_ch)

            nhead = slice_fusion_nhead(emb_ch, encoder_num_heads(self.encoder))
            self.slice_fusion = nn.TransformerEncoder(
                encoder_layer=TransformerEncoderLayer(
                    d_model=emb_ch,
                    nhead=nhead,
                    dim_feedforward=1 * emb_ch,
                    dropout=0.0,
                    batch_first=True,
                    norm_first=True,
                    rotary_positional_encoding=rotary_positional_encoding,
                ),
                num_layers=1,
                norm=nn.LayerNorm(emb_ch),
            )
            self.cls_token = nn.Parameter(torch.randn(1, 1, emb_ch))
        elif slice_fusion == "linear":
            emb_ch = emb_ch * 32
        elif slice_fusion == "average":
            pass

        self.linear = nn.Linear(emb_ch, out_ch) if enable_linear else nn.Identity()

    def unfreeze_encoder_last_blocks(self, n_blocks: int) -> int:
        """Unfreeze the last ``n_blocks`` DINOv3 ViT blocks."""
        if n_blocks <= 0:
            return 0
        if not hasattr(self.encoder, "layer"):
            print("Warning: DINOv3 encoder has no .layer attribute; cannot unfreeze.")
            return 0
        for param in self.encoder.parameters():
            param.requires_grad = False
        layers = self.encoder.layer[-n_blocks:]
        for layer in layers:
            for param in layer.parameters():
                param.requires_grad = True
        return len(layers)

    def configure_optimizers(self):
        encoder_params = []
        head_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("encoder."):
                encoder_params.append(param)
            else:
                head_params.append(param)

        opt_kwargs = dict(self.optimizer_kwargs)
        head_lr = float(opt_kwargs.pop("lr", 1e-6))
        enc_lr = float(self.encoder_lr) if self.encoder_lr is not None else head_lr * 0.1

        if encoder_params and head_params:
            param_groups = [
                {"params": encoder_params, "lr": enc_lr},
                {"params": head_params, "lr": head_lr},
            ]
        else:
            param_groups = [p for p in self.parameters() if p.requires_grad]

        optimizer = self.optimizer(param_groups, **opt_kwargs)
        if self.lr_scheduler is not None:
            lr_scheduler = self.lr_scheduler(optimizer, **self.lr_scheduler_kwargs)
            return [optimizer], [{"scheduler": lr_scheduler, "interval": "step", "frequency": 1}]
        return optimizer

    def _encode_slices(self, x: torch.Tensor, capture_attn: bool) -> torch.Tensor:
        if capture_attn:
            outputs = self.encoder(pixel_values=x, output_attentions=True, return_dict=True)
            if outputs.attentions is not None:
                self.attention_maps = [attn.detach() for attn in outputs.attentions]
            return outputs.last_hidden_state[:, 0]
        outputs = self.encoder(pixel_values=x, return_dict=True)
        return outputs.last_hidden_state[:, 0]

    def forward(self, source, save_attn=False, src_key_padding_mask=None, **kwargs):
        if save_attn:
            fastpath_enabled = torch.backends.mha.get_fastpath_enabled()
            torch.backends.mha.set_fastpath_enabled(False)
            self.attention_maps_slice = []
            self.attention_maps = []
            self.hooks = []
            self.register_hooks()

        x = source.to(self.device)
        B, C, *_ = x.shape

        x = rearrange(x, "b c d h w -> (b d c) h w")
        x = x[:, None]
        x = x.repeat(1, 3, 1, 1)

        x = self._encode_slices(x, capture_attn=save_attn)

        if hasattr(self, "bottleneck"):
            x = self.bottleneck(x)

        x = rearrange(x, "(b d) e -> b d e", b=B)

        slab_tissue_weight = kwargs.get("slab_tissue_weight")
        if slab_tissue_weight is not None:
            w = slab_tissue_weight.to(self.device).float()
            if w.dim() == 1:
                w = w.unsqueeze(0)
            x = x * w.unsqueeze(-1)

        if hasattr(self, "slice_pos_emb"):
            pos = torch.arange(0, x.shape[1], dtype=torch.long, device=x.device)
            x += self.slice_pos_emb(pos)

        if self.slice_fusion_type == "transformer":
            x = torch.concat([self.cls_token.repeat(B, 1, 1), x], dim=1)

            if src_key_padding_mask is not None:
                src_key_padding_mask = src_key_padding_mask.to(self.device)
                src_key_padding_mask_cls = torch.zeros((B, 1), device=self.device, dtype=bool)
                src_key_padding_mask = torch.concat(
                    [src_key_padding_mask_cls, src_key_padding_mask], dim=1
                )

            x = self.slice_fusion(x, src_key_padding_mask=src_key_padding_mask)
            x = x[:, 0]
        elif self.slice_fusion_type == "linear":
            x = rearrange(x, "b d e -> b (d e)")
        elif self.slice_fusion_type == "average":
            x = x.mean(dim=1, keepdim=False)

        if save_attn:
            torch.backends.mha.set_fastpath_enabled(fastpath_enabled)
            self.deregister_hooks()

        if kwargs.get("without_linear", False):
            return x
        x = self.linear(x)
        return x

    def get_slice_attention(self):
        attention_map_slice = self.attention_maps_slice[-1]
        attention_map_slice = attention_map_slice[:, :, 0, 1:]
        attention_map_slice /= attention_map_slice.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        attention_map_slice = attention_map_slice.mean(dim=1)
        attention_map_slice = attention_map_slice.view(-1)
        attention_map_slice = attention_map_slice[:, None, None]
        return attention_map_slice

    def get_plane_attention(self):
        attention_map_dino = self.attention_maps[-1]
        img_slice = slice(1 + self.num_register_tokens, None)
        attention_map_dino = attention_map_dino[:, :, 0, img_slice]
        attention_map_dino[:, :, 0] = 0
        attention_map_dino /= attention_map_dino.sum(dim=-1, keepdim=True)
        return attention_map_dino

    def get_attention_maps(self):
        attention_map_dino = self.get_plane_attention()
        attention_map_slice = self.get_slice_attention()
        attention_map = attention_map_slice * attention_map_dino
        return attention_map

    def get_attention_cls(self):
        attention_to_cls = self.attention_maps[-1]
        for attn in reversed(self.attention_maps[:-1]):
            attention_to_cls = torch.matmul(attn, attention_to_cls)
        return attention_to_cls

    def register_hooks(self):
        def enable_attention(module):
            forward_orig = module.forward

            def forward_wrap(*args, **kwargs):
                kwargs["need_weights"] = True
                kwargs["average_attn_weights"] = False
                return forward_orig(*args, **kwargs)

            module.forward = forward_wrap
            module.foward_orig = forward_orig

        def append_attention_maps(module, input, output):
            self.attention_maps_slice.append(output[1])

        for _, mod in self.slice_fusion.named_modules():
            if isinstance(mod, nn.MultiheadAttention):
                enable_attention(mod)
                self.hooks.append(mod.register_forward_hook(append_attention_maps))

    def deregister_hooks(self):
        for handle in self.hooks:
            handle.remove()

        for _, mod in self.slice_fusion.named_modules():
            if isinstance(mod, nn.MultiheadAttention) and hasattr(mod, "foward_orig"):
                mod.forward = mod.foward_orig
