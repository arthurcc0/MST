from typing import Callable
import torch
from torch import nn, Tensor


class TextHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        num_blocks: int,
        is_causal: bool,
        drop_prob: float,
        use_linear_projection: bool,
    ):
        super().__init__()
        self.num_blocks = num_blocks
        self.ln_final = nn.Identity()
        
        if num_blocks > 0:
            # Simple transformer blocks for text head
            self.blocks = nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=input_dim,
                    nhead=8,
                    dim_feedforward=input_dim * 4,
                    dropout=drop_prob,
                    batch_first=True
                )
                for _ in range(num_blocks)
            ])
            self.ln_final = nn.LayerNorm(input_dim)
        else:
            self.blocks = nn.ModuleList([nn.Identity()])
            
        self.linear_projection = nn.Identity()
        if input_dim != embed_dim or use_linear_projection:
            self.linear_projection = nn.Linear(input_dim, embed_dim, bias=False)

    def init_weights(self):
        if self.num_blocks > 0:
            self.ln_final.reset_parameters()
        if isinstance(self.linear_projection, nn.Linear):
            nn.init.normal_(self.linear_projection.weight, std=self.linear_projection.in_features**-0.5)

    def forward(self, text_tokens: Tensor) -> Tensor:
        for block in self.blocks:
            if isinstance(block, nn.Identity):
                continue
            text_tokens = block(text_tokens)
        text_tokens = self.ln_final(text_tokens)
        return self.linear_projection(text_tokens)


class TextTower(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        freeze_backbone: bool,
        embed_dim: int,
        num_head_blocks: int,
        head_blocks_is_causal: bool,
        head_blocks_drop_prob: float,
        tokens_pooler_type: str,
        use_linear_projection: bool,
    ):
        super().__init__()
        self.backbone = backbone
        self.freeze_backbone = freeze_backbone
        self.tokens_pooler_type = tokens_pooler_type
        
        backbone_out_dim = backbone.embed_dim
        self.head = TextHead(
            backbone_out_dim,
            embed_dim,
            num_head_blocks,
            head_blocks_is_causal,
            head_blocks_drop_prob,
            use_linear_projection,
        )

    def init_weights(self):
        if not self.freeze_backbone:
            self.backbone.init_weights()
        self.head.init_weights()

    def forward(self, text: Tensor) -> Tensor:
        text_features = self.backbone(text)  # [batch_size, seq_len, embed_dim]
        text_features = self.head(text_features)
        if self.tokens_pooler_type == "first":
            pooled_features = text_features[:, 0]
        elif self.tokens_pooler_type == "last":
            pooled_features = text_features[:, -1]
        elif self.tokens_pooler_type == "mean":
            pooled_features = text_features.mean(dim=1)
        elif self.tokens_pooler_type == "argmax":
            norms = torch.norm(text_features, dim=-1)
            max_indices = torch.argmax(norms, dim=1)
            pooled_features = text_features[torch.arange(text_features.size(0)), max_indices]
        else:
            raise ValueError(f"Unknown tokens pooler type: {self.tokens_pooler_type}")
            
        return pooled_features
