"""SegFormer model definitions and configuration."""

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from .decoder import SegFormerDecoder
from .mix_transformer import MixTransformer


@dataclass
class SegFormerConfig:
    """Store SegFormer encoder and decoder settings.

    Args:
        img_size: Input image size used to configure the encoder stages.
        in_chals: Number of input image channels.
        num_classes: Number of segmentation classes to predict.
        embed_dims: Channel dimensions used at each encoder stage.
        num_heads: Attention head counts for each encoder stage.
        mlp_ratios: Expansion ratios for the MLP blocks in each stage.
        qkv_bias: Whether to use bias terms in attention projections.
        qk_scale: Optional manual scaling factor for attention scores.
        drop_rate: Dropout rate used in encoder projection layers.
        attn_drop_rate: Dropout rate applied to attention weights.
        drop_path_rate: Maximum stochastic depth rate across encoder blocks.
        norm_layer: Normalization layer class used in the encoder.
        depths: Number of transformer blocks per encoder stage.
        sr_ratios: Spatial reduction ratios used by attention in each stage.
        decoder_dim: Shared decoder channel dimension.
        decoder_drop_rate: Dropout rate applied before the segmentation classifier.
    """

    img_size: int = 224
    in_chals: int = 3
    num_classes: int = 150
    embed_dims: list[int] = field(default_factory=lambda: [32, 64, 160, 256])
    num_heads: list[int] = field(default_factory=lambda: [1, 2, 5, 8])
    mlp_ratios: list[int] = field(default_factory=lambda: [4, 4, 4, 4])
    qkv_bias: bool = True
    qk_scale: float | None = None
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.0
    norm_layer: type[nn.LayerNorm] = nn.LayerNorm
    depths: list[int] = field(default_factory=lambda: [2, 2, 2, 2])
    sr_ratios: list[int] = field(default_factory=lambda: [8, 4, 2, 1])
    decoder_dim: int = 256
    decoder_drop_rate: float = 0.0


class SegFormer(nn.Module):
    """Build a SegFormer model from a transformer encoder and segmentation decoder."""

    def __init__(self, cfg: SegFormerConfig):
        """Initialize the SegFormer model components.

        Args:
            cfg: SegFormer configuration values for the encoder and decoder.
        """
        super().__init__()
        self.cfg = cfg

        self.encoder = MixTransformer(
            img_size=cfg.img_size,
            in_chals=cfg.in_chals,
            num_classes=0,
            embed_dims=cfg.embed_dims,
            num_heads=cfg.num_heads,
            mlp_ratios=cfg.mlp_ratios,
            qkv_bias=cfg.qkv_bias,
            qk_scale=cfg.qk_scale,
            drop_rate=cfg.drop_rate,
            attn_drop_rate=cfg.attn_drop_rate,
            drop_path_rate=cfg.drop_path_rate,
            norm_layer=cfg.norm_layer,
            depths=cfg.depths,
            sr_ratios=cfg.sr_ratios,
        )

        self.decoder = SegFormerDecoder(
            in_chals=cfg.embed_dims,
            embed_dim=cfg.decoder_dim,
            drop_rate=cfg.decoder_drop_rate,
            num_classes=cfg.num_classes,
        )

    def forward(self, x):
        """Run segmentation inference on an input batch.

        Args:
            x: Input image tensor of shape ``[batch, channels, height, width]``.
        """
        features = self.encoder(x)
        return self.decoder(features)


if __name__ == "__main__":
    model = SegFormer(SegFormerConfig())
    print(sum([x.numel() for x in model.parameters()]))
    x = torch.randn(1, 3, 224, 224)
    output = model(x)
    print(output.shape)
