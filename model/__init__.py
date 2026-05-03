"""SegFormer Model"""

import torch
import torch.nn as nn

from .decoder import SegFormerDecoder
from .mix_transformer import MixTransformer


class SegFormer(nn.Module):
    """Build a SegFormer model from a transformer encoder and segmentation decoder."""

    def __init__(
        self,
        img_size=224,
        in_chals=3,
        num_classes=150,
        embed_dims=[32, 64, 160, 256],
        num_heads=[1, 2, 5, 8],
        mlp_ratios=[4, 4, 4, 4],
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        depths=[2, 2, 2, 2],
        sr_ratios=[8, 4, 2, 1],
        decoder_dim=256,
    ):
        """Initialize the SegFormer model components.

        Args:
            img_size: Input image size used to configure the encoder stages.
            in_chals: Number of input image channels.
            num_classes: Number of segmentation classes to predict.
            embed_dims: Channel dimensions used at each encoder stage.
            num_heads: Attention head counts for each encoder stage.
            mlp_ratios: Expansion ratios for the MLP blocks in each stage.
            qkv_bias: Whether to use bias terms in attention projections.
            qk_scale: Optional manual scaling factor for attention scores.
            drop_rate: Dropout rate used in projection and decoder layers.
            attn_drop_rate: Dropout rate applied to attention weights.
            drop_path_rate: Maximum stochastic depth rate across encoder blocks.
            norm_layer: Normalization layer class used in the encoder.
            depths: Number of transformer blocks per encoder stage.
            sr_ratios: Spatial reduction ratios used by attention in each stage.
            decoder_dim: Shared decoder channel dimension.
        """
        super().__init__()

        self.encoder = MixTransformer(
            img_size=img_size,
            in_chals=in_chals,
            num_classes=0,
            embed_dims=embed_dims,
            num_heads=num_heads,
            mlp_ratios=mlp_ratios,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            depths=depths,
            sr_ratios=sr_ratios,
        )

        self.decoder = SegFormerDecoder(
            in_chals=embed_dims,
            embed_dim=decoder_dim,
            drop_rate=drop_rate,
            num_classes=num_classes,
        )

    def forward(self, x):
        """Run segmentation inference on an input batch.

        Args:
            x: Input image tensor of shape ``[batch, channels, height, width]``.
        """
        features = self.encoder(x)
        return self.decoder(features)


if __name__ == "__main__":
    model = SegFormer()
    print(sum([x.numel() for x in model.parameters()]))
    x = torch.randn(1, 3, 224, 224)
    output = model(x)
    print(output.shape)
