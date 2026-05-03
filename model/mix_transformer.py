"""
Mix Transformer encoder used by the SegFormer architecture.
"""

import torch
import torch.nn as nn


def to_2tuple(x):
    """
    Return x as a 2-tuple; if x is scalar, return (x, x).

    Args:
        x: Scalar or sequence value.
    """
    return x if isinstance(x, (list, tuple)) else (x, x)


class OverlapPatchEmbeddings(nn.Module):
    """Convert an image tensor into overlapping patch embeddings."""

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        """Initialize the projection and normalization layers for patch embedding.

        Args:
            img_size: Input image size used for stage configuration.
            patch_size: Convolution kernel size for extracting patches.
            stride: Convolution stride between adjacent patches.
            in_chans: Number of input channels.
            embed_dim: Output embedding dimension per patch.
        """
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        padding = patch_size[0] // 2, patch_size[1] // 2
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=padding
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """Project an image into flattened patch tokens.

        Args:
            x: Input image tensor of shape ``[batch, channels, height, width]``.
        """
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class DropPath(nn.Module):
    """Apply stochastic depth to a residual branch during training."""

    def __init__(self, drop_prob: float):
        """Store the stochastic depth probability.

        Args:
            drop_prob: Probability of dropping a residual path during training.
        """
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        """Randomly drop full residual paths for the current batch.

        Args:
            x: Input tensor to regularize.
        """
        drop_prob = self.drop_prob
        if drop_prob == 0.0 or not self.training:
            return x
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random = torch.rand(shape, dtype=x.dtype, device=x.device)
        mask = (random > drop_prob).to(dtype=x.dtype)
        out = x * mask / (1.0 - drop_prob)
        return out


class Attention(nn.Module):
    """Compute multi-head self-attention with optional spatial reduction."""

    def __init__(
        self,
        dim,
        num_heads,
        qkv_bias=False,
        qk_scale=None,
        attn_drop_rate=0.0,
        proj_drop_rate=0.0,
        sr_ratio=1,
    ):
        """Initialize attention projection layers and reduction settings.

        Args:
            dim: Token embedding dimension.
            num_heads: Number of attention heads.
            qkv_bias: Whether query, key, and value projections use bias.
            qk_scale: Optional manual scaling factor for attention scores.
            attn_drop_rate: Dropout rate applied to attention weights.
            proj_drop_rate: Dropout rate applied after the output projection.
            sr_ratio: Spatial reduction ratio for keys and values.
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        assert dim % num_heads == 0, (
            f"dim {dim} should be divisible by num_heads {num_heads}"
        )
        self.scale = qk_scale if qk_scale is not None else head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_rate)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        """Apply self-attention to a sequence of image tokens.

        Args:
            x: Token tensor of shape ``[batch, tokens, channels]``.
            H: Feature map height corresponding to the token sequence.
            W: Feature map width corresponding to the token sequence.
        """
        B, N, C = x.shape
        q = (
            self.q(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )  # B, num_heads, tokens, head_dim

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)  # B, head_dim, height, width
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)  # B, (H//sr)*(W//sr), C
            x_ = self.norm(x_)
            kv = (
                self.kv(x_)
                .reshape(B, -1, 2, self.num_heads, C // self.num_heads)
                .permute(2, 0, 3, 1, 4)
            )  # 2, B, num_heads, tokens, head_dim
        else:
            kv = (
                self.kv(x)
                .reshape(B, -1, 2, self.num_heads, C // self.num_heads)
                .permute(2, 0, 3, 1, 4)
            )  # 2, B, num_heads, tokens, head_dim
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class DWConv(nn.Module):
    """Apply depthwise convolution inside the MLP token mixing path."""

    def __init__(self, dim):
        """Initialize the depthwise convolution layer.

        Args:
            dim: Number of input and output channels.
        """
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size=3, stride=1, padding=1, bias=True, groups=dim
        )

    def forward(self, x, H, W):
        """Reshape tokens to a feature map, convolve, and flatten back.

        Args:
            x: Token tensor of shape ``[batch, tokens, channels]``.
            H: Feature map height corresponding to the token sequence.
            W: Feature map width corresponding to the token sequence.
        """
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)  # B, N, C
        return x


class Mlp(nn.Module):
    """Project tokens through a feed-forward block with depthwise mixing."""

    def __init__(
        self,
        in_features,
        hidden_features,
        out_features,
        act_layer=nn.GELU,
        drop_rate=0.0,
    ):
        """Initialize the MLP projection, activation, and dropout layers.

        Args:
            in_features: Input token channel dimension.
            hidden_features: Hidden channel dimension used inside the block.
            out_features: Output token channel dimension.
            act_layer: Activation layer class used between projections.
            drop_rate: Dropout rate applied after each projection.
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop_rate)

    def forward(self, x, H, W):
        """Transform token features with linear and depthwise operations.

        Args:
            x: Token tensor of shape ``[batch, tokens, channels]``.
            H: Feature map height corresponding to the token sequence.
            W: Feature map width corresponding to the token sequence.
        """
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    """Run one transformer block with attention and MLP residual branches."""

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        sr_ratio=1,
    ):
        """Initialize normalization, attention, and MLP submodules.

        Args:
            dim: Token embedding dimension.
            num_heads: Number of attention heads.
            mlp_ratio: Expansion ratio used inside the MLP block.
            qkv_bias: Whether attention projections use bias terms.
            qk_scale: Optional manual scaling factor for attention scores.
            drop_rate: Dropout rate used in attention and MLP projections.
            attn_drop_rate: Dropout rate applied to attention weights.
            drop_path_rate: Stochastic depth rate for residual branches.
            act_layer: Activation layer class used by the MLP.
            norm_layer: Normalization layer class used before each branch.
            sr_ratio: Spatial reduction ratio used by attention.
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=drop_rate,
            sr_ratio=sr_ratio,
        )

        self.drop_path = DropPath(drop_path_rate)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            dim, dim * mlp_ratio, dim, act_layer=act_layer, drop_rate=drop_rate
        )

    def forward(self, x, H, W):
        """Apply attention and MLP residual updates to the token sequence.

        Args:
            x: Token tensor of shape ``[batch, tokens, channels]``.
            H: Feature map height corresponding to the token sequence.
            W: Feature map width corresponding to the token sequence.
        """
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class MixTransformer(nn.Module):
    """Encode an image into multi-scale feature maps with transformer stages."""

    def __init__(
        self,
        img_size=224,
        in_chals=3,
        num_classes=1000,
        embed_dims=[32, 64, 160, 256],
        num_heads=[1, 2, 5, 8],
        mlp_ratios=[4, 4, 4, 4],
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        depths=[2, 2, 2, 2],
        sr_ratios=[8, 4, 2, 1],
    ):
        """Initialize the hierarchical Mix Transformer encoder.

        Args:
            img_size: Input image size used to configure stage resolutions.
            in_chals: Number of input image channels.
            num_classes: Number of classes for optional classification output.
            embed_dims: Channel dimensions for the encoder stages.
            num_heads: Attention head counts for the encoder stages.
            mlp_ratios: Expansion ratios for the stage MLP blocks.
            qkv_bias: Whether attention projections use bias terms.
            qk_scale: Optional manual scaling factor for attention scores.
            drop_rate: Dropout rate used in attention and MLP projections.
            attn_drop_rate: Dropout rate applied to attention weights.
            drop_path_rate: Maximum stochastic depth rate across all blocks.
            norm_layer: Normalization layer class used in the encoder.
            depths: Number of transformer blocks in each stage.
            sr_ratios: Spatial reduction ratios for the attention layers.
        """
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        inp_reductions = [1, 4, 8, 16]
        assert len(inp_reductions) == len(depths)
        assert len(num_heads) == len(depths)
        assert len(sr_ratios) == len(depths)
        assert len(embed_dims) == len(depths)

        patch_embeds, blocks, norms = [], [], []
        dims = [in_chals] + embed_dims
        depth_drop_rates = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]
        cur = 0
        for l_i in range(len(depths)):
            patch_embeds.append(
                OverlapPatchEmbeddings(
                    img_size=img_size // inp_reductions[l_i],
                    patch_size=7 if l_i == 0 else 3,
                    stride=4 if l_i == 0 else 2,
                    in_chans=dims[l_i],
                    embed_dim=dims[l_i + 1],
                )
            )
            blocks.append(
                nn.ModuleList(
                    [
                        Block(
                            dim=embed_dims[l_i],
                            num_heads=num_heads[l_i],
                            mlp_ratio=mlp_ratios[l_i],
                            qkv_bias=qkv_bias,
                            qk_scale=qk_scale,
                            drop_rate=drop_rate,
                            attn_drop_rate=attn_drop_rate,
                            drop_path_rate=depth_drop_rates[cur + b_i],
                            norm_layer=norm_layer,
                            sr_ratio=sr_ratios[l_i],
                        )
                        for b_i in range(depths[l_i])
                    ]
                )
            )
            cur += depths[l_i]
            norms.append(norm_layer(embed_dims[l_i]))

        self.patch_embeds = nn.ModuleList(patch_embeds)
        self.blocks = nn.ModuleList(blocks)
        self.norms = nn.ModuleList(norms)

        if self.num_classes > 0:
            self.head = nn.Linear(embed_dims[3], num_classes)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module):
        """Initialize supported module weights in place.

        Args:
            m: Module instance to initialize.
        """
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0.0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out = fan_out / m.groups
            nn.init.normal_(m.weight, mean=0, std=(2 / fan_out) ** 0.5)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        """Encode an input image into stage outputs or class logits.

        Args:
            x: Input image tensor of shape ``[batch, channels, height, width]``.
        """
        B = x.shape[0]
        outs = []

        for patch_embed, stage_blocks, norm in zip(
            self.patch_embeds, self.blocks, self.norms, strict=True
        ):
            x, H, W = patch_embed(x)
            for blk in stage_blocks.children():
                x = blk(x, H, W)
            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()  # B, C, H, W
            outs.append(x)

        if self.num_classes > 0:
            x = outs[-1].mean(dim=(2, 3))
            return self.head(x)

        return outs
