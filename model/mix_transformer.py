"""
Mix Transformer encoder used by the SegFormer architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def to_2tuple(x: int | tuple[int, int] | list[int]) -> tuple[int, int]:
    """
    Return x as a 2-tuple; if x is scalar, return (x, x).

    Args:
        x: Scalar or sequence value.

    Returns:
        tuple[int, int]: Two-element tuple representation of the input size.
    """
    if isinstance(x, tuple):
        return x
    if isinstance(x, list):
        return x[0], x[1]
    return x, x


class ChannelLayerNorm2d(nn.Module):
    """Apply LayerNorm over channels for ``BCHW`` feature maps."""

    def __init__(self, num_channels: int, eps: float = 1e-5) -> None:
        """Store affine channel parameters compatible with HF LayerNorm weights."""
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize each spatial location across channels.

        Express the norm via an explicit channel-last view so export paths do
        not have to specialize on implicit layout assumptions.
        """
        if x.ndim != 4:
            raise ValueError(
                f"ChannelLayerNorm2d expects rank-4 input, got shape {tuple(x.shape)}"
            )
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, (self.weight.shape[0],), self.weight, self.bias, self.eps)
        return x.permute(0, 3, 1, 2)


class OverlapPatchMerging(nn.Module):
    """Convert an image tensor into overlapping patch embeddings."""

    def __init__(
        self,
        img_size: int | tuple[int, int] = 224,
        patch_size: int | tuple[int, int] = 7,
        stride: int = 4,
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        """Initialize the projection and normalization layers for patch embedding.

        Args:
            img_size: Input image size used for stage configuration.
            patch_size: Convolution kernel size for extracting patches.
            stride: Convolution stride between adjacent patches.
            in_chans: Number of input channels.
            embed_dim: Output embedding dimension per patch.

        Returns:
            None: Initializes the patch projection layers.
        """
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        padding = patch_size[0] // 2, patch_size[1] // 2
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=padding
        )
        self.norm = ChannelLayerNorm2d(embed_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        """Project an image into normalized patch feature maps.

        Args:
            x: Input image tensor of shape ``[batch, channels, height, width]``.

        Returns:
            tuple[torch.Tensor, int, int]: Feature map and its spatial height and width.
        """
        x = self.proj(x)
        _, _, H, W = x.shape
        x = self.norm(x)
        return x, H, W


class DropPath(nn.Module):
    """Apply stochastic depth to a residual branch during training."""

    def __init__(self, drop_prob: float) -> None:
        """Store the stochastic depth probability.

        Args:
            drop_prob: Probability of dropping a residual path during training.
        """
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Randomly drop full residual paths for the current batch.

        Args:
            x: Input tensor to regularize.

        Returns:
            torch.Tensor: Regularized residual branch output.
        """
        drop_prob = self.drop_prob
        if drop_prob == 0.0 or not self.training:
            return x
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random = torch.rand(shape, dtype=x.dtype, device=x.device)
        mask = (random > drop_prob).to(dtype=x.dtype)
        out = x * mask / (1.0 - drop_prob)
        return out


class EfficientSelfAttention(nn.Module):
    """Compute multi-head self-attention with optional spatial reduction."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = False,
        attn_drop_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
        sr_ratio: int = 1,
        use_sdpa_attn: bool = True,
    ) -> None:
        """Initialize attention projection layers and reduction settings.

        Args:
            dim: Token embedding dimension.
            num_heads: Number of attention heads.
            qkv_bias: Whether query, key, and value projections use bias.
            attn_drop_rate: Dropout rate applied to attention weights.
            proj_drop_rate: Dropout rate applied after the output projection.
            sr_ratio: Spatial reduction ratio for keys and values.
            use_sdpa_attn: Unused compatibility flag retained for constructor stability.

        Returns:
            None: Initializes attention projection modules.
        """
        super().__init__()
        del use_sdpa_attn
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0, (
            f"dim {dim} should be divisible by num_heads {num_heads}"
        )
        self.scale = self.head_dim**-0.5

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)
        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.proj_drop = nn.Dropout(proj_drop_rate)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = ChannelLayerNorm2d(dim)

    def _reshape_q(self, x: torch.Tensor) -> torch.Tensor:
        """Fold heads into the batch dimension for rank-3 batched matmul."""
        B, _, H, W = x.shape
        x = x.reshape(B * self.num_heads, self.head_dim, H * W)
        return x.transpose(1, 2)

    def _reshape_k(self, x: torch.Tensor) -> torch.Tensor:
        """Fold heads into the batch dimension without increasing tensor rank."""
        B, _, H, W = x.shape
        return x.reshape(B * self.num_heads, self.head_dim, H * W)

    def _reshape_v(self, x: torch.Tensor) -> torch.Tensor:
        """Fold heads into the batch dimension for attention value aggregation."""
        B, _, H, W = x.shape
        x = x.reshape(B * self.num_heads, self.head_dim, H * W)
        return x.transpose(1, 2)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """Apply self-attention to a feature map tensor.

        Args:
            x: Feature map tensor of shape ``[batch, channels, height, width]``.
            H: Feature map height corresponding to ``x``.
            W: Feature map width corresponding to ``x``.

        Returns:
            torch.Tensor: Attention output with the same shape as ``x``.
        """
        del H, W
        B, _, feat_h, feat_w = x.shape
        q = self._reshape_q(self.q(x))

        kv_src = x
        if self.sr_ratio > 1:
            kv_src = self.sr(kv_src)
            kv_src = self.norm(kv_src)

        kv = self.kv(kv_src)
        k, v = torch.chunk(kv, chunks=2, dim=1)
        k = self._reshape_k(k)
        v = self._reshape_v(v)

        attn = torch.bmm(q, k) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = torch.bmm(attn, v)
        x = x.transpose(1, 2).reshape(B, self.dim, feat_h, feat_w)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DWConv(nn.Module):
    """Apply depthwise convolution inside the MLP token mixing path."""

    def __init__(self, dim: int) -> None:
        """Initialize the depthwise convolution layer.

        Args:
            dim: Number of input and output channels.

        Returns:
            None: Initializes the depthwise convolution layer.
        """
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size=3, stride=1, padding=1, bias=True, groups=dim
        )

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """Apply depthwise convolution to a ``BCHW`` feature map."""
        del H, W
        return self.dwconv(x)


class MixFFN(nn.Module):
    """Project feature maps through a feed-forward block with depthwise mixing."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        act_layer: type[nn.Module] = nn.GELU,
        drop_rate: float = 0.0,
    ) -> None:
        """Initialize the MLP projection, activation, and dropout layers.

        Args:
            in_features: Input feature channel dimension.
            hidden_features: Hidden channel dimension used inside the block.
            out_features: Output feature channel dimension.
            act_layer: Activation layer class used between projections.
            drop_rate: Dropout rate applied after each projection.

        Returns:
            None: Initializes the feed-forward block layers.
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1)
        self.drop = nn.Dropout(drop_rate)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """Transform feature maps with pointwise and depthwise operations.

        Args:
            x: Feature map tensor of shape ``[batch, channels, height, width]``.
            H: Feature map height corresponding to ``x``.
            W: Feature map width corresponding to ``x``.

        Returns:
            torch.Tensor: Output feature map after feed-forward mixing.
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
        dim: int,
        num_heads: int,
        mlp_ratio: int,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        act_layer: type[nn.Module] = nn.GELU,
        norm_layer: type[nn.Module] = ChannelLayerNorm2d,
        sr_ratio: int = 1,
    ) -> None:
        """Initialize normalization, attention, and MLP submodules.

        Args:
            dim: Token embedding dimension.
            num_heads: Number of attention heads.
            mlp_ratio: Expansion ratio used inside the MLP block.
            qkv_bias: Whether attention projections use bias terms.
            drop_rate: Dropout rate used in attention and MLP projections.
            attn_drop_rate: Dropout rate applied to attention weights.
            drop_path_rate: Stochastic depth rate for residual branches.
            act_layer: Activation layer class used by the MLP.
            norm_layer: Normalization layer class used before each branch.
            sr_ratio: Spatial reduction ratio used by attention.

        Returns:
            None: Initializes the transformer block modules.
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = EfficientSelfAttention(
            dim,
            num_heads,
            qkv_bias=qkv_bias,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=drop_rate,
            sr_ratio=sr_ratio,
        )

        self.drop_path = DropPath(drop_path_rate)
        self.norm2 = norm_layer(dim)
        self.mlp = MixFFN(
            dim, dim * mlp_ratio, dim, act_layer=act_layer, drop_rate=drop_rate
        )

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """Apply attention and MLP residual updates to the feature map."""
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class MixTransformerStage(nn.Module):
    """Run one encoder stage while preserving spatial metadata."""

    def __init__(
        self,
        patch_embed: OverlapPatchMerging,
        blocks: nn.ModuleList,
        norm: nn.Module,
    ) -> None:
        """Store the stage modules in execution order."""
        super().__init__()
        self.patch_embed = patch_embed
        self.blocks = blocks
        self.norm = norm

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        """Apply patch embedding, transformer blocks, and stage normalization."""
        x, H, W = self.patch_embed(x)
        for block in self.blocks:
            x = block(x, H, W)
        x = self.norm(x)
        return x, H, W


class MixTransformer(nn.Module):
    """Encode an image into multi-scale feature maps with transformer stages."""

    def __init__(
        self,
        img_size: int = 224,
        in_chals: int = 3,
        num_classes: int = 1000,
        embed_dims: list[int] = [32, 64, 160, 256],
        num_heads: list[int] = [1, 2, 5, 8],
        mlp_ratios: list[int] = [4, 4, 4, 4],
        qkv_bias: bool = False,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: type[nn.Module] = ChannelLayerNorm2d,
        depths: list[int] = [2, 2, 2, 2],
        sr_ratios: list[int] = [8, 4, 2, 1],
    ) -> None:
        """Initialize the hierarchical Mix Transformer encoder."""
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        inp_reductions = [1, 4, 8, 16]
        assert len(inp_reductions) == len(depths)
        assert len(num_heads) == len(depths)
        assert len(sr_ratios) == len(depths)
        assert len(embed_dims) == len(depths)

        stages = []
        dims = [in_chals] + embed_dims
        depth_drop_rates = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]
        cur = 0
        for l_i in range(len(depths)):
            stages.append(
                MixTransformerStage(
                    patch_embed=OverlapPatchMerging(
                        img_size=img_size // inp_reductions[l_i],
                        patch_size=7 if l_i == 0 else 3,
                        stride=4 if l_i == 0 else 2,
                        in_chans=dims[l_i],
                        embed_dim=dims[l_i + 1],
                    ),
                    blocks=nn.ModuleList(
                        [
                            Block(
                                dim=dims[l_i + 1],
                                num_heads=num_heads[l_i],
                                mlp_ratio=mlp_ratios[l_i],
                                qkv_bias=qkv_bias,
                                drop_rate=drop_rate,
                                attn_drop_rate=attn_drop_rate,
                                drop_path_rate=depth_drop_rates[cur + b_i],
                                norm_layer=norm_layer,
                                sr_ratio=sr_ratios[l_i],
                            )
                            for b_i in range(depths[l_i])
                        ]
                    ),
                    norm=norm_layer(dims[l_i + 1]),
                )
            )
            cur += depths[l_i]

        self.stages = nn.ModuleList(stages)

        if self.num_classes > 0:
            self.head = nn.Linear(embed_dims[3], num_classes)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        """Initialize supported module weights in place."""
        if isinstance(m, ChannelLayerNorm2d):
            nn.init.constant_(m.bias, 0.0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0.0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out = fan_out / m.groups
            nn.init.normal_(m.weight, mean=0, std=(2 / fan_out) ** 0.5)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor] | torch.Tensor:
        """Encode an input image into stage outputs or class logits."""
        outs = []

        for stage in self.stages:
            x, _, _ = stage(x)
            outs.append(x)

        if self.num_classes > 0:
            x = outs[-1].mean(dim=(2, 3))
            return self.head(x)

        return outs
