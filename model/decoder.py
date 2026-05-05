"""
Decoder used in SegFormer architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SegFormerDecoder(nn.Module):
    """All MLP SegFormer Decoder"""

    def __init__(
        self,
        in_chals: list[int] = [32, 64, 160, 256],
        embed_dim: int = 256,
        drop_rate: float = 0.0,
        num_classes: int = 150,
    ) -> None:
        """Initialize the decoder projection, fusion, and classifier layers.

        Args:
            in_chals: Channel sizes of the encoder feature maps.
            embed_dim: Common channel size used after per-scale projection.
            drop_rate: Dropout rate applied before the final classifier.
            num_classes: Number of segmentation classes to predict.

        Returns:
            None: Initializes decoder layers in place.
        """
        super().__init__()
        # SegFormer uses MLP layers to unify the channel dimensions of the
        # multi-scale encoder outputs. Here, I use a 1x1 convolution instead.
        #
        # A 1x1 convolution is mathematically equivalent to applying the same
        # Linear(C_in, C_out) layer independently at every spatial location.
        # It is more convenient here because the features are already in
        # image/tensor format: [B, C, H, W].
        #
        # With an MLP, we would need to reshape each feature map into tokens:
        #   [B, C, H, W]
        #   -> flatten spatial dims: [B, C, H*W]
        #   -> transpose:           [B, H*W, C]
        #   -> Linear(C_in, C_out): [B, H*W, C_out]
        #   -> transpose/reshape:   [B, C_out, H, W]
        #
        # A 1x1 convolution performs the same channel projection directly:
        #   [B, C_in, H, W] -> [B, C_out, H, W]
        #
        # This avoids extra flatten/transpose/reshape operations while preserving
        # the same per-pixel channel-unification behavior.
        self.linear_c = nn.ModuleList(
            [nn.Conv2d(c_in, embed_dim, kernel_size=1) for c_in in in_chals]
        )

        self.linear_fuse = nn.Sequential(
            nn.Conv2d(embed_dim * len(in_chals), embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(),
        )

        self.dropout = nn.Dropout(drop_rate)

        self.clfr = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        """Initialize supported module weights in place.

        Args:
            m: Module instance to initialize.

        Returns:
            None: Mutates supported module weights in place.
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

    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        """Project, align, and fuse encoder features into segmentation logits.

        Args:
            inputs: List of feature tensors ordered from high to low resolution.

        Returns:
            torch.Tensor: Decoder logits at the highest encoder resolution.
        """
        t_H, t_W = inputs[0].shape[2:]

        c_l = []
        for i in reversed(range(len(self.linear_c))):
            x = self.linear_c[i](inputs[i])
            if i > 0:
                x = F.interpolate(x, (t_H, t_W), mode="bilinear", align_corners=False)
            c_l.append(x)

        x = torch.cat(c_l, dim=1)
        x = self.linear_fuse(x)
        x = self.dropout(x)

        return self.clfr(x)
