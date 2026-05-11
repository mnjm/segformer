"""SegFormer model definitions and configuration."""

import logging
from dataclasses import dataclass, field
from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from .decoder import SegFormerDecoder
from .hf_mapper import build_encoder_state_dict, load_hf_model
from .mix_transformer import ChannelLayerNorm2d, MixTransformer

logger = logging.getLogger(__name__)


@dataclass
class SegFormerConfig:
    """Store SegFormer encoder and decoder settings.

    Args:
        name: Model name (e.g., "segformer-b0").
        img_size: Input image size used to configure the encoder stages.
        in_chals: Number of input image channels.
        num_classes: Number of segmentation classes to predict.
        embed_dims: Channel dimensions used at each encoder stage.
        num_heads: Attention head counts for each encoder stage.
        mlp_ratios: Expansion ratios for the MLP blocks in each stage.
        qkv_bias: Whether to use bias terms in attention projections.
        drop_rate: Dropout rate used in encoder projection layers.
        attn_drop_rate: Dropout rate applied to attention weights.
        drop_path_rate: Maximum stochastic depth rate across encoder blocks.
        norm_layer: Normalization layer class used in the encoder.
        depths: Number of transformer blocks per encoder stage.
        sr_ratios: Spatial reduction ratios used by attention in each stage.
        decoder_dim: Shared decoder channel dimension.
        decoder_drop_rate: Dropout rate applied before the segmentation classifier.
    """

    name: str = "segformer-b0"
    img_size: int = 224
    in_chals: int = 3
    num_classes: int = 150
    embed_dims: list[int] = field(default_factory=lambda: [32, 64, 160, 256])
    num_heads: list[int] = field(default_factory=lambda: [1, 2, 5, 8])
    mlp_ratios: list[int] = field(default_factory=lambda: [4, 4, 4, 4])
    qkv_bias: bool = True
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.0
    norm_layer: type[nn.Module] = ChannelLayerNorm2d
    depths: list[int] = field(default_factory=lambda: [2, 2, 2, 2])
    sr_ratios: list[int] = field(default_factory=lambda: [8, 4, 2, 1])
    decoder_dim: int = 256
    decoder_drop_rate: float = 0.0


class SegFormer(nn.Module):
    """Build a SegFormer model from a transformer encoder and segmentation decoder."""

    def __init__(self, cfg: SegFormerConfig, ignore_idx: int | None = None) -> None:
        """Initialize the SegFormer model components.

        Args:
            cfg: SegFormer configuration values for the encoder and decoder.
            loss_fn: Optional segmentation loss module.

        Returns:
            None: Initializes the encoder and decoder modules.
        """
        super().__init__()
        self.cfg = cfg
        if ignore_idx is not None:
            self.loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_idx)
        else:
            self.loss_fn = nn.CrossEntropyLoss()

        self.encoder = MixTransformer(
            img_size=cfg.img_size,
            in_chals=cfg.in_chals,
            num_classes=0,
            embed_dims=cfg.embed_dims,
            num_heads=cfg.num_heads,
            mlp_ratios=cfg.mlp_ratios,
            qkv_bias=cfg.qkv_bias,
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

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Run segmentation inference on an input batch.

        Args:
            x: Input image tensor of shape ``[batch, channels, height, width]``.
            y: Optional target mask tensor for loss computation.

        Returns:
            torch.Tensor | tuple[torch.Tensor, torch.Tensor]: Logits alone, or logits and loss.
        """
        features = self.encoder(x)
        logits = self.decoder(features)
        logits = F.interpolate(logits, size=x.shape[2:], mode="bilinear", align_corners=False)
        if y is not None and self.loss_fn is not None:
            loss = self.loss_fn(logits, y)
            return logits, loss
        return logits

    def load_pretrained_encoder(self) -> None:
        """Load pretrained Hugging Face weights into the local encoder.

        Returns:
            None: Updates encoder parameters in place.
        """
        hf_model = load_hf_model(self.cfg.name)
        incompatible = self.encoder.load_state_dict(
            build_encoder_state_dict(hf_model),
            strict=True,
        )
        missing_keys = incompatible.missing_keys
        unexpected_keys = incompatible.unexpected_keys
        assert len(missing_keys) == 0, f"Missing keys: {missing_keys}"
        assert len(unexpected_keys) == 0, f"Unexpected keys: {unexpected_keys}"
        logger.info(f"Loaded pretrained encoder for {self.cfg.name}")

    def configure_optimizer(
        self,
        optim_cfg: DictConfig,
        device: torch.device,
    ) -> torch.optim.Optimizer:
        """Build an optimizer with SegFormer-style parameter-wise settings.

        Args:
            optim_cfg: Config with optimizer type, base LR, and decay settings.
            device: Used to decide fused optimizer on CUDA.

        Returns:
            torch.optim.Optimizer: Configured optimizer instance.
        """
        supported_optimizers_map = {"adamw": torch.optim.AdamW}
        assert optim_cfg.type in supported_optimizers_map, (
            f"{optim_cfg.type=} optimizer not supported"
        )
        optim_init = supported_optimizers_map[optim_cfg.type]

        optim_cfg.fused = getattr(optim_cfg, "fused", False) and device.type == "cuda"
        weight_decay = getattr(optim_cfg, "weight_decay", 1e-2)
        lr = optim_cfg.lr

        optim_groups: list[dict[str, Any]] = []

        def build_weight_decay_param_groups(
            params: list[nn.Parameter],
            weight_decay: float,
            lr: float,
        ) -> list[dict[str, Any]]:
            """Split parameters into decay and no-decay optimizer groups.

            Args:
                params: Parameters to group.
                weight_decay: Weight decay applied to matrix-like parameters.
                lr: Learning rate applied to both parameter groups.

            Returns:
                list[dict[str, Any]]: Optimizer parameter group dictionaries.
            """
            decay_parmas = [p for p in params if p.dim() >= 2]
            no_decay_parmas = [p for p in params if p.dim() < 2]
            return [
                {"params": decay_parmas, "weight_decay": weight_decay, "lr": lr},
                {"params": no_decay_parmas, "weight_decay": 0.0, "lr": lr},
            ]

        encoder_params = [p for p in self.encoder.parameters() if p.requires_grad]
        decoder_params = [p for p in self.decoder.parameters() if p.requires_grad]
        optim_groups.extend(
            build_weight_decay_param_groups(encoder_params, lr=lr, weight_decay=weight_decay)
        )
        optim_groups.extend(
            build_weight_decay_param_groups(decoder_params, lr=lr * 10.0, weight_decay=weight_decay)
        )

        kwargs = cast(dict[str, Any], dict(optim_cfg))
        del kwargs["type"]
        optimizer = optim_init(optim_groups, **kwargs)
        return optimizer


if __name__ == "__main__":
    model = SegFormer(SegFormerConfig())
    model.load_pretrained_encoder()
    print(f"{sum([x.numel() for x in model.parameters()]) / 1e6:.2f}M parameters")
    model = torch.compile(model, fullgraph=True)
    print("Model Compiled")
    x = torch.randn(1, 3, 224, 224)
    output = model(x)
    print(f"{output.shape=}")
