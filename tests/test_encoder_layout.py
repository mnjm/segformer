"""Regression tests for the edge-optimized SegFormer encoder layout."""

import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from model import SegFormer, SegFormerConfig  # noqa: E402


def build_small_model() -> SegFormer:
    """Create a compact SegFormer instance for encoder layout tests."""
    cfg = SegFormerConfig(
        img_size=64,
        num_classes=3,
        embed_dims=[32, 64, 160, 256],
        num_heads=[1, 2, 5, 8],
        depths=[2, 2, 2, 2],
        sr_ratios=[8, 4, 2, 1],
        decoder_dim=64,
        decoder_drop_rate=0.0,
    )
    return SegFormer(cfg).eval()


def test_encoder_stage_outputs_use_bchw_layout() -> None:
    """Encoder stages should emit feature maps instead of token tensors."""
    model = build_small_model()
    x = torch.randn(2, 3, 64, 64)

    with torch.no_grad():
        outputs = model.encoder(x)

    assert len(outputs) == 4
    expected_shapes = [
        (2, 32, 16, 16),
        (2, 64, 8, 8),
        (2, 160, 4, 4),
        (2, 256, 2, 2),
    ]
    for output, expected_shape in zip(outputs, expected_shapes, strict=True):
        assert output.shape == expected_shape


def test_encoder_uses_conv_layout_friendly_projections() -> None:
    """The encoder should avoid tokenwise Linear layers after the rewrite."""
    model = build_small_model()
    linear_modules = [
        name
        for name, module in model.encoder.named_modules()
        if isinstance(module, nn.Linear)
    ]
    assert linear_modules == []


def test_encoder_graph_stays_at_rank_four_or_less() -> None:
    """Symbolically traced encoder graphs should not introduce rank-5 tensors."""
    model = build_small_model()
    traced = symbolic_trace(model.encoder)
    ShapeProp(traced).propagate(torch.randn(1, 3, 64, 64))

    for node in traced.graph.nodes:
        tensor_meta = node.meta.get("tensor_meta")
        if tensor_meta is None:
            continue
        shape = getattr(tensor_meta, "shape", None)
        if shape is None:
            continue
        assert len(shape) <= 4, f"{node.op} {node.target} produced rank-{len(shape)} tensor"
        assert node.target != "permute"
