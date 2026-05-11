"""HF parity tests for SegFormer model variants."""

import sys
from pathlib import Path
from typing import Any, cast

import pytest
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from model import SegFormer, SegFormerConfig  # noqa: E402
from model.hf_mapper import (  # noqa: E402
    HF_MODEL_IDS,
    build_model_state_dict,
    load_hf_model,
)

CONFIG_DIR = ROOT_DIR / "config" / "model"


def load_local_config(config_path: Path, num_classes: int) -> SegFormerConfig:
    """Load one SegFormer YAML config file.

    Args:
        config_path: Path to the YAML config for one SegFormer variant.
        num_classes: Number of labels expected by the HF checkpoint.

    Returns:
        SegFormerConfig: Parsed local model configuration.
    """
    cfg = OmegaConf.load(config_path)
    cfg.num_classes = num_classes
    raw_cfg = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(raw_cfg, dict)
    raw_cfg = cast(dict[str, Any], raw_cfg)
    return SegFormerConfig(**raw_cfg)


@pytest.mark.parametrize(
    "config_path",
    sorted(CONFIG_DIR.glob("segformer-b*.yaml")),
    ids=lambda path: path.stem,
)
def test_hf_segformer_parity(config_path: Path) -> None:
    """Check parameter count and logits parity for each SegFormer variant.

    Args:
        config_path: YAML config path provided by pytest parametrization.

    Returns:
        None: Asserts parity against the Hugging Face implementation.
    """
    assert config_path.stem in HF_MODEL_IDS
    hf_model = load_hf_model(config_path.stem)
    local_cfg = load_local_config(config_path, num_classes=hf_model.config.num_labels)
    local_model = SegFormer(local_cfg).eval()

    incompatible = local_model.load_state_dict(
        build_model_state_dict(hf_model),
        strict=True,
    )
    missing_keys = incompatible.missing_keys
    unexpected_keys = incompatible.unexpected_keys
    assert missing_keys == []
    assert unexpected_keys == []

    hf_params = sum(param.numel() for param in hf_model.parameters())
    local_params = sum(param.numel() for param in local_model.parameters())
    assert local_params == hf_params

    torch.manual_seed(7)
    pixel_values = torch.randn(2, local_cfg.in_chals, local_cfg.img_size, local_cfg.img_size)

    with torch.no_grad():
        hf_logits = hf_model(pixel_values).logits
        hf_logits = F.interpolate(
            hf_logits, size=pixel_values.shape[-2:], mode="bilinear", align_corners=False
        )
        local_logits = local_model(pixel_values)

    assert local_logits.shape == hf_logits.shape
    torch.testing.assert_close(local_logits, hf_logits, rtol=1e-4, atol=1e-4)
