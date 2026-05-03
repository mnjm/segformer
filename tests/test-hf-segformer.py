"""HF parity tests for SegFormer model variants."""

import sys
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from omegaconf import OmegaConf
from transformers import SegformerForSemanticSegmentation

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from model import SegFormer, SegFormerConfig  # noqa: E402

CONFIG_DIR = ROOT_DIR / "configs" / "models"
CACHE_DIR = ROOT_DIR / "cache"
HF_MODEL_IDS = {
    "segformer-b0": "nvidia/segformer-b0-finetuned-ade-512-512",
    "segformer-b1": "nvidia/segformer-b1-finetuned-ade-512-512",
    "segformer-b2": "nvidia/segformer-b2-finetuned-ade-512-512",
    "segformer-b3": "nvidia/segformer-b3-finetuned-ade-512-512",
    "segformer-b4": "nvidia/segformer-b4-finetuned-ade-512-512",
    "segformer-b5": "nvidia/segformer-b5-finetuned-ade-640-640",
}


def load_local_config(config_path: Path) -> SegFormerConfig:
    """Load one SegFormer YAML config file.

    Args:
        config_path: Path to the YAML config for one SegFormer variant.
    """
    raw_cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    assert isinstance(raw_cfg, dict)
    raw_cfg = cast(dict[str, Any], raw_cfg)
    return SegFormerConfig(**raw_cfg)


def build_local_state_dict(
    hf_model: SegformerForSemanticSegmentation,
) -> dict[str, torch.Tensor]:
    """Convert HF SegFormer weights into the local module layout.

    Args:
        hf_model: Loaded Hugging Face SegFormer segmentation model.
    """
    hf_state = hf_model.state_dict()
    local_state: dict[str, torch.Tensor] = {}

    num_stages = len(hf_model.config.depths)
    for stage_idx in range(num_stages):
        local_state[f"encoder.patch_embeds.{stage_idx}.proj.weight"] = hf_state[
            f"segformer.encoder.patch_embeddings.{stage_idx}.proj.weight"
        ]
        local_state[f"encoder.patch_embeds.{stage_idx}.proj.bias"] = hf_state[
            f"segformer.encoder.patch_embeddings.{stage_idx}.proj.bias"
        ]
        local_state[f"encoder.patch_embeds.{stage_idx}.norm.weight"] = hf_state[
            f"segformer.encoder.patch_embeddings.{stage_idx}.layer_norm.weight"
        ]
        local_state[f"encoder.patch_embeds.{stage_idx}.norm.bias"] = hf_state[
            f"segformer.encoder.patch_embeddings.{stage_idx}.layer_norm.bias"
        ]
        local_state[f"encoder.norms.{stage_idx}.weight"] = hf_state[
            f"segformer.encoder.layer_norm.{stage_idx}.weight"
        ]
        local_state[f"encoder.norms.{stage_idx}.bias"] = hf_state[
            f"segformer.encoder.layer_norm.{stage_idx}.bias"
        ]

        for block_idx in range(hf_model.config.depths[stage_idx]):
            base = f"segformer.encoder.block.{stage_idx}.{block_idx}"
            local_base = f"encoder.blocks.{stage_idx}.{block_idx}"

            local_state[f"{local_base}.norm1.weight"] = hf_state[
                f"{base}.layer_norm_1.weight"
            ]
            local_state[f"{local_base}.norm1.bias"] = hf_state[
                f"{base}.layer_norm_1.bias"
            ]
            local_state[f"{local_base}.norm2.weight"] = hf_state[
                f"{base}.layer_norm_2.weight"
            ]
            local_state[f"{local_base}.norm2.bias"] = hf_state[
                f"{base}.layer_norm_2.bias"
            ]

            local_state[f"{local_base}.attn.q.weight"] = hf_state[
                f"{base}.attention.self.query.weight"
            ]
            local_state[f"{local_base}.attn.q.bias"] = hf_state[
                f"{base}.attention.self.query.bias"
            ]
            local_state[f"{local_base}.attn.kv.weight"] = torch.cat(
                [
                    hf_state[f"{base}.attention.self.key.weight"],
                    hf_state[f"{base}.attention.self.value.weight"],
                ],
                dim=0,
            )
            local_state[f"{local_base}.attn.kv.bias"] = torch.cat(
                [
                    hf_state[f"{base}.attention.self.key.bias"],
                    hf_state[f"{base}.attention.self.value.bias"],
                ],
                dim=0,
            )
            local_state[f"{local_base}.attn.proj.weight"] = hf_state[
                f"{base}.attention.output.dense.weight"
            ]
            local_state[f"{local_base}.attn.proj.bias"] = hf_state[
                f"{base}.attention.output.dense.bias"
            ]

            sr_ratio = hf_model.config.sr_ratios[stage_idx]
            if sr_ratio > 1:
                local_state[f"{local_base}.attn.sr.weight"] = hf_state[
                    f"{base}.attention.self.sr.weight"
                ]
                local_state[f"{local_base}.attn.sr.bias"] = hf_state[
                    f"{base}.attention.self.sr.bias"
                ]
                local_state[f"{local_base}.attn.norm.weight"] = hf_state[
                    f"{base}.attention.self.layer_norm.weight"
                ]
                local_state[f"{local_base}.attn.norm.bias"] = hf_state[
                    f"{base}.attention.self.layer_norm.bias"
                ]

            local_state[f"{local_base}.mlp.fc1.weight"] = hf_state[
                f"{base}.mlp.dense1.weight"
            ]
            local_state[f"{local_base}.mlp.fc1.bias"] = hf_state[
                f"{base}.mlp.dense1.bias"
            ]
            local_state[f"{local_base}.mlp.dwconv.dwconv.weight"] = hf_state[
                f"{base}.mlp.dwconv.dwconv.weight"
            ]
            local_state[f"{local_base}.mlp.dwconv.dwconv.bias"] = hf_state[
                f"{base}.mlp.dwconv.dwconv.bias"
            ]
            local_state[f"{local_base}.mlp.fc2.weight"] = hf_state[
                f"{base}.mlp.dense2.weight"
            ]
            local_state[f"{local_base}.mlp.fc2.bias"] = hf_state[
                f"{base}.mlp.dense2.bias"
            ]

    for stage_idx in range(num_stages):
        local_state[f"decoder.linear_c.{stage_idx}.weight"] = (
            hf_state[f"decode_head.linear_c.{stage_idx}.proj.weight"]
            .unsqueeze(-1)
            .unsqueeze(-1)
        )
        local_state[f"decoder.linear_c.{stage_idx}.bias"] = hf_state[
            f"decode_head.linear_c.{stage_idx}.proj.bias"
        ]

    local_state["decoder.linear_fuse.0.weight"] = hf_state[
        "decode_head.linear_fuse.weight"
    ]
    local_state["decoder.linear_fuse.1.weight"] = hf_state[
        "decode_head.batch_norm.weight"
    ]
    local_state["decoder.linear_fuse.1.bias"] = hf_state["decode_head.batch_norm.bias"]
    local_state["decoder.linear_fuse.1.running_mean"] = hf_state[
        "decode_head.batch_norm.running_mean"
    ]
    local_state["decoder.linear_fuse.1.running_var"] = hf_state[
        "decode_head.batch_norm.running_var"
    ]
    local_state["decoder.linear_fuse.1.num_batches_tracked"] = hf_state[
        "decode_head.batch_norm.num_batches_tracked"
    ]
    local_state["decoder.clfr.weight"] = hf_state["decode_head.classifier.weight"]
    local_state["decoder.clfr.bias"] = hf_state["decode_head.classifier.bias"]

    return local_state


@pytest.mark.parametrize(
    "config_path",
    sorted(CONFIG_DIR.glob("segformer-b*.yaml")),
    ids=lambda path: path.stem,
)
def test_hf_segformer_parity(config_path: Path):
    """Check parameter count and logits parity for each SegFormer variant.

    Args:
        config_path: YAML config path provided by pytest parametrization.
    """
    local_cfg = load_local_config(config_path)
    hf_model_id = HF_MODEL_IDS[config_path.stem]
    hf_model = SegformerForSemanticSegmentation.from_pretrained(
        hf_model_id,
        cache_dir=str(CACHE_DIR),
    ).eval()
    local_model = SegFormer(local_cfg).eval()

    local_state = build_local_state_dict(hf_model)
    missing_keys, unexpected_keys = local_model.load_state_dict(
        local_state, strict=True
    )
    assert missing_keys == []
    assert unexpected_keys == []

    hf_params = sum(param.numel() for param in hf_model.parameters())
    local_params = sum(param.numel() for param in local_model.parameters())
    assert local_params == hf_params

    torch.manual_seed(7)
    pixel_values = torch.randn(
        2, local_cfg.in_chals, local_cfg.img_size, local_cfg.img_size
    )

    with torch.no_grad():
        hf_logits = hf_model(pixel_values).logits
        local_logits = local_model(pixel_values)

    assert local_logits.shape == hf_logits.shape
    torch.testing.assert_close(local_logits, hf_logits, rtol=1e-4, atol=1e-4)
