"""Map Hugging Face SegFormer weights onto local encoder and decoder modules."""

from pathlib import Path

import torch
from transformers import SegformerForSemanticSegmentation

HF_MODEL_IDS = {
    "segformer-b0": "nvidia/segformer-b0-finetuned-ade-512-512",
    "segformer-b1": "nvidia/segformer-b1-finetuned-ade-512-512",
    "segformer-b2": "nvidia/segformer-b2-finetuned-ade-512-512",
    "segformer-b3": "nvidia/segformer-b3-finetuned-ade-512-512",
    "segformer-b4": "nvidia/segformer-b4-finetuned-ade-512-512",
    "segformer-b5": "nvidia/segformer-b5-finetuned-ade-640-640",
}

CACHE_DIR = Path(__file__).resolve().parents[1] / "cache"


def load_hf_model(
    model_name: str,
) -> SegformerForSemanticSegmentation:
    """Load one pretrained HF SegFormer segmentation model.

    Args:
        model_name: Local variant name such as ``"segformer-b0"``.

    Returns:
        SegformerForSemanticSegmentation: Loaded pretrained Hugging Face model.
    """
    return SegformerForSemanticSegmentation.from_pretrained(
        HF_MODEL_IDS[model_name], cache_dir=CACHE_DIR
    )


def linear_to_conv2d_weight(weight: torch.Tensor) -> torch.Tensor:
    """Expand a Linear weight matrix into a ``1x1`` convolution kernel."""
    return weight.unsqueeze(-1).unsqueeze(-1)


def build_encoder_state_dict(
    hf_model: SegformerForSemanticSegmentation,
) -> dict[str, torch.Tensor]:
    """Convert HF encoder weights into the local encoder layout.

    Args:
        hf_model: Loaded Hugging Face SegFormer segmentation model.

    Returns:
        dict[str, torch.Tensor]: Encoder weights mapped to local module names.
    """
    hf_state = hf_model.state_dict()
    local_state: dict[str, torch.Tensor] = {}

    num_stages = len(hf_model.config.depths)
    for stage_idx in range(num_stages):
        stage_base = f"stages.{stage_idx}"

        local_state[f"{stage_base}.patch_embed.proj.weight"] = hf_state[
            f"segformer.encoder.patch_embeddings.{stage_idx}.proj.weight"
        ]
        local_state[f"{stage_base}.patch_embed.proj.bias"] = hf_state[
            f"segformer.encoder.patch_embeddings.{stage_idx}.proj.bias"
        ]
        local_state[f"{stage_base}.patch_embed.norm.weight"] = hf_state[
            f"segformer.encoder.patch_embeddings.{stage_idx}.layer_norm.weight"
        ]
        local_state[f"{stage_base}.patch_embed.norm.bias"] = hf_state[
            f"segformer.encoder.patch_embeddings.{stage_idx}.layer_norm.bias"
        ]
        local_state[f"{stage_base}.norm.weight"] = hf_state[
            f"segformer.encoder.layer_norm.{stage_idx}.weight"
        ]
        local_state[f"{stage_base}.norm.bias"] = hf_state[
            f"segformer.encoder.layer_norm.{stage_idx}.bias"
        ]

        for block_idx in range(hf_model.config.depths[stage_idx]):
            hf_base = f"segformer.encoder.block.{stage_idx}.{block_idx}"
            local_base = f"{stage_base}.blocks.{block_idx}"

            local_state[f"{local_base}.norm1.weight"] = hf_state[f"{hf_base}.layer_norm_1.weight"]
            local_state[f"{local_base}.norm1.bias"] = hf_state[f"{hf_base}.layer_norm_1.bias"]
            local_state[f"{local_base}.norm2.weight"] = hf_state[f"{hf_base}.layer_norm_2.weight"]
            local_state[f"{local_base}.norm2.bias"] = hf_state[f"{hf_base}.layer_norm_2.bias"]

            local_state[f"{local_base}.attn.q.weight"] = linear_to_conv2d_weight(
                hf_state[f"{hf_base}.attention.self.query.weight"]
            )
            local_state[f"{local_base}.attn.q.bias"] = hf_state[
                f"{hf_base}.attention.self.query.bias"
            ]
            local_state[f"{local_base}.attn.kv.weight"] = linear_to_conv2d_weight(
                torch.cat(
                    [
                        hf_state[f"{hf_base}.attention.self.key.weight"],
                        hf_state[f"{hf_base}.attention.self.value.weight"],
                    ],
                    dim=0,
                )
            )
            local_state[f"{local_base}.attn.kv.bias"] = torch.cat(
                [
                    hf_state[f"{hf_base}.attention.self.key.bias"],
                    hf_state[f"{hf_base}.attention.self.value.bias"],
                ],
                dim=0,
            )
            local_state[f"{local_base}.attn.proj.weight"] = linear_to_conv2d_weight(
                hf_state[f"{hf_base}.attention.output.dense.weight"]
            )
            local_state[f"{local_base}.attn.proj.bias"] = hf_state[
                f"{hf_base}.attention.output.dense.bias"
            ]

            sr_ratio = hf_model.config.sr_ratios[stage_idx]
            if sr_ratio > 1:
                local_state[f"{local_base}.attn.sr.weight"] = hf_state[
                    f"{hf_base}.attention.self.sr.weight"
                ]
                local_state[f"{local_base}.attn.sr.bias"] = hf_state[
                    f"{hf_base}.attention.self.sr.bias"
                ]
                local_state[f"{local_base}.attn.norm.weight"] = hf_state[
                    f"{hf_base}.attention.self.layer_norm.weight"
                ]
                local_state[f"{local_base}.attn.norm.bias"] = hf_state[
                    f"{hf_base}.attention.self.layer_norm.bias"
                ]

            local_state[f"{local_base}.mlp.fc1.weight"] = linear_to_conv2d_weight(
                hf_state[f"{hf_base}.mlp.dense1.weight"]
            )
            local_state[f"{local_base}.mlp.fc1.bias"] = hf_state[f"{hf_base}.mlp.dense1.bias"]
            local_state[f"{local_base}.mlp.dwconv.dwconv.weight"] = hf_state[
                f"{hf_base}.mlp.dwconv.dwconv.weight"
            ]
            local_state[f"{local_base}.mlp.dwconv.dwconv.bias"] = hf_state[
                f"{hf_base}.mlp.dwconv.dwconv.bias"
            ]
            local_state[f"{local_base}.mlp.fc2.weight"] = linear_to_conv2d_weight(
                hf_state[f"{hf_base}.mlp.dense2.weight"]
            )
            local_state[f"{local_base}.mlp.fc2.bias"] = hf_state[f"{hf_base}.mlp.dense2.bias"]

    return local_state


def build_decoder_state_dict(
    hf_model: SegformerForSemanticSegmentation,
) -> dict[str, torch.Tensor]:
    """Convert HF decoder weights into the local decoder layout.

    Args:
        hf_model: Loaded Hugging Face SegFormer segmentation model.

    Returns:
        dict[str, torch.Tensor]: Decoder weights mapped to local module names.
    """
    hf_state = hf_model.state_dict()
    local_state: dict[str, torch.Tensor] = {}

    num_stages = len(hf_model.config.depths)
    for stage_idx in range(num_stages):
        local_state[f"linear_c.{stage_idx}.weight"] = (
            hf_state[f"decode_head.linear_c.{stage_idx}.proj.weight"].unsqueeze(-1).unsqueeze(-1)
        )
        local_state[f"linear_c.{stage_idx}.bias"] = hf_state[
            f"decode_head.linear_c.{stage_idx}.proj.bias"
        ]

    local_state["linear_fuse.0.weight"] = hf_state["decode_head.linear_fuse.weight"]
    local_state["linear_fuse.1.weight"] = hf_state["decode_head.batch_norm.weight"]
    local_state["linear_fuse.1.bias"] = hf_state["decode_head.batch_norm.bias"]
    local_state["linear_fuse.1.running_mean"] = hf_state["decode_head.batch_norm.running_mean"]
    local_state["linear_fuse.1.running_var"] = hf_state["decode_head.batch_norm.running_var"]
    local_state["linear_fuse.1.num_batches_tracked"] = hf_state[
        "decode_head.batch_norm.num_batches_tracked"
    ]
    local_state["clfr.weight"] = hf_state["decode_head.classifier.weight"]
    local_state["clfr.bias"] = hf_state["decode_head.classifier.bias"]

    return local_state


def build_model_state_dict(
    hf_model: SegformerForSemanticSegmentation,
) -> dict[str, torch.Tensor]:
    """Convert HF weights into the local full-model layout.

    Args:
        hf_model: Loaded Hugging Face SegFormer segmentation model.

    Returns:
        dict[str, torch.Tensor]: Combined encoder and decoder state dict.
    """
    encoder_state = {
        f"encoder.{key}": value for key, value in build_encoder_state_dict(hf_model).items()
    }
    decoder_state = {
        f"decoder.{key}": value for key, value in build_decoder_state_dict(hf_model).items()
    }
    return encoder_state | decoder_state
