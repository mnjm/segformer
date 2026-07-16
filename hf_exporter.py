"""
Export a training checkpoint as a Hugging Face SegFormer model.
Exported model can be loaded  with 'SegformerForSemanticSegmentation.from_pretrained'
"""

import argparse
from pathlib import Path
from typing import Any, Mapping, cast

import torch
from omegaconf import OmegaConf
from transformers import SegformerConfig, SegformerForSemanticSegmentation, SegformerImageProcessor

from utils import torch_compile_ckpt_fix

def load_dataset_cfg(dataset_name: str) -> dict[str, Any]:
    """Load a dataset YAML configuration.

    Args:
        dataset_name: Dataset config name without the .yaml suffix.

    Returns:
        Resolved dataset configuration values.
    """
    cfg_path = Path("./config/dataset") / f"{dataset_name}.yaml"
    assert cfg_path.exists(), f"Dataset config not found: {cfg_path}"
    return cast(dict[str, Any], OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True))

def build_hf_config(model_cfg: dict[str, Any], dataset_cfg: dict[str, Any]) -> SegformerConfig:
    """Build a config matching this repository's model.

    Args:
        model_cfg: Model configuration from the checkpoint.
        dataset_cfg: Dataset configuration for labels and ignore index.

    Returns:
        Transformers SegFormer configuration.
    """
    labels = list(cast(list[str], dataset_cfg["classes"]))
    id2label = dict(enumerate(labels))
    label2id = {label: index for index, label in id2label.items()}

    return SegformerConfig(
        num_channels=int(model_cfg["in_chals"]),
        depths=list(cast(list[int], model_cfg["depths"])),
        sr_ratios=list(cast(list[int], model_cfg["sr_ratios"])),
        hidden_sizes=list(cast(list[int], model_cfg["embed_dims"])),
        num_attention_heads=list(cast(list[int], model_cfg["num_heads"])),
        mlp_ratios=list(cast(list[int], model_cfg["mlp_ratios"])),
        hidden_dropout_prob=float(model_cfg["drop_rate"]),
        attention_probs_dropout_prob=float(model_cfg["attn_drop_rate"]),
        classifier_dropout_prob=float(model_cfg["decoder_drop_rate"]),
        drop_path_rate=float(model_cfg["drop_path_rate"]),
        decoder_hidden_size=int(model_cfg["decoder_dim"]),
        semantic_loss_ignore_index=int(dataset_cfg.get("ignore_idx", 255)),
        id2label=id2label,
        label2id=label2id,
    )

def build_hf_state_dict(
    local_state: Mapping[str, torch.Tensor], hf_cfg: SegformerConfig
) -> dict[str, torch.Tensor]:
    """Map repository checkpoint parameters to Transformers names.

    Args:
        local_state: Checkpoint model state dictionary.
        hf_cfg: Transformers SegFormer configuration describing stage depths.

    Returns:
        State dictionary compatible with SegformerForSemanticSegmentation
    """
    state = dict(local_state)
    hf_state: dict[str, torch.Tensor] = {}

    def take(local_key: str, hf_key: str) -> None:
        try:
            hf_state[hf_key] = state.pop(local_key)
        except KeyError as error:
            raise KeyError(f"Checkpoint is missing expected parameter '{local_key}'.") from error

    for stage_idx, depth in enumerate(hf_cfg.depths):
        local_stage = f"encoder.stages.{stage_idx}"
        hf_stage = "segformer.encoder"
        take(
            f"{local_stage}.patch_embed.proj.weight",
            f"{hf_stage}.patch_embeddings.{stage_idx}.proj.weight",
        )
        take(
            f"{local_stage}.patch_embed.proj.bias",
            f"{hf_stage}.patch_embeddings.{stage_idx}.proj.bias",
        )
        take(
            f"{local_stage}.patch_embed.norm.weight",
            f"{hf_stage}.patch_embeddings.{stage_idx}.layer_norm.weight",
        )
        take(
            f"{local_stage}.patch_embed.norm.bias",
            f"{hf_stage}.patch_embeddings.{stage_idx}.layer_norm.bias",
        )
        take(f"{local_stage}.norm.weight", f"{hf_stage}.layer_norm.{stage_idx}.weight")
        take(f"{local_stage}.norm.bias", f"{hf_stage}.layer_norm.{stage_idx}.bias")

        for block_idx in range(depth):
            local_block = f"{local_stage}.blocks.{block_idx}"
            hf_block = f"{hf_stage}.block.{stage_idx}.{block_idx}"
            take(f"{local_block}.norm1.weight", f"{hf_block}.layer_norm_1.weight")
            take(f"{local_block}.norm1.bias", f"{hf_block}.layer_norm_1.bias")
            take(f"{local_block}.norm2.weight", f"{hf_block}.layer_norm_2.weight")
            take(f"{local_block}.norm2.bias", f"{hf_block}.layer_norm_2.bias")
            take(f"{local_block}.attn.q.weight", f"{hf_block}.attention.self.query.weight")
            take(f"{local_block}.attn.q.bias", f"{hf_block}.attention.self.query.bias")

            kv_weight = state.pop(f"{local_block}.attn.kv.weight")
            kv_bias = state.pop(f"{local_block}.attn.kv.bias")
            key_weight, value_weight = kv_weight.chunk(2, dim=0)
            key_bias, value_bias = kv_bias.chunk(2, dim=0)
            hf_state[f"{hf_block}.attention.self.key.weight"] = key_weight
            hf_state[f"{hf_block}.attention.self.value.weight"] = value_weight
            hf_state[f"{hf_block}.attention.self.key.bias"] = key_bias
            hf_state[f"{hf_block}.attention.self.value.bias"] = value_bias
            take(f"{local_block}.attn.proj.weight", f"{hf_block}.attention.output.dense.weight")
            take(f"{local_block}.attn.proj.bias", f"{hf_block}.attention.output.dense.bias")

            if hf_cfg.sr_ratios[stage_idx] > 1:
                take(f"{local_block}.attn.sr.weight", f"{hf_block}.attention.self.sr.weight")
                take(f"{local_block}.attn.sr.bias", f"{hf_block}.attention.self.sr.bias")
                take(
                    f"{local_block}.attn.norm.weight",
                    f"{hf_block}.attention.self.layer_norm.weight",
                )
                take(f"{local_block}.attn.norm.bias", f"{hf_block}.attention.self.layer_norm.bias")

            take(f"{local_block}.mlp.fc1.weight", f"{hf_block}.mlp.dense1.weight")
            take(f"{local_block}.mlp.fc1.bias", f"{hf_block}.mlp.dense1.bias")
            take(f"{local_block}.mlp.dwconv.dwconv.weight", f"{hf_block}.mlp.dwconv.dwconv.weight")
            take(f"{local_block}.mlp.dwconv.dwconv.bias", f"{hf_block}.mlp.dwconv.dwconv.bias")
            take(f"{local_block}.mlp.fc2.weight", f"{hf_block}.mlp.dense2.weight")
            take(f"{local_block}.mlp.fc2.bias", f"{hf_block}.mlp.dense2.bias")

    for stage_idx in range(len(hf_cfg.depths)):
        hf_state[f"decode_head.linear_c.{stage_idx}.proj.weight"] = (
            state.pop(f"decoder.linear_c.{stage_idx}.weight").squeeze(-1).squeeze(-1)
        )
        take(f"decoder.linear_c.{stage_idx}.bias", f"decode_head.linear_c.{stage_idx}.proj.bias")

    take("decoder.linear_fuse.0.weight", "decode_head.linear_fuse.weight")
    take("decoder.linear_fuse.1.weight", "decode_head.batch_norm.weight")
    take("decoder.linear_fuse.1.bias", "decode_head.batch_norm.bias")
    take("decoder.linear_fuse.1.running_mean", "decode_head.batch_norm.running_mean")
    take("decoder.linear_fuse.1.running_var", "decode_head.batch_norm.running_var")
    take("decoder.linear_fuse.1.num_batches_tracked", "decode_head.batch_norm.num_batches_tracked")
    take("decoder.clfr.weight", "decode_head.classifier.weight")
    take("decoder.clfr.bias", "decode_head.classifier.bias")

    if state:
        unknown = ", ".join(sorted(state))
        raise ValueError(f"Checkpoint has unsupported model parameters: {unknown}")
    return hf_state

def main() -> None:
    """Parse arguments and export a checkpoint to Transformers format.

    Returns:
        None.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Training .pt checkpoint written by train.py")
    parser.add_argument("output_dir", type=Path, help="Directory to write the Hugging Face model")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]
    dataset_name = cfg.dataset.name
    model_cfg = cast(dict[str, Any], OmegaConf.to_container(cfg.model, resolve=True))
    dataset_cfg = load_dataset_cfg(dataset_name)
    hf_cfg = build_hf_config(model_cfg, dataset_cfg)
    dataset_classes = list(cast(list[str], dataset_cfg["classes"]))
    assert int(model_cfg["num_classes"]) == len(dataset_classes), (
        "Number of labels does not match the configured classifier size."
    )

    local_state = cast(
        Mapping[str, torch.Tensor], torch_compile_ckpt_fix(dict(checkpoint["model"]))
    )
    hf_model = SegformerForSemanticSegmentation(hf_cfg)
    hf_state = build_hf_state_dict(local_state, hf_cfg)
    hf_model.load_state_dict(hf_state, strict=True)

    normalization = cast(
        dict[str, Any], OmegaConf.to_container(cfg.dataset.input_normalization, resolve=True)
    )
    input_size = int(cfg.model.img_size)
    processor = SegformerImageProcessor(
        do_resize=True,
        size={"height": input_size, "width": input_size},
        do_reduce_labels=False,
        image_mean=[float(value) for value in cast(list[float], normalization["mean"])],
        image_std=[float(value) for value in cast(list[float], normalization["std"])],
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    hf_model.save_pretrained(args.output_dir, safe_serialization=True)
    processor.save_pretrained(args.output_dir)
    print(f"Exported Transformers SegFormer model to {args.output_dir}")

if __name__ == "__main__":
    main()
