"""Create a random SegFormer checkpoint for export and edge benchmarking."""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, cast

from omegaconf import DictConfig, OmegaConf
import torch

from model import SegFormer, SegFormerConfig


def parse_args() -> Namespace:
    """Parse CLI arguments for random checkpoint generation."""
    parser = ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Path to the Hydra config used to construct the model.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="random-segformer-b0.pt",
        help="Path to the output checkpoint file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used before model initialization.",
    )
    return parser.parse_args()


def load_cfg(config_path: Path) -> DictConfig:
    """Load the project config and locally compose its model/dataset defaults."""
    cfg = cast(DictConfig, OmegaConf.load(config_path))
    config_dir = config_path.parent
    merged_cfg = OmegaConf.create()

    defaults = cast(list[Any], cfg.get("defaults", []))
    for item in defaults:
        if not isinstance(item, (dict, DictConfig)):
            continue
        for group_name, choice in cast(dict[str, Any], dict(item)).items():
            if group_name == "_self_":
                continue
            group_cfg = OmegaConf.load(config_dir / group_name / f"{choice}.yaml")
            merged_cfg[group_name] = group_cfg

    cfg_keys = [str(key) for key in cfg.keys() if key not in {"defaults", "hydra"}]
    cfg_no_defaults = OmegaConf.masked_copy(cfg, cfg_keys)
    cfg = cast(DictConfig, OmegaConf.merge(merged_cfg, cfg_no_defaults))
    OmegaConf.resolve(cfg)
    return cfg


def main(args: Namespace) -> None:
    """Build a random SegFormer model and save a training-style checkpoint."""
    torch.manual_seed(args.seed)

    config_path = Path(args.config)
    output_path = Path(args.output)
    cfg = load_cfg(config_path)

    model_cfg = SegFormerConfig(**cast(dict[str, Any], OmegaConf.to_container(cfg.model, resolve=True)))
    ignore_idx = getattr(cfg.dataset, "ignore_idx", None)
    model = SegFormer(model_cfg, ignore_idx=ignore_idx).eval()

    ckpt = {
        "model": model.state_dict(),
        "config": cfg,
        "epoch": 0,
        "optimizer": None,
        "lr_scheduler": None,
        "wandb_id": None,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, output_path)
    print(f"Saved random checkpoint to {output_path}")


if __name__ == "__main__":
    main(parse_args())
