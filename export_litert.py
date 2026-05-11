"""Post-training LiteRT/TFLite export pipeline for SegFormer checkpoints."""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from importlib import import_module
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from model import SegFormer, SegFormerConfig
from utils import torch_compile_ckpt_fix

try:
    litert_torch = import_module("litert_torch")
    litert_quantize = import_module("litert_torch.quantize")
    torchao_pt2e = import_module("torchao.quantization.pt2e")
    torchao_quantize_pt2e = import_module("torchao.quantization.pt2e.quantize_pt2e")
except ImportError as exc:  # pragma: no cover - depends on optional export stack.
    litert_torch: Any | None = None
    litert_quantize: Any | None = None
    torchao_pt2e: Any | None = None
    torchao_quantize_pt2e: Any | None = None
    EXPORT_IMPORT_ERROR = exc
else:
    EXPORT_IMPORT_ERROR = None

np.random.seed(42)
torch.manual_seed(42)


class LiteRTExportWrapper(nn.Module):
    """Wrap SegFormer for LiteRT export from ``NHWC`` uint8-like image inputs."""

    def __init__(
        self,
        model: nn.Module,
        mean: torch.Tensor,
        std: torch.Tensor,
        n_classes: int,
    ) -> None:
        """Register preprocessing buffers used at inference time."""
        super().__init__()
        self.model = model
        self.n_classes = int(n_classes)
        self.register_buffer("mean", mean.view(1, 3, 1, 1))
        self.register_buffer("std", std.view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize ``NHWC`` input and return per-pixel class probabilities."""
        mean = cast(torch.Tensor, self.mean)
        std = cast(torch.Tensor, self.std)
        x = x.permute(0, 3, 1, 2)
        x = x / 255.0
        x = (x - mean) / std

        logits = self.model(x)
        assert logits.ndim == 4 and logits.shape[1] == self.n_classes
        return F.softmax(logits, dim=1)


def ensure_export_deps() -> None:
    """Fail early when LiteRT export dependencies are unavailable."""
    if EXPORT_IMPORT_ERROR is not None:
        raise ImportError(
            "LiteRT export dependencies are not installed. "
            "Install `litert_torch` and `torchao` in this environment."
        ) from EXPORT_IMPORT_ERROR


def get_quant_modules() -> tuple[Any, Any, Any, Any, Any]:
    """Return optional export modules after dependency validation."""
    ensure_export_deps()
    assert litert_torch is not None
    assert litert_quantize is not None
    assert torchao_pt2e is not None
    assert torchao_quantize_pt2e is not None
    return (
        litert_torch,
        litert_quantize.pt2e_quantizer,
        litert_quantize.quant_config,
        torchao_pt2e.move_exported_model_to_eval,
        (
            torchao_quantize_pt2e.convert_pt2e,
            torchao_quantize_pt2e.prepare_pt2e,
        ),
    )


def parse_args() -> Namespace:
    """Parse command-line arguments for LiteRT export."""
    parser = ArgumentParser()
    parser.add_argument("checkpoint_path", type=str)
    parser.add_argument(
        "-i8",
        "--int8",
        action="store_true",
        help="Convert the model to int8 with static PTQ and save it alongside the float model.",
    )
    parser.add_argument(
        "--calibration-batches",
        type=int,
        default=5,
        help="Number of random calibration batches for PTQ. Default: 5.",
    )
    parser.add_argument(
        "--calibration-batch-size",
        type=int,
        default=1,
        help="Batch size for each random PTQ calibration batch. Default: 1.",
    )
    return parser.parse_args()


def _checkpoint_cfg(ckpt: dict[str, Any]) -> DictConfig:
    """Return the Hydra config stored in a checkpoint."""
    cfg = ckpt.get("config", ckpt.get("cfg"))
    if cfg is None:
        raise KeyError("checkpoint does not contain training config under key 'config' or 'cfg'")
    if isinstance(cfg, DictConfig):
        return cfg
    return cast(DictConfig, OmegaConf.create(cfg))


def _resolve_cfg(cfg: DictConfig) -> DictConfig:
    """Return a resolved copy of the training config."""
    cfg_copy = cast(DictConfig, OmegaConf.create(OmegaConf.to_container(cfg, resolve=True)))
    OmegaConf.resolve(cfg_copy)
    return cfg_copy


def _build_model_from_checkpoint(ckpt: dict[str, Any]) -> tuple[SegFormer, DictConfig]:
    """Instantiate and load a SegFormer model from a training checkpoint."""
    cfg = _resolve_cfg(_checkpoint_cfg(ckpt))
    model_cfg = SegFormerConfig(**cast(dict[str, Any], OmegaConf.to_container(cfg.model, resolve=True)))
    ignore_idx = getattr(cfg.dataset, "ignore_idx", None)
    model = SegFormer(model_cfg, ignore_idx=ignore_idx)
    model.load_state_dict(torch_compile_ckpt_fix(ckpt["model"]), strict=True)
    model.eval()
    return model, cfg


def _make_random_export_sample(img_size: tuple[int, int]) -> tuple[torch.Tensor]:
    """Create one random ``NHWC`` sample for export validation."""
    sample = torch.randint(0, 256, (1, img_size[0], img_size[1], 3), dtype=torch.float32)
    return (sample,)


def _make_random_calibration_batch(
    batch_size: int,
    img_size: tuple[int, int],
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """Generate a random normalized ``NCHW`` batch for PTQ calibration."""
    x = torch.randint(0, 256, (batch_size, img_size[0], img_size[1], 3), dtype=torch.float32)
    x = x.permute(0, 3, 1, 2)
    x = x / 255.0
    x = (x - mean.view(1, 3, 1, 1)) / std.view(1, 3, 1, 1)
    return x


def convert_to_litert(model: nn.Module, cfg: DictConfig, save_to: Path) -> None:
    """Export a float SegFormer checkpoint to LiteRT and compare outputs."""
    litert_mod, _, _, _, _ = get_quant_modules()
    mean = torch.as_tensor(cfg.data.input_normalization.mean, dtype=torch.float32)
    std = torch.as_tensor(cfg.data.input_normalization.std, dtype=torch.float32)
    n_classes = int(cfg.dataset.num_classes)
    wrapped_model = LiteRTExportWrapper(model, mean, std, n_classes).eval()

    img_size = (int(cfg.data.input_size), int(cfg.data.input_size))
    sample_inputs = _make_random_export_sample(img_size)

    with torch.no_grad():
        pth_outputs = wrapped_model(*sample_inputs)
    edge_model = litert_mod.convert(wrapped_model, sample_inputs)
    litert_output = edge_model(*sample_inputs)

    if np.allclose(
        pth_outputs.detach().numpy(), np.asarray(litert_output), atol=1e-5, rtol=1e-5
    ):
        print("Inference result with PyTorch and LiteRT was within tolerance")
    else:
        print("Something is wrong with PyTorch -> LiteRT conversion")

    edge_model.export(str(save_to))
    print(f"Model was exported to {str(save_to)!r}")


def convert_to_litert_i8(
    model: nn.Module,
    cfg: DictConfig,
    save_to: Path,
    calibration_batches: int = 5,
    calibration_batch_size: int = 1,
) -> None:
    """Run PTQ int8 calibration on random data and export a quantized LiteRT model."""
    litert_mod, pt2e_quantizer, quant_config, move_exported_model_to_eval, pt2e_fns = (
        get_quant_modules()
    )
    convert_pt2e, prepare_pt2e = pt2e_fns
    model.eval()

    img_size = (int(cfg.data.input_size), int(cfg.data.input_size))
    mean = torch.as_tensor(cfg.data.input_normalization.mean, dtype=torch.float32)
    std = torch.as_tensor(cfg.data.input_normalization.std, dtype=torch.float32)
    n_classes = int(cfg.dataset.num_classes)

    q_config = pt2e_quantizer.get_symmetric_quantization_config(
        is_dynamic=False,
        is_per_channel=True,
    )
    quantizer = pt2e_quantizer.PT2EQuantizer().set_global(q_config)

    sample_inputs = (torch.randn(1, 3, *img_size),)
    exported_model = torch.export.export(model, sample_inputs)
    prepared_model = prepare_pt2e(exported_model.module(), quantizer)

    with torch.no_grad():
        for _ in range(calibration_batches):
            x_batch = _make_random_calibration_batch(
                batch_size=calibration_batch_size,
                img_size=img_size,
                mean=mean,
                std=std,
            )
            prepared_model(x_batch)

    quantized_model = convert_pt2e(prepared_model, fold_quantize=False)
    move_exported_model_to_eval(quantized_model)

    wrapped_model = LiteRTExportWrapper(
        quantized_model,
        mean,
        std,
        n_classes,
    )
    wrapped_model.training = False
    sample_inputs = _make_random_export_sample(img_size)

    with torch.no_grad():
        pth_outputs = wrapped_model(*sample_inputs).detach().numpy()

    edge_model = litert_mod.convert(
        wrapped_model,
        sample_inputs,
        quant_config=quant_config.QuantConfig(pt2e_quantizer=quantizer),
    )
    litert_output = edge_model(*sample_inputs)

    if np.allclose(pth_outputs, np.asarray(litert_output), atol=1e-5, rtol=1e-5):
        print("[Int8] Inference result with PyTorch and LiteRT was within tolerance")
    else:
        error = np.abs(pth_outputs - np.asarray(litert_output)).sum()
        print(f"[Int8] PyTorch -> LiteRT mismatch, absolute error sum: {error}")

    edge_model.export(str(save_to))
    print(f"[Int8] Model was exported to {str(save_to)!r}")


def main(args: Namespace) -> None:
    """Load a checkpoint and export LiteRT models based on CLI flags."""
    ckpt_path = Path(args.checkpoint_path)
    assert ckpt_path.is_file(), f"Checkpoint not found: {ckpt_path}"

    ckpt = cast(dict[str, Any], torch.load(ckpt_path, map_location="cpu", weights_only=False))
    print(f"Loaded checkpoint from {ckpt_path}")
    model, cfg = _build_model_from_checkpoint(ckpt)

    convert_to_litert(model, cfg, ckpt_path.parent / f"{ckpt_path.stem}.tflite")

    if args.int8:
        tflite_ptq_i8_path = ckpt_path.parent / f"{ckpt_path.stem}_i8_ptq.tflite"
        convert_to_litert_i8(
            model,
            cfg,
            tflite_ptq_i8_path,
            calibration_batches=args.calibration_batches,
            calibration_batch_size=args.calibration_batch_size,
        )


if __name__ == "__main__":
    main(parse_args())
