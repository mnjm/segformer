"""Entrypoint for SegFormer training."""

import logging
import math
from contextlib import nullcontext
from pathlib import Path
from typing import Any, cast

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from data import init_dataloader, plot_samples
from model import SegFormer, SegFormerConfig
from utils import (
    SegmentationMetrics,
    WandBLogger,
    get_ist_time_now,
    poly_with_linear_warmup_lr_scheduler,
    timer,
    torch_compile_ckpt_fix,
    torch_get_device,
    torch_set_seed,
)

OmegaConf.register_new_resolver("now_ist", get_ist_time_now)
logger = logging.getLogger("train")
seed = 42


@hydra.main(version_base=None, config_path="config", config_name="default")
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    device = torch_get_device(cfg.device_type)
    logger.info(f"Using {device}")
    hydra_cfg = HydraConfig.get()
    log_dir = Path(hydra_cfg.runtime.output_dir)
    run_name = hydra_cfg.job.name
    torch_autocast_dtype = {"f32": torch.float32, "bf16": torch.bfloat16}[cfg.autocast_dtype]
    torch_set_seed(42)

    train_loader = init_dataloader(cfg, "train")
    val_loader = init_dataloader(cfg, "val")
    palette_map = getattr(train_loader.dataset, "palette_map")
    if getattr(cfg, "show_batch_initially", False):
        sample = next(iter(train_loader))
        plot_samples(sample[0], sample[1], cast(dict[int, torch.Tensor], palette_map))

    ckpt: dict[str, Any] | None = None
    start_epoch = 1
    ignore_idx = getattr(cfg.dataset, "ignore_idx", None)
    num_classes = len(getattr(train_loader.dataset, "class_map", {}))
    assert num_classes > 0, f"Expected num_classes > 0, got {num_classes}"
    wandb_id = None
    if cfg.init_from == "scratch":
        model_cfg = SegFormerConfig(**cfg.model)
        model = SegFormer(model_cfg, ignore_idx=ignore_idx)
        model.load_pretrained_encoder()
        model.to(device)
    else:
        loaded_ckpt = torch.load(cfg.init_from, map_location=device, weights_only=False)
        ckpt = cast(dict[str, Any], loaded_ckpt)
        ckpt_config = cast(DictConfig, ckpt["config"])
        model_cfg = SegFormerConfig(**ckpt_config.model)
        model = SegFormer(model_cfg, ignore_idx=ignore_idx)
        model.to(device)
        model.load_state_dict(torch_compile_ckpt_fix(ckpt["model"]))
        logger.info(f"Loaded checkpoint from {cfg.init_from}")
        start_epoch = ckpt["epoch"] + 1
        wandb_id = ckpt.get("wandb_id")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"Model: {model_cfg.name} | Params: {total_params / 1e6:.2f}M | Trainable: {trainable_params / 1e6:.2f}M"
    )

    if cfg.torch_compile:
        model = cast(SegFormer, torch.compile(model, fullgraph=True))

    optimizer = model.configure_optimizer(cfg.optimizer, device)
    n_epochs = cfg.n_epochs
    grad_accum_steps = cfg.grad_accum_steps

    lr_scheduler = None
    if cfg.lr_scheduler is not None:
        if cfg.lr_scheduler.type == "poly_with_linear_warmup":
            kwargs = dict(cfg.lr_scheduler)
            del kwargs["type"]
            lr_scheduler = poly_with_linear_warmup_lr_scheduler(
                optimizer,
                total_steps=n_epochs * math.ceil(len(train_loader) / grad_accum_steps),
                **kwargs,
            )
        else:
            raise ValueError(f"Unknown lr scheduler type: {cfg.lr_scheduler.type}")

    if cfg.init_from != "scratch":
        assert ckpt is not None
        optimizer.load_state_dict(ckpt["optimizer"])
        lr_schdlr_state = ckpt.get("lr_scheduler", None)
        if lr_schdlr_state and lr_scheduler:
            lr_scheduler.load_state_dict(lr_schdlr_state)

    if cfg.enable_tf32:
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True

    autocast_ctx = (
        torch.amp.autocast(device_type=device.type, dtype=torch_autocast_dtype)
        if device.type == "cuda" and torch_autocast_dtype == torch.bfloat16
        else nullcontext()
    )

    wb_logger = WandBLogger(
        project=cfg.logging.wandb.project,
        run=run_name,
        config=cast(
            dict[str, Any],
            OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        ),
        tags=("train", "val"),
        metrics=("loss", "accuracy", "miou", "time"),
        run_id=wandb_id,
        enable=cfg.logging.wandb.enable,
    )

    optimizer.zero_grad(set_to_none=True)

    @timer
    def train_epoch() -> dict:
        model.train()
        progress_bar = tqdm(train_loader, desc="Train", dynamic_ncols=True, leave=False)
        metrics = SegmentationMetrics(
            num_classes=num_classes,
            ignore_index=ignore_idx,
            device=device,
        )

        step = -1
        for step, batch in enumerate(progress_bar):
            img, mask = batch
            img = img.to(device)
            mask = mask.to(device)
            with autocast_ctx:
                logits, loss_i = model(img, mask)

            (loss_i / grad_accum_steps).backward()
            if (step + 1) % grad_accum_steps == 0:
                if cfg.clip_grad_norm_1:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if lr_scheduler is not None:
                    lr_scheduler.step()

            metrics.update(logits, mask, loss_i)

        if (step + 1) % grad_accum_steps != 0:
            if cfg.clip_grad_norm_1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if lr_scheduler is not None:
                lr_scheduler.step()

        progress_bar.close()
        if device.type == "cuda":
            torch.cuda.synchronize()

        return metrics.compute()

    @timer
    @torch.no_grad()
    def val_epoch() -> dict:
        model.eval()
        progress_bar = tqdm(val_loader, desc="Val", dynamic_ncols=True, leave=False)
        metrics = SegmentationMetrics(
            num_classes=num_classes,
            ignore_index=ignore_idx,
            device=device,
        )
        for step, batch in enumerate(progress_bar):
            img, mask = batch
            img = img.to(device)
            mask = mask.to(device)
            with autocast_ctx:
                logits, loss_i = model(img, mask)
            metrics.update(logits, mask, loss_i)
        progress_bar.close()
        if device.type == "cuda":
            torch.cuda.synchronize()

        return metrics.compute()

    t, stats = val_epoch()
    logger.info(
        "Initial Val " + " ".join(f"{k}={v:.4f}" for k, v in stats.items()) + f" Time={t:.2f}s"
    )
    wb_logger.log("val", {"epoch": start_epoch - 1, **stats, "time": t})
    for epoch in range(start_epoch, n_epochs + 1):
        last_epoch = epoch == cfg.n_epochs
        logger.info(f"Epoch: {epoch}/{cfg.n_epochs}")

        t, stats = train_epoch()
        logger.info(
            f"{'Train':<5} "
            + " ".join(f"{k}={v:.4f}" for k, v in stats.items())
            + f" Time={t:.2f}s"
        )
        wb_logger.log("train", {"epoch": epoch, **stats, "time": t})

        if last_epoch or epoch % cfg.val_every_epoch == 0:
            t, stats = val_epoch()
            logger.info(
                f"{'Val':<5} "
                + " ".join(f"{k}={v:.4f}" for k, v in stats.items())
                + f" Time={t:.2f}s"
            )
            wb_logger.log("val", {"epoch": epoch, **stats, "time": t})

        if last_epoch or epoch % cfg.save_every_epoch == 0:
            ckpt_path = log_dir / f"{cfg.model.name}.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": cfg,
                    "epoch": epoch,
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict() if lr_scheduler is not None else None,
                    "wandb_id": wb_logger.run_id,
                },
                ckpt_path,
            )
            logger.info(f"Saved checkpoint to {str(ckpt_path)}")


if __name__ == "__main__":
    main()
