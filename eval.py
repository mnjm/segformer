"""Evaluate a trained SegFormer checkpoint on a dataset split."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import torch
from bidict import bidict
from omegaconf import DictConfig
from tabulate import tabulate
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import decode_mask, init_dataset
from model import SegFormer, SegFormerConfig
from utils import torch_compile_ckpt_fix, torch_get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Path to a checkpoint saved by train.py.")
    parser.add_argument("-s", "--split", choices=("train", "val"), default="val")
    parser.add_argument(
        "-cm",
        "--compute_metrics",
        action="store_true",
        help="If set, computes overall and per-class metrics for all classes.",
    )
    parser.add_argument(
        "-v",
        "--visualize",
        type=str,
        default="",
        help="Comma-separated class names or class ids to visualize.",
    )
    parser.add_argument(
        "-t",
        "--top",
        type=int,
        default=None,
        help="Show the N lowest-loss matching samples.",
    )
    parser.add_argument(
        "-b",
        "--bottom",
        type=int,
        default=None,
        help="Show the N highest-loss matching samples.",
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Disable torch.compile even if the checkpoint config enabled it.",
    )
    return parser.parse_args()


def resolve_class_names(class_map: bidict[int, str], num_classes: int) -> list[str]:
    return [class_map[idx] for idx in range(num_classes)]


def resolve_requested_classes(
    requested: str,
    class_map: bidict[int, str],
    valid_class_ids: set[int],
) -> set[int]:
    if not requested.strip():
        return set()

    requested_items = [item.strip() for item in requested.split(",") if item.strip()]
    resolved = set[int]()
    for item in requested_items:
        if item.isdigit():
            class_id = int(item)
            if class_id not in valid_class_ids:
                raise ValueError(f"Class id {class_id} is not valid for this dataset")
            resolved.add(class_id)
            continue

        if item not in class_map.inv:
            valid_names = ", ".join(class_map[idx] for idx in sorted(valid_class_ids))
            raise ValueError(f"Unknown class name {item!r}. Valid classes: {valid_names}")
        class_id = class_map.inv[item]
        if class_id not in valid_class_ids:
            raise ValueError(f"Class name {item!r} resolves to ignored class id {class_id}")
        resolved.add(class_id)
    return resolved


def sample_path_for_index(dataset: Any, sample_idx: int) -> str:
    if hasattr(dataset, "samples"):
        return str(dataset.samples[sample_idx][0])
    if hasattr(dataset, "datasets") and hasattr(dataset, "cumulative_sizes"):
        subdataset_idx = int(np.searchsorted(dataset.cumulative_sizes, sample_idx, side="right"))
        prev_size = 0 if subdataset_idx == 0 else dataset.cumulative_sizes[subdataset_idx - 1]
        return sample_path_for_index(dataset.datasets[subdataset_idx], sample_idx - prev_size)
    return f"sample_{sample_idx}"


def build_eval_dataloader(cfg: DictConfig, split: str) -> DataLoader[Any]:
    dataset = init_dataset(cfg, split=split, apply_augmentations=False)
    dataloader_kwargs = dict(cfg.dataloader)
    dataloader_kwargs["shuffle"] = False
    dataloader_kwargs["drop_last"] = False
    dataloader_kwargs["num_workers"] = 0
    return DataLoader(dataset, **dataloader_kwargs)


def build_metric_tables(
    confmat: torch.Tensor,
    total_loss: float,
    total_samples: int,
    class_names: list[str],
) -> tuple[list[list[Any]], list[list[Any]]]:
    intersection = torch.diag(confmat)
    predicted = confmat.sum(dim=0)
    target = confmat.sum(dim=1)
    union = predicted + target - intersection
    true_negative = confmat.sum() - (
        intersection + predicted - intersection + target - intersection
    )

    iou = intersection / union.clamp(min=1)
    dice = (2 * intersection) / (predicted + target).clamp(min=1)
    precision = intersection / predicted.clamp(min=1)
    recall = intersection / target.clamp(min=1)
    specificity = true_negative / (true_negative + predicted - intersection).clamp(min=1)
    class_acc = recall
    f1 = (2 * precision * recall) / (precision + recall).clamp(min=1e-12)

    valid = union > 0
    valid_target = target > 0

    mean_loss = total_loss / max(total_samples, 1)
    pix_acc = intersection.sum() / confmat.sum().clamp(min=1)
    miou = iou[valid].mean() if valid.any() else torch.tensor(0.0, device=confmat.device)
    mean_dice = dice[valid].mean() if valid.any() else torch.tensor(0.0, device=confmat.device)
    mean_precision = (
        precision[valid_target].mean()
        if valid_target.any()
        else torch.tensor(0.0, device=confmat.device)
    )
    mean_recall = (
        recall[valid_target].mean()
        if valid_target.any()
        else torch.tensor(0.0, device=confmat.device)
    )
    mean_specificity = (
        specificity[valid_target].mean()
        if valid_target.any()
        else torch.tensor(0.0, device=confmat.device)
    )
    mean_f1 = f1[valid].mean() if valid.any() else torch.tensor(0.0, device=confmat.device)
    mean_class_acc = (
        class_acc[valid_target].mean()
        if valid_target.any()
        else torch.tensor(0.0, device=confmat.device)
    )
    freq = target / target.sum().clamp(min=1)
    fw_iou = (
        (freq[valid] * iou[valid]).sum()
        if valid.any()
        else torch.tensor(0.0, device=confmat.device)
    )

    summary_rows = [
        ["loss", mean_loss],
        ["pix_acc", pix_acc.item()],
        ["mean_acc", mean_class_acc.item()],
        ["miou", miou.item()],
        ["fw_iou", fw_iou.item()],
        ["dice", mean_dice.item()],
        ["f1", mean_f1.item()],
        ["precision", mean_precision.item()],
        ["recall", mean_recall.item()],
        ["specificity", mean_specificity.item()],
    ]

    class_rows = []
    for class_id, class_name in enumerate(class_names):
        class_rows.append(
            [
                class_id,
                class_name,
                int(target[class_id].item()),
                int(predicted[class_id].item()),
                iou[class_id].item(),
                dice[class_id].item(),
                precision[class_id].item(),
                recall[class_id].item(),
                specificity[class_id].item(),
                f1[class_id].item(),
            ]
        )
    return summary_rows, class_rows


def visualize_ranked_samples(
    ranked_samples: list[dict[str, Any]],
    palette_map: dict[int, torch.Tensor],
    class_map: bidict[int, str],
    title: str,
) -> None:
    n = len(ranked_samples)
    if n == 0:
        return

    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    axes = np.atleast_2d(axes)
    for row_idx, sample in enumerate(ranked_samples):
        image = sample["image"].permute(1, 2, 0).numpy()
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        mask = decode_mask(sample["mask"].to(torch.uint8), palette_map).numpy() / 255.0
        pred = decode_mask(sample["pred"].to(torch.uint8), palette_map).numpy() / 255.0

        matched_names = ", ".join(class_map[class_id] for class_id in sample["matched_classes"])
        sample_title = (
            f"loss={sample['loss']:.4f}\n"
            f"{Path(sample['sample_path']).name}\n"
            f"classes={matched_names}"
        )

        axes[row_idx, 0].imshow(image)
        axes[row_idx, 0].set_title(sample_title)
        axes[row_idx, 0].axis("off")

        axes[row_idx, 1].imshow(image * 0.3 + mask * 0.7)
        axes[row_idx, 1].set_title("ground truth")
        axes[row_idx, 1].axis("off")

        axes[row_idx, 2].imshow(image * 0.3 + pred * 0.7)
        axes[row_idx, 2].set_title("prediction")
        axes[row_idx, 2].axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    plt.show()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if args.top is not None and args.bottom is not None:
        raise ValueError("Only one of --top or --bottom can be provided")
    if not args.compute_metrics and not args.visualize.strip():
        raise ValueError("Provide at least one of --compute_metrics or --visualize")
    if args.visualize.strip() and args.top is None and args.bottom is None:
        raise ValueError("Visualization requires either --top N or --bottom N")
    if not args.visualize.strip() and (args.top is not None or args.bottom is not None):
        raise ValueError("--top/--bottom require --visualize")

    ckpt_raw = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ckpt = cast(dict[str, Any], ckpt_raw)
    cfg = cast(DictConfig, ckpt["config"])

    device = torch.device("cuda") if torch.cuda.is_available() else torch_get_device("auto")

    dataloader = build_eval_dataloader(cfg.dataset, split=args.split)
    dataset = dataloader.dataset
    class_map = cast(bidict[int, str], getattr(dataset, "class_map"))
    palette_map = cast(dict[int, torch.Tensor], getattr(dataset, "palette_map"))
    ignore_index = class_map.inv.get("ignore", None)
    num_classes = len(class_map) - (1 if ignore_index is not None else 0)
    class_names = resolve_class_names(class_map, num_classes)
    valid_class_ids = set(range(num_classes))
    requested_classes = resolve_requested_classes(args.visualize, class_map, valid_class_ids)

    loss_fn = (
        torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
        if ignore_index is not None
        else torch.nn.CrossEntropyLoss()
    )
    model_cfg = SegFormerConfig(**cfg.model)
    model = SegFormer(model_cfg, loss_fn=loss_fn)
    model.load_state_dict(torch_compile_ckpt_fix(ckpt["model"]))
    model.to(device)
    model.eval()

    if bool(getattr(cfg, "torch_compile", False)) and not args.no_compile:
        model = cast(SegFormer, torch.compile(model, dynamic=True))

    torch_autocast_dtype = {"f32": torch.float32, "bf16": torch.bfloat16}[cfg.autocast_dtype]
    autocast_ctx = (
        torch.amp.autocast(device_type=device.type, dtype=torch_autocast_dtype)
        if device.type == "cuda" and torch_autocast_dtype == torch.bfloat16
        else nullcontext()
    )

    confmat_dtype = torch.float32 if device.type == "mps" else torch.float64
    confmat = torch.zeros((num_classes, num_classes), dtype=confmat_dtype, device=device)
    total_loss = 0.0
    total_samples = 0
    ranked_samples: list[dict[str, Any]] = []
    seen_samples = 0

    progress_bar = tqdm(
        dataloader,
        total=len(dataloader),
        desc=f"Evaluating {args.split}",
        dynamic_ncols=True,
    )

    per_sample_loss_fn = (
        torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduction="none")
        if ignore_index is not None
        else torch.nn.CrossEntropyLoss(reduction="none")
    )

    for images, masks in progress_bar:
        batch_size = images.shape[0]
        sample_indices = list(range(seen_samples, seen_samples + batch_size))
        seen_samples += batch_size
        images = images.to(device)
        masks = masks.to(device)
        with autocast_ctx:
            logits, batch_loss = model(images, masks)

        preds = torch.argmax(logits, dim=1)
        total_loss += batch_loss.item() * images.shape[0]
        total_samples += images.shape[0]

        flat_preds = preds.reshape(-1)
        flat_targets = masks.reshape(-1)
        if ignore_index is not None:
            valid_mask = flat_targets != ignore_index
            flat_preds = flat_preds[valid_mask]
            flat_targets = flat_targets[valid_mask]
        valid_mask = (
            (flat_targets >= 0)
            & (flat_targets < num_classes)
            & (flat_preds >= 0)
            & (flat_preds < num_classes)
        )
        flat_preds = flat_preds[valid_mask]
        flat_targets = flat_targets[valid_mask]
        indices = num_classes * flat_targets + flat_preds
        confmat += torch.bincount(indices, minlength=num_classes**2).reshape(
            num_classes, num_classes
        )

        if requested_classes:
            pixel_losses = per_sample_loss_fn(logits, masks)
            for batch_idx, sample_idx in enumerate(sample_indices):
                sample_mask = masks[batch_idx]
                present_classes = set(torch.unique(sample_mask).tolist())
                matched_classes = sorted(
                    valid_class_ids.intersection(requested_classes, present_classes)
                )
                if not matched_classes:
                    continue

                valid_pixels = torch.ones_like(sample_mask, dtype=torch.bool)
                if ignore_index is not None:
                    valid_pixels = sample_mask != ignore_index
                sample_loss = pixel_losses[batch_idx][valid_pixels].mean().item()
                ranked_samples.append(
                    {
                        "loss": sample_loss,
                        "sample_idx": sample_idx,
                        "sample_path": sample_path_for_index(dataset, sample_idx),
                        "matched_classes": matched_classes,
                        "image": images[batch_idx].cpu(),
                        "mask": masks[batch_idx].cpu(),
                        "pred": preds[batch_idx].cpu(),
                    }
                )

    if args.compute_metrics:
        summary_rows, class_rows = build_metric_tables(
            confmat, total_loss, total_samples, class_names
        )
        print("\nSummary Metrics")
        print(
            tabulate(summary_rows, headers=["metric", "value"], tablefmt="github", floatfmt=".6f")
        )
        print("\nPer-Class Metrics")
        print(
            tabulate(
                class_rows,
                headers=[
                    "id",
                    "class",
                    "target_px",
                    "pred_px",
                    "iou",
                    "dice",
                    "precision",
                    "recall",
                    "specificity",
                    "f1",
                ],
                tablefmt="github",
                floatfmt=".6f",
            )
        )

    if args.top is not None or args.bottom is not None:
        if not ranked_samples:
            print("\nNo matching samples were found for the requested classes.")
            return

        reverse = args.bottom is not None
        k = args.bottom if args.bottom is not None else args.top
        assert k is not None
        ranked_samples.sort(key=lambda item: item["loss"], reverse=reverse)
        chosen = ranked_samples[:k]

        print("\nRanked Samples")
        print(
            tabulate(
                [
                    [
                        item["sample_idx"],
                        Path(item["sample_path"]).name,
                        ", ".join(class_map[class_id] for class_id in item["matched_classes"]),
                        item["loss"],
                    ]
                    for item in chosen
                ],
                headers=["index", "sample", "matched_classes", "loss"],
                tablefmt="github",
                floatfmt=".6f",
            )
        )
        ranking_name = "Top" if args.top is not None else "Bottom"
        visualize_ranked_samples(
            chosen, palette_map, class_map, f"{ranking_name} {len(chosen)} Samples"
        )


if __name__ == "__main__":
    main()
