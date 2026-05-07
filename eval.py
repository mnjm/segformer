"""Evaluate a trained SegFormer checkpoint on a dataset split."""

import argparse
from contextlib import nullcontext
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
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
    parser.add_argument("-b", "--batch-size", type=int, default=4)
    parser.add_argument(
        "-cm",
        "--compute_metrics",
        action="store_true",
        help="If set, computes overall and per-class metrics for all classes.",
    )
    parser.add_argument(
        "--class-names",
        type=str,
        default="",
        help="Comma-separated class labels to rank. Empty means all classes.",
    )
    parser.add_argument(
        "-t",
        "--top",
        type=int,
        default=None,
        help="Show the N lowest-loss samples for the selected classes.",
    )
    parser.add_argument(
        "--bottom",
        type=int,
        default=None,
        help="Show the N highest-loss samples for the selected classes.",
    )
    return parser.parse_args()


def build_metric_tables(
    confmat: torch.Tensor,
    total_loss: float,
    total_samples: int,
    class_names: list[str],
) -> tuple[list[list[Any]], list[list[Any]]]:
    """Build summary and per-class metric tables.

    Args:
        confmat: Confusion matrix over the evaluated split.
        total_loss: Sum of batch losses weighted by batch size.
        total_samples: Number of evaluated samples.
        class_names: Class labels ordered by class id.

    Returns:
        Summary metric rows and per-class metric rows.
    """
    intersection = torch.diag(confmat)
    predicted = confmat.sum(dim=0)
    target = confmat.sum(dim=1)
    union = predicted + target - intersection
    true_negative = confmat.sum() - (
        intersection + predicted - intersection + target - intersection
    )

    iou = intersection / union.clamp(min=1)
    precision = intersection / predicted.clamp(min=1)
    recall = intersection / target.clamp(min=1)
    specificity = true_negative / (true_negative + predicted - intersection).clamp(min=1)
    class_acc = recall
    f1 = (2 * precision * recall) / (precision + recall).clamp(min=1e-12)
    avg_px_area = target / max(total_samples, 1)

    valid = union > 0
    valid_target = target > 0

    mean_loss = total_loss / max(total_samples, 1)
    pix_acc = intersection.sum() / confmat.sum().clamp(min=1)
    miou = iou[valid].mean() if valid.any() else torch.tensor(0.0, device=confmat.device)
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
                avg_px_area[class_id].item(),
                class_acc[class_id].item(),
                iou[class_id].item(),
                precision[class_id].item(),
                recall[class_id].item(),
                specificity[class_id].item(),
                f1[class_id].item(),
            ]
        )
    return summary_rows, class_rows


def visualize_ranked_samples(
    ranked_sample: dict[str, Any],
    palette_map: dict[int, torch.Tensor],
    class_map: bidict[int, str],
    output_path: Path,
    ranking_name: str,
    sample_rank: int,
) -> Path:
    """Save one ranked sample figure.

    Args:
        ranked_sample: Ranked sample payload to display.
        palette_map: Mapping from class id to RGB color tensor.
        class_map: Mapping from class id to class label.
        output_path: Destination image path.
        ranking_name: Human-readable ranking label.
        sample_rank: One-based sample rank within the saved ordering.

    Returns:
        Path to the saved image.
    """
    fig, axes = plt.subplots(1, 3, figsize=(10, 4), squeeze=False)
    matched_names = ", ".join(class_map[class_id] for class_id in ranked_sample["matched_classes"])
    image_i = ranked_sample["image"].permute(1, 2, 0).numpy()
    image_i = (image_i - image_i.min()) / (image_i.max() - image_i.min() + 1e-8)
    mask_i = decode_mask(ranked_sample["mask"].to(torch.uint8), palette_map).numpy() / 255.0
    pred_i = decode_mask(ranked_sample["pred"].to(torch.uint8), palette_map).numpy() / 255.0
    overlay_gt = image_i * 0.3 + mask_i * 0.7
    overlay_pred = image_i * 0.3 + pred_i * 0.7

    axes[0, 0].imshow(image_i)
    axes[0, 0].set_title(f"classes={matched_names}\nloss={ranked_sample['loss']:.4f}")
    axes[0, 0].axis("off")
    axes[0, 1].imshow(overlay_gt)
    axes[0, 1].set_title("ground truth")
    axes[0, 1].axis("off")
    axes[0, 2].imshow(overlay_pred)
    axes[0, 2].set_title("prediction")
    axes[0, 2].axis("off")

    fig.suptitle(f"{ranking_name} Sample {sample_rank}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def build_ranking_stem(class_labels: list[str], ranking_kind: str, k: int) -> str:
    """Build the output filename stem for ranked visualizations."""
    safe_labels = ["".join(ch if ch.isalnum() else "_" for ch in label) for label in class_labels]
    class_part = "_".join(safe_labels) if safe_labels else "all"
    return f"{class_part}_{ranking_kind}_{k}"


def plot_per_class_iou(class_rows: list[list[Any]], miou: float, output_path: Path) -> Path:
    """Save a paper-friendly horizontal bar chart of per-class IoU.

    Args:
        class_rows: Per-class metric table rows from ``build_metric_tables``.
        miou: Mean intersection over union for the evaluated split.
        output_path: Destination image path.

    Returns:
        Path to the saved plot.
    """
    class_names = [str(row[1]) for row in class_rows]
    ious = [float(row[6]) for row in class_rows]

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )
    fig_height = max(4.8, 0.23 * len(class_names))
    fig, ax = plt.subplots(figsize=(6.8, fig_height))
    ax.barh(class_names, ious, height=0.38, color="#4C78A8", edgecolor="none")
    ax.set_xlabel("IoU")
    ax.set_title(f"mIoU {miou:.6f}", pad=10)
    ax.set_xlim(0.0, 1.0)
    ax.invert_yaxis()
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#B0B0B0")
    ax.spines["bottom"].set_color("#B0B0B0")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if not args.compute_metrics and args.top is None and args.bottom is None:
        raise ValueError("Provide --compute_metrics, --top N, or --bottom N")
    should_rank = args.top is not None or args.bottom is not None

    ckpt_raw = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ckpt = cast(dict[str, Any], ckpt_raw)
    cfg = cast(DictConfig, ckpt["config"])

    device = torch.device("cuda") if torch.cuda.is_available() else torch_get_device("auto")

    dataset = init_dataset(cfg, split=args.split, apply_augmentations=False)
    dataloader_kwargs = dict(cfg.data.dataloader)
    dataloader_kwargs["shuffle"] = False
    dataloader_kwargs["drop_last"] = False
    dataloader_kwargs["num_workers"] = 0
    dataloader_kwargs["pin_memory"] = False
    dataloader = DataLoader(dataset, **dataloader_kwargs)
    dataset = dataloader.dataset
    class_map = cast(bidict[int, str], getattr(dataset, "class_map"))
    palette_map = cast(dict[int, torch.Tensor], getattr(dataset, "palette_map"))
    ignore_index = getattr(cfg.dataset, "ignore_idx", None)
    num_classes = len(class_map)
    class_names = [class_map[idx] for idx in range(num_classes)]
    valid_class_ids = set(range(num_classes))
    requested_classes: set[int] = set()
    requested_labels: list[str] = []
    if should_rank:
        requested_labels = [item.strip() for item in args.class_names.split(",") if item.strip()]
        if requested_labels:
            invalid_labels = [label for label in requested_labels if label not in class_map.inv]
            if invalid_labels:
                valid_names = ", ".join(class_names)
                invalid_names = ", ".join(repr(label) for label in invalid_labels)
                raise ValueError(
                    f"Unknown class name(s) {invalid_names}. Valid classes: {valid_names}"
                )
            requested_classes = {class_map.inv[label] for label in requested_labels}
        else:
            requested_classes = set(valid_class_ids)

    model_cfg = SegFormerConfig(**cfg.model)
    model = SegFormer(model_cfg, ignore_idx=ignore_index)
    model.load_state_dict(torch_compile_ckpt_fix(ckpt["model"]))
    model.to(device)
    model.eval()

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

        if should_rank:
            pixel_losses = per_sample_loss_fn(logits, masks)
            for batch_idx, sample_idx in enumerate(sample_indices):
                sample_mask = masks[batch_idx]
                present_classes = {
                    int(class_id)
                    for class_id in torch.unique(sample_mask).tolist()
                    if class_id >= 0
                }
                matched_classes = sorted(requested_classes.intersection(present_classes))
                if not matched_classes:
                    continue

                valid_pixels = torch.isin(
                    sample_mask,
                    torch.tensor(
                        matched_classes, device=sample_mask.device, dtype=sample_mask.dtype
                    ),
                )
                if ignore_index is not None:
                    valid_pixels &= sample_mask != ignore_index
                if not valid_pixels.any():
                    continue

                sample_loss = pixel_losses[batch_idx][valid_pixels].mean().item()
                ranked_samples.append(
                    {
                        "loss": sample_loss,
                        "sample_idx": sample_idx,
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
                    "avg_px_area",
                    "pix_acc",
                    "iou",
                    "precision",
                    "recall",
                    "specificity",
                    "f1",
                ],
                tablefmt="github",
                floatfmt=".6f",
            )
        )
        output_dir = Path("eval_outputs")
        output_dir.mkdir(parents=True, exist_ok=True)
        miou = next(float(value) for metric, value in summary_rows if metric == "miou")
        plot_path = plot_per_class_iou(class_rows, miou, output_dir / "iou_barplot.png")
        print(f"\nSaved per-class IoU plot to {plot_path}")

    if args.top is not None or args.bottom is not None:
        if not ranked_samples:
            print("\nNo matching samples were found for the requested classes.")
            return

        output_dir = Path("eval_outputs")
        output_dir.mkdir(parents=True, exist_ok=True)

        ranking_jobs: list[tuple[str, int, bool, str]] = []
        if args.top is not None:
            ranking_jobs.append(("top", args.top, False, "Top"))
        if args.bottom is not None:
            ranking_jobs.append(("bottom", args.bottom, True, "Bottom"))

        for ranking_kind, k, reverse, ranking_name in ranking_jobs:
            ranked_samples.sort(key=lambda item: item["loss"], reverse=reverse)
            chosen = ranked_samples[:k]
            output_stem = build_ranking_stem(requested_labels, ranking_kind, k)
            ranking_dir = output_dir / output_stem
            ranking_dir.mkdir(parents=True, exist_ok=True)
            for sample_rank, ranked_sample in enumerate(chosen, start=1):
                output_path = ranking_dir / f"{sample_rank}.png"
                visualize_ranked_samples(
                    ranked_sample=ranked_sample,
                    palette_map=palette_map,
                    class_map=class_map,
                    output_path=output_path,
                    ranking_name=ranking_name,
                    sample_rank=sample_rank,
                )
            print(
                f"\nSaved {len(chosen)} {ranking_name.lower()} ranked visualizations to {ranking_dir}"
            )


if __name__ == "__main__":
    main()
