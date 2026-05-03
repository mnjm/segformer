# This file contains Pascal VOC semantic-segmentation dataset utilities for training and visualization.

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.v2 as T
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import tv_tensors
from torchvision.io import read_image

VOC_CLASSES: tuple[str, ...] = (
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
)
"""Pascal VOC semantic-segmentation class names."""

VOC_COLOR_MAP = torch.tensor(
    [
        [0, 0, 0],
        [128, 0, 0],
        [0, 128, 0],
        [128, 128, 0],
        [0, 0, 128],
        [128, 0, 128],
        [0, 128, 128],
        [128, 128, 128],
        [64, 0, 0],
        [192, 0, 0],
        [64, 128, 0],
        [192, 128, 0],
        [64, 0, 128],
        [192, 0, 128],
        [64, 128, 128],
        [192, 128, 128],
        [0, 64, 0],
        [128, 64, 0],
        [0, 192, 0],
        [128, 192, 0],
        [0, 64, 128],
    ],
    dtype=torch.uint8,
)
"""Pascal VOC color map indexed by class id."""

VOC_IGNORE_INDEX = 255
"""Pascal VOC ignore label used in segmentation masks."""


def _resolve_voc_roots(dataset_root: Path, years: Sequence[str]) -> list[Path]:
    """Resolve Pascal VOC year directories.

    Args:
        dataset_root: Root that contains `VOC2007` and `VOC2012`.
        years: Pascal VOC year suffixes such as `("2007", "2012")`.
    """
    roots = [dataset_root / f"VOC{year}" for year in years]
    missing = [str(root) for root in roots if not root.is_dir()]
    if missing:
        missing_joined = ", ".join(missing)
        raise FileNotFoundError(f"Missing VOC dataset directories: {missing_joined}")
    return roots


def build_voc_segmentation_transforms(
    image_size: tuple[int, int],
    train: bool,
) -> T.Compose:
    """Build joint image and mask transforms for semantic segmentation.

    Args:
        image_size: Final `(height, width)` used by the model.
        train: Whether to include training-time augmentation.
    """
    transforms: list[torch.nn.Module] = []
    if train:
        transforms.append(T.RandomHorizontalFlip(p=0.5))
    transforms.append(T.Resize(image_size))
    return T.Compose(transforms)


class VOCSemanticSegmentationDataset(Dataset[tuple[Tensor, Tensor]]):
    """Load Pascal VOC semantic-segmentation samples.

    Args:
        dataset_root: Root that contains the VOC year directories.
        split: Dataset split from `ImageSets/Segmentation`.
        years: Pascal VOC year suffixes such as `("2007", "2012")`.
        transforms: Optional joint image and mask transforms.
        mean: Channel-wise normalization mean for the image tensor.
        std: Channel-wise normalization std for the image tensor.
    """

    def __init__(
        self,
        dataset_root: str | Path = "dataset/voc-datasets",
        split: str = "train",
        years: Sequence[str] = ("2007", "2012"),
        transforms: T.Compose | None = None,
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.transforms = transforms
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
        self.samples: list[tuple[Path, Path]] = []

        for voc_root in _resolve_voc_roots(self.dataset_root, years):
            split_path = voc_root / "ImageSets" / "Segmentation" / f"{split}.txt"
            if not split_path.is_file():
                raise FileNotFoundError(f"Missing split file: {split_path}")

            image_dir = voc_root / "JPEGImages"
            mask_dir = voc_root / "SegmentationClass"
            with split_path.open("r", encoding="utf-8") as handle:
                image_ids = [line.strip() for line in handle if line.strip()]

            for image_id in image_ids:
                image_path = image_dir / f"{image_id}.jpg"
                mask_path = mask_dir / f"{image_id}.png"
                if image_path.is_file() and mask_path.is_file():
                    self.samples.append((image_path, mask_path))

        if not self.samples:
            raise RuntimeError(
                f"No VOC segmentation samples found for split={split!r} under {self.dataset_root}"
            )

    def __len__(self) -> int:
        """Return the number of segmentation samples.

        Args:
            None.
        """
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        """Load a normalized image tensor and its segmentation mask.

        Args:
            index: Position of the sample inside the dataset.
        """
        image_path, mask_path = self.samples[index]
        image = read_image(str(image_path))
        mask_np = np.array(Image.open(mask_path), dtype=np.uint8, copy=True)
        mask = torch.from_numpy(mask_np)

        image_tensor = tv_tensors.Image(image)
        mask_tensor = tv_tensors.Mask(mask)
        if self.transforms is not None:
            image_tensor, mask_tensor = self.transforms(image_tensor, mask_tensor)

        image_float = image_tensor.to(torch.float32).div_(255.0)
        image_float = (image_float - self.mean) / self.std
        mask_long = mask_tensor.to(torch.long)
        return image_float, mask_long


def segmentation_collate_fn(
    batch: Sequence[tuple[Tensor, Tensor]],
) -> tuple[Tensor, Tensor]:
    """Stack segmentation samples into dense mini-batches.

    Args:
        batch: Sequence of `(image, mask)` samples.
    """
    images, masks = zip(*batch, strict=True)
    return torch.stack(images, dim=0), torch.stack(masks, dim=0)


def build_voc_segmentation_dataloader(
    dataset_root: str | Path = "dataset/voc-datasets",
    split: str = "train",
    image_size: tuple[int, int] = (512, 512),
    batch_size: int = 8,
    num_workers: int = 4,
    years: Sequence[str] = ("2007", "2012"),
) -> DataLoader[tuple[Tensor, Tensor]]:
    """Build a Pascal VOC semantic-segmentation dataloader.

    Args:
        dataset_root: Root that contains the VOC year directories.
        split: Dataset split from `ImageSets/Segmentation`.
        image_size: Final `(height, width)` used by the model.
        batch_size: Number of samples per batch.
        num_workers: Worker count used by the dataloader.
        years: Pascal VOC year suffixes such as `("2007", "2012")`.
    """
    dataset = VOCSemanticSegmentationDataset(
        dataset_root=dataset_root,
        split=split,
        years=years,
        transforms=build_voc_segmentation_transforms(
            image_size=image_size,
            train=split == "train",
        ),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=split == "train",
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        collate_fn=segmentation_collate_fn,
    )


def decode_voc_mask(mask: Tensor) -> Tensor:
    """Convert a class-index mask into a color image.

    Args:
        mask: Segmentation mask tensor with shape `(H, W)`.
    """
    mask_long = mask.to(torch.long)
    color_mask = torch.zeros((*mask_long.shape, 3), dtype=torch.uint8)

    valid = (mask_long >= 0) & (mask_long < len(VOC_CLASSES))
    color_mask[valid] = VOC_COLOR_MAP[mask_long[valid]]
    color_mask[mask_long == VOC_IGNORE_INDEX] = torch.tensor(
        [255, 255, 255], dtype=torch.uint8
    )
    return color_mask


def plot_voc_samples(
    dataset: Dataset[tuple[Tensor, Tensor]],
    sample_count: int = 4,
    output_path: str | Path | None = None,
) -> Path | None:
    """Plot image, mask, and overlay panels for a few VOC samples.

    Args:
        dataset: Dataset that returns `(image, mask)` pairs.
        sample_count: Number of samples to render from the start of the dataset.
        output_path: Optional file path used to save the rendered figure.
    """
    figure, axes = plt.subplots(sample_count, 3, figsize=(12, 4 * sample_count))
    axes_array = np.atleast_2d(axes)

    for row in range(sample_count):
        image, mask = dataset[row]
        image_vis = image.detach().cpu()
        image_vis = image_vis * torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)
        image_vis = image_vis + torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
        image_vis = image_vis.clamp(0.0, 1.0).permute(1, 2, 0).numpy()

        mask_vis = decode_voc_mask(mask.detach().cpu()).numpy()
        overlay = 0.55 * image_vis + 0.45 * (mask_vis.astype(np.float32) / 255.0)

        axes_array[row, 0].imshow(image_vis)
        axes_array[row, 0].set_title("image")
        axes_array[row, 1].imshow(mask_vis)
        axes_array[row, 1].set_title("mask")
        axes_array[row, 2].imshow(overlay)
        axes_array[row, 2].set_title("overlay")

        for col in range(3):
            axes_array[row, col].axis("off")

    figure.tight_layout()
    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=180, bbox_inches="tight")
        plt.close(figure)
        return output

    plt.show()
    plt.close(figure)
    return None


if __name__ == "__main__":
    dataset = VOCSemanticSegmentationDataset(
        split="train",
        transforms=build_voc_segmentation_transforms(image_size=(512, 512), train=True),
    )
    saved_path = plot_voc_samples(
        dataset=dataset,
        sample_count=4,
        output_path="cache/voc-segmentation-samples.png",
    )
    print(f"Saved sample plot to {saved_path}")
