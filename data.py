"""Pascal VOC semantic-segmentation data loading and visualization helpers."""

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.v2 as T
from bidict import bidict
from hydra.utils import instantiate
from omegaconf import DictConfig
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import tv_tensors
from torchvision.io import read_image

from utils import to_2tuple

logger = logging.getLogger(__name__)
supported_dataset = ["voc_2007_2012"]
voc_path = Path("./dataset/voc-datasets")


def init_dataset(
    cfg: DictConfig,
    split: str,
    apply_augmentations: bool = False,
) -> Dataset[Any]:
    """Build one dataset split and attach dataset metadata.

    Args:
        cfg: Dataset configuration.
        split: Dataset split name.
        apply_augmentations: Whether to enable training augmentations.

    Returns:
        Dataset[Any]: Configured dataset instance for the requested split.
    """
    dataset_name = cfg.name
    assert dataset_name in supported_dataset, f"Dataset {dataset_name} is not supported"
    assert split in ["train", "val"]
    transforms = build_transforms(cfg, apply_augmentations)
    if dataset_name == "voc_2007_2012":
        if split == "train":
            dataset_l = [
                VOCSemanticSegmentationDataset(
                    voc_path / sub_name, split="trainval", transforms=transforms
                )
                for sub_name in ["VOC2007", "VOC2012"]
            ]
            dataset = ConcatDataset(dataset_l)
        else:
            dataset = VOCSemanticSegmentationDataset(
                voc_path / "VOC2007", split="test", transforms=transforms
            )
    else:
        raise NotImplementedError(f"Dataset {dataset_name} is not supported")
    class_map = init_class_map(dataset_name)
    palette_map = init_palette_map(dataset_name, class_map)
    setattr(dataset, "class_map", class_map)
    setattr(dataset, "palette_map", palette_map)
    return dataset


def init_dataloader(cfg: DictConfig, split: str) -> DataLoader[Any]:
    """Build one dataloader for the requested dataset split.

    Args:
        cfg: Dataset configuration.
        split: Dataset split name.

    Returns:
        DataLoader[Any]: Configured dataloader for the split.
    """
    assert split in ["train", "val"], f"Split {split} is not supported"
    dataset = init_dataset(cfg, split, apply_augmentations=(split == "train"))
    dataloader_kwargs = dict(cfg.dataloader)
    dataloader_kwargs["drop_last"] = split == "train" and dataloader_kwargs.get("drop_last", False)
    dataloader_kwargs["shuffle"] = split == "train"
    dataloader = DataLoader(dataset, **dataloader_kwargs)
    return dataloader


def build_transforms(cfg: DictConfig, apply_augmentations: bool = False) -> T.Compose:
    """Create the image and mask transform pipeline.

    Args:
        cfg: Dataset configuration.
        apply_augmentations: Whether to include augmentation transforms.

    Returns:
        T.Compose: Composed torchvision v2 transform pipeline.
    """
    transforms: list[Any] = []
    resized_b = False
    if apply_augmentations:
        for aug in cfg.augmentations:
            if (
                aug.get("_target_") == "torchvision.transforms.v2.RandomResizedCrop"
                and aug.get("size") == cfg.input_size
            ):
                resized_b = True
            transforms.append(instantiate(aug, _convert_="all"))
    if not resized_b:
        transforms.append(T.Resize(to_2tuple(cfg.input_size)))
    mean = cfg.input_normalization.mean
    std = cfg.input_normalization.std
    transforms.extend(
        [
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=mean, std=std),
        ]
    )
    return T.Compose(transforms)


class VOCSemanticSegmentationDataset(Dataset):
    """Load Pascal VOC semantic segmentation image and mask pairs."""

    def __init__(self, dataset_path: Path, split: str, transforms: T.Compose | None) -> None:
        """Initialize dataset metadata and available samples.

        Args:
            dataset_path: Root directory for one Pascal VOC dataset split.
            split: Split file stem to read from ``ImageSets/Segmentation``.
            transforms: Optional transform pipeline applied to image and mask.
        """
        self.dataset_path = dataset_path
        self.split = split
        self.transforms = transforms
        self.samples: list[tuple[Path, Path]] = []

        img_id_file = dataset_path / "ImageSets" / "Segmentation" / f"{split}.txt"
        with open(img_id_file, "r") as f:
            img_ids = [line.strip() for line in f.readlines()]

        for img_id in img_ids:
            img_path = dataset_path / "JPEGImages" / f"{img_id}.jpg"
            mask_path = dataset_path / "SegmentationClass" / f"{img_id}.png"
            if not img_path.is_file():
                logger.warning(f"Missing image for {img_id}")
                continue
            if not mask_path.is_file():
                logger.warning(f"Missing mask for {img_id}")
                continue
            self.samples.append((img_path, mask_path))

    def __len__(self) -> int:
        """Return the number of valid image and mask pairs.

        Returns:
            int: Dataset length.
        """
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[tv_tensors.Image, tv_tensors.Mask]:
        """Load one image and segmentation mask pair.

        Args:
            idx: Sample index.

        Returns:
            tuple[tv_tensors.Image, tv_tensors.Mask]: Loaded image and mask tensors.
        """
        img_path, mask_path = self.samples[idx]
        img = read_image(str(img_path))
        img = tv_tensors.Image(img)
        mask = Image.open(mask_path)
        assert mask.mode == "P", f"{str(mask_path)!r} is not a palette image, got {mask.mode=}"
        mask_np = np.array(mask, dtype=np.uint8)
        mask = tv_tensors.Mask(torch.from_numpy(mask_np))
        if self.transforms is not None:
            img, mask = self.transforms(img, mask)
        return img, mask


def init_class_map(dataset_name: str) -> bidict[int, str]:
    """Build the class-id to class-name mapping for one dataset.

    Args:
        dataset_name: Supported dataset name.

    Returns:
        bidict[int, str]: Bi-directional mapping of class IDs and class names.
    """
    assert dataset_name in supported_dataset
    if dataset_name == "voc_2007_2012":
        voc_classes = [
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
        ]
        class_map = bidict({i: cls for i, cls in enumerate(voc_classes)})
        class_map[255] = "ignore"
        return class_map
    return bidict()


def init_palette_map(
    dataset_name: str,
    class_map: bidict[int, str],
) -> dict[int, torch.Tensor]:
    """Build the RGB palette lookup for dataset class IDs.

    Args:
        dataset_name: Supported dataset name.
        class_map: Class ID to name mapping.

    Returns:
        dict[int, torch.Tensor]: Mapping from class ID to RGB color tensor.
    """
    assert dataset_name in supported_dataset
    sample_mask_path = next((voc_path / "VOC2012" / "SegmentationClass").glob("*.png"))
    sample_mask = Image.open(sample_mask_path)
    assert sample_mask.mode == "P", f"{sample_mask_path} is not a palette image"
    palette = sample_mask.getpalette()
    assert palette is not None, f"{sample_mask_path} does not have a palette"
    palette = torch.tensor(
        [palette[i : i + 3] for i in range(0, len(palette), 3)],
        dtype=torch.uint8,
    )
    return {class_id: palette[class_id] for class_id in class_map if class_id != 255}


def decode_mask(mask: torch.Tensor, palette: dict[int, torch.Tensor]) -> torch.Tensor:
    assert mask.dtype == torch.uint8 and mask.ndim == 2, "mask must be uint8 and 2D"
    color_mask = torch.zeros(mask.shape[0], mask.shape[1], 3, dtype=torch.uint8)
    for class_id, color in palette.items():
        color_mask[mask == class_id] = color
    return color_mask


def plot_samples(
    images: torch.Tensor,
    masks: torch.Tensor,
    palette: dict[int, torch.Tensor],
    predictions: torch.Tensor | None = None,
    overlay_r: float = 0.3,
) -> None:
    """Visualize images with ground-truth and optional predicted masks.

    Args:
        images: Batch of image tensors shaped ``[batch, 3, height, width]``.
        masks: Batch of mask tensors shaped ``[batch, height, width]``.
        palette: Mapping from class ID to RGB color tensor.
        predictions: Optional predicted masks shaped ``[batch, height, width]``.
        overlay_r: Image blending ratio used for overlays.

    Returns:
        None: Displays the visualization and closes the figure.
    """
    assert images.ndim == 4 and images.size(1) == 3, "images must be B x 3 x H x W"
    assert masks.ndim == 3 and masks.size(0) == images.size(0), "masks must be B x H x W"
    assert predictions is None or predictions.ndim == 3 and predictions.size(0) == images.size(0), (
        "predictions must be B x H x W"
    )
    n = images.size(0)
    fig, axes = plt.subplots(n, 2 if predictions is None else 3, figsize=(10, 10))
    axes = np.atleast_2d(axes)
    for i in range(n):
        image_i = images[i].permute(1, 2, 0).numpy()
        image_i = (image_i - image_i.min()) / (image_i.max() - image_i.min() + 1e-8)
        axes[i, 0].imshow(image_i)
        axes[i, 0].axis("off")
        mask_i = decode_mask(masks[i].to(torch.uint8), palette).numpy() / 255.0
        overlay_i = image_i * overlay_r + mask_i * (1 - overlay_r)
        axes[i, 1].imshow(overlay_i)
        axes[i, 1].axis("off")
        if predictions is not None:
            pred_i = predictions[i]
            pred_i = decode_mask(pred_i.to(torch.uint8), palette).numpy() / 255.0
            overlay_pred_i = image_i * overlay_r + pred_i * (1 - overlay_r)
            axes[i, 2].imshow(overlay_pred_i)
            axes[i, 2].axis("off")
    fig.tight_layout()
    plt.show()
