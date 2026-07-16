"""
Run an exported Hugging Face SegFormer model on one image for sanity checking.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import torch
from PIL import Image
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

from utils import torch_get_device


def generate_palette(num_labels: int) -> list[int]:
    """Create a deterministic RGB palette with black reserved for class zero.

    Args:
        num_labels: Number of semantic labels in the model configuration.

    Returns:
        Flat PIL palette containing 256 RGB entries.
    """
    palette = [0] * (256 * 3)
    for class_id in range(1, min(num_labels, 256)):
        palette[class_id * 3 : class_id * 3 + 3] = [
            (class_id * 37) % 256,
            (class_id * 17) % 256,
            (class_id * 97) % 256,
        ]
    return palette

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", help="Directory created by hf_exporter.py or Hub model ID")
    parser.add_argument("image", type=Path, help="Image to segment")
    parser.add_argument(
        "--output", type=Path, default=Path("segmentation.png"), help="Output mask PNG"
    )
    args = parser.parse_args()

    device = torch_get_device("cuda" if torch.cuda.is_available() else "auto")
    processor = AutoImageProcessor.from_pretrained(args.model_dir)
    model = cast(
        SegformerForSemanticSegmentation,
        SegformerForSemanticSegmentation.from_pretrained(args.model_dir),
    )
    torch.nn.Module.to(model, device)
    model.eval()
    image = Image.open(args.image).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.inference_mode():
        outputs = model(**inputs)
    prediction = processor.post_process_semantic_segmentation(
        outputs, target_sizes=[(image.height, image.width)]
    )[0].to("cpu", torch.uint8)

    output = Image.fromarray(prediction.numpy(), mode="P")
    output.putpalette(generate_palette(model.config.num_labels))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.save(args.output)
    print(f"Saved segmentation mask to {args.output}")

if __name__ == "__main__":
    main()
