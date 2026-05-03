# This file contains a small script for loading and inspecting a Hugging Face SegFormer model.
# Author: mnjm
# %%
from pathlib import Path

from transformers import SegformerForSemanticSegmentation

MODEL_ID = "nvidia/segformer-b0-finetuned-ade-512-512"
cache_dir = Path("./cache")
cache_dir.mkdir(parents=True, exist_ok=True)

model = SegformerForSemanticSegmentation.from_pretrained(
    MODEL_ID,
    cache_dir=str(cache_dir),
)
print(model.config)

# %%
print(model)
# %%
params = sum(p.numel() for p in model.parameters())
print(f"{params=}")
