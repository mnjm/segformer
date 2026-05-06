# SegFormer

PyTorch implementation of SegFormer for semantic segmentation, with Pascal VOC training/evaluation, Hydra-based configuration, and parity tests against Hugging Face checkpoints.

## What is in this repo

- Local implementation of the SegFormer encoder/decoder stack in [`model/`](./model)
- Pretrained encoder initialization from Hugging Face SegFormer checkpoints
- Pascal VOC 2007 + 2012 data pipeline in [`data.py`](./data.py)
- Training entrypoint in [`train.py`](./train.py)
- Evaluation and visualization entrypoint in [`eval.py`](./eval.py)
- Variant configs for `segformer-b0` through `segformer-b5` in [`config/model/`](./config/model)
- Parity test to verify this implementation matches Hugging Face logits in [`tests/test_parity_hf.py`](./tests/test_parity_hf.py)

## Requirements

- Python 3.12+
- `uv` for environment management
- `curl` and `unzip` for dataset download
- CUDA is the default training target in [`config/default.yaml`](./config/default.yaml)

The project pins PyTorch through `uv`:

- Linux: CUDA 12.9 wheels
- Non-Linux: CPU wheels

## Setup

Install dependencies:

```bash
uv sync
```

If you want to enable Weights & Biases logging, create a `.env` file with:

```bash
WANDB_API_KEY=...
```

W&B logging is off by default. Enable it with a Hydra override:

```bash
uv run python train.py logging.wandb.enable=true
```

## Dataset

This repo expects Pascal VOC under `./dataset/voc-datasets` and ships a helper script:

```bash
chmod +x download-voc-dataset.sh
./download-voc-dataset.sh
```

Expected layout after extraction:

```text
dataset/
  voc-datasets/
    VOC2007/
    VOC2012/
```

Training uses:

- `VOC2007/trainval`
- `VOC2012/trainval`

Validation uses:

- `VOC2007/test`

The default dataset config is:

- Dataset name: `voc_2007_2012`
- Input size: `512`
- Classes: `21` foreground/background classes, with VOC ignore label `255`

## Training

Run training with the default config:

```bash
uv run python train.py
```

Default behavior:

- Model: `segformer-b0`
- Device: `cuda`
- Epochs: `300`
- Mixed precision dtype: `bf16`
- Gradient accumulation: `4`
- Validation every `10` epochs
- Checkpoint every `20` epochs
- Hydra outputs under `./logs/<run-name>/`

On a fresh run, `train.py` initializes the encoder from the matching Hugging Face checkpoint and trains the decoder plus encoder locally.

Useful overrides:

```bash
uv run python train.py model=segformer-b2
uv run python train.py device_type=cpu
uv run python train.py device_type=auto
uv run python train.py dataset.dataloader.batch_size=4
uv run python train.py init_from=/absolute/path/to/checkpoint.pt
uv run python train.py torch_compile=true
```

Notes:

- `device_type=cuda` asserts that CUDA is available.
- `device_type=auto` is intended for non-CUDA environments and will prefer MPS, then XLA, then CPU.
- Resuming from `init_from=...` restores model, optimizer, scheduler, epoch, and W&B run id.

## Evaluation

`eval.py` operates on checkpoints produced by `train.py`.

Compute metrics on the validation split:

```bash
uv run python eval.py ./logs/<run-name>/segformer-b0.pt --compute_metrics
```

Evaluate on the training split instead:

```bash
uv run python eval.py ./logs/<run-name>/segformer-b0.pt --split train --compute_metrics
```

Visualize the best matching samples for selected classes:

```bash
uv run python eval.py ./logs/<run-name>/segformer-b0.pt \
  --class-names person,dog \
  --top 8
```

Show the worst matching samples for a class:

```bash
uv run python eval.py ./logs/<run-name>/segformer-b0.pt \
  --class-names bicycle \
  --bottom 8
```

Show the best samples across all classes:

```bash
uv run python eval.py ./logs/<run-name>/segformer-b0.pt --top 8
```

Important evaluation flags:

- `--compute_metrics`: overall and per-class metrics
- `--split {train,val}`: dataset split, default `val`
- `--class-names ...`: comma-separated class labels; empty means all classes
- `--top N`: lowest-loss samples by loss on the selected classes
- `--bottom N`: highest-loss samples by loss on the selected classes
- `--no-compile`: ignore `torch_compile` from the saved config

## Model Variants

Available configs:

- `segformer-b0`
- `segformer-b1`
- `segformer-b2`
- `segformer-b3`
- `segformer-b4`
- `segformer-b5`

Each variant is defined in [`config/model/`](./config/model) and can be selected with a Hydra override:

```bash
uv run python train.py model=segformer-b5
```

## Testing

Run the parity test suite:

```bash
uv run pytest tests/test_parity_hf.py
```

This test:

- Loads each local `segformer-b*` config
- Maps Hugging Face weights into the local implementation
- Verifies parameter count parity
- Verifies output logits match Hugging Face within tolerance

The first run may download Hugging Face model weights into `./cache/`.

## Project Structure

```text
.
├── config/
│   ├── default.yaml
│   ├── device/
│   └── model/
├── model/
│   ├── decoder.py
│   ├── hf_mapper.py
│   └── mix_transformer.py
├── tests/
│   └── test_parity_hf.py
├── data.py
├── eval.py
├── train.py
├── utils.py
└── download-voc-dataset.sh
```

## Output Artifacts

Training runs are written to:

- `logs/<hydra-job-name>/`

Typical contents:

- `<model-name>.pt`: checkpoint
- `config/`: Hydra-resolved config snapshot

Hugging Face downloads are cached in:

- `cache/`
