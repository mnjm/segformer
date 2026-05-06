# SegFormer

A minimal implementation of [SegFormer](https://arxiv.org/pdf/2105.15203). Validated on VOC 2007 and 2012 datasets.

## Structure

- SegFormer implementation in [`model/`](./model)
- Pascal VOC semantic segmentation data pipeline in [`data.py`](./data.py)
- Training entrypoint in [`train.py`](./train.py)
- Evaluation and visualization in [`eval.py`](./eval.py)
- Model configs for `segformer-b0` through `segformer-b5` in [`config/model/`](./config/model)
- A parity test in [`tests/test_parity_hf.py`](./tests/test_parity_hf.py)

## Setup

Requirements:

- Python 3.12+
- `uv`
- `curl` and `unzip` for the dataset helper script

Install dependencies:

```bash
uv sync
```


## Dataset

The code expects Pascal VOC under `./dataset/voc`. The helper script will download and unpack it:

```bash
./download-voc-dataset.sh
```

Expected layout:

```text
dataset/
  voc/
    VOC2007/
    VOC2012/
```

Training uses `VOC2007/trainval` and `VOC2012/trainval`. Validation uses `VOC2007/test`.

## Training

Run the default config:

```bash
uv run python train.py
```

Defaults are `segformer-b0`, `cuda`, `300` epochs, `bf16`, gradient accumulation `4`. Outputs go under `logs/<run-name>/`.

A fresh run initializes the encoder from the matching Hugging Face checkpoint.

Common overrides:

```bash
uv run python train.py model=segformer-b2
uv run python train.py device_type=cpu
uv run python train.py device_type=auto
uv run python train.py dataset.dataloader.batch_size=4
uv run python train.py init_from=/absolute/path/to/checkpoint.pt
uv run python train.py torch_compile=true
```

`device_type=cuda` requires CUDA. `device_type=auto` falls back to MPS, then XLA, then CPU. `init_from=...` resumes model, optimizer, scheduler, epoch, and W&B state (if enabled).

If you want Weights & Biases logging, add this to `.env`:

```bash
WANDB_API_KEY=...
```

Then enable it when training:

```bash
uv run python train.py logging.wandb.enable=true
```

## Evaluation

Evaluate a checkpoint and compute metrics:

```bash
uv run python eval.py ./logs/<run-name>/segformer-b0.pt --compute_metrics
```

Run on the training split instead:

```bash
uv run python eval.py ./logs/<run-name>/segformer-b0.pt --split train --compute_metrics
```

Show best examples for selected classes:

```bash
uv run python eval.py ./logs/<run-name>/segformer-b0.pt \
  --class-names person,dog \
  --top 8
```

Show worst examples for a class:

```bash
uv run python eval.py ./logs/<run-name>/segformer-b0.pt \
  --class-names bicycle \
  --bottom 8
```

Useful flags:

- `--compute_metrics`
- `--split {train,val}`
- `--class-names ...`
- `--top N`
- `--bottom N`

## Testing

Run the Hugging Face parity test:

```bash
uv run pytest tests/test_parity_hf.py
```

It checks weight mapping, parameter counts, and output logits across the model variants. The first run may populate `./cache/`.

```
@misc{xie2021segformer,
    title   = {SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers},
    author  = {Enze Xie and Wenhai Wang and Zhiding Yu and Anima Anandkumar and Jose M. Alvarez and Ping Luo},
    year    = {2021},
    eprint  = {2105.15203},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
```