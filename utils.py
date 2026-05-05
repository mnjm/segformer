"""Small utility helpers shared by training scripts."""

import os
from datetime import datetime
from functools import wraps
from time import time
from typing import Any, Callable, ParamSpec, TypeVar

import pytz
import torch
import wandb
from dotenv import load_dotenv
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

P = ParamSpec("P")
R = TypeVar("R")


def to_2tuple(x: Any) -> Any:
    """
    Return x as a 2-tuple; if x is scalar, return (x, x).

    Args:
        x: Scalar or sequence value.

    Returns:
        Any: Original sequence value or a duplicated 2-tuple.
    """
    return x if isinstance(x, (list, tuple)) else (x, x)


def torch_get_device(device_type: str) -> torch.device:
    """
    Resolve a device string to a torch.device.

    Args:
        device_type: One of "cuda", "auto", or "cpu". "auto" uses MPS or XLA
            if available, otherwise CPU.

    Returns:
        torch.device: The resolved device.

    Raises:
        AssertionError: If device_type is "cuda" and CUDA is not available, or
            "auto" but CUDA is available (to avoid accidental CPU fallback).
    """
    if device_type == "cuda":
        assert torch.cuda.is_available(), "CUDA is not available :(, `python train.py +device=auto`"
        device = torch.device("cuda")
    elif device_type == "auto":
        assert not torch.cuda.is_available(), "CUDA is available :), switch to cuda"
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device("mps")
        else:
            try:
                import torch_xla.core.xla_model as xm  # type: ignore

                device = xm.xla_device()
            except ImportError:
                device = torch.device("cpu")
    else:
        device = torch.device("cpu")
    return device


def torch_set_seed(seed: int) -> None:
    """
    Set PyTorch and CUDA random seeds for reproducibility.

    Args:
        seed: Integer seed value. Also enables deterministic cuDNN when CUDA
            is available.

    Returns:
        None: Seeds PyTorch random number generators in place.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def torch_compile_ckpt_fix(state_dict: dict[str, Any]) -> dict[str, Any]:
    """
    Remove '_orig_mod.' prefix from state_dict keys added by torch.compile.

    Args:
        state_dict: State dict (modified in place).

    Returns:
        dict[str, Any]: The same state dict with corrected parameter names.
    """
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    return state_dict


def get_ist_time_now(fmt: str = "%d-%m-%Y-%H%M%S") -> str:
    """
    Return current time in Asia/Kolkata (IST) as a formatted string.

    Args:
        fmt: strftime format string. Default is "%d-%m-%Y-%H%M%S".

    Returns:
        str: Formatted datetime string in IST.
    """
    tz = pytz.timezone("Asia/Kolkata")
    now_ist = datetime.now(tz)
    return now_ist.strftime(fmt)


def timer(func: Callable[P, R]) -> Callable[P, tuple[float, R]]:
    """
    Decorator that times a function and returns (elapsed_seconds, return_value).

    Args:
        func: Function to wrap.

    Returns:
        Callable[P, tuple[float, R]]: Wrapped callable returning elapsed time and value.
    """

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> tuple[float, R]:
        """
        Run the wrapped function and measure elapsed time.

        Args:
            *args: Positional arguments passed to func.
            **kwargs: Keyword arguments passed to func.

        Returns:
            tuple[float, R]: Elapsed time in seconds and the wrapped return value.
        """
        t0 = time()
        ret = func(*args, **kwargs)
        t = time() - t0
        return t, ret

    return wrapper


def poly_with_linear_warmup_lr_scheduler(
    optimizer: Optimizer,
    total_steps: int,
    warmup_pct: float = 0.075,
    warmup_ratio: float = 1e-6,
    power: float = 1.0,
    min_lr: float = 0.0,
) -> LambdaLR:
    """
    Learning rate schedule:
    1. Warmup: linear increase from (warmup_ratio * base_lr) → base_lr
    2. Poly decay: lr = (base_lr - min_lr) * (1 - step/total_steps)^power + min_lr

    Args:
        optimizer: Optimizer whose learning rate will be scheduled.
        total_steps: Total number of scheduler steps.
        warmup_pct: Fraction of steps used for warmup.
        warmup_ratio: Initial warmup LR ratio relative to base LR.
        power: Polynomial decay exponent.
        min_lr: Minimum LR ratio reached at the end of decay.

    Returns:
        LambdaLR: Configured warmup-plus-polynomial scheduler.
    """
    warmup_iters = int(warmup_pct * total_steps)

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_iters:
            alpha = float(current_step) / float(max(1, warmup_iters))
            return warmup_ratio + alpha * (1.0 - warmup_ratio)

        progress = float(current_step - warmup_iters) / float(max(1, total_steps - warmup_iters))
        progress = min(progress, 1.0)

        poly_decay = (1 - progress) ** power
        return poly_decay * (1 - min_lr) + min_lr

    return LambdaLR(optimizer, lr_lambda)


class SegmentationMetrics:
    """Track segmentation loss, mIoU, and pixel accuracy across batches."""

    def __init__(
        self, num_classes: int, device: torch.device, ignore_index: int | None = None
    ) -> None:
        """Initialize metric accumulators.

        Args:
            num_classes: Number of valid classes included in metrics.
            device: Device used to store the confusion matrix.
            ignore_index: Optional label ID excluded from evaluation.
        """
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.device = device

        self.reset()

    def reset(self) -> None:
        """Reset all running metric accumulators.

        Returns:
            None: Clears losses, sample counts, and confusion matrix state.
        """
        self.total_loss = 0.0
        self.total_samples = 0
        confmat_dtype = torch.float32 if self.device.type == "mps" else torch.float64

        self.confmat = torch.zeros(
            (self.num_classes, self.num_classes), dtype=confmat_dtype, device=self.device
        )

    @torch.no_grad()
    def update(self, logits: torch.Tensor, targets: torch.Tensor, loss: torch.Tensor) -> None:
        """Accumulate metrics for one batch.

        Args:
            logits: Predicted logits shaped ``[batch, classes, height, width]``.
            targets: Target masks shaped ``[batch, height, width]``.
            loss: Batch loss tensor.

        Returns:
            None: Updates running totals in place.
        """
        # accumulate loss
        batch_size = logits.shape[0]
        self.total_loss += loss.item() * batch_size
        self.total_samples += batch_size

        # predictions
        preds = torch.argmax(logits, dim=1)

        preds = preds.view(-1)
        targets = targets.view(-1)

        if self.ignore_index is not None:
            mask = targets != self.ignore_index
            preds = preds[mask]
            targets = targets[mask]

        # valid class mask
        mask = (
            (targets >= 0)
            & (targets < self.num_classes)
            & (preds >= 0)
            & (preds < self.num_classes)
        )
        preds = preds[mask]
        targets = targets[mask]

        # confusion matrix update
        indices = self.num_classes * targets + preds
        self.confmat += torch.bincount(indices, minlength=self.num_classes**2).reshape(
            self.num_classes, self.num_classes
        )

    def compute(self) -> dict[str, float]:
        """Compute aggregate metrics from accumulated state.

        Returns:
            dict[str, float]: Mean loss, mean IoU, and pixel accuracy.
        """
        # mean loss
        mean_loss = self.total_loss / max(self.total_samples, 1)

        # confusion matrix components
        intersection = torch.diag(self.confmat)
        union = self.confmat.sum(0) + self.confmat.sum(1) - intersection

        # IoU per class
        iou = intersection / union.clamp(min=1)

        # mean IoU (ignore empty classes)
        valid = union > 0
        miou = iou[valid].mean() if valid.any() else torch.tensor(0.0, device=self.device)

        # pixel accuracy
        correct = intersection.sum()
        total = self.confmat.sum()
        acc = correct / total.clamp(min=1)

        return {"loss": mean_loss, "miou": miou.item(), "accuracy": acc.item()}


class WandBLogger:
    """Weights & Biases logger for train/val metrics with configurable tags."""

    def __init__(
        self,
        project: str,
        run: str,
        config: dict[str, Any],
        tags: tuple[str, ...],
        metrics: tuple[str, ...],
        run_id: str | None = None,
        enable: bool = False,
    ) -> None:
        """
        Initialize W&B run and define metrics.

        Args:
            project: W&B project name.
            run: Run name.
            config: Dict-like config to log.
            tags: Tuple of tag names (e.g. ("train", "val")).
            metrics: Tuple of metric names to define per tag.
            run_id: Optional run ID for resuming.
            enable: If False, log calls are no-ops.

        Returns:
            None: Initializes the optional W&B run state.
        """
        self.enable = enable
        self.run_id = run_id
        if self.enable:
            load_dotenv()
            wnb_key = os.getenv("WANDB_API_KEY")
            assert wnb_key is not None, "WANDB_API_KEY not loaded in env"
            wandb.login(key=wnb_key)
            wandb_run = wandb.init(
                project=project, name=run, config=config, id=self.run_id, resume="allow"
            )
            assert wandb_run is not None
            self.run_id = wandb_run.id
            self.tags = tags
            self.metrics = metrics
            wandb.define_metric("epoch")
            for tag in tags:
                for metric in metrics:
                    wandb.define_metric(f"{tag}/{metric}", step_metric="epoch")

    def log(self, tag: str, data: dict[str, float]) -> None:
        """
        Log metrics for a tag. data must contain 'epoch' and metric keys from
        self.metrics.

        Args:
            tag: One of the tags passed at init (e.g. "train", "val").
            data: Dict with "epoch" and metric name -> value.

        Returns:
            None: Sends metrics to W&B when logging is enabled.
        """
        if self.enable:
            assert tag in self.tags, f"{tag=} not created"
            new_data = {"epoch": data["epoch"]}
            for metric, val in data.items():
                if metric == "epoch":
                    continue
                assert metric in self.metrics, f"{metric=} not created"
                new_data[f"{tag}/{metric}"] = val
            wandb.log(new_data)
