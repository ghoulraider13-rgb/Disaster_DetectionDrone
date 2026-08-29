"""Training loop: fine-tune YOLOv11n on a downloaded Roboflow dataset."""

from __future__ import annotations

import logging
from pathlib import Path

from .datasets import DatasetSpec

logger = logging.getLogger(__name__)

#: Base weights every model fine-tunes from (COCO-pretrained nano variant).
BASE_MODEL = "yolo11n.pt"

#: Training defaults shared by every disaster model. Config ``train`` blocks
#: override any of these per dataset.
DEFAULT_TRAIN_ARGS: dict = {
    "epochs": 50,
    "imgsz": 640,
    "batch": -1,  # auto-batch: pick the largest batch that fits VRAM
    "patience": 20,  # early stopping (notebook trained blind for 50 epochs)
    "plots": True,
    "seed": 42,  # reproducibility (absent from the original notebook)
    "cos_lr": True,  # cosine LR decay, gentler than default linear
    "device": 0,  # CUDA:0 when available
}


def _resolve_device(value) -> int | str:
    """Map a device setting to something Ultralytics accepts.

    The config ships ``device: 0`` (CUDA). On CPU-only boxes we fall back to
    ``cpu`` instead of crashing, but only when torch actually lacks CUDA.
    """
    if value in (None, "", "auto", 0):
        try:
            import torch

            if torch.cuda.is_available():
                return 0
        except ImportError:  # torch missing: let ultralytics decide
            return 0
        return "cpu"
    return value


def train_model(
    spec: DatasetSpec,
    data_yaml: str | Path,
    base_model: str = BASE_MODEL,
    output_root: str | Path = "runs",
    train_args: dict | None = None,
) -> Path:
    """Fine-tune the base YOLOv11n model on one disaster dataset.

    Args:
        spec: Dataset spec providing the run name and per-dataset overrides.
        data_yaml: Path to the dataset's ``data.yaml``.
        base_model: Ultralytics weights to start from (COCO-pretrained YOLOv11n).
        output_root: Directory where run folders are created.
        train_args: Extra keyword arguments to ``YOLO.train()``; merged over
            :data:`DEFAULT_TRAIN_ARGS`.

    Returns:
        Path to the run directory containing ``weights/best.pt``.

    Raises:
        RuntimeError: If training completed but ``best.pt`` is missing.
    """
    from ultralytics import YOLO

    overrides = {**DEFAULT_TRAIN_ARGS, **(spec.train or {}), **(train_args or {})}
    device = _resolve_device(overrides.pop("device", 0))
    overrides["device"] = device

    run_root = Path(output_root) / "drone_models" / spec.run_name
    logger.info(
        "Training %s: base=%s data=%s device=%s epochs=%s -> %s",
        spec.name, base_model, data_yaml, device, overrides.get("epochs"), run_root,
    )
    model = YOLO(base_model)
    model.train(data=str(data_yaml), project=str(Path(output_root) / "drone_models"), name=spec.run_name, **overrides)

    best = run_root / "weights" / "best.pt"
    if not best.exists():
        raise RuntimeError(f"Training finished but {best} was not produced")
    logger.info("Best weights for %s at %s", spec.name, best)
    return best
