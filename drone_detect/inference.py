"""Inference helpers: run trained models on images, videos, or camera streams."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

#: Detections below this confidence are dropped at inference time.
DEFAULT_CONF = 0.25

#: Non-maximum suppression IoU threshold.
DEFAULT_IOU = 0.6


def predict(
    weights: str | Path,
    source: str | int,
    conf: float = DEFAULT_CONF,
    iou: float = DEFAULT_IOU,
    save: bool = True,
    extra_args: dict | None = None,
) -> list[dict]:
    """Run detection on an image/video/stream and return per-frame results.

    Args:
        weights: Path to trained ``best.pt``.
        source: File path, directory, URL, webcam index (``0``), or RTSP URL.
        conf: Confidence threshold.
        iou: NMS IoU threshold.
        save: Whether Ultralytics should write annotated media to ``runs/``.
        extra_args: Extra keyword arguments to ``YOLO.predict()``.

    Returns:
        A list of per-frame dicts::

            {"frame": i, "image": path_or_None, "boxes": [
                {"class": "flood", "conf": 0.87, "xyxy": [x1, y1, x2, y2]}
            ]}

    Raises:
        FileNotFoundError: If ``weights`` does not exist.
    """
    from ultralytics import YOLO

    weights = Path(weights)
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")

    model = YOLO(str(weights))
    results = model.predict(source=source, conf=conf, iou=iou, save=save, **(extra_args or {}))

    frames: list[dict] = []
    for i, res in enumerate(results):
        boxes = []
        for box in res.boxes:
            boxes.append(
                {
                    "class": model.names.get(int(box.cls), str(box.cls)),
                    "conf": float(box.conf),
                    "xyxy": [round(float(v), 1) for v in box.xyxy[0].tolist()],
                }
            )
        frames.append(
            {
                "frame": i,
                "image": str(Path(res.path)) if getattr(res, "path", None) else None,
                "boxes": boxes,
            }
        )
    return frames
