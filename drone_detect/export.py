"""Export trained weights to deployment formats for onboard drone inference.

Ultralytics handles the actual conversion; this module wraps it with sane,
format-aware defaults for edge GPUs (Jetson via TensorRT, laptops/NUCs via
ONNX/OpenVINO). Ultralytics 8.4+ validates export arguments per format, so
each format gets its own allow-listed argument profile instead of one
shared dict (e.g. ``opset``/``simplify`` are ONNX/TensorRT-only).
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

#: Export formats worth shipping for drone payloads (Ultralytics argument names).
SUPPORTED_FORMATS = ("onnx", "openvino", "engine", "torchscript")

#: Per-format argument profiles for ``YOLO.export()``. Only arguments the format
#: actually supports are passed (validated against Ultralytics' export table).
#: ``imgsz`` is universal; precision is left at the Ultralytics default so the
#: same code works on ultralytics>=8.3 (``half``) and >=8.4 (``quantize``).
FORMAT_ARGS: dict[str, dict] = {
    "onnx": {"imgsz": 640, "dynamic": False, "simplify": True, "opset": 12},
    "engine": {"imgsz": 640, "dynamic": False, "simplify": True, "opset": 12},
    "openvino": {"imgsz": 640, "dynamic": False},
    "torchscript": {"imgsz": 640, "dynamic": False},
}


def export_model(
    weights: str | Path,
    fmt: str = "onnx",
    output_dir: str | Path | None = None,
    export_args: dict | None = None,
) -> Path:
    """Export a trained ``best.pt`` to a deployment format.

    Args:
        weights: Path to trained Ultralytics weights (``best.pt``).
        fmt: One of ``onnx``, ``openvino``, ``engine`` (TensorRT), ``torchscript``.
        output_dir: Where to write artifacts (defaults to next to the weights).
        export_args: Extra keyword arguments to ``YOLO.export()`` merged over
            the format's profile in :data:`FORMAT_ARGS`.

    Returns:
        Path to the exported artifact.

    Raises:
        FileNotFoundError: If ``weights`` does not exist.
        ValueError: If ``fmt`` is not a supported export format.
    """
    from ultralytics import YOLO

    weights = Path(weights)
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")
    fmt = fmt.lower()
    if fmt not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported format {fmt!r}; expected one of {SUPPORTED_FORMATS}")

    overrides = {**FORMAT_ARGS.get(fmt, {}), **(export_args or {})}
    # Drop args the target format does not support (Ultralytics >=8.4 asserts on these).
    supported = {"imgsz", "batch", "data", "nms", "fraction", *FORMAT_ARGS.get(fmt, {})}
    dropped = {k: v for k, v in overrides.items() if k not in supported}
    if dropped:
        logger.warning("Ignoring args not supported by format %s: %s", fmt, sorted(dropped))
    overrides = {k: v for k, v in overrides.items() if k in supported}

    logger.info("Exporting %s to %s (args: %s)", weights, fmt, overrides)
    model = YOLO(str(weights))
    path = model.export(format=fmt, **overrides)

    exported = Path(path)
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        dest = output_dir / exported.name
        exported = exported.replace(dest)
        logger.info("Moved export artifact to %s", dest)
    return exported
