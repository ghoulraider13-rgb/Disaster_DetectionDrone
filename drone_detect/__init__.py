"""Train YOLOv11 models for UAV disaster detection and survivor search.

A small, opinionated pipeline built on Ultralytics YOLOv11 for disaster-response
drones: download Roboflow datasets, fine-tune compact nano models, and export
TensorRT / ONNX / OpenVINO artifacts for onboard edge inference.
"""

from __future__ import annotations

import logging
from pathlib import Path

__version__ = "1.0.0"

#: Package-wide logger. Libraries configure a NullHandler; the CLI attaches a
#: console handler so log lines only appear when a user explicitly runs a command.
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

#: Default project root (two parents up from this file).
ROOT = Path(__file__).resolve().parent.parent
