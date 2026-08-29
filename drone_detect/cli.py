"""Command-line interface for the drone_detect package.

Subcommands::

    python -m drone_detect datasets list
    python -m drone_detect datasets download [--only flood]
    python -m drone_detect train [--only flood] [--epochs N] [--device cpu]
    python -m drone_detect export --weights runs/drone_models/flood_model/weights/best.pt --format onnx
    python -m drone_detect predict --weights ... --source image.jpg [--conf 0.25]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from . import __version__
from .datasets import (
    DatasetSpec,
    download_dataset,
    load_config,
    resolve_data_yaml,
)
from .train import train_model

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "datasets.yaml"
DEFAULT_DATASETS_DIR = "datasets"
DEFAULT_RUNS_DIR = "runs"


def _get_api_key() -> str:
    """Read the Roboflow API key from the environment or fail with guidance."""
    key = os.environ.get("ROBOFLOW_API_KEY", "")
    if not key:
        sys.exit(
            "Error: ROBOFLOW_API_KEY is not set.\n"
            "Get a key from https://app.roboflow.com/settings/account and export it:\n"
            "  export ROBOFLOW_API_KEY=your_key   (bash)\n"
            "  set ROBOFLOW_API_KEY=your_key      (Windows cmd)\n"
            "Or put it in a .env file (see .env.example)."
        )
    return key


def _spec_by_name(specs: list[DatasetSpec], name: str) -> DatasetSpec:
    for spec in specs:
        if spec.name == name:
            return spec
    valid = ", ".join(s.name for s in specs)
    sys.exit(f"Error: unknown dataset '{name}'. Valid names: {valid}")


# --------------------------------------------------------------------------- #
# Subcommand handlers
# --------------------------------------------------------------------------- #

def cmd_datasets_list(args: argparse.Namespace) -> None:
    specs = load_config(args.config)
    print(f"{'name':<12} {'roboflow slug':<50} {'run':<16} description")
    print("-" * 100)
    for s in specs:
        print(f"{s.name:<12} {s.slug:<50} {s.run_name:<16} {s.description}")


def cmd_datasets_download(args: argparse.Namespace) -> None:
    specs = load_config(args.config)
    if args.only:
        specs = [_spec_by_name(specs, args.only)]
    key = _get_api_key()
    for spec in specs:
        download_dataset(spec, api_key=key, location=args.datasets_dir)


def cmd_train(args: argparse.Namespace) -> None:
    specs = load_config(args.config)
    if args.only:
        specs = [_spec_by_name(specs, args.only)]
    key = None
    if any(not resolve_data_yaml_if_present(s, args.datasets_dir) for s in specs):
        key = _get_api_key()
    for spec in specs:
        if not resolve_data_yaml_if_present(spec, args.datasets_dir):
            download_dataset(spec, api_key=key, location=args.datasets_dir)
        data_yaml = resolve_data_yaml(spec, args.datasets_dir)
        overrides = {"epochs": args.epochs} if args.epochs else {}
        if args.device:
            overrides["device"] = args.device
        best = train_model(spec, data_yaml=data_yaml, output_root=args.runs_dir, train_args=overrides)
        print(f"[OK] {spec.name}: best weights -> {best}")


def resolve_data_yaml_if_present(spec: DatasetSpec, datasets_root: str) -> Path | None:
    try:
        return resolve_data_yaml(spec, datasets_root)
    except FileNotFoundError:
        return None


def cmd_export(args: argparse.Namespace) -> None:
    from .export import export_model

    path = export_model(
        weights=args.weights,
        fmt=args.format,
        output_dir=args.output_dir,
    )
    print(f"[OK] exported -> {path}")


def cmd_predict(args: argparse.Namespace) -> None:
    from .inference import predict

    frames = predict(
        weights=args.weights,
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        save=not args.no_save,
    )
    for frame in frames:
        for box in frame["boxes"]:
            print(
                f"frame {frame['frame']:>4}  {box['class']:<20} "
                f"conf={box['conf']:.2f}  xyxy={box['xyxy']}"
            )
    if not any(f["boxes"] for f in frames):
        print("No detections above threshold.")


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="drone_detect",
        description="Train YOLOv11 models for UAV disaster detection.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    # datasets -------------------------------------------------------------
    p_ds = sub.add_parser("datasets", help="manage Roboflow datasets")
    ds_sub = p_ds.add_subparsers(dest="ds_command", required=True)

    p_list = ds_sub.add_parser("list", help="list datasets in the config")
    p_list.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p_list.set_defaults(func=cmd_datasets_list)

    p_dl = ds_sub.add_parser("download", help="download datasets to disk")
    p_dl.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p_dl.add_argument("--only", help="download a single dataset by name")
    p_dl.add_argument("--datasets-dir", default=DEFAULT_DATASETS_DIR)
    p_dl.set_defaults(func=cmd_datasets_download)

    # train ------------------------------------------------------------------
    p_train = sub.add_parser("train", help="fine-tune YOLOv11n on one or all datasets")
    p_train.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p_train.add_argument("--only", help="train a single dataset by name")
    p_train.add_argument("--datasets-dir", default=DEFAULT_DATASETS_DIR)
    p_train.add_argument("--runs-dir", default=DEFAULT_RUNS_DIR)
    p_train.add_argument("--epochs", type=int, help="override epochs (default: config)")
    p_train.add_argument("--device", help="override device, e.g. 'cpu' or 0 (default: config)")
    p_train.set_defaults(func=cmd_train)

    # export -----------------------------------------------------------------
    p_exp = sub.add_parser("export", help="export trained weights for edge deployment")
    p_exp.add_argument("--weights", required=True, type=Path, help="path to best.pt")
    p_exp.add_argument("--format", default="onnx", help="onnx | openvino | engine | torchscript")
    p_exp.add_argument("--output-dir", default=None, help="where to place the artifact")
    p_exp.set_defaults(func=cmd_export)

    # predict ----------------------------------------------------------------
    p_pred = sub.add_parser("predict", help="run detection on an image/video/stream")
    p_pred.add_argument("--weights", required=True, type=Path)
    p_pred.add_argument("--source", required=True, help="image, video, dir, URL, or webcam index")
    p_pred.add_argument("--conf", type=float, default=0.25)
    p_pred.add_argument("--iou", type=float, default=0.6)
    p_pred.add_argument("--no-save", action="store_true", help="do not write annotated media")
    p_pred.set_defaults(func=cmd_predict)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
