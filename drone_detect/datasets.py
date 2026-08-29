"""Dataset registry and Roboflow download helpers.

Each disaster type maps to a pinned Roboflow dataset (workspace / project /
version). Versions are pinned so training runs remain reproducible; bump the
version number in ``configs/*.yaml`` when a new dataset version is published.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

DEFAULT_WORKSPACE = "ashwins-workspace-zz9gf"


@dataclass(frozen=True)
class DatasetSpec:
    """A pinned Roboflow dataset version plus local training hyperparameters.

    Attributes:
        name: Short slug used as the local run name (e.g. ``flood``).
        workspace: Roboflow workspace id.
        project: Roboflow project slug.
        version: Pinned dataset version number.
        run_name: Name of the training run folder under ``runs/<run_name>``.
        description: Human-readable summary shown in logs and docs.
        train: Extra keyword arguments forwarded to ``YOLO.train()``.
    """

    name: str
    workspace: str
    project: str
    version: int
    run_name: str
    description: str = ""
    train: dict = field(default_factory=dict)

    @property
    def slug(self) -> str:
        """Roboflow dataset identifier in ``workspace/project/version`` form."""
        return f"{self.workspace}/{self.project}/{self.version}"


def load_config(path: str | Path) -> list[DatasetSpec]:
    """Load dataset specs from a YAML config file.

    Example config (see ``configs/datasets.yaml``)::

        workspace: ashwins-workspace-zz9gf
        datasets:
          - name: flood
            project: ai-disaster-survival-drone
            version: 2
            run_name: flood_model
            train: {epochs: 50, imgsz: 640}

    Args:
        path: Path to the YAML config file.

    Returns:
        A list of :class:`DatasetSpec`, one per entry.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If a dataset entry is missing required keys.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    workspace = cfg.get("workspace", DEFAULT_WORKSPACE)

    specs: list[DatasetSpec] = []
    for entry in cfg.get("datasets", []):
        missing = {"name", "project", "version"} - entry.keys()
        if missing:
            raise ValueError(
                f"Dataset entry missing required key(s) {sorted(missing)}: {entry}"
            )
        specs.append(
            DatasetSpec(
                name=entry["name"],
                workspace=entry.get("workspace", workspace),
                project=entry["project"],
                version=int(entry["version"]),
                run_name=entry.get("run_name", entry["name"]),
                description=entry.get("description", ""),
                train=entry.get("train", {}),
            )
        )
    logger.debug("Loaded %d dataset spec(s) from %s", len(specs), path)
    return specs


def download_dataset(spec: DatasetSpec, api_key: str, location: str | Path) -> Path:
    """Download a pinned Roboflow dataset version in YOLOv11 format.

    Downloads to ``<location>/<project>-<version>``. If that folder already
    exists, Roboflow's SDK reuses it instead of re-downloading, so repeated
    training runs against the same pinned version are cheap.

    Args:
        spec: The dataset spec to download.
        api_key: Roboflow API key (from ``ROBOFLOW_API_KEY`` env var).
        location: Root directory for dataset downloads.

    Returns:
        Path to the extracted dataset folder (the one containing ``data.yaml``).

    Raises:
        RuntimeError: If the Roboflow API rejects the key or the dataset.
    """
    from roboflow import Roboflow

    root = Path(location)
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"{spec.project}-v{spec.version}"
    if (target / "data.yaml").exists():
        logger.info("Dataset %s already present at %s, skipping download", spec.name, target)
        return target

    logger.info("Downloading %s from Roboflow (%s)", spec.name, spec.slug)
    rf = Roboflow(api_key=api_key)
    dataset = (
        rf.workspace(spec.workspace).project(spec.project).version(spec.version).download(
            model_format="yolov11",
            location=str(target),
            overwrite=False,
        )
    )
    loc = Path(dataset.location)
    if not (loc / "data.yaml").exists():
        raise RuntimeError(
            f"Downloaded dataset at {loc} does not contain data.yaml "
            f"(project={spec.project!r}, version={spec.version})"
        )
    logger.info("Dataset %s ready at %s", spec.name, loc)
    return loc


def resolve_data_yaml(spec: DatasetSpec, datasets_root: str | Path) -> Path:
    """Locate the dataset's ``data.yaml`` under the download root without downloading.

    Args:
        spec: The dataset spec to look up.
        datasets_root: Directory containing downloaded datasets.

    Returns:
        Path to ``data.yaml``.

    Raises:
        FileNotFoundError: If the dataset has not been downloaded yet.
    """
    root = Path(datasets_root)
    for candidate in (
        root / f"{spec.project}-v{spec.version}" / "data.yaml",
        root / spec.project / "data.yaml",
    ):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Dataset '{spec.name}' not found under {root}. "
        f"Run `drone_detect datasets download` first."
    )
