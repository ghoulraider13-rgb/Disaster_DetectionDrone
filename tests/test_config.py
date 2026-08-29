"""Tests for config loading and dataset resolution logic.

These tests deliberately avoid importing ultralytics/roboflow/torch (the
modules import them lazily inside functions), so they run in seconds on any
environment with just pyyaml installed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from drone_detect.datasets import (
    DEFAULT_WORKSPACE,
    DatasetSpec,
    load_config,
    resolve_data_yaml,
)

CONFIG = ROOT / "configs" / "datasets.yaml"


# --------------------------------------------------------------------------- #
# DatasetSpec
# --------------------------------------------------------------------------- #

def test_spec_defaults():
    spec = DatasetSpec(name="x", workspace="w", project="p", version=3, run_name="r")
    assert spec.description == ""
    assert spec.train == {}
    assert spec.slug == "w/p/3"


def test_config_loads_all_three_models():
    specs = load_config(CONFIG)
    names = [s.name for s in specs]
    assert names == ["flood", "landslide", "earthquake"]


def test_config_versions_are_pinned():
    specs = load_config(CONFIG)
    pinned = {s.name: s.version for s in specs}
    assert pinned == {"flood": 2, "landslide": 1, "earthquake": 1}


def test_config_train_args_forwarded():
    specs = load_config(CONFIG)
    for spec in specs:
        assert spec.train.get("epochs") == 50
        assert spec.train.get("imgsz") == 640


def test_config_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")


def test_config_missing_keys_raise(tmp_path: Path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "datasets:\n  - name: flood\n    version: 2\n", encoding="utf-8"
    )  # missing 'project'
    with pytest.raises(ValueError, match="project"):
        load_config(bad)


def test_default_workspace_matches_notebook():
    assert DEFAULT_WORKSPACE == "ashwins-workspace-zz9gf"


# --------------------------------------------------------------------------- #
# resolve_data_yaml
# --------------------------------------------------------------------------- #

def _make_dataset(root: Path, spec: DatasetSpec) -> Path:
    d = root / f"{spec.project}-v{spec.version}"
    (d / "train").mkdir(parents=True, exist_ok=True)
    (d / "data.yaml").write_text("names: [a]\n", encoding="utf-8")
    return d


def test_resolve_data_yaml_finds_pinned_dir(tmp_path: Path):
    spec = DatasetSpec(name="flood", workspace="w", project="proj", version=2, run_name="r")
    _make_dataset(tmp_path, spec)
    assert resolve_data_yaml(spec, tmp_path) == tmp_path / "proj-v2" / "data.yaml"


def test_resolve_data_yaml_falls_back_to_bare_project(tmp_path: Path):
    spec = DatasetSpec(name="flood", workspace="w", project="proj", version=9, run_name="r")
    d = tmp_path / spec.project
    d.mkdir()
    (d / "data.yaml").write_text("names: [a]\n", encoding="utf-8")
    assert resolve_data_yaml(spec, tmp_path) == d / "data.yaml"


def test_resolve_data_yaml_missing_raises(tmp_path: Path):
    spec = DatasetSpec(name="flood", workspace="w", project="proj", version=1, run_name="r")
    with pytest.raises(FileNotFoundError, match="drone_detect datasets download"):
        resolve_data_yaml(spec, tmp_path)


# --------------------------------------------------------------------------- #
# CLI parser (no heavy imports involved)
# --------------------------------------------------------------------------- #

def test_cli_parser_builds(tmp_path: Path):
    from drone_detect.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(["datasets", "list", "--config", str(tmp_path / "x.yaml")])
    assert args.command == "datasets"
    assert args.ds_command == "list"
