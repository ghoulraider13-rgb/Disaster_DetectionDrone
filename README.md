<div align="center">

# 🚁 Disaster Detection Drone

**YOLOv11-based aerial detection of floods, landslides, and earthquake damage for UAV search-and-rescue.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![YOLOv11](https://img.shields.io/badge/model-YOLOv11s-00B0FF?logo=ultralytics)](https://docs.ultralytics.com/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Datasets](https://img.shields.io/badge/datasets-Roboflow-9C27B0?logo=roboflow)](configs/datasets.yaml)

</div>

## 📖 Overview

During disaster response, the first hours — the *golden hour* — decide who survives. Drones equipped
with onboard computer vision can survey dangerous or inaccessible terrain far faster than ground
teams, but only if detection runs in real time at the edge. This project fine-tunes compact
**YOLOv11n** models on aerial disaster imagery so they can run directly on drone companion
computers (Jetson-class hardware) instead of streaming video to a human operator.

Three models are trained, one per disaster type:

| Model | Task | Source dataset |
|---|---|---|
| `flood_model_v2` | Flood extent + survivor detection on flooded terrain | [AI disaster survival drone v2](https://universe.roboflow.com/ashwins-workspace-zz9gf/ai-disaster-survival-drone) |
| `landslide_model_v2` | Landslide / slope-failure detection from UAV imagery | [AI disaster drone - Landslide v1](https://universe.roboflow.com/ashwins-workspace-zz9gf/ai-disaster-drone-landslide) |
| `earthquake_model_v2` | Rubble and structural damage assessment | [AI disaster drone - Earthquakes v1](https://universe.roboflow.com/ashwins-workspace-zz9gf/ai-disaster-drone-earthquakes) |

All models start from COCO-pretrained `yolo11s.pt` (small variant: ~11M params, ~137 GFLOPs) — the
right size/accuracy trade-off for onboard inference on disaster drones where small objects
(survivors, rubble) need higher resolution (1280×1280) and model capacity.

## 🗂️ Project structure

```
Disaster_DetectionDrone/
├── configs/
│   └── datasets.yaml        # Pinned Roboflow dataset versions + per-model hyperparameters
├── drone_detect/
│   ├── __init__.py          # Package metadata
│   ├── __main__.py          # python -m drone_detect entry point
│   ├── cli.py               # CLI: datasets / train / export / predict
│   ├── datasets.py          # Roboflow download helpers + config loader
│   ├── train.py             # Fine-tuning loop with reproducible defaults
│   ├── export.py            # ONNX / TensorRT / OpenVINO export for edge devices
│   └── inference.py         # Prediction helpers returning structured detections
├── notebooks/
│   └── training_pipeline.ipynb   # Kaggle-compatible end-to-end walkthrough
├── weights/                 # Trained weights (download from Releases, not committed)
├── tests/
│   └── test_config.py       # Config/schema unit tests
├── requirements.txt
├── .env.example             # Template for ROBOFLOW_API_KEY
└── README.md
```

## 🚀 Quickstart

### 1. Install

```bash
git clone https://github.com/ghoulraider13-rgb/Disaster_DetectionDrone.git
cd Disaster_DetectionDrone
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure the Roboflow key

Datasets are fetched from Roboflow, which requires a (free) API key:

```bash
cp .env.example .env                  # then edit .env: ROBOFLOW_API_KEY=...
export ROBOFLOW_API_KEY=your_key      # or set it in your shell
```

The key is **never** hardcoded in the code — it is read from the environment.

### 3. Train all three models

```bash
python -m drone_detect datasets list               # inspect the pinned datasets
python -m drone_detect train                       # downloads data, then trains all models
python -m drone_detect train --only flood --device cpu   # single model, CPU fallback
```

Training defaults (per model): 150 epochs, 1280×1280 images, AdamW optimizer with cosine LR,
early stopping with `patience=30`, fixed `seed=42` for reproducibility, batch size 16,
and enhanced augmentation (mosaic, mixup, copy_paste) for small aerial objects.
Override anything per-dataset in [`configs/datasets.yaml`](configs/datasets.yaml) or on the
command line (`--epochs`, `--device`, `--model`).

Each run writes to `runs/drone_models/<name>/` with `weights/best.pt`, confusion matrix,
`results.png`, and validation predictions.

### 4. Export for onboard deployment

```bash
python -m drone_detect export \
    --weights runs/drone_models/flood_model/weights/best.pt \
    --format onnx          # or: engine (Jetson TensorRT), openvino, torchscript
```

### 5. Run inference

```bash
python -m drone_detect predict \
    --weights runs/drone_models/flood_model/weights/best.pt \
    --source mission_footage.mp4 --conf 0.25
```

Prints one line per detection (`class`, confidence, bounding box) and saves annotated media.

## 📓 Notebooks

[`notebooks/training_pipeline.ipynb`](notebooks/training_pipeline.ipynb) reproduces the full
pipeline (download → train → visualize confusion matrices) and runs as-is on Kaggle/Colab GPUs.
It is the cleaned-up version of the original research notebook: no hardcoded credentials,
no duplicated cells, and corrected output paths.

## ⚙️ Reproducibility & design choices

- **Pinned dataset versions** — each Roboflow dataset is pinned to an exact version so a rerun
  in six months trains on the same data.
- **Fixed seed + early stopping** — the original notebook trained blind for a fixed 50 epochs;
  the pipeline now stops when validation mAP plateaus and is repeatable.
- **Nano models only** — disaster drones carry Jetson-class companion computers; the paper this
  project accompanies shows YOLOv11n keeps accuracy within ~1–2 mAP of larger variants at a
  fraction of the latency.
- **CLI-first** — everything is scriptable for CI and fleet training runs; the notebook is a
  convenience, not the interface.

## 📊 Results

Training runs (confusion matrices, PR curves, per-class mAP) are generated under
`runs/drone_models/<model>/` on every train. Baseline comparison against classical segmentation
(k-means, region growing) is reported in the accompanying paper — YOLOv11n outperforms both by
9–20 accuracy points on flood detection, with recall gains mattering most for SAR.

### Current Model Status (All Trained)

| Model | Classes | Weights Location |
|-------|---------|------------------|
| `flood_model_v2` | Flooded regions, Pool | `runs/drone_models/flood_model_v2/weights/best.pt` |
| `landslide_model_v2` | landslide, landslide- | `runs/drone_models/landslide_model_v2/weights/best.pt` |
| `earthquake_model_v2` | Buildings-Collapsed, Buildings-Damaged, Road-Damaged | `runs/drone_models/earthquake_model_v2/weights/best.pt` |

All models trained on YOLOv11s (COCO-pretrained) at 1280×1280 resolution with enhanced augmentation for small aerial objects. Ready for export to ONNX/TensorRT/OpenVINO for edge deployment on Jetson-class hardware.

## 🛣️ Roadmap

- [x] Complete all three model training runs (flood, landslide, earthquake)
- [ ] ONNX/TensorRT benchmark table on Jetson Orin Nano
- [ ] Temporal smoothing across video frames to fight occlusion-driven misses
- [ ] Thermal-visual fusion for survivor detection at night
- [ ] Swarm mode: multi-drone detection merging

## 🤝 Contributing

Issues and PRs welcome. For dataset changes, bump the version in `configs/datasets.yaml` and
note the delta in the PR description.

## 📄 License

[MIT](LICENSE) — trained weights included in Releases are provided under the same license,
subject to the source datasets' terms on Roboflow Universe.

---

<div align="center">
Built for faster SAR response. Every minute saved is a life.
</div>
