# Apple-Fruit-Disease-Detection

This project includes a YOLO11 **image-classification** application for four
apple categories: Apple Blotch, Healthy Apple, Apple Rot, and Apple Scab. The
Streamlit interface provides a responsive light-theme dashboard, safe image
upload, CPU/CUDA inference, prediction history, and model-output-based
explanations.

The application does not perform object detection, disease-region
localization, severity measurement, or validated four-view recognition.
Confidence is a relative model score, not diagnostic certainty.

YOLO11 image-classification project for categorizing apple images. The project
is designed for a B.Tech Data Science final-year submission and is being
developed phase by phase.

## Current status

**Phase 11 - UI redesign and explainable prediction experience: complete with warnings**

The supplied dataset was prepared read-only from
`C:\Users\ACER\Downloads\datasets\ds`. It is a folder-based classification
dataset, not a YOLO object-detection dataset: it contains 5,566 readable
images in four class folders and no bounding-box annotation files. A derived,
exact-duplicate-safe classification dataset is available under
`dataset_processed/`, with 3,897 train, 835 validation, and 834 test images.
The label conflict audit found 290 exact-image groups with conflicting source
class labels. All 580 affected images were quarantined without modifying the
source. The rebuilt derived dataset contains 4,986 images and is ready for a
single-image classification baseline with warnings, subject to review of
near-duplicates and quality flags.
The approved Phase 6 checkpoint is used without retraining. Phase 7 held-out
evaluation, Phase 9 hardening, and Phase 10 CUDA verification are documented
under `results/`. See `APP_README.md` and
`results/phase11/PHASE11_UI_REDESIGN_REPORT.md` for the current application
workflow.

## Project structure

```text
Apple-Fruit-Diseases-Detection/
├── app/                    # Streamlit application (later phase)
├── config/settings.yaml    # Paths and model settings
├── dataset/                # Local dataset; ignored by Git
│   ├── images/{train,val,test}/
│   └── labels/{train,val,test}/
├── models/                 # Local model weights; weights are ignored
├── notebooks/              # Analysis and experiment notebooks
├── results/                # Generated reports and plots; ignored
├── src/                    # Analysis, preprocessing, training, evaluation
├── data.yaml               # Ultralytics dataset configuration template
├── requirements.txt
└── PROJECT_PROGRESS.md
```

## Supported dataset layout

Place a YOLO-format dataset under `dataset/` without modifying the original
source files:

```text
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

Each label line must contain:

```text
class_id x_center y_center width height
```

Coordinates must be normalized to `[0, 1]`. If the provided dataset uses
another annotation format, Phase 2 will identify it and propose a conversion
without changing the original files.

## Installation

Python **3.11** is the supported runtime. A Python 3.14.7 interpreter is
currently available on the development machine, but the compatibility of the
deep-learning stack with Python 3.14 is not assumed.

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install PyTorch using the command recommended for the host at
<https://pytorch.org/get-started/locally/>, then install the remaining
packages:

```powershell
python -m pip install -r requirements.txt
```

Verify the environment:

```powershell
python -c "import torch, ultralytics; print('torch', torch.__version__); print('cuda available', torch.cuda.is_available()); print('ultralytics', ultralytics.__version__)"
```

## Planned workflow

1. Obtain reviewed bounding-box annotations or change the project scope to image classification.
2. Resolve the documented cross-split duplicate leakage risks.
3. Produce leakage-safe detection splits without altering original data.
4. Train YOLO11n, with YOLO11s as an optional comparison.
5. Evaluate validation and final test metrics separately.
6. Add image prediction and the Apple-Fruit-Disease-Detection Streamlit interface.
7. Document tests, limitations, and reproducible results.

Training must not begin until dataset inspection and annotation validation are
complete.

## License and data

Dataset licensing and attribution must be documented after the dataset source
is provided. Do not commit private datasets, credentials, virtual
environments, caches, or large model weights.
