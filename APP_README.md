# Apple-Fruit-Disease-Detection Application

Phase 11 provides a responsive light-theme dashboard and an explainable
prediction experience. The application remains a four-class YOLO11 image
classifier, not an object-detection system.

Phase 11.1 adds a premium application shell with Disease Library, Model
Performance, and About Project pages, richer dashboard cards, a CSS agricultural
illustration, responsive result panels, and a 10 MB Streamlit upload boundary.

## Setup

Use Python 3.11 and the existing virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-app.txt
```

The application uses the approved checkpoint at
`runs/appleguard/classification/full_training_v1/weights/best.pt`.

## Launch

```powershell
streamlit run app.py
```

Upload a JPG, JPEG, or PNG from **Disease Prediction**, then select **Analyze
image**. The app validates the file, displays it, runs YOLO11 classification
at 224 pixels, and shows the predicted class, confidence band, actual device,
explanation, and all four probabilities sorted from highest to lowest.
Uploaded images are not stored permanently; only prediction metadata is saved
to a per-Streamlit-session SQLite file under `results/phase9/history/`.

Confidence bands are presentation labels only: high (85% or more), moderate
(65% to below 85%), and low (below 65%). They are not calibrated diagnostic
thresholds. Explanations describe the relative model outputs and do not claim
to identify visual causes.

## CPU and GPU

CPU is the safe default and is used automatically when CUDA is unavailable or
cannot initialize. If a CUDA-enabled PyTorch installation is available, the
sidebar may offer `cuda:0`. The app does not install drivers or force CUDA.

## Troubleshooting

- **Missing checkpoint:** confirm the Phase 6 `best.pt` path exists.
- **Unsupported image:** use JPG, JPEG, or PNG below 10 MB.
- **Model loading failure:** activate `.venv` and verify PyTorch and
  Ultralytics installation.
- **CUDA failure or memory error:** select CPU and retry.
- **History:** use **Prediction History → Clear history** to remove local
  prediction metadata.

This is an image-classification application, not an object-detection system.
It does not produce disease-region bounding boxes or validate four-view
recognition of the same physical apple. Predictions are informational and do
not replace qualified agricultural advice.
