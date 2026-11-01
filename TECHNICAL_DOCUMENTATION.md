# Technical Documentation

## Architecture

`app.py` provides Streamlit pages and session-level prediction caching.
`app/utils/validation.py` bounds and decodes uploaded images.
`app/utils/inference.py` loads the trusted Ultralytics YOLO11 classification
checkpoint, validates class ordering, selects CPU/CUDA safely, and returns
normalized probabilities. `app/utils/history.py` stores sanitized prediction
metadata in a per-session SQLite database with a unique prediction signature.

The model uses its Ultralytics preprocessing pipeline with image size 224.
The approved checkpoint is Phase 6 `best.pt`; Phase 7 metrics are static
held-out evaluation results and are not recomputed during app use.

## Environment

Python 3.11 is supported. Install `requirements-app.txt` inside `.venv`.
CPU PyTorch is verified. CUDA is conditional and currently unavailable to
PyTorch despite an NVIDIA driver being visible to `nvidia-smi`.

## Error handling and security

Uploads are limited to 10 MB, restricted to JPG/JPEG/PNG extensions, decoded
with Pillow, checked for malformed/truncated content, and limited by pixel
count. Uploaded names are sanitized and never used as filesystem paths.
SQLite parameter binding, a unique signature constraint, and sanitized
filenames protect history writes. User-facing model errors are intentionally
bounded.

## Data and model boundaries

The application does not read `dataset_final` for predictions and does not
modify datasets, training outputs, or Phase 7 evaluation results. It is a
whole-image classifier, not an object detector. Four-view same-apple
recognition is not validated.
