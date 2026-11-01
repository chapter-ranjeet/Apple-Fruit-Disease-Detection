# Deployment Guide

## Scope

This application is an experimental YOLO11 image-classification aid. It
classifies a complete uploaded image into four classes; it is not an object
detection system and does not validate four-view recognition.

## Local deployment

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-app.txt
streamlit run app.py
```

The approved checkpoint must exist at:

`runs/appleguard/classification/full_training_v1/weights/best.pt`

The checkpoint is intentionally not committed by Git. Supply it through a
trusted artifact store or a secure deployment volume. Do not include
`dataset_final/`, training runs, or Phase 7 raw prediction artifacts in a
deployment image.

## Hosting recommendation

For a demonstration, use Streamlit Community Cloud or a private VM only after
adding external authentication and supplying the checkpoint as a protected
artifact. The current shared prediction history is suitable for trusted local
use, not an unauthenticated multi-user public service.

For production-like hosting, use a private container or VM with:

- Python 3.11 and the verified dependency set.
- At least 4 GB RAM for CPU inference and model loading.
- Persistent storage only for explicitly approved per-session prediction
  metadata.
- External authentication and authorization.
- Restricted network access and HTTPS termination.
- A protected model artifact mount.

No public deployment was performed in Phase 9.

## CPU/GPU behavior

CPU is the verified default. CUDA is selected only when
`torch.cuda.is_available()` is true and device allocation succeeds. The
current environment reports an NVIDIA GPU through `nvidia-smi` but uses a
CPU-only PyTorch build, so production deployment must plan for CPU unless a
separately validated CUDA environment is installed.

## Operations checklist

1. Install Python 3.11 dependencies.
2. Place the trusted `best.pt` at the approved path.
3. Run `pytest -q`.
4. Start with `streamlit run app.py`.
5. Verify the Streamlit health endpoint.
6. Confirm history storage permissions and retention.
7. Keep the service private until an external access-control boundary is
   configured.
