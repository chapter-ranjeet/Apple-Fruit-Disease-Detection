# GPU Setup Guide

## Hardware verified

- NVIDIA GeForce RTX 3050 Laptop GPU
- 4 GB VRAM
- NVIDIA driver 591.97
- Driver-reported CUDA compatibility: 13.1

## Separate environment

The CPU environment `.venv` is preserved. Create/use `.venv-gpu` with Python
3.11:

```powershell
py -3.11 -m venv .venv-gpu
.\.venv-gpu\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements-gpu.txt
```

The verified package set is PyTorch 2.14.1+cu126, torchvision 0.29.1+cu126,
torchaudio 2.11.0+cu126, and Ultralytics 8.4.197. The official PyTorch CUDA
wheels provide their runtime dependencies; a separate CUDA Toolkit is not
required for this inference setup.

Verify:

```powershell
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable')"
python -m pip check
```

## Run the application

GPU environment:

```powershell
.\.venv-gpu\Scripts\Activate.ps1
streamlit run app.py
```

CPU environment:

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

The app offers `cuda:0` only when PyTorch reports CUDA availability. If CUDA
initialization or inference fails, the current prediction falls back to CPU
and displays the actual device used.

Do not retrain the model, install arbitrary CUDA Toolkit versions, or expose
the application publicly without deployment security controls.
