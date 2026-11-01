"""Configuration, dataset, and readiness helpers for YOLO classification."""

from __future__ import annotations

import importlib.util
import os
import shutil
from pathlib import Path
from typing import Any

import yaml

EXPECTED_CLASSES = ("Blotch_Apple", "Normal_Apple", "Rot_Apple", "Scab_Apple")
SPLITS = ("train", "val", "test")


def load_settings(path: Path = Path("config/settings.yaml")) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        settings = yaml.safe_load(stream)
    if not isinstance(settings, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    return settings


def dataset_root(settings: dict[str, Any], project_root: Path = Path(".")) -> Path:
    return (project_root / settings["paths"]["dataset_root"]).resolve()


def validate_dataset(root: Path, class_names: tuple[str, ...] = EXPECTED_CLASSES,
                     required_splits: tuple[str, ...] = SPLITS) -> dict[str, Any]:
    errors: list[str] = []
    counts: dict[str, dict[str, int]] = {}
    if not root.is_dir():
        errors.append(f"dataset root does not exist: {root}")
    for split in SPLITS:
        split_root = root / split
        if not split_root.is_dir() and split in required_splits:
            errors.append(f"missing split directory: {split_root}")
        counts[split] = {}
        for class_name in class_names:
            class_root = split_root / class_name
            if not class_root.is_dir() and split in required_splits:
                errors.append(f"missing class directory: {class_root}")
                counts[split][class_name] = 0
                continue
            if not class_root.is_dir():
                counts[split][class_name] = 0
                continue
            counts[split][class_name] = sum(
                1 for path in class_root.iterdir() if path.is_file()
            )
            if counts[split][class_name] == 0 and split in required_splits:
                errors.append(f"empty class directory: {class_root}")
    return {
        "root": str(root),
        "classes": list(class_names),
        "counts": counts,
        "total": sum(sum(split.values()) for split in counts.values()),
        "required_splits": list(required_splits),
        "valid": not errors,
        "errors": errors,
    }


def choose_device(configured: str = "auto") -> str:
    if configured not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"Unsupported device: {configured}")
    if configured == "cpu":
        return "cpu"
    try:
        import torch
        cuda_available = bool(torch.cuda.is_available())
    except ImportError:
        cuda_available = False
    if configured == "cuda" and not cuda_available:
        raise RuntimeError("CUDA was requested but is unavailable.")
    return "cuda:0" if cuda_available else "cpu"


def build_train_args(settings: dict[str, Any], run_dir: Path,
                     dataset: Path | None = None) -> dict[str, Any]:
    model = settings["model"]
    paths = settings["paths"]
    return {
        "data": str(dataset or paths["dataset_root"]),
        "task": "classify",
        "imgsz": int(model["image_size"]),
        "epochs": int(model["epochs"]),
        "batch": int(model["batch"]),
        "patience": int(model["patience"]),
        "device": choose_device(str(model.get("device", "auto"))),
        "workers": int(model.get("workers", 0)),
        "seed": int(settings["project"]["seed"]),
        "project": str(run_dir.parent.resolve()),
        "name": run_dir.name,
        "exist_ok": False,
        "pretrained": True,
        "save": True,
        "verbose": True,
    }


def dependency_status() -> dict[str, Any]:
    modules = ("torch", "torchvision", "torchaudio", "ultralytics", "yaml")
    available = {name: importlib.util.find_spec(name) is not None for name in modules}
    python = os.sys.version_info
    return {
        "python": f"{python.major}.{python.minor}.{python.micro}",
        "python_311": python[:2] == (3, 11),
        "modules": available,
        "pip_available": shutil.which("pip") is not None,
    }
