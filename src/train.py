"""YOLO11 classification training pipeline.

This module does not train when imported. Use ``--check-only`` for the
readiness gate; training requires an explicit ``--train`` argument.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from .utils.training_utils import (
        EXPECTED_CLASSES, build_train_args, dataset_root, dependency_status,
        load_settings, validate_dataset,
    )
except ImportError:
    from utils.training_utils import (
        EXPECTED_CLASSES, build_train_args, dataset_root, dependency_status,
        load_settings, validate_dataset,
    )


def readiness_report(settings_path: Path = Path("config/settings.yaml")) -> dict[str, Any]:
    project_root = settings_path.resolve().parent.parent
    settings = load_settings(settings_path)
    dependencies = dependency_status()
    root = dataset_root(settings, project_root)
    dataset = validate_dataset(
        root,
        tuple(settings["model"]["class_names"]),
        tuple(settings.get("required_splits", ("train", "val", "test"))),
    )
    model_path = Path(settings["model"]["baseline"])
    model_available = model_path.is_file() or bool(shutil.which("yolo"))
    errors = list(dataset["errors"])
    if not dependencies["python_311"]:
        errors.append("Python 3.11 is required.")
    if not all(dependencies["modules"].get(name, False) for name in ("torch", "ultralytics")):
        errors.append("PyTorch and Ultralytics must be installed.")
    if settings["model"]["task"] != "classify":
        errors.append("model.task must be classify.")
    if settings["model"]["classes"] != len(EXPECTED_CLASSES):
        errors.append("model.classes does not equal four.")
    if tuple(settings["model"]["class_names"]) != EXPECTED_CLASSES:
        errors.append("class mapping does not match the approved classes.")
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset,
        "dependencies": dependencies,
        "model": {
            "task": settings["model"]["task"],
            "checkpoint": settings["model"]["baseline"],
            "available_for_loading": model_available,
        },
        "configuration": settings["model"],
        "ready": not errors,
        "errors": errors,
    }


def run_training(settings_path: Path = Path("config/settings.yaml"),
                 run_dir: Path = Path("runs/appleguard/classification")) -> dict[str, Any]:
    gate = readiness_report(settings_path)
    if not gate["ready"]:
        raise RuntimeError("Training readiness gate failed: " + "; ".join(gate["errors"]))
    from ultralytics import YOLO
    settings = load_settings(settings_path)
    args = build_train_args(settings, run_dir, dataset_root(settings, settings_path.parent.parent))
    model = YOLO(settings["model"]["baseline"])
    result = model.train(**args)
    summary = {"run_dir": str(run_dir), "arguments": args, "result": str(result)}
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true", help="Run readiness checks without training.")
    parser.add_argument("--train", action="store_true", help="Explicitly start training.")
    parser.add_argument("--settings", default=Path("config/settings.yaml"), type=Path)
    parser.add_argument("--run-dir", default=Path("runs/appleguard/classification"), type=Path)
    args = parser.parse_args()
    if not args.check_only and not args.train:
        parser.error("Choose --check-only or --train.")
    gate = readiness_report(args.settings)
    print(json.dumps(gate, indent=2))
    if args.train:
        print(json.dumps(run_training(args.settings, args.run_dir), indent=2))
    return 0 if gate["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
