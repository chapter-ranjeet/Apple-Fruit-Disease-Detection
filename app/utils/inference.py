from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image

CLASS_NAMES = ("Blotch_Apple", "Normal_Apple", "Rot_Apple", "Scab_Apple")
DISPLAY_NAMES = {
    "Blotch_Apple": "Apple Blotch",
    "Normal_Apple": "Healthy Apple",
    "Rot_Apple": "Apple Rot",
    "Scab_Apple": "Apple Scab",
}


def available_devices() -> list[str]:
    return ["cuda:0", "cpu"] if torch.cuda.is_available() else ["cpu"]


def resolve_device(preferred: str = "cpu") -> str:
    if preferred.startswith("cuda"):
        if not torch.cuda.is_available():
            return "cpu"
        try:
            torch.empty(1, device=preferred)
            return preferred
        except (RuntimeError, AssertionError):
            return "cpu"
    return "cpu"


def load_model(checkpoint: Path, device: str = "cpu") -> Any:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint}")
    try:
        from ultralytics import YOLO
        model = YOLO(str(checkpoint))
        names = tuple(model.names[index] for index in range(len(model.names)))
        if names != CLASS_NAMES:
            raise ValueError(f"Unexpected checkpoint class mapping: {names}")
        model.to(resolve_device(device))
        return model
    except (OSError, RuntimeError, ValueError, ImportError) as exc:
        raise RuntimeError(f"Unable to load classification checkpoint: {exc}") from exc


def predict_image(model: Any, image: Image.Image, device: str = "cpu") -> dict[str, Any]:
    try:
        actual_device = resolve_device(device)
        with torch.inference_mode():
            result = model.predict(
                source=image,
                imgsz=224,
                device=actual_device,
                verbose=False,
                save=False,
            )[0]
        probabilities = result.probs.data.detach().cpu().numpy().astype(float).tolist()
        if len(probabilities) != len(CLASS_NAMES):
            raise ValueError("Model returned an invalid probability vector.")
        total = sum(probabilities)
        if total <= 0:
            raise ValueError("Model returned invalid probabilities.")
        probabilities = [value / total for value in probabilities]
        predicted_index = max(range(len(probabilities)), key=probabilities.__getitem__)
        predicted_class = CLASS_NAMES[predicted_index]
        return {
            "class_name": predicted_class,
            "display_name": DISPLAY_NAMES[predicted_class],
            "confidence": probabilities[predicted_index],
            "probabilities": dict(zip(CLASS_NAMES, probabilities)),
            "device": actual_device,
        }
    except (AttributeError, IndexError, RuntimeError, TypeError, ValueError) as exc:
        if device.startswith("cuda"):
            try:
                model.to("cpu")
                return predict_image(model, image, "cpu")
            except (AttributeError, RuntimeError, TypeError, ValueError) as fallback_exc:
                raise RuntimeError(f"Inference failed on CUDA and CPU fallback: {fallback_exc}") from exc
        raise RuntimeError(f"Inference failed: {exc}") from exc
